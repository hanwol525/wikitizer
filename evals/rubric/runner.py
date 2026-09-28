"""Orchestration: parse a WOF, run the all-model rubric checks (or emit every criterion
``skipped`` when the model is off/unavailable), and build the ``RubricResult``.

Error policy mirrors the pipeline's "degrade toward less-complete, never corrupt": a
request-time error on one check emits ``skipped`` items for exactly that check's criteria +
a ``[REVIEW]`` log, never aborting. Because ``summary.complete`` is DERIVED from the items
(``skipped == 0``), any such degrade surfaces as ``complete=false`` -- never a silent "full"
run.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from evals.common import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TEMPERATURE
from evals.common.models import Engine, Evidence, Item, Status
from evals.common.parse import parse_wof
from evals.rubric import ENTITY_CATEGORIES
from evals.rubric.adjudicator import RubricAdjudicator
from evals.rubric.emit import build_result
from evals.rubric.models import RubricResult
from evals.rubric.resolve import (
    cited_quotes,
    entity_slug,
    entry_candidates,
    section_candidates,
    section_key,
)

logger = logging.getLogger("evals.rubric.runner")


# --- criterion enumeration (drives the --no-model path AND per-check degrade slices) --- #

def _section_criteria(rubric) -> List[Tuple[str, str]]:
    return [(f"rubric.sections.{entity_slug(r.name)}", f"section {r.name!r} present")
            for r in rubric.sections]


def _presence_criteria(rubric, attr, token) -> List[Tuple[str, str]]:
    return [(f"rubric.presence.{token}.{entity_slug(r.name)}", f"{r.name!r} present under {token}")
            for r in getattr(rubric, attr)]


def _sourcing_criteria(rubric, attr, token) -> List[Tuple[str, str]]:
    return [(f"rubric.sourcing.{token}.{entity_slug(r.name)}", f"{r.name!r} sourcing under {token}")
            for r in getattr(rubric, attr)]


def _pc_criteria(rubric) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for r in rubric.characters:
        if r.pc:
            slug = entity_slug(r.name)
            out.append((f"rubric.pc-stated.{slug}", f"{r.name!r} PC status stated"))
            out.append((f"rubric.pc-marked.{slug}", f"{r.name!r} marked as PC (trailing *)"))
    return out


def _enumerate_criteria(rubric) -> List[Tuple[str, str]]:
    crit = list(_section_criteria(rubric))
    for attr, token, _section in ENTITY_CATEGORIES:
        crit += _presence_criteria(rubric, attr, token)
    crit += _pc_criteria(rubric)
    for attr, token, _section in ENTITY_CATEGORIES:
        crit += _sourcing_criteria(rubric, attr, token)
    return crit


def _skipped_item(cid: str, desc: str, detail: str = "model disabled or unavailable") -> Item:
    return Item(id=cid, description=desc, engine=Engine.MODEL, status=Status.SKIPPED,
                evidence=Evidence(lines=[], detail=detail))


def _na_item(cid: str, desc: str, detail: str) -> Item:
    return Item(id=cid, description=desc, engine=Engine.MODEL, status=Status.NA,
                evidence=Evidence(lines=[], detail=detail))


def _model_cfg(model_client, votes: int) -> dict:
    # Always populated (the rubric's single grader is a ModelGrader). When there's no client
    # (the --no-model path) the defaults stand in, so the grader entry is still valid.
    return {
        "name": getattr(model_client, "model", None) or DEFAULT_MODEL,
        "provider": getattr(model_client, "provider", None) or DEFAULT_PROVIDER,
        "temperature": float(getattr(model_client, "temperature", DEFAULT_TEMPERATURE)),
        "thinking": bool(getattr(model_client, "thinking", False)),
        "votes": max(1, int(votes)),
        "endpoint": getattr(model_client, "base_url", None),
    }


def _entity_keys(ent) -> List[str]:
    """Normalized join keys (name + aliases) for matching a category's section name."""
    keys = [section_key(ent.name)] + [section_key(a) for a in ent.aliases]
    return [k for k in keys if k]


def grade(wof_text: str, wof_name: str, rubric, votes: int = 3, use_model: bool = True,
          model_client=None, now: Optional[datetime] = None) -> RubricResult:
    now = now or datetime.now(timezone.utc)
    parsed = parse_wof(wof_text)
    model_cfg = _model_cfg(model_client, votes)

    # --- no-model: every criterion skipped (complete derives to False) ------ #
    if not use_model or model_client is None:
        items = [_skipped_item(cid, desc) for cid, desc in _enumerate_criteria(rubric)]
        items.sort(key=lambda it: it.id)
        return build_result(wof_name, items, now, model_cfg)

    adj = RubricAdjudicator(model_client, votes=votes)
    items: List[Item] = []
    wof_titles = section_candidates(parsed)

    # 1. sections
    sections_degraded = False
    try:
        section_items, section_map = adj.judge_sections(rubric.sections, wof_titles)
    except Exception as exc:                          # noqa: BLE001 -- degrade, never abort
        logger.warning("[REVIEW] rubric sections check errored (%s); skipping", exc)
        section_items = [_skipped_item(cid, desc, f"[REVIEW] check errored: {exc}")
                         for cid, desc in _section_criteria(rubric)]
        section_map = [None] * len(rubric.sections)
        sections_degraded = True
    items += section_items

    # normalized rubric-section name/alias -> its index (for the category->section join)
    sec_index_by_key = {}
    for idx, r in enumerate(rubric.sections):
        for key in _entity_keys(r):
            sec_index_by_key.setdefault(key, idx)

    # 2. presence per category (scoped by the section map, joined by normalized name)
    entity_entries = {}                               # (attr, k) -> Entry or None
    for attr, token, section_name in ENTITY_CATEGORIES:
        required = getattr(rubric, attr)
        if not required:
            continue
        if sections_degraded:
            # The sections judge errored (an infra failure), so we don't KNOW which WOF section
            # holds this category -- skip presence (excluded from the denominator) rather than
            # FAIL it as "section absent" (a content verdict it was never evaluated for).
            items += [_skipped_item(cid, desc, "[REVIEW] section check errored; presence not evaluated")
                      for cid, desc in _presence_criteria(rubric, attr, token)]
            for k in range(len(required)):
                entity_entries[(attr, k)] = None
            continue
        sec_idx = sec_index_by_key.get(section_key(section_name))
        wof_idx = section_map[sec_idx] if (sec_idx is not None and sec_idx < len(section_map)) else None
        if sec_idx is None:
            logger.warning("[REVIEW] rubric: category %s has required entities but no matching "
                           "required section (%r); presence will FAIL as 'section absent'",
                           attr, section_name)
        entries = entry_candidates(parsed, wof_idx)
        try:
            pres_items, emap = adj.judge_presence(token, required, entries)
        except Exception as exc:                      # noqa: BLE001
            logger.warning("[REVIEW] rubric presence %s errored (%s); skipping", token, exc)
            pres_items = [_skipped_item(cid, desc, f"[REVIEW] check errored: {exc}")
                          for cid, desc in _presence_criteria(rubric, attr, token)]
            emap = [None] * len(required)
        items += pres_items
        for k in range(len(required)):
            entity_entries[(attr, k)] = emap[k] if k < len(emap) else None

    # 3. PCs: present ones get judged together; absent/unmapped ones -> na (both items)
    present_pcs = [(r, entity_entries[("characters", k)])
                   for k, r in enumerate(rubric.characters)
                   if r.pc and entity_entries.get(("characters", k)) is not None]
    try:
        pc_items = adj.judge_pcs(present_pcs)
    except Exception as exc:                          # noqa: BLE001
        logger.warning("[REVIEW] rubric pcs check errored (%s); skipping", exc)
        pc_items = []
        for r, entry in present_pcs:
            slug = entity_slug(r.name)
            pc_items.append(_skipped_item(f"rubric.pc-stated.{slug}", f"{r.name!r} PC status stated",
                                          f"[REVIEW] check errored: {exc}"))
            pc_items.append(_skipped_item(f"rubric.pc-marked.{slug}", f"{r.name!r} marked as PC (trailing *)",
                                          f"[REVIEW] check errored: {exc}"))
    items += pc_items
    for k, r in enumerate(rubric.characters):
        if r.pc and entity_entries.get(("characters", k)) is None:
            slug = entity_slug(r.name)
            items.append(_na_item(f"rubric.pc-stated.{slug}", f"{r.name!r} PC status stated", "character absent"))
            items.append(_na_item(f"rubric.pc-marked.{slug}", f"{r.name!r} marked as PC (trailing *)", "character absent"))

    # 4. sourcing per present entity (gates: absent -> na; cites-nothing -> FAIL no-call)
    for attr, token, _section in ENTITY_CATEGORIES:
        required = getattr(rubric, attr)
        for k, r in enumerate(required):
            cid = f"rubric.sourcing.{token}.{entity_slug(r.name)}"
            desc = f"{r.name!r} sourcing under {token}"
            entry = entity_entries.get((attr, k))
            if entry is None:
                items.append(_na_item(cid, desc, "entity absent (presence failed); sourcing not applicable"))
                continue
            quotes = cited_quotes(parsed, entry)
            if not quotes:
                items.append(Item(id=cid, description=desc, engine=Engine.MODEL, status=Status.FAIL,
                                  evidence=Evidence(lines=[entry.lineno], detail="entry cites no footnote")))
                continue
            try:
                items.append(adj.judge_support(r, token, entry, quotes))
            except Exception as exc:                  # noqa: BLE001
                logger.warning("[REVIEW] rubric sourcing %s %r errored (%s); skipping", token, r.name, exc)
                items.append(_skipped_item(cid, desc, f"[REVIEW] check errored: {exc}"))

    items.sort(key=lambda it: it.id)
    return build_result(wof_name, items, now, model_cfg)


def grade_file(path, rubric, votes: int = 3, use_model: bool = True, model_client=None,
               now: Optional[datetime] = None) -> RubricResult:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return grade(text, p.name, rubric, votes=votes, use_model=use_model,
                 model_client=model_client, now=now)
