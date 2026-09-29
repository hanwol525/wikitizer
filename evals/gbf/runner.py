"""Orchestration: parse the WOF, pair it against the gold, run the all-model GBF checks (or emit
every criterion ``skipped`` when the model is off/unavailable), and build the ``GbfResult``.

Error policy mirrors the pipeline's + rubric's "degrade toward less-complete, never corrupt": a
request-time error on one faithfulness call skips just that item; a pairing failure skips the still-
unmatched faithfulness items AND ordering (no reliable map). Because ``summary.complete`` is DERIVED
from the items (``skipped == 0``), any degrade surfaces as ``complete=false`` -- never a silent
"full" run.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from evals.common import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TEMPERATURE
from evals.common.models import Engine, Evidence, Item, Status
from evals.common.parse import ParsedWOF, parse_wof
from evals.gbf import DEFAULT_GBF_VOTES, DEFAULT_MAX_WORKERS
from evals.gbf.adjudicator import GbfAdjudicator
from evals.gbf.emit import build_result
from evals.gbf.models import GbfResult
from evals.gbf.ordering import score_ordering
from evals.gbf.pair import gold_slug, pair, real_entries

logger = logging.getLogger("evals.gbf.runner")

_ORDERING_ID = "gbf.ordering"
_ORDERING_DESC = "History events in chronological order"


def _faith_id(slug: str) -> str:
    return f"gbf.faithfulness.{slug}"


def _faith_desc(name: str) -> str:
    return f"{name!r} faithful in meaning to the gold"


def _enumerate_criteria(gold: ParsedWOF) -> List[Tuple[str, str]]:
    """Every criterion GBF would emit for this gold: one ``gbf.faithfulness.<slug>`` per real gold
    entry (the universe -- the gold defines what could be graded) + the single ``gbf.ordering``.
    Drives the ``--no-model`` path AND a total-pairing-failure degrade."""
    crit = [(_faith_id(gold_slug(e)), _faith_desc(e.name)) for e in real_entries(gold)]
    crit.append((_ORDERING_ID, _ORDERING_DESC))
    return crit


def _skipped_item(cid: str, desc: str, detail: str = "model disabled or unavailable") -> Item:
    return Item(id=cid, description=desc, engine=Engine.MODEL, status=Status.SKIPPED,
                evidence=Evidence(lines=[], detail=detail))


def _na_item(cid: str, desc: str, detail: str) -> Item:
    return Item(id=cid, description=desc, engine=Engine.MODEL, status=Status.NA,
                evidence=Evidence(lines=[], detail=detail))


def _model_cfg(model_client, votes: int) -> dict:
    # Always populated (GBF's single grader is a ModelGrader). When there's no client (the --no-model
    # path) the defaults stand in, so the grader entry is still schema-valid.
    return {
        "name": getattr(model_client, "model", None) or DEFAULT_MODEL,
        "provider": getattr(model_client, "provider", None) or DEFAULT_PROVIDER,
        "temperature": float(getattr(model_client, "temperature", DEFAULT_TEMPERATURE)),
        "thinking": bool(getattr(model_client, "thinking", False)),
        "votes": max(1, int(votes)),
        "endpoint": getattr(model_client, "base_url", None),
    }


def _judge_one(adj, cid: str, desc: str, ge, wof_entry) -> Item:
    """Judge one matched pair, catching any error into a ``skipped`` item (the same per-item degrade
    the sequential loop used). It NEVER raises, so ``future.result()`` can't abort the run."""
    try:
        return adj.judge_faithfulness(ge, wof_entry)
    except Exception as exc:                            # noqa: BLE001 -- degrade, never abort
        logger.warning("[REVIEW] gbf faithfulness %r errored (%s); skipping", ge.name, exc)
        return _skipped_item(cid, desc, f"[REVIEW] check errored: {exc}")


def _run_faithfulness(adj, tasks: List[Tuple[str, str, object, object]], max_workers: int) -> List[Item]:
    """Judge the matched (cid, desc, gold_entry, wof_entry) tasks. Sequential when ``max_workers <= 1``
    or there's <=1 task (a pool-free deterministic path); otherwise the independent, network-bound
    calls fan out over a ThreadPoolExecutor (the same basis as the extractor fan-out in
    orchestrator.py: ``GbfAdjudicator`` is stateless per call and the model client is thread-safe).
    Ordering is irrelevant -- the caller sorts all items by id."""
    if not tasks:
        return []
    if max_workers <= 1 or len(tasks) == 1:
        return [_judge_one(adj, *t) for t in tasks]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = [pool.submit(_judge_one, adj, *t) for t in tasks]
        return [f.result() for f in futures]           # _judge_one already caught its errors


def grade(wof_text: str, wof_name: str, gold: ParsedWOF, ranks, votes: int = DEFAULT_GBF_VOTES,
          use_model: bool = True, model_client=None, now: Optional[datetime] = None,
          max_workers: int = DEFAULT_MAX_WORKERS) -> GbfResult:
    now = now or datetime.now(timezone.utc)
    parsed_wof = parse_wof(wof_text)
    model_cfg = _model_cfg(model_client, votes)

    # --- no-model: every criterion skipped (complete derives to False) ------ #
    if not use_model or model_client is None:
        items = [_skipped_item(cid, desc) for cid, desc in _enumerate_criteria(gold)]
        items.sort(key=lambda it: it.id)
        return build_result(wof_name, items, now, model_cfg)

    adj = GbfAdjudicator(model_client, votes=votes)

    # Pairing is its own pass. ``pair`` guards its own model call (residual), but wrap defensively so
    # an unexpected error still degrades (every criterion skipped) instead of aborting.
    try:
        pm = pair(parsed_wof, gold, adj)
    except Exception as exc:                           # noqa: BLE001 -- degrade, never abort
        logger.warning("[REVIEW] gbf pairing errored (%s); skipping every check", exc)
        items = [_skipped_item(cid, desc, f"[REVIEW] pairing errored: {exc}")
                 for cid, desc in _enumerate_criteria(gold)]
        items.sort(key=lambda it: it.id)
        return build_result(wof_name, items, now, model_cfg)

    residual_set = set(pm.residual_indices)
    items: List[Item] = []

    # --- faithfulness per gold entry --------------------------------------- #
    # na/skipped items are cheap (no network) and built inline; the matched entries are the
    # independent, network-bound calls, collected into a task list and fanned out (see
    # _run_faithfulness). Final ordering is by id, so the concurrency changes nothing but wall-clock.
    matched_tasks: List[Tuple[str, str, object, object]] = []
    for i, ge in enumerate(pm.gold_entries):
        cid = _faith_id(pm.gold_slugs[i])
        desc = _faith_desc(ge.name)
        wof_entry = pm.matched[i]
        if wof_entry is not None:
            matched_tasks.append((cid, desc, ge, wof_entry))
        elif pm.residual_failed and i in residual_set:
            # We could not pair it (the residual call errored) -> we cannot judge it. Skip (not na).
            items.append(_skipped_item(cid, desc, "[REVIEW] pairing errored; faithfulness not evaluated"))
        else:
            items.append(_na_item(cid, desc, "entity absent from WOF; faithfulness not applicable"))
    items.extend(_run_faithfulness(adj, matched_tasks, max_workers))

    # --- ordering (reuses the pair map; no new model call) ------------------ #
    if pm.residual_failed:
        items.append(_skipped_item(_ORDERING_ID, _ORDERING_DESC,
                                   "[REVIEW] pairing errored; ordering not evaluated"))
    else:
        try:
            items.append(score_ordering(gold, parsed_wof, pm, ranks))
        except Exception as exc:                       # noqa: BLE001
            logger.warning("[REVIEW] gbf ordering errored (%s); skipping", exc)
            items.append(_skipped_item(_ORDERING_ID, _ORDERING_DESC, f"[REVIEW] check errored: {exc}"))

    items.sort(key=lambda it: it.id)
    return build_result(wof_name, items, now, model_cfg)


def grade_file(path, gold: ParsedWOF, ranks, votes: int = DEFAULT_GBF_VOTES, use_model: bool = True,
               model_client=None, now: Optional[datetime] = None,
               max_workers: int = DEFAULT_MAX_WORKERS) -> GbfResult:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return grade(text, p.name, gold, ranks, votes=votes, use_model=use_model,
                 model_client=model_client, now=now, max_workers=max_workers)
