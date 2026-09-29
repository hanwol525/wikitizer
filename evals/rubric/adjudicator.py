"""The rubric model checks, run through the shared voting engine.

Every criterion is judged by the model over a tiny, ``ParsedWOF``-narrowed candidate set
(numbered names, or an entry's clipped body + its own cited quotes) -- never raw markdown and
never the whole file. Each check builds ``Item``s DIRECTLY (per-entity ``description`` inline),
so it does not use ``BaseAdjudicator.describe``/``_judge`` (those are FC's per-criterion fold).

It adds a record-shaped voting engine (``_vote_records``) mirroring ``common._vote``'s hygiene
for the ``{"results":[{...}, ...]}`` reply shape:
  * discard a reply whose ``results`` length != n as an abstain (FIX #5),
  * fail closed (``None``) if EVERY reply is malformed,
  * ``votes`` is odd, and a per-index tie resolves toward the STRICT (bad) label + ``[REVIEW]``
    -- in every rubric check the positive/pass direction is ``True``, so ``majority = trues*2 > total``
    (a tie -> ``False``) is exactly the strict-on-tie rule.
"""

import logging
from typing import List, Optional, Tuple

from evals.common.adjudicator import BaseAdjudicator
from evals.common.models import Engine, Evidence, Item, Status
from evals.rubric.resolve import entity_slug

logger = logging.getLogger("evals.rubric.adjudicator")

_BODY_CLIP = 400

SECTIONS_SYSTEM = (
    "You are checking whether a fantasy-wiki markdown file contains the REQUIRED top-level "
    "sections. You are given a numbered list of the file's ACTUAL section titles, and a list "
    "of required sections (each with acceptable alternative names). For each required section "
    "decide whether it is present among the actual titles (allowing the listed alternatives "
    "and obvious synonyms). "
    'Output ONLY JSON: {"results":[{"present":true|false,"match":<1-based index into the '
    'actual titles, or null>}, ...]} with one object per required section, in order. No prose.'
)

PRESENCE_SYSTEM = (
    "You are checking whether REQUIRED entities appear in a section of a fantasy-wiki. You are "
    "given the numbered entry names ACTUALLY present in the section, and a list of required "
    "entities (each with acceptable alternative names/spellings). For each required entity "
    "decide whether it is present among the section's entries (allowing the listed aliases and "
    "obvious spelling variants). "
    'Output ONLY JSON: {"results":[{"present":true|false,"match":<1-based index into the '
    'section entries, or null>}, ...]} one per required entity, in order. No prose.'
)

PCS_SYSTEM = (
    "You are checking player-character (PC) marking in a fantasy-wiki. Each item is a character "
    "entry: its raw heading line and a snippet of its body. For each item decide TWO things:\n"
    "  pc_stated = does the text state or clearly imply this character is a player character "
    "(a PC / played by a player)?\n"
    "  asterisk = does the HEADING line carry a trailing asterisk '*' marker?\n"
    'Output ONLY JSON: {"results":[{"pc_stated":true|false,"asterisk":true|false}, ...]} one '
    "per item, in order. No prose."
)

SUPPORT_SYSTEM = (
    "You are checking whether a fantasy-wiki entry's claims are SUPPORTED by its OWN cited "
    "quotes. You are given the entry's body text and ONLY the quotes it cites (its footnotes). "
    "Decide whether at least one cited quote actually supports the entry's central claim(s). "
    "If yes, return the supporting quote text in 'span'; if no, 'span' is empty. Judge ONLY "
    "against the provided quotes -- do not use outside knowledge. "
    'Output ONLY JSON: {"results":[{"supported":true|false,"span":"<supporting quote text or '
    'empty>"}]}. No prose.'
)


def _clip(text: str, n: int = _BODY_CLIP) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _required_row(i: int, ent) -> str:
    alts = f" (also: {', '.join(ent.aliases)})" if ent.aliases else ""
    return f"{i + 1}. {ent.name}{alts}"


class RubricAdjudicator(BaseAdjudicator):
    """Runs the rubric model checks against an injected ``model_client``. Inherits
    ``__init__(model_client, votes)`` from ``BaseAdjudicator``."""

    # --- item construction + record voting --------------------------------- #

    def _item(self, cid: str, description: str, status: Status, evidence: Evidence) -> Item:
        return Item(id=cid, description=description, engine=Engine.MODEL, status=status,
                    evidence=evidence)

    # ``_vote_records`` / ``_fold_bool`` / ``_companion`` are the shared record-voting engine,
    # promoted to ``BaseAdjudicator`` (both rubric + gbf inherit them). ``_valid_match`` stays
    # here -- it is rubric-specific (a 1-based section/entry match index).

    @staticmethod
    def _valid_match(votes: List[List[dict]], i: int, upper: int) -> Optional[int]:
        """The first valid 1-based ``match`` index (``type is int`` -- a bool is not an index --
        and within ``1..upper``) among the votes that said ``present: true``, else ``None``."""
        for v in votes:
            if v[i].get("present") is True:
                match = v[i].get("match")
                if type(match) is int and 1 <= match <= upper:
                    return match
        return None

    # --- the checks --------------------------------------------------------- #

    def judge_sections(self, required, wof_titles) -> Tuple[List[Item], List[Optional[int]]]:
        """One call. Returns (items, section_map) where ``section_map[j]`` is the 1-based WOF
        section index a present required-section matched, else ``None`` -- a list PARALLEL to
        ``required`` by index (the category->section join reads it by index, not by name)."""
        n = len(required)
        ids = [f"rubric.sections.{entity_slug(r.name)}" for r in required]
        descs = [f"section {r.name!r} present" for r in required]
        if n == 0:
            return [], []
        rows = "\n".join(_required_row(i, r) for i, r in enumerate(required))
        titles = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(wof_titles)) or "(none)"
        user = f"Actual section titles:\n{titles}\n\nRequired sections:\n{rows}"
        votes = self._vote_records(SECTIONS_SYSTEM, user, n)
        if votes is None:
            logger.warning("[REVIEW] rubric sections: no parseable verdicts; failing closed")
            items = [self._item(ids[j], descs[j], Status.FAIL, Evidence(
                lines=[], detail="[REVIEW] model returned no parseable verdicts; failing closed"))
                for j in range(n)]
            return items, [None] * n
        items: List[Item] = []
        section_map: List[Optional[int]] = []
        for j in range(n):
            present, split = self._fold_bool(votes, j, "present")
            note = " ([REVIEW] split vote)" if split else ""
            if split:
                logger.warning("[REVIEW] rubric section %r: split vote", required[j].name)
            if present:
                wof_idx = self._valid_match(votes, j, len(wof_titles))
                section_map.append(wof_idx)
                if wof_idx is None:
                    # "present" but pointing at no real section: a PASS here would contradict
                    # the per-category presence items (which would read the section as absent).
                    logger.warning("[REVIEW] rubric section %r: voted present with no valid match; failing closed",
                                   required[j].name)
                    items.append(self._item(ids[j], descs[j], Status.FAIL, Evidence(
                        lines=[], detail="[REVIEW] model said present but gave no valid section match; "
                                         "failing closed" + note)))
                    continue
                detail = f"section present (WOF section #{wof_idx})" + note
                items.append(self._item(ids[j], descs[j], Status.PASS, Evidence(lines=[], detail=detail)))
            else:
                section_map.append(None)
                items.append(self._item(ids[j], descs[j], Status.FAIL, Evidence(lines=[], detail="section absent" + note)))
        return items, section_map

    def judge_presence(self, category_token, required, entries) -> Tuple[List[Item], List]:
        """One call over the entries under the category's matched section. Returns
        (items, entity_entry_map) where ``entity_entry_map[k]`` is the ``Entry`` a present
        required entity matched, else ``None``. Empty ``entries`` (section absent) -> no call,
        all FAIL."""
        n = len(required)
        ids = [f"rubric.presence.{category_token}.{entity_slug(r.name)}" for r in required]
        descs = [f"{r.name!r} present under {category_token}" for r in required]
        if n == 0:
            return [], []
        if not entries:
            items = [self._item(ids[j], descs[j], Status.FAIL,
                                Evidence(lines=[], detail="section absent")) for j in range(n)]
            return items, [None] * n
        rows = "\n".join(_required_row(i, r) for i, r in enumerate(required))
        enames = "\n".join(f"{i + 1}. {e.name}" for i, e in enumerate(entries))
        user = f"Section entries:\n{enames}\n\nRequired entities:\n{rows}"
        votes = self._vote_records(PRESENCE_SYSTEM, user, n)
        if votes is None:
            logger.warning("[REVIEW] rubric presence %s: no parseable verdicts; failing closed", category_token)
            items = [self._item(ids[j], descs[j], Status.FAIL, Evidence(
                lines=[], detail="[REVIEW] model returned no parseable verdicts; failing closed"))
                for j in range(n)]
            return items, [None] * n
        items = []
        emap: List = []
        for j in range(n):
            present, split = self._fold_bool(votes, j, "present")
            note = " ([REVIEW] split vote)" if split else ""
            if split:
                logger.warning("[REVIEW] rubric presence %s %r: split vote", category_token, required[j].name)
            if present:
                match = self._valid_match(votes, j, len(entries))
                entry = entries[match - 1] if match is not None else None
                emap.append(entry)
                if entry is None:
                    # "present" but pointing at no real entry: a PASS here would leave sourcing
                    # silently unscorable (na, "entity absent") downstream.
                    logger.warning("[REVIEW] rubric presence %s %r: voted present with no valid match; "
                                   "failing closed", category_token, required[j].name)
                    items.append(self._item(ids[j], descs[j], Status.FAIL, Evidence(
                        lines=[], detail="[REVIEW] model said present but gave no valid entry match; "
                                         "failing closed" + note)))
                    continue
                detail = f"present (entry: {entry.name!r})" + note
                items.append(self._item(ids[j], descs[j], Status.PASS,
                                        Evidence(lines=[entry.lineno], detail=detail)))
            else:
                emap.append(None)
                items.append(self._item(ids[j], descs[j], Status.FAIL,
                                        Evidence(lines=[], detail="not found under the section" + note)))
        return items, emap

    def judge_pcs(self, present_pcs) -> List[Item]:
        """One call over PCs that resolved to an entry -- ``present_pcs`` is a list of
        ``(RubricEntity, Entry)``. Emits TWO items per PC (``pc-stated`` + ``pc-marked``). The
        heading passed is ``entry.raw`` (preserves a trailing ``*``). Absent PCs get their
        ``na`` items from the runner, not here."""
        items: List[Item] = []
        n = len(present_pcs)
        if n == 0:
            return items
        rows = "\n".join(
            f"{i + 1}. heading: {entry.raw.strip()!r} | body: {_clip(entry.body)!r}"
            for i, (_ent, entry) in enumerate(present_pcs))
        user = "Items:\n" + rows
        votes = self._vote_records(PCS_SYSTEM, user, n)
        for j, (ent, entry) in enumerate(present_pcs):
            slug = entity_slug(ent.name)
            stated_id = f"rubric.pc-stated.{slug}"
            marked_id = f"rubric.pc-marked.{slug}"
            stated_desc = f"{ent.name!r} PC status stated"
            marked_desc = f"{ent.name!r} marked as PC (trailing *)"
            if votes is None:
                logger.warning("[REVIEW] rubric pcs: no parseable verdicts; failing closed")
                ev = Evidence(lines=[entry.lineno],
                              detail="[REVIEW] model returned no parseable verdicts; failing closed")
                items.append(self._item(stated_id, stated_desc, Status.FAIL, ev))
                items.append(self._item(marked_id, marked_desc, Status.FAIL, ev))
                continue
            stated, s_split = self._fold_bool(votes, j, "pc_stated")
            marked, m_split = self._fold_bool(votes, j, "asterisk")
            items.append(self._item(
                stated_id, stated_desc, Status.PASS if stated else Status.FAIL,
                Evidence(lines=[entry.lineno],
                         detail=("stated as a PC" if stated else "PC status not stated")
                         + (" ([REVIEW] split vote)" if s_split else ""))))
            items.append(self._item(
                marked_id, marked_desc, Status.PASS if marked else Status.FAIL,
                Evidence(lines=[entry.lineno],
                         detail=("carries a trailing '*' marker" if marked else "no trailing '*' marker")
                         + (" ([REVIEW] split vote)" if m_split else ""))))
        return items

    def judge_support(self, entity, category_token, entry, quotes) -> Item:
        """One entity per call, voted. ``quotes`` is the entity's own cited ``(n, text)`` pairs
        (assumed non-empty -- the runner gates absent/no-cite before calling). PASS when a cited
        quote supports the central claim (evidence names the backing ``[^n]`` + the span), else
        FAIL. Tie/all-malformed -> the strict (unsupported) label + ``[REVIEW]``."""
        cid = f"rubric.sourcing.{category_token}.{entity_slug(entity.name)}"
        desc = f"{entity.name!r} sourcing under {category_token}"
        qlines = "\n".join(f"  [^{n}] {_clip(q, 300)}" for n, q in quotes)
        user = (f"Entry: {entity.name!r}\nBody: {_clip(entry.body)!r}\nCited quotes:\n{qlines}\n"
                "Does a cited quote support the entry's central claim?")
        votes = self._vote_records(SUPPORT_SYSTEM, user, 1)
        if votes is None:
            logger.warning("[REVIEW] rubric sourcing %r: no parseable verdicts; failing closed", entity.name)
            return self._item(cid, desc, Status.FAIL, Evidence(
                lines=[entry.lineno], detail="[REVIEW] model returned no parseable verdicts; failing closed"))
        supported, split = self._fold_bool(votes, 0, "supported")
        note = " ([REVIEW] split vote)" if split else ""
        if supported:
            span = self._companion(votes, 0, "supported", "span", True) or ""
            cited = ", ".join(f"[^{n}]" for n, _ in quotes)
            span_txt = _clip(str(span), 200)
            return self._item(cid, desc, Status.PASS, Evidence(
                lines=[entry.lineno], detail=f"claim backed by a cited quote ({cited})" + note,
                found=span_txt))
        return self._item(cid, desc, Status.FAIL, Evidence(
            lines=[entry.lineno], detail="claim not supported by its cited quotes" + note))
