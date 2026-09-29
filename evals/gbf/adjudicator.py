"""The GBF model checks, run through the shared voting engine.

Two model methods, each over a tiny ``ParsedWOF``-narrowed candidate set (never raw markdown, never
the whole file):

  * ``pair_residual`` -- match the entries Python couldn't confidently lock, over the record-voting
    engine promoted to ``BaseAdjudicator`` (``_vote_records`` -> ``pair.majority_index``).
  * ``judge_faithfulness`` -- ONE gold/WOF pair per call, voted (default 5x, the fuzziest judgment;
    never batched -- batching invites the exhaustive-context degradation we design around). The reply
    is claim-level (``affirmed``/``contradicted``/``omitted`` per gold claim + WOF-only ``extra``),
    folded DETERMINISTICALLY in Python into one ``Item`` status.

Claim counts vary per vote, so ``judge_faithfulness`` cannot use the fixed-length ``_vote_records``;
it uses a small local ``_vote_dicts`` loop and folds per vote. ``[EXTRA]`` notes are orthogonal to
the status -- they surface WOF claims the gold is silent on but never move the score.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional

from evals.common.adjudicator import BaseAdjudicator, _safe_parse
from evals.common.models import Engine, Evidence, Item, Status
from evals.gbf.pair import gold_slug, majority_index

logger = logging.getLogger("evals.gbf.adjudicator")

_BODY_CLIP = 2000        # faithfulness needs the substance; per-pair narrowing keeps context bounded
_SNIPPET_CLIP = 200      # residual pairing only needs enough to recognize the same entity
_SPAN_CLIP = 160

PAIR_SYSTEM = (
    "You are matching entries between two versions of the same fantasy-wiki: a GOLD reference and a "
    "candidate (WOF). You are given the GOLD entries and the WOF entries that could NOT be matched "
    "automatically. For EACH gold entry, decide which WOF entry (if any) describes the SAME entity "
    "(same place/person/group/thing/event), allowing spelling and phrasing variants. Judge ONLY the "
    "provided names and snippets; do not invent entries. "
    'Output ONLY JSON: {"results":[{"gold_index":<int>,"wof_index":<int or null>}, ...]} with one '
    "object per GOLD entry, in the given order, using the 0-based indices shown. No prose."
)

FAITHFULNESS_SYSTEM = (
    "You are checking whether a candidate fantasy-wiki entry (WOF) is FAITHFUL IN MEANING to the "
    "GOLD entry for the same entity. Judge ONLY the two prose bodies provided -- use no outside "
    "knowledge. Steps:\n"
    "1. Extract the GOLD entry's key factual claims (the substantive assertions, not filler).\n"
    "2. For each gold claim, decide its status in the WOF body:\n"
    "   - 'affirmed'    : the WOF states the same fact (quote the supporting WOF span in 'span').\n"
    "   - 'contradicted': the WOF states something incompatible (quote the conflicting WOF span).\n"
    "   - 'omitted'     : the WOF does not mention it (quote the GOLD span in 'span').\n"
    "3. List any WOF claims the GOLD is silent on as 'extra' (these are notes only).\n"
    'Output ONLY JSON: {"claims":[{"claim":"<gold claim>","verdict":"affirmed|contradicted|omitted",'
    '"span":"<quoted span>"}], "extra":["<wof-only claim>", ...]}. No prose.'
)


def _clip(text: str, n: int = _BODY_CLIP) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _body(entry) -> str:
    """An entry's prose for meaning judgment: its body, falling back to the raw heading line when the
    body is empty (a name-only entry still gives the model something to compare)."""
    return (entry.body or "").strip() or (entry.raw or "").strip()


class GbfAdjudicator(BaseAdjudicator):
    """Runs the GBF model checks against an injected ``model_client``. Inherits
    ``__init__(model_client, votes)`` + the record-voting engine from ``BaseAdjudicator``. Builds
    ``Item``s directly (per-pair inline ``description``), like ``RubricAdjudicator``."""

    def _item(self, cid: str, description: str, status: Status, evidence: Evidence) -> Item:
        return Item(id=cid, description=description, engine=Engine.MODEL, status=status,
                    evidence=evidence)

    # --- residual pairing --------------------------------------------------- #

    def pair_residual(self, gold_residual, wof_residual) -> Dict[int, Optional[int]]:
        """One voted call. Returns ``{gold_local_index: wof_local_index or None}`` over the residual
        gold list (indices are 0-based into the lists passed in). Empty residual on either side ->
        no call, all ``None``."""
        n = len(gold_residual)
        if n == 0 or not wof_residual:
            return {i: None for i in range(n)}
        g_rows = "\n".join(f"{i}. {e.name}  ::  {_clip(_body(e), _SNIPPET_CLIP)}"
                           for i, e in enumerate(gold_residual))
        w_rows = "\n".join(f"{j}. {e.name}  ::  {_clip(_body(e), _SNIPPET_CLIP)}"
                           for j, e in enumerate(wof_residual))
        user = (f"GOLD entries (unmatched):\n{g_rows}\n\nWOF entries (unmatched):\n{w_rows}\n\n"
                "For each GOLD entry, give the 0-based WOF index of the SAME entity, or null.")
        votes = self._vote_records(PAIR_SYSTEM, user, n)
        if votes is None:
            # ALL votes were unparseable/wrong-length -- an infra/parse failure, NOT a considered
            # "no match". Raise so pair()'s degrade wrap sets residual_failed=True and the runner
            # marks these entries SKIPPED (we couldn't judge), never a false na on a "complete" run.
            # (A genuine model-said-no-match is a valid vote with wof_index=null -> majority None ->
            # left unmatched -> na; that path is unaffected.)
            raise RuntimeError("residual pairing: no parseable verdicts across all votes")
        mapping_votes: List[Dict[int, Optional[int]]] = []
        for v in votes:
            d: Dict[int, Optional[int]] = {}
            for pos, rec in enumerate(v):
                gi = rec.get("gold_index")
                gi = gi if (type(gi) is int and 0 <= gi < n) else pos   # explicit index, else position
                d[gi] = rec.get("wof_index")
            mapping_votes.append(d)
        return majority_index(mapping_votes, n)

    # --- faithfulness ------------------------------------------------------- #

    def _vote_dicts(self, system: str, user: str) -> List[dict]:
        """Call the model ``self.votes`` times, keeping each reply that parses to a dict. Used for
        the faithfulness reply whose ``claims`` length varies per vote (so ``_vote_records``' fixed-n
        guard doesn't apply)."""
        out: List[dict] = []
        for _ in range(self.votes):
            parsed = _safe_parse(self.model_client.complete(system, user))
            if isinstance(parsed, dict):
                out.append(parsed)
        return out

    @staticmethod
    def _vote_label(vote: dict) -> Optional[Status]:
        """Derive one vote's faithfulness label from its claim verdicts, or ``None`` (abstain) when
        the vote yields NO usable claim verdicts -- a missing/non-list ``claims``, an empty list, or
        a list with no dict records. Any contradiction -> FAIL; all affirmed -> PASS; else (some
        omitted, none contradicted) -> PARTIAL. An abstain is filtered before the fold, so a judge
        that systematically extracts nothing fails CLOSED (all-abstain -> the no-verdicts path) rather
        than being recorded as a quiet PARTIAL on a ``complete`` run."""
        claims = vote.get("claims")
        if not isinstance(claims, list):
            return None
        verdicts = [c.get("verdict") for c in claims if isinstance(c, dict)]
        if not verdicts:
            return None
        if any(v == "contradicted" for v in verdicts):
            return Status.FAIL
        if all(v == "affirmed" for v in verdicts):
            return Status.PASS
        return Status.PARTIAL

    def judge_faithfulness(self, gold_entry, wof_entry) -> Item:
        """One gold/WOF pair, voted. Folds per-vote labels into a majority Item status
        (tie/all-malformed -> FAIL + ``[REVIEW]``, strict). Evidence names the contradicted/omitted
        gold claims (with spans) and appends any WOF-only ``extra`` as ``[EXTRA]`` notes (which never
        change the status)."""
        slug = gold_slug(gold_entry)
        cid = f"gbf.faithfulness.{slug}"
        desc = f"{gold_entry.name!r} faithful in meaning to the gold"
        lines = [wof_entry.lineno] if getattr(wof_entry, "lineno", None) else []
        user = (f"GOLD entry: {gold_entry.name!r}\nGOLD body:\n{_clip(_body(gold_entry))}\n\n"
                f"WOF entry: {wof_entry.name!r}\nWOF body:\n{_clip(_body(wof_entry))}")

        votes = self._vote_dicts(FAITHFULNESS_SYSTEM, user)
        labelled = [(v, self._vote_label(v)) for v in votes]
        labelled = [(v, lab) for v, lab in labelled if lab is not None]
        if not labelled:
            return self._item(cid, desc, Status.FAIL, Evidence(
                lines=lines,
                detail="[REVIEW] model returned no parseable faithfulness verdicts; failing closed"))

        labels = [lab for _v, lab in labelled]
        top_label, top_count = Counter(labels).most_common(1)[0]
        # Require a STRICT majority (> half of the usable votes), mirroring _fold_bool's
        # ``trues*2 > total``. A mere plurality (e.g. 2 PASS vs 1 FAIL + 1 PARTIAL on an even
        # usable-vote count from an abstain) is NOT enough and fails CLOSED, so a dissenting
        # contradiction is never silently outvoted by a non-majority PASS.
        no_majority = top_count * 2 <= len(labels)
        majority = Status.FAIL if no_majority else top_label
        if no_majority:
            logger.warning("[REVIEW] gbf faithfulness %s: no strict majority across votes -> failing strict",
                           slug)

        # Prefer a vote matching the resolved label; on the strict-fallback FAIL (which may have no
        # FAIL vote) surface the most concerning dissent so its claims land in the evidence.
        chosen = next((v for v, lab in labelled if lab == majority), None)
        if chosen is None:
            chosen = (next((v for v, lab in labelled if lab == Status.FAIL), None)
                      or next((v for v, lab in labelled if lab == Status.PARTIAL), None)
                      or labelled[0][0])
        claims = [c for c in (chosen.get("claims") or []) if isinstance(c, dict)]
        bad = []
        for c in claims:
            verd = c.get("verdict")
            if verd in ("contradicted", "omitted"):
                txt = str(c.get("claim") or "").strip()
                span = str(c.get("span") or "").strip()
                piece = f"{verd}: {txt}" if txt else verd
                if span:
                    piece += f" [span: {_clip(span, _SPAN_CLIP)}]"
                bad.append(piece)
        extras = [str(x).strip() for x in (chosen.get("extra") or []) if str(x).strip()]

        if majority == Status.PASS:
            detail = f"all {len(claims)} gold claim(s) affirmed in the WOF"
        elif bad:
            detail = "; ".join(bad)
        else:
            detail = f"faithfulness: {majority.value}"
        if extras:
            detail += " | " + " ".join(f"[EXTRA] {e}" for e in extras)
        if no_majority:
            detail += " ([REVIEW] no strict majority across votes)"
        return self._item(cid, desc, majority, Evidence(lines=lines, detail=detail))
