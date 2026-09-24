"""The 3 genuine-judgment FC checks, run through the LLM adjudicator.

Each check gets its tiny pre-extracted candidate set from ``fc_lint`` (never the whole
file), a tight single-purpose prompt, and a forced-JSON reply of one verdict per candidate.
For stability the whole set is judged ``votes`` times and a per-candidate majority is taken;
the verdicts fold into ONE ``FCItem`` (``engine="model"``).

FIX #5 (vote hygiene): a vote that returns the wrong number of verdicts is discarded as an
abstain, so misaligned indices can never silently corrupt the per-candidate majority; if
EVERY vote is malformed the check fails CLOSED with a ``[REVIEW]`` note (never a false pass);
``votes`` is odd so a per-candidate majority rarely ties, and a tie is resolved toward the
"needs a human" (bad) label and flagged ``[REVIEW]``.
"""

import json
import logging
from collections import Counter
from typing import List, Optional, Tuple

from agents.base import loads_tolerant

from evals.fc.fc_lint import CRITERION_TEXT
from evals.fc.models import Engine, Evidence, FCItem, Status

logger = logging.getLogger("evals.fc.adjudicator")

GROUPING_ID = "fc.entries.grouping-no-anchor"
TILDE_ID = "fc.list-entries.approx-tilde"
TBD_ID = "fc.markers.tbd"

GROUPING_SYSTEM = (
    "You are grading the FORMATTING of a fantasy-wiki markdown file. "
    "Each item is a sub-heading that carries NO anchor tag, plus the entry names listed "
    "beneath it. For each item decide exactly one label:\n"
    "  group_label  = a heading that only CLUSTERS or labels a group of entries (e.g. "
    "'Could Not Place', 'Uncategorized', 'Minor NPCs') -- correctly has no anchor.\n"
    "  missing_anchor = a heading that names a single real linkable entry which SHOULD have "
    "its own anchor but is missing one.\n"
    'Output ONLY JSON: {"verdicts": ["group_label" or "missing_anchor", ...]} with one '
    "verdict per item, in the same order. No prose."
)

TILDE_SYSTEM = (
    "You are grading the FORMATTING of a fantasy-wiki markdown file. "
    "Each item is a numeric figure and the sentence it appears in. For each item decide "
    "exactly one label:\n"
    "  approximate_unmarked = the figure is an ESTIMATE or rough/approximate date (e.g. "
    "'about 200 years ago', 'roughly 40', 'around 120') that SHOULD be marked with a leading "
    "tilde '~' but is not.\n"
    "  exact_ok = the figure is EXACT/precise (a stated year, an exact count or quantity, an "
    "identifier) and is fine WITHOUT a tilde.\n"
    'Output ONLY JSON: {"verdicts": ["approximate_unmarked" or "exact_ok", ...]} one per '
    "item, in order. No prose."
)

TBD_SYSTEM = (
    "You are grading the FORMATTING of a fantasy-wiki markdown file. "
    "Each item is a wiki entry: a name and its body text. For each item decide exactly one "
    "label:\n"
    "  stub_needs_tbd = the entry is an unfinished STUB with no real established content (a "
    "placeholder like 'no details yet', 'to be determined', 'unknown') that SHOULD carry a "
    "'[TBD]' marker but does NOT.\n"
    "  ok = the entry has real content, OR it already contains a [TBD] marker.\n"
    'Output ONLY JSON: {"verdicts": ["stub_needs_tbd" or "ok", ...]} one per item, in order. '
    "No prose."
)

_BODY_CLIP = 400


def _clip(text: str, n: int = _BODY_CLIP) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _model_item(cid: str, status: Status, evidence: Evidence) -> FCItem:
    return FCItem(id=cid, description=CRITERION_TEXT.get(cid), engine=Engine.MODEL,
                  status=status, evidence=evidence)


def _safe_parse(raw: str):
    """Parse a model reply into a dict, tolerating fences/preambles; None on failure."""
    try:
        parsed = loads_tolerant(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class Adjudicator:
    """Runs the 3 model checks against an injected ``model_client`` (anything with
    ``complete(system, user) -> str``)."""

    def __init__(self, model_client, votes: int = 3):
        self.model_client = model_client
        self.votes = max(1, votes)

    # --- voting ------------------------------------------------------------- #

    def _vote(self, system: str, user: str, n: int, allowed: set,
              bad_label: str) -> Tuple[Optional[List[str]], List[int]]:
        """Return (per-candidate majority labels, disagreement indices). ``labels`` is None
        when EVERY vote was malformed (caller fails closed). FIX #5 lives here."""
        valid_votes: List[List[str]] = []
        for _ in range(self.votes):
            raw = self.model_client.complete(system, user)
            parsed = _safe_parse(raw)
            verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
            if not isinstance(verdicts, list) or len(verdicts) != n:
                logger.debug("discarding a malformed vote (verdicts=%r)", verdicts)
                continue
            valid_votes.append([str(v).strip() for v in verdicts])
        if not valid_votes:
            return None, list(range(n))

        labels: List[str] = []
        disagreements: List[int] = []
        for i in range(n):
            col = [v[i] for v in valid_votes if v[i] in allowed]
            if not col:
                labels.append(bad_label)
                disagreements.append(i)
                continue
            counts = Counter(col).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                labels.append(bad_label)          # tie -> surface for review
                disagreements.append(i)
            else:
                labels.append(counts[0][0])
                if len(set(col)) > 1:
                    disagreements.append(i)       # a non-unanimous but decided split
        return labels, disagreements

    # --- the three checks --------------------------------------------------- #

    def _judge(self, cid: str, system: str, user: str, candidates: List[dict],
               allowed: set, bad_label: str, offender_label: str) -> FCItem:
        labels, disagreements = self._vote(system, user, len(candidates), allowed, bad_label)
        lines = [c.get("lineno") for c in candidates if c.get("lineno")]
        if labels is None:
            return _model_item(cid, Status.FAIL, Evidence(
                lines=lines, detail="[REVIEW] model returned no parseable verdicts; failing closed"))
        offenders = [c for c, lab in zip(candidates, labels) if lab == offender_label]
        note = ""
        if disagreements:
            note = f" ({len(disagreements)} candidate(s) had split votes -- [REVIEW])"
            logger.warning("[REVIEW] %s: %d candidate(s) had split votes across %d votes",
                           cid, len(disagreements), self.votes)
        if not offenders:
            return _model_item(cid, Status.PASS, Evidence(
                lines=[], detail=f"{len(candidates)} candidate(s) judged, none flagged" + note))
        off_lines = [c.get("lineno") for c in offenders if c.get("lineno")]
        names = [c.get("label") or c.get("name") or c.get("figure") or "?" for c in offenders]
        return _model_item(cid, Status.FAIL, Evidence(
            lines=off_lines,
            detail=f"{len(offenders)} of {len(candidates)} candidate(s) flagged" + note,
            found="; ".join(str(x) for x in names[:8])))

    def judge_grouping(self, candidates: List[dict]) -> FCItem:
        if not candidates:
            # Nothing to judge -> NA (excluded from the scoring denominator), matching the
            # mechanical checks' "nothing applies" convention rather than inflating pass.
            return _model_item(GROUPING_ID, Status.NA,
                               Evidence(lines=[], detail="no anchorless entry-level headings to judge"))
        rows = [f"{i + 1}. label: {c['label']!r} | entries beneath: "
                f"{', '.join(c['entries_beneath']) or '(none)'}"
                for i, c in enumerate(candidates)]
        user = "Items:\n" + "\n".join(rows)
        return self._judge(GROUPING_ID, GROUPING_SYSTEM, user, candidates,
                           {"group_label", "missing_anchor"}, "missing_anchor", "missing_anchor")

    def judge_approx_tilde(self, candidates: List[dict]) -> FCItem:
        if not candidates:
            return _model_item(TILDE_ID, Status.NA,
                               Evidence(lines=[], detail="no numeric figures to check"))
        rows = [f"{i + 1}. figure: {c['figure']!r} | sentence: {_clip(c['sentence'], 200)!r}"
                for i, c in enumerate(candidates)]
        user = "Items:\n" + "\n".join(rows)
        return self._judge(TILDE_ID, TILDE_SYSTEM, user, candidates,
                           {"approximate_unmarked", "exact_ok"}, "approximate_unmarked",
                           "approximate_unmarked")

    def judge_tbd(self, candidates: List[dict]) -> FCItem:
        if not candidates:
            return _model_item(TBD_ID, Status.NA,
                               Evidence(lines=[], detail="no entries to check"))
        rows = [f"{i + 1}. name: {c['name']!r} | body: {_clip(c['body'])!r}"
                for i, c in enumerate(candidates)]
        user = "Items:\n" + "\n".join(rows)
        return self._judge(TBD_ID, TBD_SYSTEM, user, candidates,
                           {"stub_needs_tbd", "ok"}, "stub_needs_tbd", "stub_needs_tbd")
