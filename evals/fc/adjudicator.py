"""The 3 genuine-judgment FC checks, run through the shared voting engine.

Each check gets its tiny pre-extracted candidate set from ``fc_lint`` (never the whole
file), a tight single-purpose prompt, and a forced-JSON reply of one verdict per candidate.
The voting/fold machinery (self-consistency votes, per-candidate majority, FIX #5 vote
hygiene, fail-closed) lives in ``evals.common.adjudicator.BaseAdjudicator``; this module
supplies only the FC-specific prompts, candidate payloads, and label vocab, plus the
``describe`` override so a folded ``Item`` carries its FC criterion prose.
"""

import logging
from typing import List

from evals.common.adjudicator import BaseAdjudicator
from evals.common.models import Evidence, Item, Status

from evals.fc.fc_lint import CRITERION_TEXT

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


class Adjudicator(BaseAdjudicator):
    """Runs the 3 FC model checks against an injected ``model_client`` (anything with
    ``complete(system, user) -> str``)."""

    def describe(self, cid: str):
        return CRITERION_TEXT.get(cid)

    def judge_grouping(self, candidates: List[dict]) -> Item:
        if not candidates:
            # Nothing to judge -> NA (excluded from the scoring denominator), matching the
            # mechanical checks' "nothing applies" convention rather than inflating pass.
            return self._model_item(GROUPING_ID, Status.NA,
                                    Evidence(lines=[], detail="no anchorless entry-level headings to judge"))
        rows = [f"{i + 1}. label: {c['label']!r} | entries beneath: "
                f"{', '.join(c['entries_beneath']) or '(none)'}"
                for i, c in enumerate(candidates)]
        user = "Items:\n" + "\n".join(rows)
        return self._judge(GROUPING_ID, GROUPING_SYSTEM, user, candidates,
                           {"group_label", "missing_anchor"}, "missing_anchor", "missing_anchor")

    def judge_approx_tilde(self, candidates: List[dict]) -> Item:
        if not candidates:
            return self._model_item(TILDE_ID, Status.NA,
                                    Evidence(lines=[], detail="no numeric figures to check"))
        rows = [f"{i + 1}. figure: {c['figure']!r} | sentence: {_clip(c['sentence'], 200)!r}"
                for i, c in enumerate(candidates)]
        user = "Items:\n" + "\n".join(rows)
        return self._judge(TILDE_ID, TILDE_SYSTEM, user, candidates,
                           {"approximate_unmarked", "exact_ok"}, "approximate_unmarked",
                           "approximate_unmarked")

    def judge_tbd(self, candidates: List[dict]) -> Item:
        if not candidates:
            return self._model_item(TBD_ID, Status.NA,
                                    Evidence(lines=[], detail="no entries to check"))
        rows = [f"{i + 1}. name: {c['name']!r} | body: {_clip(c['body'])!r}"
                for i, c in enumerate(candidates)]
        user = "Items:\n" + "\n".join(rows)
        return self._judge(TBD_ID, TBD_SYSTEM, user, candidates,
                           {"stub_needs_tbd", "ok"}, "stub_needs_tbd", "stub_needs_tbd")
