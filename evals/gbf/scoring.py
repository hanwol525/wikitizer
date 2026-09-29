"""Bucket the GBF items into faithfulness vs ordering and build the two-sub-score summary.

``complete`` is DERIVED here from the items (``skipped == 0``), like the rubric -- so a degraded run
(a check errored -> a ``skipped`` item) correctly reports ``complete=false`` instead of a
caller-supplied literal.

**Partial earns HALF credit in GBF (GBF-only).** Unlike FC/rubric -- which keep partial=fail via the
shared ``evals.common.scoring.score_block`` -- a GBF ``partial`` is a faithful-but-abridged entry
(some gold claims omitted, NONE contradicted) or a nearly-ordered timeline: not the same failure as a
meaning contradiction. So GBF uses its OWN ``_weighted_subscore`` (below), which weights partial at
``_PARTIAL_WEIGHT``; the shared ``score_block`` is deliberately left untouched so FC/rubric are
unaffected. The ``pass/partial/fail/na/applicable`` counts are unweighted, so the full breakdown
stays visible in the separate ``SubScore`` fields -- only ``score`` + ``score_display`` carry the
half-credit.
"""

from typing import List, Optional, Tuple

from evals.common.models import Item, Status
from evals.gbf.models import GbfSummary, SubScore

_ORDERING_PREFIX = "gbf.ordering"
_PARTIAL_WEIGHT = 0.5   # GBF-only: a partial (faithful-but-abridged / nearly-ordered) is not a fail


def bucket(items: List[Item]) -> Tuple[List[Item], List[Item]]:
    """(faithfulness, ordering). Ordering = ids starting ``gbf.ordering``; faithfulness =
    everything else (``gbf.faithfulness.*``)."""
    faithfulness: List[Item] = []
    ordering: List[Item] = []
    for it in items:
        (ordering if it.id.startswith(_ORDERING_PREFIX) else faithfulness).append(it)
    return faithfulness, ordering


def _weighted_subscore(items: List[Item]) -> SubScore:
    """The GBF applicable-only tally, with ``partial`` weighted at ``_PARTIAL_WEIGHT`` (half credit).

    Mirrors ``evals.common.scoring.score_block`` in shape (unweighted counts + ``applicable = pass +
    partial + fail``, ``na``/``skipped`` excluded), but the numerator is ``pass + 0.5*partial`` rather
    than ``pass`` alone. ``score`` is ``None`` when nothing is applicable; ``score_display`` uses
    ``:g`` so it stays an integer fraction when there are no partials (``"3/4"``, ``"0/0"``) and shows
    the half only when a partial is present (``"7.5/20"``, ``"1.5/3"``)."""
    counts = {s: 0 for s in Status}
    for it in items:
        counts[it.status] += 1
    n_pass = counts[Status.PASS]
    partial = counts[Status.PARTIAL]
    fail = counts[Status.FAIL]
    na = counts[Status.NA]
    applicable = n_pass + partial + fail
    credit = n_pass + _PARTIAL_WEIGHT * partial
    score: Optional[float] = round(credit / applicable, 4) if applicable else None
    return SubScore(
        n_pass=n_pass,
        partial=partial,
        fail=fail,
        na=na,
        applicable=applicable,
        score=score,
        score_display=f"{credit:g}/{applicable}",
    )


def build_gbf_summary(items: List[Item]) -> GbfSummary:
    faithfulness, ordering = bucket(items)
    skipped = sum(1 for it in items if it.status == Status.SKIPPED)
    return GbfSummary(
        complete=(skipped == 0),
        faithfulness=_weighted_subscore(faithfulness),
        ordering=_weighted_subscore(ordering),
    )
