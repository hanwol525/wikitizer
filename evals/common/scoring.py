"""Scoring: tally the graded items into a ``Summary``.

The denominator is APPLICABLE-only: ``applicable = pass + partial + fail`` (``na`` and
``skipped`` excluded). ``partial`` sits in the denominator but not the numerator, so it
scores as a fail while staying counted separately for the review. ``score = pass /
applicable`` (``None`` when nothing is applicable). ``complete`` is false whenever anything
was skipped, so a provisional run is visibly provisional.
"""

from typing import List

from evals.common.models import Item, Status, Summary


def build_summary(items: List[Item]) -> Summary:
    counts = {s: 0 for s in Status}
    for it in items:
        counts[it.status] += 1
    n_pass = counts[Status.PASS]
    partial = counts[Status.PARTIAL]
    fail = counts[Status.FAIL]
    na = counts[Status.NA]
    skipped = counts[Status.SKIPPED]
    applicable = n_pass + partial + fail
    score = round(n_pass / applicable, 4) if applicable else None
    return Summary(
        n_pass=n_pass,
        partial=partial,
        fail=fail,
        na=na,
        skipped=skipped,
        applicable=applicable,
        score=score,
        score_display=f"{n_pass}/{applicable}",
        complete=(skipped == 0),
    )
