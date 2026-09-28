"""Scoring: tally graded items into a ``ScoreBlock`` (the shared applicable-only math) and,
for FC, a full ``Summary`` on top of it.

The denominator is APPLICABLE-only: ``applicable = pass + partial + fail`` (``na`` and
``skipped`` excluded). ``partial`` sits in the denominator but not the numerator, so it
scores as a fail while staying counted separately for the review. ``score = pass /
applicable`` (``None`` when nothing is applicable).

``score_block`` is the reusable per-bucket scorer (the rubric eval calls it once per
sub-score bucket); ``build_summary`` layers ``skipped`` + ``complete`` on top for FC.
``complete`` is false whenever anything was skipped, so a provisional run is visibly
provisional.
"""

from dataclasses import dataclass
from typing import List, Optional

from evals.common.models import Item, Status, Summary


@dataclass
class ScoreBlock:
    """The applicable-only tally for a set of items, WITHOUT the run-level ``skipped`` /
    ``complete`` (those are added by ``build_summary`` for FC; the rubric's ``SubScore``
    mirrors these fields exactly and carries neither)."""

    n_pass: int
    partial: int
    fail: int
    na: int
    applicable: int
    score: Optional[float]
    score_display: str


def score_block(items: List[Item]) -> ScoreBlock:
    counts = {s: 0 for s in Status}
    for it in items:
        counts[it.status] += 1
    n_pass = counts[Status.PASS]
    partial = counts[Status.PARTIAL]
    fail = counts[Status.FAIL]
    na = counts[Status.NA]
    applicable = n_pass + partial + fail
    score = round(n_pass / applicable, 4) if applicable else None
    return ScoreBlock(
        n_pass=n_pass,
        partial=partial,
        fail=fail,
        na=na,
        applicable=applicable,
        score=score,
        score_display=f"{n_pass}/{applicable}",
    )


def build_summary(items: List[Item]) -> Summary:
    block = score_block(items)
    skipped = sum(1 for it in items if it.status == Status.SKIPPED)
    return Summary(
        n_pass=block.n_pass,
        partial=block.partial,
        fail=block.fail,
        na=block.na,
        skipped=skipped,
        applicable=block.applicable,
        score=block.score,
        score_display=block.score_display,
        complete=(skipped == 0),
    )
