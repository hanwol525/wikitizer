"""Bucket the rubric items into presence vs sourcing and build the two-sub-score summary.

``complete`` is DERIVED here from the items (``skipped == 0``), exactly like
``evals.common.scoring.build_summary`` -- so a degraded run (a check errored -> a ``skipped``
item) correctly reports ``complete=false`` instead of a caller-supplied literal.
"""

from dataclasses import asdict
from typing import List, Tuple

from evals.common.models import Item, Status
from evals.common.scoring import score_block
from evals.rubric.models import RubricSummary, SubScore

_SOURCING_PREFIX = "rubric.sourcing."


def bucket(items: List[Item]) -> Tuple[List[Item], List[Item]]:
    """(presence, sourcing). Sourcing = ids starting ``rubric.sourcing.``; presence =
    everything else (``rubric.sections.*`` / ``rubric.presence.*`` / ``rubric.pc-*.*``)."""
    presence: List[Item] = []
    sourcing: List[Item] = []
    for it in items:
        (sourcing if it.id.startswith(_SOURCING_PREFIX) else presence).append(it)
    return presence, sourcing


def build_rubric_summary(items: List[Item]) -> RubricSummary:
    presence, sourcing = bucket(items)
    skipped = sum(1 for it in items if it.status == Status.SKIPPED)
    return RubricSummary(
        complete=(skipped == 0),
        presence=SubScore(**asdict(score_block(presence))),
        sourcing=SubScore(**asdict(score_block(sourcing))),
    )
