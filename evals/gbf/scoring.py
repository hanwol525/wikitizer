"""Bucket the GBF items into faithfulness vs ordering and build the two-sub-score summary.

``complete`` is DERIVED here from the items (``skipped == 0``), exactly like
``evals.common.scoring.build_summary`` and the rubric -- so a degraded run (a check errored -> a
``skipped`` item) correctly reports ``complete=false`` instead of a caller-supplied literal.
"""

from dataclasses import asdict
from typing import List, Tuple

from evals.common.models import Item, Status
from evals.common.scoring import score_block
from evals.gbf.models import GbfSummary, SubScore

_ORDERING_PREFIX = "gbf.ordering"


def bucket(items: List[Item]) -> Tuple[List[Item], List[Item]]:
    """(faithfulness, ordering). Ordering = ids starting ``gbf.ordering``; faithfulness =
    everything else (``gbf.faithfulness.*``)."""
    faithfulness: List[Item] = []
    ordering: List[Item] = []
    for it in items:
        (ordering if it.id.startswith(_ORDERING_PREFIX) else faithfulness).append(it)
    return faithfulness, ordering


def build_gbf_summary(items: List[Item]) -> GbfSummary:
    faithfulness, ordering = bucket(items)
    skipped = sum(1 for it in items if it.status == Status.SKIPPED)
    return GbfSummary(
        complete=(skipped == 0),
        faithfulness=SubScore(**asdict(score_block(faithfulness))),
        ordering=SubScore(**asdict(score_block(ordering))),
    )
