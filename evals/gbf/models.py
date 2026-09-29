"""The GBF result models: the ``GbfResult`` envelope + its split ``faithfulness`` / ``ordering``
sub-scores.

Reuses the shared envelope pieces (``Status`` / ``Engine`` / ``Evidence`` / ``Item`` / graders /
``SubScore``) from ``evals.common.models`` and re-exports them so ``from evals.gbf.models import
...`` call sites + tests keep working. Only the summary shape is GBF-specific: two independent
``SubScore``s under a run-level ``complete``, with no combined top score.

``SubScore`` (now shared, in ``evals.common.models``) carries the JSON key ``"pass"`` via the same
reserved-keyword fix (``Field(alias="pass")`` + ``populate_by_name=True``); serialize ``by_alias=True``.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.common import SCHEMA_VERSION
from evals.common.models import (  # re-exported for gbf call sites + tests
    Engine,
    Evidence,
    Grader,
    Item,
    MechanicalGrader,
    ModelGrader,
    Status,
    SubScore,
)

__all__ = [
    "Engine",
    "Evidence",
    "Grader",
    "GbfResult",
    "GbfSummary",
    "Item",
    "MechanicalGrader",
    "ModelGrader",
    "Status",
    "SubScore",
]


class GbfSummary(BaseModel):
    """GBF's two independent sub-scores under a run-level ``complete`` (false whenever any item was
    ``skipped``). There is deliberately NO combined top score -- faithfulness and ordering measure
    different things and must not be averaged into a single number."""

    model_config = ConfigDict(extra="forbid")

    complete: bool
    faithfulness: SubScore
    ordering: SubScore


class GbfResult(BaseModel):
    """One GBF grading result for one WOF -- the top-level object emitted to JSON.

    ``eval`` is a ``Literal["gbf"]`` const so this validates strictly as the GBF instance of the
    shared envelope (fc/rubric are sibling schemas). ``wof`` is the join key tying the fc/rubric/gbf
    results for one run together. Every item is ``engine="model"``.
    """

    model_config = ConfigDict(extra="forbid")

    eval: Literal["gbf"] = "gbf"
    schema_version: str = SCHEMA_VERSION
    wof: str
    graded_at: datetime
    graders: list[Grader]
    summary: GbfSummary
    items: list[Item]
