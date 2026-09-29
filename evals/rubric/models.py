"""The rubric result models: the ``RubricResult`` envelope + its split ``presence`` /
``sourcing`` sub-scores.

Reuses the shared envelope pieces (``Status`` / ``Engine`` / ``Evidence`` / ``Item`` /
graders) from ``evals.common.models`` and re-exports them so ``from evals.rubric.models
import ...`` call sites + tests keep working. Only the summary shape is rubric-specific:
two independent ``SubScore``s (mirroring ``ScoreBlock``) under a run-level ``complete``, with
no combined top score.

``SubScore.n_pass`` carries the JSON key ``"pass"`` -- the same reserved-keyword fix as the
FC ``Summary`` (``Field(alias="pass")`` + ``populate_by_name=True``). Serialize with
``by_alias=True`` so the wire key is ``"pass"``.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.common import SCHEMA_VERSION
from evals.common.models import (  # re-exported for rubric call sites + tests
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
    "Item",
    "MechanicalGrader",
    "ModelGrader",
    "RubricResult",
    "RubricSummary",
    "Status",
    "SubScore",
]


class RubricSummary(BaseModel):
    """The rubric's two independent sub-scores under a run-level ``complete`` (false whenever
    any item was ``skipped``). There is deliberately NO combined top score."""

    model_config = ConfigDict(extra="forbid")

    complete: bool
    presence: SubScore
    sourcing: SubScore


class RubricResult(BaseModel):
    """One rubric grading result for one WOF -- the top-level object emitted to JSON.

    ``eval`` is a ``Literal["rubric"]`` const so this validates strictly as the rubric instance
    of the shared envelope (fc/gbf are sibling schemas). ``wof`` is the join key tying the
    fc/rubric/gbf results for one run together. Every item is ``engine="model"``.
    """

    model_config = ConfigDict(extra="forbid")

    eval: Literal["rubric"] = "rubric"
    schema_version: str = SCHEMA_VERSION
    wof: str
    graded_at: datetime
    graders: list[Grader]
    summary: RubricSummary
    items: list[Item]
