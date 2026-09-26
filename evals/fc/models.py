"""FC-specific result model: the top-level ``FCResult`` that assembles the shared envelope.

The shared envelope pieces (``Status`` / ``Engine`` / ``Evidence`` / ``Item`` / graders /
``Summary``) live in ``evals.common.models``. This module re-exports them so existing
``from evals.fc.models import ...`` call sites keep working, defines the FC-specific
``FCResult``, and keeps ``FCItem`` as a thin alias for the shared ``Item`` (FC has always
called a criterion-result an ``FCItem``).

Serialization uses ``by_alias=True`` so ``Summary.n_pass`` emits the schema key ``"pass"``.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.common.models import (  # re-exported for FC call sites + tests
    Engine,
    Evidence,
    Grader,
    Item,
    MechanicalGrader,
    ModelGrader,
    Status,
    Summary,
)

# FC refers to a criterion-result ``Item`` as ``FCItem``; keep the name as a thin alias.
FCItem = Item

# The shared pieces are re-exported on purpose (call sites + tests import them from here);
# ``__all__`` marks that so an import-cleanup tool won't strip the "unused" imports.
__all__ = [
    "Engine",
    "Evidence",
    "FCItem",
    "FCResult",
    "Grader",
    "Item",
    "MechanicalGrader",
    "ModelGrader",
    "Status",
    "Summary",
]


class FCResult(BaseModel):
    """One FC grading result for one WOF -- the top-level object emitted to JSON.

    ``eval`` is a ``Literal["fc"]`` const so this validates strictly as the FC instance of
    the shared envelope (rubric/gbf are sibling schemas). ``wof`` is the join key tying the
    fc/rubric/gbf results for one run together.
    """

    model_config = ConfigDict(extra="forbid")

    eval: Literal["fc"] = "fc"
    schema_version: str = "1.0"
    wof: str
    graded_at: datetime
    graders: list[Grader]
    summary: Summary
    items: list[FCItem]
