"""Pydantic v2 models that mirror ``fc-result.schema.json`` exactly.

These models ARE the FC result contract on the Python side: the emitter builds an
``FCResult``, pydantic validates it on construction, and ``model_dump(mode="json",
by_alias=True)`` produces the schema-valid JSON. Keeping the schema and these models
in lock-step means a mismatch fails loudly at emit time rather than silently shipping
an off-spec result.

Two shapes here are subtler than they look, both to satisfy the schema without fighting
Python:

  * ``Summary.n_pass`` -- the JSON key is ``"pass"``, but ``pass`` is a reserved Python
    keyword (you cannot write ``pass: int`` as a field, access ``summary.pass``, or put
    it in an f-string). So the field is named ``n_pass`` with ``Field(alias="pass")``;
    ``populate_by_name=True`` lets callers build it either way, and the emitter dumps
    ``by_alias=True`` so the wire key is still ``"pass"``. (``eval`` needs no such trick
    -- it's a builtin, not a keyword.)
  * ``Grader`` is a real discriminated union on ``engine`` -- the schema's ``oneOf`` of a
    mechanical vs a model grader. The ``Literal`` engine consts are the discriminator, so
    a grader dict is routed to the right model even if the two ever grow overlapping fields.

Python 3.9.6: ``Optional[X]`` / ``Union`` / ``Literal`` / ``Annotated`` from ``typing``;
builtin generics (``list[int]``) are fine (PEP 585); no ``X | Y`` unions.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Status(str, Enum):
    """Result for one criterion.

    ``pass``/``fail`` as expected. ``partial`` = partially met; it SCORES as a fail but
    is surfaced separately so a reviewer can still see it. ``na`` = the criterion does not
    apply to this WOF (excluded from the scoring denominator). ``skipped`` = not evaluated
    this run (e.g. the model pass was disabled or no key); excluded from the denominator AND
    it flips ``summary.complete`` to false, so a partial run is never mistaken for a full one.

    The ``(str, Enum)`` mix-in (3.9's stand-in for ``StrEnum``) makes ``Status.PASS == "pass"``
    true, so it JSON-serializes to the bare string while pydantic still rejects any off-menu value.
    """

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    NA = "na"
    SKIPPED = "skipped"


class Engine(str, Enum):
    """Which engine judged an item: deterministic Python linter vs the LLM adjudicator."""

    MECHANICAL = "mechanical"
    MODEL = "model"


class Evidence(BaseModel):
    """Why a status was assigned. Required on EVERY item, including passes (kept terse there:
    a count or "present"; specific on fails/partials: line numbers + the offending value)."""

    model_config = ConfigDict(extra="forbid")

    # Line number(s) in the WOF the finding refers to. Empty for whole-document checks.
    lines: list[int] = Field(default_factory=list)
    # Terse finding for mechanical checks (a count / the offending value); a one-line
    # rationale for model checks.
    detail: str
    # Optional, for mechanical comparisons: what was expected vs what was actually present.
    expected: Optional[str] = None
    found: Optional[str] = None


class FCItem(BaseModel):
    """One criterion's grading result. ``id`` is the ONLY join key (the dotted ``fc.*``
    slug); ``description`` is an informational, non-authoritative copy of the criterion
    prose and may be omitted."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: Optional[str] = None
    engine: Engine
    status: Status
    evidence: Evidence


class Summary(BaseModel):
    """Tally over ``items[]``. ``items`` is the source of truth; a consumer should be able
    to recompute this and flag any mismatch.

    ``n_pass`` carries the JSON key ``"pass"`` (see the module docstring). ``applicable =
    n_pass + partial + fail`` is the scoring denominator (``na`` and ``skipped`` excluded);
    ``score = n_pass / applicable`` in 0..1, or ``None`` when ``applicable == 0``.
    ``complete`` is false whenever ``skipped > 0``.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    n_pass: int = Field(alias="pass")
    partial: int
    fail: int
    na: int
    skipped: int
    applicable: int
    score: Optional[float] = None
    score_display: str
    complete: bool


class MechanicalGrader(BaseModel):
    """The deterministic linter, as recorded in a result's ``graders`` list."""

    model_config = ConfigDict(extra="forbid")

    engine: Literal["mechanical"] = "mechanical"
    tool: str
    version: str


class ModelGrader(BaseModel):
    """The LLM adjudicator, as recorded in a result's ``graders`` list. Present only when
    the model actually ran (the emitter omits it on a fully-skipped run)."""

    model_config = ConfigDict(extra="forbid")

    engine: Literal["model"] = "model"
    name: str
    provider: str
    endpoint: Optional[str] = None
    temperature: float
    thinking: bool
    votes: int


# A grader is a mechanical OR a model grader, discriminated on ``engine`` (mirrors the
# schema's ``oneOf``). The ``Literal`` engine consts above are what pydantic keys on.
Grader = Annotated[Union[MechanicalGrader, ModelGrader], Field(discriminator="engine")]


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
