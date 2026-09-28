"""Tests for evals/common/models.py -- the shared result-envelope pieces. Fully offline.

Covers extra=forbid on the envelope models, string serialization of the Status/Engine
enums, and the discriminated Grader union (FIX #7's Literal-const discriminator), all
exercised through the shared pieces directly (no eval-specific result object). FC's own
FCResult/Summary-alias tests live in tests/test_fc_models.py.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from evals.common.models import (
    Engine,
    Evidence,
    Grader,
    Item,
    MechanicalGrader,
    ModelGrader,
    Status,
)

_GRADER = TypeAdapter(Grader)


def test_extra_forbidden_on_every_model():
    with pytest.raises(ValidationError):
        Evidence(detail="x", bogus=1)
    with pytest.raises(ValidationError):
        Item(id="x", engine=Engine.MODEL, status=Status.PASS,
             evidence=Evidence(detail="d"), bogus=1)


def test_status_and_engine_serialize_as_strings():
    it = Item(id="fc.x", engine=Engine.MODEL, status=Status.SKIPPED,
              evidence=Evidence(detail="d"))
    d = it.model_dump(mode="json")
    assert d["engine"] == "model" and d["status"] == "skipped"


# --- FIX #7: Literal consts + discriminated union --------------------------- #

def test_grader_discriminated_union_parses_both():
    mech = _GRADER.validate_python(
        MechanicalGrader(tool="fc_lint.py", version="0.1.0").model_dump(mode="json"))
    model = _GRADER.validate_python(
        ModelGrader(name="qwen/qwen3-8b", provider="openrouter", temperature=0.6,
                    thinking=False, votes=3).model_dump(mode="json"))
    assert isinstance(mech, MechanicalGrader)
    assert isinstance(model, ModelGrader)
    assert model.name == "qwen/qwen3-8b"
