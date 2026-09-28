"""Tests for evals/rubric/emit.py -- assembly + serialization. Fully offline."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals.common.models import Engine, Evidence, Item, Status
from evals.rubric.emit import build_result, result_to_json, write_result

_NOW = datetime(2026, 9, 23, 14, 32, 10, tzinfo=timezone.utc)
_MODEL_CFG = dict(name="qwen/qwen3-8b", provider="openrouter", temperature=0.6, thinking=False,
                  votes=3, endpoint=None)


def _items():
    return [
        Item(id="rubric.sections.locations", engine=Engine.MODEL, status=Status.PASS,
             evidence=Evidence(detail="present")),
        Item(id="rubric.sourcing.locations.aldenburg", engine=Engine.MODEL, status=Status.NA,
             evidence=Evidence(detail="entity absent")),
    ]


def test_model_grader_always_present():
    result = build_result("w.md", _items(), _NOW, _MODEL_CFG)
    assert len(result.graders) == 1
    assert result.graders[0].engine == "model"        # ModelGrader.engine is a Literal str
    assert result.graders[0].name == "qwen/qwen3-8b"


def test_result_to_json_readds_both_nullable_scores():
    # both buckets have 0 applicable here (pass-only in presence? no -> presence has 1 pass).
    # Force both to None by using only na/skipped:
    items = [
        Item(id="rubric.sections.x", engine=Engine.MODEL, status=Status.NA, evidence=Evidence(detail="d")),
        Item(id="rubric.sourcing.x.y", engine=Engine.MODEL, status=Status.NA, evidence=Evidence(detail="d")),
    ]
    result = build_result("w.md", items, _NOW, _MODEL_CFG)
    data = json.loads(result_to_json(result))
    # exclude_none would have dropped these; they must be re-added as explicit null.
    assert data["summary"]["presence"]["score"] is None
    assert data["summary"]["sourcing"]["score"] is None
    assert "score" in data["summary"]["presence"] and "score" in data["summary"]["sourcing"]


def test_result_to_json_omits_absent_optional_strings():
    result = build_result("w.md", _items(), _NOW, _MODEL_CFG)
    data = json.loads(result_to_json(result))
    # endpoint is None -> omitted (not null); description absent -> omitted
    assert "endpoint" not in data["graders"][0]
    assert "description" not in data["items"][0]


def test_write_result_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "r.json"
    result = build_result("w.md", _items(), _NOW, _MODEL_CFG)
    write_result(result, out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["eval"] == "rubric"


def test_emitted_result_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parent.parent / "rubric-result.schema.json"
    if not schema_path.exists():
        pytest.skip("rubric-result.schema.json not found")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    result = build_result("w.md", _items(), _NOW, _MODEL_CFG)
    jsonschema.validate(json.loads(result_to_json(result)), schema)
