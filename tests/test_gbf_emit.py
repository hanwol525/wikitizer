"""Tests for evals/gbf/emit.py -- assembly + serialization. Fully offline."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals.common.models import Engine, Evidence, Item, Status
from evals.gbf.emit import build_result, result_to_json, write_result

_NOW = datetime(2026, 9, 23, 14, 32, 10, tzinfo=timezone.utc)
_MODEL_CFG = dict(name="deepseek/deepseek-v4-pro-0813", provider="openrouter", temperature=0.6,
                  thinking=False, votes=5, endpoint=None)


def _items():
    return [
        Item(id="gbf.faithfulness.aldenburg", engine=Engine.MODEL, status=Status.PASS,
             evidence=Evidence(detail="all claims affirmed")),
        Item(id="gbf.ordering", engine=Engine.MODEL, status=Status.PASS,
             evidence=Evidence(detail="0 inversions")),
    ]


def test_model_grader_always_present():
    result = build_result("w.md", _items(), _NOW, _MODEL_CFG)
    assert len(result.graders) == 1
    assert result.graders[0].engine == "model"
    assert result.graders[0].name == "deepseek/deepseek-v4-pro-0813"


def test_result_to_json_readds_both_nullable_scores():
    items = [
        Item(id="gbf.faithfulness.x", engine=Engine.MODEL, status=Status.NA, evidence=Evidence(detail="d")),
        Item(id="gbf.ordering", engine=Engine.MODEL, status=Status.NA, evidence=Evidence(detail="d")),
    ]
    result = build_result("w.md", items, _NOW, _MODEL_CFG)
    data = json.loads(result_to_json(result))
    assert data["summary"]["faithfulness"]["score"] is None
    assert data["summary"]["ordering"]["score"] is None
    assert "score" in data["summary"]["faithfulness"] and "score" in data["summary"]["ordering"]


def test_result_to_json_omits_absent_optional_strings():
    result = build_result("w.md", _items(), _NOW, _MODEL_CFG)
    data = json.loads(result_to_json(result))
    assert "endpoint" not in data["graders"][0]        # endpoint None -> omitted, not null


def test_write_result_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "g.json"
    result = build_result("w.md", _items(), _NOW, _MODEL_CFG)
    write_result(result, out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["eval"] == "gbf"


def test_emitted_result_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parent.parent / "gbf-result.schema.json"
    if not schema_path.exists():
        pytest.skip("gbf-result.schema.json not found")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    result = build_result("w.md", _items(), _NOW, _MODEL_CFG)
    jsonschema.validate(json.loads(result_to_json(result)), schema)
