"""Tests for evals/fc/emit.py -- result assembly + serialization. Fully offline.

Validates against fc-result.schema.json when `jsonschema` is installed (importorskip); the
always-on guarantee is the Pydantic round-trip.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals.fc.emit import build_result, result_to_json, write_result
from evals.fc.models import Engine, Evidence, FCItem, FCResult, Status

NOW = datetime(2026, 9, 23, 14, 32, 10, tzinfo=timezone.utc)


def _mech(cid, status=Status.PASS):
    return FCItem(id=cid, engine=Engine.MECHANICAL, status=status,
                  evidence=Evidence(detail="ok"))


def _model(cid, status):
    return FCItem(id=cid, engine=Engine.MODEL, status=status, evidence=Evidence(detail="ok"))


MODEL_CFG = {"name": "qwen/qwen3-8b", "provider": "openrouter", "temperature": 0.6,
             "thinking": False, "votes": 3, "endpoint": "https://x/api/v1"}


def test_model_grader_present_only_when_model_ran():
    items = [_mech("fc.anchors.unique-ids"), _model("fc.markers.tbd", Status.PASS)]
    result = build_result("w.md", items, NOW, MODEL_CFG)
    engines = [g.engine for g in result.graders]
    assert engines == ["mechanical", "model"]


def test_model_grader_absent_when_all_model_items_skipped():
    items = [_mech("fc.anchors.unique-ids"), _model("fc.markers.tbd", Status.SKIPPED)]
    result = build_result("w.md", items, NOW, MODEL_CFG)
    assert [g.engine for g in result.graders] == ["mechanical"]
    assert result.summary.complete is False        # a skipped item forces provisional


def test_result_to_json_and_pydantic_round_trip():
    items = [_mech("fc.anchors.unique-ids")]
    result = build_result("w.md", items, NOW, None)
    text = result_to_json(result)
    data = json.loads(text)
    assert data["eval"] == "fc" and data["summary"]["pass"] == 1
    # Always-on guarantee: the emitted JSON re-validates as an FCResult.
    FCResult.model_validate(data)


def test_write_result_to_disk(tmp_path):
    result = build_result("w.md", [_mech("fc.anchors.unique-ids")], NOW, None)
    out = tmp_path / "fc.json"
    write_result(result, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["wof"] == "w.md"


def test_emitted_result_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parent.parent / "fc-result.schema.json"
    if not schema_path.exists():
        pytest.skip("fc-result.schema.json not present")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    items = [_mech("fc.anchors.unique-ids"), _model("fc.markers.tbd", Status.PASS)]
    result = build_result("w.md", items, NOW, MODEL_CFG)
    jsonschema.validate(json.loads(result_to_json(result)), schema)
