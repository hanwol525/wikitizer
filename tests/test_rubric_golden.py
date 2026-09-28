"""Offline golden bar for the all-model rubric eval: run --no-model over the real golden WOF
and confirm the pipeline emits an all-skipped, schema-valid result. Skips when the gitignored
golden WOF or output/rubric.yaml is absent (a fresh clone). No model, no key."""

import json
from pathlib import Path

import pytest

from evals.rubric.emit import result_to_json
from evals.rubric.loader import load_rubric
from evals.rubric.runner import grade_file

_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN = _ROOT / "output" / "gol-lore-full.md"
_RUBRIC = _ROOT / "output" / "rubric.yaml"

pytestmark = pytest.mark.skipif(
    not (_GOLDEN.exists() and _RUBRIC.exists()),
    reason="golden WOF (output/gol-lore-full.md) or output/rubric.yaml not present",
)


def test_no_model_golden_is_all_skipped_and_schema_valid():
    jsonschema = pytest.importorskip("jsonschema")
    rubric = load_rubric(_RUBRIC)
    result = grade_file(_GOLDEN, rubric, use_model=False)
    assert result.summary.complete is False
    assert result.items and all(it.status.value == "skipped" for it in result.items)
    assert result.summary.presence.score_display == "0/0"
    assert result.summary.sourcing.score_display == "0/0"
    schema = json.loads((_ROOT / "rubric-result.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(result_to_json(result)), schema)
