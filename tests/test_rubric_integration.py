"""Live end-to-end rubric test against the real judge (Qwen via OpenRouter). Opt-in only:
``pytest -m integration -k rubric``. Skips when LLM_OPENAI_* is absent or the gitignored golden
WOF / output/rubric.yaml is missing. Assertions are loose because LLM output isn't deterministic.
"""

import json
import os
from pathlib import Path

import pytest

from evals.common.model_client import build_model_client
from evals.rubric.emit import result_to_json
from evals.rubric.loader import load_rubric
from evals.rubric.runner import grade_file

_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN = _ROOT / "output" / "gol-lore-full.md"
_RUBRIC = _ROOT / "output" / "rubric.yaml"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("LLM_OPENAI_API_KEY") or not os.environ.get("LLM_OPENAI_BASE_URL"),
        reason="LLM_OPENAI_API_KEY / LLM_OPENAI_BASE_URL not set",
    ),
    pytest.mark.skipif(
        not (_GOLDEN.exists() and _RUBRIC.exists()),
        reason="golden WOF or output/rubric.yaml not present",
    ),
]


@pytest.mark.integration
def test_live_rubric_grades_golden():
    jsonschema = pytest.importorskip("jsonschema")
    client = build_model_client(prefix="RUBRIC")
    assert client is not None
    rubric = load_rubric(_RUBRIC)
    result = grade_file(_GOLDEN, rubric, votes=3, use_model=True, model_client=client)

    assert result.eval == "rubric"
    assert result.summary.complete is True
    # the golden IS the reference roster, so presence should be high
    assert result.summary.presence.score is not None and result.summary.presence.score >= 0.7
    assert result.summary.sourcing.score is not None            # sourcing resolved (not all na)
    assert all(it.evidence.detail for it in result.items)       # every item carries evidence

    schema = json.loads((_ROOT / "rubric-result.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(result_to_json(result)), schema)
