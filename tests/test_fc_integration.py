"""Live FC adjudicator test against the real judge (default Qwen3-8B via OpenRouter).

Opt-in only: marked ``integration`` (deselected by plain ``pytest`` via pytest.ini) and
``skipif``-ed when the OpenAI-compat credentials are absent. Assertions are deliberately loose
-- LLM output isn't deterministic -- and it needs the gitignored golden file, so it also skips
when that's missing.

Run it with:  pytest -m integration -k fc
"""

import os
from pathlib import Path

import pytest

from evals.fc.model_client import build_model_client
from evals.fc.models import Status
from evals.fc.runner import grade_file

_GOLDEN = Path(__file__).resolve().parent.parent / "output" / "gol-lore-full.md"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.environ.get("LLM_OPENAI_API_KEY") and os.environ.get("LLM_OPENAI_BASE_URL")),
        reason="LLM_OPENAI_API_KEY / LLM_OPENAI_BASE_URL not set",
    ),
    pytest.mark.skipif(not _GOLDEN.exists(),
                       reason="output/gol-lore-full.md not present (gitignored)"),
]


@pytest.mark.integration
def test_live_judge_grades_golden():
    client = build_model_client()
    assert client is not None
    result = grade_file(_GOLDEN, votes=3, use_model=True, model_client=client)

    model_items = [i for i in result.items if i.engine.value == "model"]
    assert len(model_items) == 3
    # The judge actually ran: none skipped, the run is complete, a model grader is recorded.
    assert all(i.status != Status.SKIPPED for i in model_items)
    assert result.summary.complete is True
    assert any(g.engine == "model" for g in result.graders)
    # Every item resolved to a real status with evidence.
    assert all(i.evidence.detail for i in result.items)
