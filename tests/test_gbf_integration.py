"""Live end-to-end GBF test against the real judge (DeepSeek/Qwen via OpenRouter). Opt-in only:
``pytest -m integration -k gbf``. Skips when LLM_OPENAI_* is absent or the gitignored gold / ranks
are missing. Assertions are loose because LLM output isn't deterministic.
"""

import json
import os
from pathlib import Path

import pytest

from evals.common.model_client import build_model_client
from evals.gbf.emit import result_to_json
from evals.gbf.reference import load_gold, load_ranks
from evals.gbf.runner import grade_file

_ROOT = Path(__file__).resolve().parent.parent
_GOLD = _ROOT / "output" / "gol-lore-full.md"
_RANKS = _ROOT / "output" / "gbf_event_ranks.yaml"

def _int_env(name, default):
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("LLM_OPENAI_API_KEY") or not os.environ.get("LLM_OPENAI_BASE_URL"),
        reason="LLM_OPENAI_API_KEY / LLM_OPENAI_BASE_URL not set",
    ),
    pytest.mark.skipif(
        not (_GOLD.exists() and _RANKS.exists()),
        reason="gold or output/gbf_event_ranks.yaml not present",
    ),
]


@pytest.mark.integration
def test_live_gbf_grades_gold_against_itself():
    # The gold graded against ITSELF is the strongest self-check: faithfulness should be near-perfect
    # and ordering should pass.
    jsonschema = pytest.importorskip("jsonschema")
    client = build_model_client(prefix="GBF")
    assert client is not None
    gold = load_gold(_GOLD)
    ranks = load_ranks(_RANKS)
    # Honor the documented knobs so `WIKITIZER_GBF_VOTES=1` actually trims the run, and fan the
    # (independent, ~20s-each) faithfulness calls out over a thread pool so this finishes in minutes.
    votes = _int_env("WIKITIZER_GBF_VOTES", 3)
    max_workers = _int_env("WIKITIZER_GBF_MAX_WORKERS", 6)
    result = grade_file(_GOLD, gold, ranks, votes=votes, use_model=True, model_client=client,
                        max_workers=max_workers)

    assert result.eval == "gbf"
    assert result.summary.complete is True
    # gold vs itself: every entry pairs and faithfulness should be high
    assert result.summary.faithfulness.score is not None and result.summary.faithfulness.score >= 0.8
    ordering = next(it for it in result.items if it.id == "gbf.ordering")
    assert ordering.status.value == "pass"
    assert all(it.evidence.detail for it in result.items)

    schema = json.loads((_ROOT / "gbf-result.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(result_to_json(result)), schema)
