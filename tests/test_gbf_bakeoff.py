"""The GBF eval-of-the-eval: a hand-labeled adversarial bake-off (Brief C §4).

Runs candidate judge models over the fabricated WOF variants in tests/fixtures/gbf/ (graded against
the REAL gold) and reports agreement with the hand labels in expected.yaml + vote stability. This is
how the judge is chosen, the ranks proven, and GBF's ongoing regression guard. Opt-in only:
``pytest -m integration -k bakeoff``. Skips when LLM_OPENAI_* or the gitignored gold/ranks are absent.

Because pairing locks these fixtures by exact anchor, ORDERING is fully deterministic (no model) --
so ordering agreement must be perfect for every candidate; a miss there is a real regression.
FAITHFULNESS is the fuzzy, model-dependent signal the bake-off exists to measure.
"""

import os
from pathlib import Path

import pytest
import yaml

from evals.common.model_client import OpenAICompatModelClient
from evals.gbf.reference import load_gold, load_ranks
from evals.gbf.runner import grade_file

_ROOT = Path(__file__).resolve().parent.parent
_GOLD = _ROOT / "output" / "gol-lore-full.md"
_RANKS = _ROOT / "output" / "gbf_event_ranks.yaml"
_FIXTURES = _ROOT / "tests" / "fixtures" / "gbf"
_EXPECTED = _FIXTURES / "expected.yaml"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("LLM_OPENAI_API_KEY") or not os.environ.get("LLM_OPENAI_BASE_URL"),
        reason="LLM_OPENAI_API_KEY / LLM_OPENAI_BASE_URL not set",
    ),
    pytest.mark.skipif(not (_GOLD.exists() and _RANKS.exists()), reason="gold or ranks not present"),
]

# The candidates to compare. The pinned DeepSeek is the intended default; Qwen is the shared default
# baseline. Add "deepseek/deepseek-v4.1-flash" here to include the cheaper DeepSeek.
_CANDIDATES = ["qwen/qwen3-8b", "deepseek/deepseek-v4-pro-0813"]
_VOTES = int(os.environ.get("WIKITIZER_GBF_VOTES", "3"))
_MAX_WORKERS = int(os.environ.get("WIKITIZER_GBF_MAX_WORKERS", "6"))


def _client(model):
    return OpenAICompatModelClient(
        model=model, temperature=0.6, thinking=False,
        base_url=os.environ["LLM_OPENAI_BASE_URL"], api_key=os.environ["LLM_OPENAI_API_KEY"])


def _expected():
    return yaml.safe_load(_EXPECTED.read_text(encoding="utf-8"))


def _grade(client, votes):
    """Grade every variant, returning {variant: {item_id: status_str}}."""
    gold = load_gold(_GOLD)
    ranks = load_ranks(_RANKS)
    out = {}
    for variant in _expected():
        result = grade_file(_FIXTURES / variant, gold, ranks, votes=votes, use_model=True,
                            model_client=client, max_workers=_MAX_WORKERS)
        out[variant] = {it.id: it.status.value for it in result.items}
    return out


def _agreement(graded, expected):
    """Return (ordering_hits, ordering_total, faith_hits, faith_total, misses)."""
    o_hit = o_tot = f_hit = f_tot = 0
    misses = []
    for variant, spec in expected.items():
        got = graded[variant]
        exp_ord = spec.get("ordering")
        if exp_ord is not None:
            o_tot += 1
            if got.get("gbf.ordering") == exp_ord:
                o_hit += 1
            else:
                misses.append(f"{variant} ordering: expected {exp_ord}, got {got.get('gbf.ordering')}")
        for slug, exp in (spec.get("faithfulness") or {}).items():
            f_tot += 1
            got_status = got.get(f"gbf.faithfulness.{slug}")
            if got_status == exp:
                f_hit += 1
            else:
                misses.append(f"{variant} faithfulness.{slug}: expected {exp}, got {got_status}")
    return o_hit, o_tot, f_hit, f_tot, misses


def test_bakeoff_reports_agreement_and_ordering_is_deterministic(capsys):
    expected = _expected()
    best_faith = 0.0
    ran_any = False
    with capsys.disabled():
        print("\n=== GBF bake-off (votes=%d) ===" % _VOTES)
        for model in _CANDIDATES:
            try:
                graded = _grade(_client(model), _VOTES)
            except Exception as exc:                      # noqa: BLE001 -- a model may be unavailable
                print(f"  {model}: unavailable ({exc})")
                continue
            ran_any = True
            o_hit, o_tot, f_hit, f_tot, misses = _agreement(graded, expected)
            faith_frac = (f_hit / f_tot) if f_tot else 0.0
            best_faith = max(best_faith, faith_frac)
            print(f"  {model}: ordering {o_hit}/{o_tot}, faithfulness {f_hit}/{f_tot} "
                  f"({faith_frac:.0%})")
            for m in misses:
                print(f"      - {m}")
            # Ordering is deterministic (pairing locks by anchor) -> it must be perfect.
            assert o_hit == o_tot, f"{model} ordering regressed: {[m for m in misses if 'ordering' in m]}"

    if not ran_any:
        pytest.skip("no candidate model was reachable")
    # A gross faithfulness failure (best judge below half agreement) is a real problem worth failing.
    assert best_faith >= 0.5, f"best faithfulness agreement was only {best_faith:.0%}"
