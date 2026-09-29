"""The GBF eval-of-the-eval: a hand-labeled adversarial bake-off (Brief C §4).

Runs candidate judge models over the fabricated WOF variants in tests/fixtures/gbf/ (graded against
the REAL gold) and reports agreement with the hand labels in expected.yaml + vote stability. This is
how the judge is chosen, the ranks proven, and GBF's ongoing regression guard. Opt-in only:
``pytest -m integration -k bakeoff``. Skips when LLM_OPENAI_* or the gitignored gold/ranks are absent.

Because pairing locks these fixtures by exact anchor, ORDERING is fully deterministic (no model) --
so ordering agreement must be perfect for every candidate; a miss there is a real regression.
FAITHFULNESS is the fuzzy, model-dependent signal the bake-off exists to measure. It is scored
FAIL-AWARE: the fixtures are abridged paraphrases of the rich real gold, so a faithful judge legitimately
returns ``partial`` (omitted gold claims) where a label optimistically says ``pass`` -- and ``pass`` /
``partial`` are one "faithful, no contradiction" class, with ``fail`` the distinct signal (see
``_faith_class``). The asserted number is that fail-aware agreement (does the judge catch
contradictions and not hallucinate them); the strict 3-way exact-match agreement + the partial<->pass
"slips" are printed for calibration but never fail the build. Because that fail-aware bar is clearable
by a contradiction-BLIND judge (the fixtures fold to ~18 "ok" vs 2 "fail"), the bake-off ALSO asserts a
contradiction-recall floor -- the best candidate must flag every entry labeled `fail`.
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


def _faith_class(status):
    """Fold a faithfulness status into the axis the bake-off asserts on. ``pass`` and ``partial`` are
    BOTH "faithful, no contradiction" -- a partial (faithful but abridged) is not the same error as a
    fail (a meaning contradiction), so the two must not be lumped. ``fail`` is the distinct signal a
    faithfulness judge must get right. Anything else (``na``/``skipped``/``None`` on a present entity
    -- a pairing miss) is kept distinct so it can never silently pass as "ok"."""
    if status == "fail":
        return "fail"
    if status in ("pass", "partial"):
        return "ok"
    return status or "missing"


def _agreement(graded, expected):
    """Return (ordering_hits, ordering_total, faith_hits, faith_total, strict_hits, strict_total,
    misses, slips).

    The ASSERTED faithfulness number (``faith_hits``/``faith_total``) is fail-aware: it agrees when
    ``_faith_class(got) == _faith_class(exp)`` -- so a judge PARTIAL where the label expected PASS is a
    hit (both "ok"), while a missed or hallucinated contradiction, or a pairing na/skipped on a present
    entity, is a real miss. ``strict_hits``/``strict_total`` is the informational 3-way exact-match
    agreement; ``slips`` are the harmless partial<->pass differences (reported, never fatal)."""
    o_hit = o_tot = f_hit = f_tot = strict_hit = strict_tot = 0
    misses = []
    slips = []
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
            got_status = got.get(f"gbf.faithfulness.{slug}")
            strict_tot += 1
            if got_status == exp:
                strict_hit += 1
            f_tot += 1
            if _faith_class(got_status) == _faith_class(exp):
                f_hit += 1
                if got_status != exp:            # same class, different label -> a partial<->pass slip
                    slips.append(f"{variant} faithfulness.{slug}: {exp} -> {got_status}")
            else:
                misses.append(f"{variant} faithfulness.{slug}: expected {exp} "
                              f"({_faith_class(exp)}), got {got_status} ({_faith_class(got_status)})")
    return o_hit, o_tot, f_hit, f_tot, strict_hit, strict_tot, misses, slips


def _contradiction_recall(graded, expected):
    """Return (hits, total, missed): of the entries LABELED ``fail`` (real, blatant contradictions),
    how many did the judge return ``fail``? This is the ONE thing the fail-aware >=0.5 bar can't
    measure -- the fixtures fold to ~18 "ok" vs 2 "fail", so a contradiction-BLIND judge (all pass)
    still clears 0.5. So the bake-off also asserts the best candidate recalls every contradiction."""
    hit = tot = 0
    missed = []
    for variant, spec in expected.items():
        got = graded[variant]
        for slug, exp in (spec.get("faithfulness") or {}).items():
            if exp != "fail":
                continue
            tot += 1
            got_status = got.get(f"gbf.faithfulness.{slug}")
            if got_status == "fail":
                hit += 1
            else:
                missed.append(f"{variant} faithfulness.{slug}: contradiction NOT caught, got {got_status}")
    return hit, tot, missed


def test_bakeoff_reports_agreement_and_ordering_is_deterministic(capsys):
    expected = _expected()
    best_faith = 0.0
    best_recall = 0                                            # max contradictions caught by any candidate
    recall_total = 0
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
            o_hit, o_tot, f_hit, f_tot, s_hit, s_tot, misses, slips = _agreement(graded, expected)
            faith_frac = (f_hit / f_tot) if f_tot else 0.0        # fail-aware (asserted)
            strict_frac = (s_hit / s_tot) if s_tot else 0.0       # 3-way exact (informational)
            best_faith = max(best_faith, faith_frac)
            c_hit, c_tot, c_missed = _contradiction_recall(graded, expected)
            recall_total = c_tot                              # constant across candidates (same fixtures)
            best_recall = max(best_recall, c_hit)
            print(f"  {model}: ordering {o_hit}/{o_tot}, faithfulness (fail-aware) {f_hit}/{f_tot} "
                  f"({faith_frac:.0%}); strict 3-way {s_hit}/{s_tot} ({strict_frac:.0%}); "
                  f"contradiction recall {c_hit}/{c_tot}")
            for m in misses:
                print(f"      - MISS {m}")
            for cm in c_missed:
                print(f"      - CONTRADICTION {cm}")
            for sl in slips:                                      # partial<->pass, tolerated
                print(f"      - slip {sl}")
            # Ordering is deterministic (pairing locks by anchor) -> it must be perfect.
            assert o_hit == o_tot, f"{model} ordering regressed: {[m for m in misses if 'ordering' in m]}"

    if not ran_any:
        pytest.skip("no candidate model was reachable")
    # The bake-off asserts on the fail-aware axis: catching contradictions (and not hallucinating
    # them) is the judge's real job; a partial-vs-pass slip on an abridged fixture is not a failure.
    assert best_faith >= 0.5, f"best fail-aware faithfulness agreement was only {best_faith:.0%}"
    # ...but the >=0.5 bar alone is clearable by a contradiction-blind judge (~18 "ok" vs 2 "fail"), so
    # ALSO require the best candidate to actually catch every labeled contradiction. These fixtures are
    # blatant (a relocated city denying its family tie; a bodyguard rewritten as the prince's son), so a
    # usable judge must flag them; a miss here is a real judge-selection failure, not fixture noise.
    if recall_total:
        assert best_recall == recall_total, (
            f"best candidate caught only {best_recall}/{recall_total} labeled contradictions")
