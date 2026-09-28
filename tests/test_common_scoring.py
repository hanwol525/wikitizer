"""Tests for evals/common/scoring.py -- the applicable-only summary math. Fully offline."""

from evals.common.models import Engine, Evidence, Item, Status
from evals.common.scoring import ScoreBlock, build_summary, score_block


def _items(**counts):
    """Build a list of Items with the given per-status counts."""
    out = []
    n = 0
    for status_name, k in counts.items():
        status = Status(status_name)
        for _ in range(k):
            out.append(Item(id=f"fc.x.{n}", engine=Engine.MECHANICAL, status=status,
                              evidence=Evidence(detail="d")))
            n += 1
    return out


def test_applicable_excludes_na_and_skipped():
    s = build_summary(_items(**{"pass": 22, "partial": 1, "fail": 2, "na": 1, "skipped": 0}))
    assert s.applicable == 25          # 22 + 1 + 2, na excluded
    assert s.na == 1 and s.skipped == 0


def test_score_and_display_match_schema_example():
    s = build_summary(_items(**{"pass": 22, "partial": 1, "fail": 2, "na": 1, "skipped": 0}))
    assert s.score == 0.88
    assert s.score_display == "22/25"


def test_partial_scores_as_fail():
    # 3 pass + 1 partial => applicable 4, score 3/4 (the partial does NOT count as a pass).
    s = build_summary(_items(**{"pass": 3, "partial": 1}))
    assert s.applicable == 4 and s.score == 0.75 and s.score_display == "3/4"


def test_score_none_when_nothing_applicable():
    s = build_summary(_items(**{"na": 2, "skipped": 3}))
    assert s.applicable == 0 and s.score is None and s.score_display == "0/0"


def test_complete_reflects_skipped():
    assert build_summary(_items(**{"pass": 5}))
    assert build_summary(_items(**{"pass": 5})).complete is True
    assert build_summary(_items(**{"pass": 5, "skipped": 3})).complete is False


def test_all_pass_scores_one():
    s = build_summary(_items(**{"pass": 4}))
    assert s.score == 1.0 and s.score_display == "4/4" and s.complete is True


# --- score_block (the reusable per-bucket scorer the rubric eval consumes) --- #

def test_score_block_applicable_math_matches_build_summary():
    items = _items(**{"pass": 22, "partial": 1, "fail": 2, "na": 1, "skipped": 3})
    block = score_block(items)
    assert isinstance(block, ScoreBlock)
    assert block.n_pass == 22 and block.partial == 1 and block.fail == 2 and block.na == 1
    assert block.applicable == 25 and block.score == 0.88 and block.score_display == "22/25"
    # score_block agrees with build_summary on every shared field (it IS build_summary's core).
    s = build_summary(items)
    assert (block.n_pass, block.partial, block.fail, block.na, block.applicable,
            block.score, block.score_display) == (
        s.n_pass, s.partial, s.fail, s.na, s.applicable, s.score, s.score_display)


def test_score_block_has_no_skipped_or_complete():
    # ScoreBlock carries only the applicable-only tally; skipped/complete are run-level.
    fields = set(ScoreBlock.__dataclass_fields__)
    assert "skipped" not in fields and "complete" not in fields


def test_score_block_none_when_nothing_applicable():
    block = score_block(_items(**{"na": 2, "skipped": 3}))
    assert block.applicable == 0 and block.score is None and block.score_display == "0/0"


def test_score_block_ignores_skipped_in_applicable():
    # skipped items don't count toward applicable (only pass/partial/fail do).
    block = score_block(_items(**{"pass": 3, "skipped": 5}))
    assert block.applicable == 3 and block.score == 1.0 and block.score_display == "3/3"
