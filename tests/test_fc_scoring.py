"""Tests for evals/fc/scoring.py -- the applicable-only summary math. Fully offline."""

from evals.fc.models import Engine, Evidence, FCItem, Status
from evals.fc.scoring import build_summary


def _items(**counts):
    """Build a list of FCItems with the given per-status counts."""
    out = []
    n = 0
    for status_name, k in counts.items():
        status = Status(status_name)
        for _ in range(k):
            out.append(FCItem(id=f"fc.x.{n}", engine=Engine.MECHANICAL, status=status,
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
