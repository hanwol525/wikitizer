"""Tests for evals/gbf/scoring.py -- bucketing + the derived two-sub-score summary. Fully offline."""

from evals.common.models import Engine, Evidence, Item, Status
from evals.gbf.scoring import bucket, build_gbf_summary


def _it(cid, status):
    return Item(id=cid, engine=Engine.MODEL, status=status, evidence=Evidence(detail="d"))


def test_bucket_splits_ordering_from_faithfulness():
    items = [
        _it("gbf.faithfulness.aldenburg", Status.PASS),
        _it("gbf.faithfulness.citadel", Status.FAIL),
        _it("gbf.ordering", Status.PASS),
    ]
    faithfulness, ordering = bucket(items)
    assert [i.id for i in faithfulness] == ["gbf.faithfulness.aldenburg", "gbf.faithfulness.citadel"]
    assert [i.id for i in ordering] == ["gbf.ordering"]


def test_summary_math_is_independent_per_bucket():
    items = [
        _it("gbf.faithfulness.a", Status.PASS),
        _it("gbf.faithfulness.b", Status.PARTIAL),
        _it("gbf.faithfulness.c", Status.FAIL),
        _it("gbf.faithfulness.d", Status.NA),
        _it("gbf.ordering", Status.PASS),
    ]
    s = build_gbf_summary(items)
    # faithfulness: pass=1 partial=1 fail=1 na=1 -> applicable=3; partial earns HALF credit (GBF-only)
    # so credit = 1 + 0.5 = 1.5 -> score = 1.5/3 = 0.5. The unweighted counts stay visible.
    assert s.faithfulness.n_pass == 1 and s.faithfulness.partial == 1 and s.faithfulness.applicable == 3
    assert s.faithfulness.score == 0.5
    assert s.faithfulness.score_display == "1.5/3"
    # ordering: one pass -> 1/1
    assert s.ordering.n_pass == 1 and s.ordering.applicable == 1 and s.ordering.score == 1.0


def test_complete_is_derived_from_skipped():
    ok = [_it("gbf.faithfulness.a", Status.PASS), _it("gbf.ordering", Status.PASS)]
    assert build_gbf_summary(ok).complete is True
    degraded = [_it("gbf.faithfulness.a", Status.SKIPPED), _it("gbf.ordering", Status.PASS)]
    assert build_gbf_summary(degraded).complete is False


def test_empty_bucket_scores_none_and_zero_display():
    # only an ordering item -> faithfulness bucket is empty
    s = build_gbf_summary([_it("gbf.ordering", Status.NA)])
    assert s.faithfulness.applicable == 0 and s.faithfulness.score is None
    assert s.faithfulness.score_display == "0/0"
    # ordering na -> also 0 applicable
    assert s.ordering.applicable == 0 and s.ordering.score is None
