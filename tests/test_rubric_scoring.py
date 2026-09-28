"""Tests for evals/rubric/scoring.py -- bucket() + build_rubric_summary. Fully offline."""

from evals.common.models import Engine, Evidence, Item, Status
from evals.rubric.scoring import bucket, build_rubric_summary


def _item(cid, status):
    return Item(id=cid, engine=Engine.MODEL, status=status, evidence=Evidence(detail="d"))


def test_bucket_splits_on_sourcing_prefix():
    items = [
        _item("rubric.sections.locations", Status.PASS),
        _item("rubric.presence.locations.aldenburg", Status.PASS),
        _item("rubric.pc-stated.aerin", Status.PASS),
        _item("rubric.sourcing.locations.aldenburg", Status.FAIL),
        _item("rubric.sourcing.characters.aerin", Status.PASS),
    ]
    presence, sourcing = bucket(items)
    assert {it.id for it in sourcing} == {"rubric.sourcing.locations.aldenburg",
                                          "rubric.sourcing.characters.aerin"}
    assert len(presence) == 3


def test_summary_math_presence_and_sourcing_independent():
    items = [
        # presence bucket: 2 pass, 1 fail, 1 na -> applicable 3, score 2/3
        _item("rubric.sections.locations", Status.PASS),
        _item("rubric.presence.locations.a", Status.PASS),
        _item("rubric.presence.locations.b", Status.FAIL),
        _item("rubric.pc-marked.x", Status.NA),
        # sourcing bucket: 1 pass, 1 fail -> applicable 2, score 1/2
        _item("rubric.sourcing.locations.a", Status.PASS),
        _item("rubric.sourcing.locations.b", Status.FAIL),
    ]
    s = build_rubric_summary(items)
    assert s.complete is True                          # no skipped
    assert s.presence.applicable == 3 and s.presence.n_pass == 2 and s.presence.score_display == "2/3"
    assert s.sourcing.applicable == 2 and s.sourcing.n_pass == 1 and s.sourcing.score_display == "1/2"
    assert s.presence.na == 1 and s.sourcing.na == 0


def test_empty_bucket_scores_none_and_zero_display():
    # sourcing entirely na -> applicable 0 -> score None, "0/0"
    items = [
        _item("rubric.sections.locations", Status.PASS),
        _item("rubric.sourcing.locations.a", Status.NA),
    ]
    s = build_rubric_summary(items)
    assert s.sourcing.applicable == 0 and s.sourcing.score is None and s.sourcing.score_display == "0/0"


def test_complete_false_when_any_skipped():
    items = [
        _item("rubric.sections.locations", Status.SKIPPED),
        _item("rubric.presence.locations.a", Status.PASS),
    ]
    s = build_rubric_summary(items)
    assert s.complete is False
