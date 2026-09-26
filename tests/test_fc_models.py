"""Tests for evals/fc/models.py -- the FC-specific FCResult + the Summary `pass` alias.

Fully offline. Covers FIX #1 (the `pass` keyword -> `n_pass` alias), the `eval="fc"` /
schema_version consts, the full FCResult round-trip, and graded_at ISO serialization. The
shared envelope pieces (extra=forbid, enum serialization, the discriminated Grader union)
are tested in tests/test_common_models.py.
"""

from datetime import datetime, timezone

from evals.fc.models import (
    Engine,
    Evidence,
    FCItem,
    FCResult,
    MechanicalGrader,
    Status,
    Summary,
)


def _summary(**over):
    base = dict(n_pass=22, partial=1, fail=2, na=1, skipped=0, applicable=25,
                score=0.88, score_display="22/25", complete=True)
    base.update(over)
    return Summary(**base)


def _result(**over):
    base = dict(
        wof="w.md",
        graded_at=datetime(2026, 9, 23, 14, 32, 10, tzinfo=timezone.utc),
        graders=[MechanicalGrader(tool="fc_lint.py", version="0.1.0")],
        summary=_summary(),
        items=[FCItem(id="fc.anchors.unique-ids", engine=Engine.MECHANICAL,
                      status=Status.PASS, evidence=Evidence(detail="0 collisions"))],
    )
    base.update(over)
    return FCResult(**base)


# --- FIX #1: the `pass` keyword alias --------------------------------------- #

def test_summary_pass_alias_both_construction_forms():
    by_name = Summary(n_pass=3, partial=0, fail=0, na=0, skipped=0, applicable=3,
                      score=1.0, score_display="3/3", complete=True)
    by_alias = Summary(**{"pass": 3, "partial": 0, "fail": 0, "na": 0, "skipped": 0,
                          "applicable": 3, "score": 1.0, "score_display": "3/3", "complete": True})
    assert by_name.n_pass == by_alias.n_pass == 3


def test_summary_json_key_is_pass_not_n_pass():
    d = _summary().model_dump(mode="json", by_alias=True)
    assert d["pass"] == 22 and "n_pass" not in d


# --- eval/schema consts + round-trip + serialization ------------------------ #

def test_eval_and_engine_consts():
    r = _result()
    assert r.eval == "fc" and r.schema_version == "1.0"
    assert r.graders[0].engine == "mechanical"


def test_full_round_trip_is_stable():
    r = _result()
    d = r.model_dump(mode="json", by_alias=True)
    again = FCResult.model_validate(d)
    assert again.model_dump(mode="json", by_alias=True) == d


def test_graded_at_serializes_iso():
    d = _result().model_dump(mode="json", by_alias=True)
    assert d["graded_at"].startswith("2026-09-23T14:32:10")


def test_score_may_be_null():
    s = _summary(n_pass=0, partial=0, fail=0, applicable=0, score=None, score_display="0/0")
    assert s.model_dump(mode="json", by_alias=True)["score"] is None
