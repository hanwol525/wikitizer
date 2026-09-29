"""Tests for evals/gbf/models.py -- GbfSummary/GbfResult (+ the shared SubScore). Fully offline."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evals.common.models import Engine, Evidence, Item, ModelGrader, Status
from evals.gbf.models import GbfResult, GbfSummary, SubScore


def _sub(**over):
    base = dict(n_pass=2, partial=0, fail=1, na=1, applicable=3, score=0.6667, score_display="2/3")
    base.update(over)
    return SubScore(**base)


def _result(**over):
    base = dict(
        wof="w.md",
        graded_at=datetime(2026, 9, 23, 14, 32, 10, tzinfo=timezone.utc),
        graders=[ModelGrader(name="deepseek/deepseek-v4-pro-0813", provider="openrouter",
                             temperature=0.6, thinking=False, votes=5)],
        summary=GbfSummary(complete=True, faithfulness=_sub(),
                           ordering=_sub(n_pass=1, fail=0, na=0, applicable=1, score=1.0,
                                         score_display="1/1")),
        items=[Item(id="gbf.ordering", engine=Engine.MODEL, status=Status.PASS,
                    evidence=Evidence(detail="ordered"))],
    )
    base.update(over)
    return GbfResult(**base)


def test_subscore_pass_alias_both_forms():
    by_name = _sub()
    by_alias = SubScore(**{"pass": 2, "partial": 0, "fail": 1, "na": 1, "applicable": 3,
                           "score": 0.6667, "score_display": "2/3"})
    assert by_name.n_pass == by_alias.n_pass == 2


def test_subscore_json_key_is_pass_not_n_pass():
    d = _sub().model_dump(mode="json", by_alias=True)
    assert d["pass"] == 2 and "n_pass" not in d


def test_extra_forbidden_everywhere():
    with pytest.raises(ValidationError):
        GbfSummary(complete=True, faithfulness=_sub(), ordering=_sub(), bogus=1)
    with pytest.raises(ValidationError):
        _result(bogus=1)


def test_eval_const_and_schema_version():
    r = _result()
    assert r.eval == "gbf" and r.schema_version == "1.0"


def test_full_round_trip_is_stable():
    r = _result()
    d = r.model_dump(mode="json", by_alias=True)
    again = GbfResult.model_validate(d)
    assert again.model_dump(mode="json", by_alias=True) == d


def test_both_nullable_subscore_scores_serialize():
    r = _result(summary=GbfSummary(
        complete=False,
        faithfulness=_sub(n_pass=0, partial=0, fail=0, na=0, applicable=0, score=None, score_display="0/0"),
        ordering=_sub(n_pass=0, partial=0, fail=0, na=0, applicable=0, score=None, score_display="0/0"),
    ))
    d = r.model_dump(mode="json", by_alias=True)
    assert d["summary"]["faithfulness"]["score"] is None
    assert d["summary"]["ordering"]["score"] is None


def test_graded_at_serializes_iso():
    d = _result().model_dump(mode="json", by_alias=True)
    assert d["graded_at"].startswith("2026-09-23T14:32:10")
