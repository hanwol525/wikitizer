"""Tests for evals/fc/adjudicator.py -- the 3 FC judge checks (grouping / tilde / tbd) and
their 0-candidate NA short-circuits. Fully offline (a FakeModelClient feeds canned JSON
replies). The shared voting engine (FIX #5 vote hygiene, fail-closed, majority) is tested in
tests/test_common_adjudicator.py.
"""

import json

from evals.fc.adjudicator import Adjudicator
from evals.fc.models import Status


class FakeModelClient:
    """Pops one queued reply per ``complete`` call and records the prompts it saw."""

    model = "fake/qwen"
    provider = "openrouter"
    temperature = 0.6
    thinking = False
    base_url = "https://example/api/v1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else ""


def verdicts(*labels):
    return json.dumps({"verdicts": list(labels)})


# --- grouping --------------------------------------------------------------- #

def test_grouping_all_group_labels_pass():
    fake = FakeModelClient([verdicts("group_label")] * 3)
    item = Adjudicator(fake, votes=3).judge_grouping(
        [{"label": "Could Not Place", "lineno": 5, "entries_beneath": ["X"]}])
    assert item.status == Status.PASS
    assert item.engine.value == "model"
    assert len(fake.calls) == 3


def test_grouping_missing_anchor_majority_fails():
    fake = FakeModelClient([verdicts("missing_anchor"), verdicts("group_label"),
                            verdicts("missing_anchor")])
    item = Adjudicator(fake, votes=3).judge_grouping(
        [{"label": "Aldenburg", "lineno": 9, "entries_beneath": []}])
    assert item.status == Status.FAIL
    assert "Aldenburg" in (item.evidence.found or "")
    assert item.evidence.lines == [9]


def test_zero_candidates_short_circuits_to_na_without_a_call():
    # 0 candidates -> NA (excluded from the denominator, matching the mechanical convention),
    # and no model call is made.
    fake = FakeModelClient([])
    item = Adjudicator(fake, votes=3).judge_grouping([])
    assert item.status == Status.NA and fake.calls == []


# --- tilde ------------------------------------------------------------------ #

def test_tilde_flags_approximate_unmarked():
    fake = FakeModelClient([verdicts("approximate_unmarked")] * 3)
    item = Adjudicator(fake, votes=3).judge_approx_tilde(
        [{"figure": "200", "sentence": "about 200 years ago", "lineno": 12}])
    assert item.status == Status.FAIL


def test_tilde_exact_ok_passes():
    fake = FakeModelClient([verdicts("exact_ok")] * 3)
    item = Adjudicator(fake, votes=3).judge_approx_tilde(
        [{"figure": "1424", "sentence": "the year is 1424", "lineno": 12}])
    assert item.status == Status.PASS


# --- tbd -------------------------------------------------------------------- #

def test_tbd_stub_fails_and_ok_passes():
    stub = Adjudicator(FakeModelClient([verdicts("stub_needs_tbd")] * 3), votes=3).judge_tbd(
        [{"name": "Ned", "body": "no details yet", "lineno": 20}])
    assert stub.status == Status.FAIL
    ok = Adjudicator(FakeModelClient([verdicts("ok")] * 3), votes=3).judge_tbd(
        [{"name": "Ned", "body": "a real backstory [TBD]", "lineno": 20}])
    assert ok.status == Status.PASS
