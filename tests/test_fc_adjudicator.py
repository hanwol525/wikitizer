"""Tests for evals/fc/adjudicator.py -- the 3 model checks + voting. Fully offline.

A FakeModelClient feeds canned JSON replies (queued in order), so the voting/majority logic,
the FIX #5 length-guard + fail-closed path, and the 0-candidate short-circuit are all exercised
without touching the network or needing a key.
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


# --- FIX #5: vote hygiene --------------------------------------------------- #

def test_wrong_length_vote_is_discarded_as_abstain():
    # Two candidates; one vote returns the wrong count (abstain), the other two decide.
    cands = [{"name": "A", "body": "x", "lineno": 1}, {"name": "B", "body": "y", "lineno": 2}]
    fake = FakeModelClient([
        verdicts("ok"),                       # wrong length (1, not 2) -> discarded
        verdicts("ok", "ok"),
        verdicts("ok", "ok"),
    ])
    item = Adjudicator(fake, votes=3).judge_tbd(cands)
    assert item.status == Status.PASS


def test_all_malformed_votes_fail_closed():
    fake = FakeModelClient(["not json", "{}", verdicts("ok", "ok")])   # none valid for 1 candidate
    item = Adjudicator(fake, votes=3).judge_tbd([{"name": "A", "body": "x", "lineno": 1}])
    assert item.status == Status.FAIL
    assert "[REVIEW]" in item.evidence.detail


def test_split_vote_is_decided_by_majority_and_flagged(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    # 2 ok vs 1 stub -> majority ok (pass), but the split is flagged [REVIEW].
    fake = FakeModelClient([verdicts("ok"), verdicts("stub_needs_tbd"), verdicts("ok")])
    item = Adjudicator(fake, votes=3).judge_tbd([{"name": "A", "body": "x", "lineno": 1}])
    assert item.status == Status.PASS
    assert "[REVIEW]" in item.evidence.detail
    assert any("[REVIEW]" in r.getMessage() for r in caplog.records)


def test_unknown_labels_ignored_then_fail_closed_when_all_unknown():
    fake = FakeModelClient([verdicts("weird"), verdicts("also_weird"), verdicts("nope")])
    item = Adjudicator(fake, votes=3).judge_grouping(
        [{"label": "X", "lineno": 3, "entries_beneath": []}])
    # No recognized label for the candidate -> resolved to the bad label -> fail.
    assert item.status == Status.FAIL
