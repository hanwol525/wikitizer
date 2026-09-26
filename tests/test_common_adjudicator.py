"""Tests for evals/common/adjudicator.py -- the shared self-consistency voting engine.

Exercises ``BaseAdjudicator._vote`` / ``_judge`` directly through a tiny concrete subclass
and a FakeModelClient: the FIX #5 length-guard (a wrong-count vote is discarded as an
abstain), the fail-closed-when-all-malformed path, per-candidate majority + ``[REVIEW]``
flagging on a split, and allowed-set filtering falling to the bad label. Fully offline. The
FC-specific judges built on this engine are tested in tests/test_fc_adjudicator.py.
"""

import json
import logging

from evals.common.adjudicator import BaseAdjudicator
from evals.common.models import Status


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


class _Judge(BaseAdjudicator):
    """A minimal concrete adjudicator: one generic judge that drives the shared ``_judge``
    over an arbitrary candidate set + label vocab. The eval-specific prompt text is
    irrelevant to the voting machinery under test."""

    def judge(self, cid, candidates, allowed, bad_label, offender_label):
        user = "Items:\n" + "\n".join(str(c) for c in candidates)
        return self._judge(cid, "system", user, candidates, allowed, bad_label, offender_label)


# Two representative label vocabularies (mirroring FC's tbd / grouping checks).
_TBD = ({"stub_needs_tbd", "ok"}, "stub_needs_tbd", "stub_needs_tbd")
_GROUP = ({"group_label", "missing_anchor"}, "missing_anchor", "missing_anchor")


def test_wrong_length_vote_is_discarded_as_abstain():
    # Two candidates; one vote returns the wrong count (abstain), the other two decide.
    cands = [{"name": "A", "body": "x", "lineno": 1}, {"name": "B", "body": "y", "lineno": 2}]
    fake = FakeModelClient([
        verdicts("ok"),                       # wrong length (1, not 2) -> discarded
        verdicts("ok", "ok"),
        verdicts("ok", "ok"),
    ])
    item = _Judge(fake, votes=3).judge("c.tbd", cands, *_TBD)
    assert item.status == Status.PASS


def test_all_malformed_votes_fail_closed():
    fake = FakeModelClient(["not json", "{}", verdicts("ok", "ok")])   # none valid for 1 candidate
    item = _Judge(fake, votes=3).judge("c.tbd", [{"name": "A", "body": "x", "lineno": 1}], *_TBD)
    assert item.status == Status.FAIL
    assert "[REVIEW]" in item.evidence.detail


def test_split_vote_is_decided_by_majority_and_flagged(caplog):
    caplog.set_level(logging.WARNING)
    # 2 ok vs 1 stub -> majority ok (pass), but the split is flagged [REVIEW].
    fake = FakeModelClient([verdicts("ok"), verdicts("stub_needs_tbd"), verdicts("ok")])
    item = _Judge(fake, votes=3).judge("c.tbd", [{"name": "A", "body": "x", "lineno": 1}], *_TBD)
    assert item.status == Status.PASS
    assert "[REVIEW]" in item.evidence.detail
    assert any("[REVIEW]" in r.getMessage() for r in caplog.records)


def test_unknown_labels_ignored_then_fail_closed_when_all_unknown():
    fake = FakeModelClient([verdicts("weird"), verdicts("also_weird"), verdicts("nope")])
    item = _Judge(fake, votes=3).judge("c.group", [{"label": "X", "lineno": 3}], *_GROUP)
    # No recognized label for the candidate -> resolved to the bad label -> fail.
    assert item.status == Status.FAIL
