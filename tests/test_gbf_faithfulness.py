"""Tests for evals/gbf/adjudicator.py judge_faithfulness -- the claim-level fold. Fully offline."""

import json

from evals.common.parse import Entry
from evals.gbf.adjudicator import GbfAdjudicator
from evals.common.models import Status


class FakeModelClient:
    model = "fake/deepseek"; provider = "openrouter"; temperature = 0.6; thinking = False; base_url = None

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else ""


def _claim(text, verdict, span=""):
    return {"claim": text, "verdict": verdict, "span": span}


def faith(claims, extra=None):
    return json.dumps({"claims": claims, "extra": extra or []})


_GOLD = Entry(name="Aldenburg", anchor_id="aldenburg", lineno=9, raw="### Aldenburg",
              body="Aldenburg is a town in the north, ruled by Baron Aldric.", is_bullet=False)
_WOF = Entry(name="Aldenburg", anchor_id="aldenburg", lineno=42, raw="### Aldenburg",
             body="Aldenburg is a northern town.", is_bullet=False)


def _judge(replies, votes):
    adj = GbfAdjudicator(FakeModelClient(replies), votes=votes)
    return adj.judge_faithfulness(_GOLD, _WOF)


def test_contradiction_fails():
    item = _judge([faith([_claim("in the north", "affirmed", "northern town"),
                          _claim("ruled by Baron Aldric", "contradicted", "ruled by a council")])], votes=1)
    assert item.status == Status.FAIL
    assert item.id == "gbf.faithfulness.aldenburg"
    assert item.evidence.lines == [42]


def test_all_affirmed_passes():
    item = _judge([faith([_claim("in the north", "affirmed", "northern town")])], votes=1)
    assert item.status == Status.PASS
    assert "affirmed" in item.evidence.detail


def test_omission_partials():
    item = _judge([faith([_claim("in the north", "affirmed", "northern town"),
                          _claim("ruled by Baron Aldric", "omitted", "ruled by Baron Aldric")])], votes=1)
    assert item.status == Status.PARTIAL
    assert "omitted" in item.evidence.detail


def test_majority_over_five_votes():
    reps = [faith([_claim("x", "affirmed", "x")])] * 3 + \
           [faith([_claim("x", "contradicted", "y")])] * 2
    item = _judge(reps, votes=5)
    assert item.status == Status.PASS                    # 3 pass beats 2 fail


def test_no_strict_majority_fails_strict_with_review():
    # 3 votes, 3 distinct labels -> no label > half -> strict FAIL + [REVIEW]
    reps = [faith([_claim("x", "affirmed", "x")]),
            faith([_claim("x", "omitted", "x")]),
            faith([_claim("x", "contradicted", "y")])]
    item = _judge(reps, votes=3)
    assert item.status == Status.FAIL
    assert "[REVIEW]" in item.evidence.detail and "majority" in item.evidence.detail


def test_plurality_without_strict_majority_fails_closed():
    # An abstain (unparseable reply) leaves 4 usable votes [PASS, PASS, FAIL, PARTIAL]: PASS is a
    # plurality (2 of 4) but NOT a strict majority, and a contradiction dissents -> must FAIL, not PASS.
    reps = ["not parseable",
            faith([_claim("x", "affirmed", "x")]),
            faith([_claim("x", "affirmed", "x")]),
            faith([_claim("x", "contradicted", "y")]),
            faith([_claim("x", "omitted", "x")])]
    item = _judge(reps, votes=5)
    assert item.status == Status.FAIL
    assert "[REVIEW]" in item.evidence.detail


def test_empty_claims_votes_abstain_and_fail_closed():
    # Structurally valid replies that extract NO claims are abstains, not PARTIAL votes; an all-abstain
    # judge fails closed rather than being recorded as a quiet PARTIAL on a complete run.
    item = _judge([faith([]), faith([]), faith([])], votes=3)
    assert item.status == Status.FAIL
    assert "[REVIEW]" in item.evidence.detail


def test_extra_surfaces_and_does_not_change_pass():
    item = _judge([faith([_claim("in the north", "affirmed", "northern town")],
                         extra=["has a famous night market"])], votes=1)
    assert item.status == Status.PASS
    assert "[EXTRA]" in item.evidence.detail and "market" in item.evidence.detail


def test_evidence_carries_span_and_lineno():
    item = _judge([faith([_claim("ruled by Baron Aldric", "contradicted",
                                 "Aldenburg is ruled by a merchant council")])], votes=1)
    assert item.status == Status.FAIL
    assert "merchant council" in item.evidence.detail
    assert item.evidence.lines == [42]


def test_cased_and_padded_verdicts_are_normalized():
    item = _judge([faith([_claim("in the north", "Affirmed ", "northern town"),
                          _claim("a town", " AFFIRMED", "town")])], votes=1)
    assert item.status == Status.PASS


def test_cased_contradiction_fails_with_evidence():
    # A "Contradicted" vote must FAIL *and* surface its claim/span (the evidence loop normalizes too).
    item = _judge([faith([_claim("ruled by Baron Aldric", "Contradicted",
                                 "ruled by a merchant council")])], votes=1)
    assert item.status == Status.FAIL
    assert "contradicted: ruled by Baron Aldric" in item.evidence.detail
    assert "merchant council" in item.evidence.detail


def test_cased_omission_partials_with_evidence():
    item = _judge([faith([_claim("in the north", "affirmed", "northern town"),
                          _claim("ruled by Baron Aldric", " Omitted", "ruled by Baron Aldric")])], votes=1)
    assert item.status == Status.PARTIAL
    assert "omitted: ruled by Baron Aldric" in item.evidence.detail


def test_garbage_verdict_abstains_and_fails_closed():
    # An off-menu verdict is unusable, not a quiet PARTIAL -> abstain -> fail closed with [REVIEW].
    item = _judge([faith([_claim("in the north", "affirmed", "northern town"),
                          _claim("ruled by Baron Aldric", "garbage", "")])], votes=1)
    assert item.status == Status.FAIL
    assert "[REVIEW]" in item.evidence.detail


def test_missing_verdict_abstains_and_fails_closed():
    item = _judge([json.dumps({"claims": [{"claim": "in the north", "span": "northern town"}]})],
                  votes=1)
    assert item.status == Status.FAIL
    assert "[REVIEW]" in item.evidence.detail


def test_contradiction_beside_garbage_verdict_still_fails_on_the_contradiction():
    # A real contradiction is decisive even when a sibling verdict is junk -- it is a FAIL vote,
    # not an abstain, so its evidence is reported (no no-verdicts [REVIEW]).
    item = _judge([faith([_claim("ruled by Baron Aldric", "contradicted", "ruled by a council"),
                          _claim("in the north", "garbage", "")])], votes=1)
    assert item.status == Status.FAIL
    assert "contradicted: ruled by Baron Aldric" in item.evidence.detail
    assert "no parseable" not in item.evidence.detail


def test_garbage_vote_abstains_and_majority_forms_from_the_rest():
    # One junk-verdict vote abstains; the remaining two PASS votes are a strict majority.
    reps = [faith([_claim("x", "garbage", "x")]),
            faith([_claim("x", "affirmed", "x")]),
            faith([_claim("x", "affirmed", "x")])]
    item = _judge(reps, votes=3)
    assert item.status == Status.PASS


def test_all_malformed_fails_closed():
    item = _judge(["[]", json.dumps({"foo": 1})], votes=2)   # non-dict, then dict-with-no-claims
    assert item.status == Status.FAIL
    assert "[REVIEW]" in item.evidence.detail
