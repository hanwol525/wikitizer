"""Union of overlapping merge groups (run-1.16 §A).

The LLM (even Opus) routinely lists one entry in TWO merge groups -- a hub, e.g. "Lake
Mundi" paired with both "Mundi" and something else. The validator flagged the reused
index and the salvage DROPPED the whole group, silently losing legit merges (Lake
Mundi+Mundi, Skjoldr, Ambrose). Now overlapping groups are UNIONED into one component
before validation ({A,B}+{B,C} -> {A,B,C}), with a size-cap firebreak against a runaway
chain, and character player-name vetoes still strand a clashing component. Offline, no API
except the FakeClient boundary.
"""

import json
import logging

from agents.reconciler import Reconciler, _coalesce_overlapping_groups, MERGE_COMPONENT_CAP
from models.lore import Character, Location
from models.reconcile import MergeGroup


# --- fake client (mirrors the other reconciler test modules) ---------------- #
class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class _Messages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _Resp(spec)


class FakeClient:
    def __init__(self, responses): self.messages = _Messages(responses)
    @property
    def call_count(self): return len(self.messages.calls)


def loc(n): return Location(name=n)
def ch(n, player=None): return Character(name=n, is_pc=True, player_name=player)


def _decision(groups, canonical="X"):
    return json.dumps({
        "merges": [{"members": m, "canonical": c, "conflicts": []} for m, c in groups],
        "possible_duplicates": [],
    })


# --- unit: the coalesce itself ---------------------------------------------- #
def test_overlapping_groups_union_into_one_component():
    entries = [loc("A"), loc("B"), loc("C"), loc("D")]
    merges = [MergeGroup(members=[0, 1], canonical="A"), MergeGroup(members=[1, 2], canonical="A")]
    out = _coalesce_overlapping_groups(merges, entries, "Location")
    assert [sorted(g.members) for g in out] == [[0, 1, 2]]


def test_non_overlapping_groups_unchanged():
    entries = [loc(x) for x in "ABCD"]
    merges = [MergeGroup(members=[0, 1], canonical="A"), MergeGroup(members=[2, 3], canonical="C")]
    out = _coalesce_overlapping_groups(merges, entries, "Location")
    assert sorted(sorted(g.members) for g in out) == [[0, 1], [2, 3]]


def test_over_cap_component_is_dropped_with_review(caplog):
    entries = [loc(str(i)) for i in range(MERGE_COMPONENT_CAP + 1)]
    # two clean groups overlapping on the last/first index -> union exceeds the cap
    a = list(range(0, MERGE_COMPONENT_CAP))
    b = list(range(MERGE_COMPONENT_CAP - 1, MERGE_COMPONENT_CAP + 1))
    merges = [MergeGroup(members=a, canonical="0"), MergeGroup(members=b, canonical="0")]
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        out = _coalesce_overlapping_groups(merges, entries, "Location")
    assert out == []                                   # runaway component dropped
    assert "suspected runaway chain" in caplog.text


def test_broken_group_passed_through_not_unioned():
    # A clean [0,1] and a BROKEN [0,9] (out of range for 3 entries) share index 0, but the
    # broken one must NOT drag the clean merge into a doomed component -- it's passed
    # through untouched for validate->salvage to drop.
    entries = [loc("A"), loc("B"), loc("C")]
    merges = [MergeGroup(members=[0, 1], canonical="A"), MergeGroup(members=[0, 9], canonical="A")]
    out = _coalesce_overlapping_groups(merges, entries, "Location")
    assert [sorted(g.members) for g in out] == [[0, 1], [0, 9]]   # clean unchanged, broken kept


# --- end-to-end via reconcile(): the Lake-Mundi hub now merges -------------- #
def test_reconcile_merges_hub_listed_in_two_groups():
    entries = [loc("Lake Mundi"), loc("Mundi"), loc("The Lake of Mundi")]
    rec = Reconciler(client=FakeClient([_decision([([0, 1], "Lake Mundi"),
                                                   ([0, 2], "Lake Mundi")])]))
    out = rec.reconcile(entries)
    assert rec.client.call_count == 1                  # clean on the first try (no wasted retries)
    assert [e.name for e in out] == ["Lake Mundi"]     # all three folded into the hub
    aliases = {a.text for a in out[0].aliases}
    assert "Mundi" in aliases and "The Lake of Mundi" in aliases


def test_player_name_clash_strands_the_unioned_component():
    # Union {0,1,2} but 0 and 1 carry DIFFERENT real players -> the component is vetoed and
    # left unmerged (the safe direction), not fabricated into one PC.
    entries = [ch("Aerin", player="Hannah"), ch("Aerin Wakestrider", player="Conrad"),
               ch("Aerin the Guide")]
    rec = Reconciler(client=FakeClient([_decision([([0, 2], "Aerin"), ([1, 2], "Aerin")])]))
    out = rec.reconcile(entries)
    assert len(out) == 3                                # clashing component kept separate
