"""Cross-type resolution arbiter (run-1.14, note #3/#7/#8).

One real entity extracted under several types (Scoia'tael as both a People AND a Location)
made duplicate pages and killed cross-links. An LLM arbiter picks the correct type(s);
Python folds the losers' facts into the winner and drops the wrong-type pages. A
Location+Organization dual (a realm) is left alone; any LLM failure degrades to a no-op.
Offline, FakeClient, no API.
"""

import json

from agents.reconciler import Reconciler
from models.lore import Detail, Location, Organization, PeopleAndCultures


# --- fake client with call capture ------------------------------------------ #
class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Msgs:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return type("R", (), {"content": [_Block(spec)]})


class FakeClient:
    def __init__(self, responses):
        self.messages = _Msgs(responses)


def det(t):
    return Detail(text=t, source_files=["g.txt"])


def loc(name, *facts):
    return Location(name=name, details=[det(f) for f in facts])


def peo(name, *facts):
    return PeopleAndCultures(name=name, details=[det(f) for f in facts])


def org(name, *facts):
    return Organization(name=name, details=[det(f) for f in facts])


def _choices(*pairs):
    return json.dumps({"choices": [{"cluster": c, "keep_types": t} for c, t in pairs]})


# --- resolution ------------------------------------------------------------- #
def test_arbiter_keeps_winner_drops_loser_and_folds_facts():
    by_type = {
        "locations": [loc("Scoia'tael", "a resistance base"), loc("Riverton", "a town")],
        "people": [peo("Scoia'tael", "elven guerrillas", "hate the empire")],
        "organizations": [], "characters": [], "items": [], "history": ["hist"],
    }
    rec = Reconciler(client=FakeClient([_choices((0, ["people"]))]))
    out = rec.resolve_cross_type(by_type)
    assert [e.name for e in out["locations"]] == ["Riverton"]          # loser dropped
    assert [e.name for e in out["people"]] == ["Scoia'tael"]
    assert len(out["people"][0].details) == 3                          # loser's fact folded in
    assert out["history"] == ["hist"]                                  # non-noun key passthrough
    assert len(rec.client.messages.calls) == 1


def test_location_org_dual_is_left_alone_no_call():
    by_type = {
        "locations": [loc("Krieger Imperium", "a realm")],
        "organizations": [org("Krieger Imperium", "the ruling body")],
        "people": [], "characters": [], "items": [],
    }
    rec = Reconciler(client=FakeClient(["{\"choices\": []}"]))
    out = rec.resolve_cross_type(by_type)
    assert len(out["locations"]) == 1 and len(out["organizations"]) == 1
    assert rec.client.messages.calls == []                            # a pure dual isn't arbitrated


def test_realm_wins_both_but_people_copy_dropped():
    by_type = {
        "locations": [loc("Maltraav", "a territory")],
        "organizations": [org("Maltraav", "a noble house in power")],
        "people": [peo("Maltraav", "misfiled as a people")],
        "characters": [], "items": [],
    }
    rec = Reconciler(client=FakeClient([_choices((0, ["locations", "organizations"]))]))
    out = rec.resolve_cross_type(by_type)
    assert len(out["locations"]) == 1 and len(out["organizations"]) == 1
    assert out["people"] == []                                        # the misfiled copy dropped


def test_no_cross_type_conflict_makes_no_call():
    by_type = {"locations": [loc("A")], "people": [peo("B")],
               "organizations": [], "characters": [], "items": []}
    rec = Reconciler(client=FakeClient(["{\"choices\": []}"]))
    out = rec.resolve_cross_type(by_type)
    assert rec.client.messages.calls == []
    assert [e.name for e in out["locations"]] == ["A"]


def test_degrades_to_noop_on_llm_failure():
    by_type = {
        "locations": [loc("Scoia'tael", "base")], "people": [peo("Scoia'tael", "people")],
        "organizations": [], "characters": [], "items": [],
    }
    rec = Reconciler(client=FakeClient(["not json", "still not", "nope"]))
    out = rec.resolve_cross_type(by_type)
    # nothing dropped -- both copies survive (safe under-merge direction)
    assert len(out["locations"]) == 1 and len(out["people"]) == 1
