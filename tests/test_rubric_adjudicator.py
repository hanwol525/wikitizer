"""Tests for evals/rubric/adjudicator.py -- the model checks + record vote-hygiene. Offline.

A FakeModelClient feeds canned {"results":[...]} replies (queued in order). Entries come from
a fabricated WOF via parse_wof, so the checks run on real Entry objects.
"""

import json

from evals.common.models import Status
from evals.common.parse import parse_wof
from evals.rubric.adjudicator import RubricAdjudicator
from evals.rubric.loader import RubricEntity


class FakeModelClient:
    model = "fake/qwen"
    provider = "openrouter"
    temperature = 0.6
    thinking = False
    base_url = None

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else ""


def R(*objs):
    return json.dumps({"results": list(objs)})


def E(name, **kw):
    return RubricEntity(name=name, **kw)


WOF = """\
# Wiki

subtitle

---

## Locations

### <a id="alpha"></a>Alpha

Alpha the fortress.[^1]

### <a id="beta"></a>Beta

Beta the port.

## Characters

### <a id="aerin"></a>Aerin*

A ranger played by a party member.[^2]

## Footnotes

[^1]: `Alpha the fortress` — DM, dm.txt
[^2]: `Aerin is my PC` — Sam, dm.txt
"""


def _entries(section_title):
    p = parse_wof(WOF)
    for s in p.sections:
        if s.title == section_title:
            return s.entries
    return []


# --- sections --------------------------------------------------------------- #

def test_judge_sections_present_absent_and_index_map():
    fake = FakeModelClient([R({"present": True, "match": 1}, {"present": False, "match": None})])
    items, smap = RubricAdjudicator(fake, votes=1).judge_sections(
        [E("Locations"), E("Missing")], ["Locations", "Characters"])
    assert items[0].status == Status.PASS and items[1].status == Status.FAIL
    assert items[0].id == "rubric.sections.locations"
    assert smap == [1, None]                           # parallel to required, by index


# --- presence --------------------------------------------------------------- #

def test_judge_presence_maps_entry_and_marks_absent():
    entries = _entries("Locations")                    # [Alpha, Beta]
    fake = FakeModelClient([R({"present": True, "match": 1}, {"present": False, "match": None})])
    items, emap = RubricAdjudicator(fake, votes=1).judge_presence(
        "locations", [E("Alpha"), E("Ghost")], entries)
    assert items[0].status == Status.PASS and items[1].status == Status.FAIL
    assert items[0].id == "rubric.presence.locations.alpha"
    assert emap[0] is entries[0] and emap[1] is None


def test_judge_presence_section_absent_no_call():
    fake = FakeModelClient([])
    items, emap = RubricAdjudicator(fake, votes=3).judge_presence("locations", [E("Alpha")], [])
    assert items[0].status == Status.FAIL and "section absent" in items[0].evidence.detail
    assert emap == [None] and fake.calls == []          # no model call when the section is absent


# --- PCs (two items each; correction #3: the raw heading with '*' is shown) --- #

def test_judge_pcs_two_items_and_sees_asterisk_via_raw():
    aerin_entry = _entries("Characters")[0]            # heading "### <a ...></a>Aerin*"
    fake = FakeModelClient([R({"pc_stated": True, "asterisk": True})])
    items = RubricAdjudicator(fake, votes=1).judge_pcs([(E("Aerin"), aerin_entry)])
    by_id = {it.id: it for it in items}
    assert set(by_id) == {"rubric.pc-stated.aerin", "rubric.pc-marked.aerin"}
    assert by_id["rubric.pc-stated.aerin"].status == Status.PASS
    assert by_id["rubric.pc-marked.aerin"].status == Status.PASS
    # the trailing '*' actually reached the model (entry.raw was passed, not a stripped name)
    assert "*" in fake.calls[0][1]


# --- record vote-hygiene ---------------------------------------------------- #

def test_wrong_length_vote_is_discarded_as_abstain():
    fake = FakeModelClient([
        R({"present": True, "match": 1}),                                   # len 1 != 2 -> abstain
        R({"present": True, "match": 1}, {"present": True, "match": 2}),
        R({"present": True, "match": 1}, {"present": True, "match": 2}),
    ])
    items, smap = RubricAdjudicator(fake, votes=3).judge_sections([E("A"), E("B")], ["A", "B"])
    assert items[0].status == Status.PASS and items[1].status == Status.PASS
    assert smap == [1, 2]


def test_all_malformed_votes_fail_closed():
    fake = FakeModelClient(["not json", "{}", "also not json"])              # none parse to results
    items, smap = RubricAdjudicator(fake, votes=3).judge_sections([E("A")], ["A"])
    assert items[0].status == Status.FAIL and "[REVIEW]" in items[0].evidence.detail
    assert smap == [None]


def test_split_vote_decided_by_majority_and_flagged():
    fake = FakeModelClient([
        R({"present": True, "match": 1}),
        R({"present": False, "match": None}),
        R({"present": True, "match": 1}),
    ])
    items, _ = RubricAdjudicator(fake, votes=3).judge_sections([E("A")], ["A"])
    assert items[0].status == Status.PASS                # 2 vs 1 -> present
    assert "[REVIEW]" in items[0].evidence.detail


# --- sourcing --------------------------------------------------------------- #

def test_judge_support_pass_names_footnote_and_span():
    alpha = _entries("Locations")[0]
    fake = FakeModelClient([R({"supported": True, "span": "Alpha the fortress"})])
    item = RubricAdjudicator(fake, votes=1).judge_support(
        E("Alpha"), "locations", alpha, [(1, "Alpha the fortress")])
    assert item.status == Status.PASS
    assert item.id == "rubric.sourcing.locations.alpha"
    assert "[^1]" in item.evidence.detail
    assert item.evidence.found == "Alpha the fortress"
    assert item.evidence.lines == [alpha.lineno]


def test_judge_support_unsupported_fails():
    alpha = _entries("Locations")[0]
    fake = FakeModelClient([R({"supported": False, "span": ""})])
    item = RubricAdjudicator(fake, votes=1).judge_support(
        E("Alpha"), "locations", alpha, [(1, "an unrelated quote")])
    assert item.status == Status.FAIL


def test_judge_support_all_malformed_fail_closed():
    alpha = _entries("Locations")[0]
    fake = FakeModelClient(["nope"])
    item = RubricAdjudicator(fake, votes=1).judge_support(
        E("Alpha"), "locations", alpha, [(1, "q")])
    assert item.status == Status.FAIL and "[REVIEW]" in item.evidence.detail


def test_judge_pcs_all_malformed_fail_closed():
    aerin_entry = _entries("Characters")[0]
    fake = FakeModelClient(["nope", "{}", "also bad"])          # none parse to a length-1 results
    items = RubricAdjudicator(fake, votes=3).judge_pcs([(E("Aerin"), aerin_entry)])
    by_id = {it.id: it for it in items}
    assert by_id["rubric.pc-stated.aerin"].status == Status.FAIL
    assert by_id["rubric.pc-marked.aerin"].status == Status.FAIL
    assert "[REVIEW]" in by_id["rubric.pc-stated.aerin"].evidence.detail


def test_match_bool_is_not_treated_as_index():
    # the bool/int trap: a JSON bool for "match" must NOT be accepted as index 1 -- and a
    # "present" with no valid match fails closed rather than passing with nothing mapped.
    fake = FakeModelClient([R({"present": True, "match": True})])
    items, smap = RubricAdjudicator(fake, votes=1).judge_sections([E("A")], ["A"])
    assert items[0].status == Status.FAIL and "[REVIEW]" in items[0].evidence.detail
    assert smap == [None]

    entries = _entries("Locations")
    fake2 = FakeModelClient([R({"present": True, "match": True})])
    items2, emap = RubricAdjudicator(fake2, votes=1).judge_presence("locations", [E("Alpha")], entries)
    assert items2[0].status == Status.FAIL and "[REVIEW]" in items2[0].evidence.detail
    assert emap == [None]


def test_present_with_out_of_range_or_null_match_fails_closed():
    fake = FakeModelClient([R({"present": True, "match": 99}, {"present": True, "match": None})])
    items, smap = RubricAdjudicator(fake, votes=1).judge_sections([E("A"), E("B")], ["A", "B"])
    assert [it.status for it in items] == [Status.FAIL, Status.FAIL]
    assert all("[REVIEW]" in it.evidence.detail for it in items)
    assert smap == [None, None]

    entries = _entries("Locations")
    fake2 = FakeModelClient([R({"present": True, "match": 0}, {"present": True, "match": 3})])
    items2, emap = RubricAdjudicator(fake2, votes=1).judge_presence(
        "locations", [E("Alpha"), E("Beta")], entries)
    assert [it.status for it in items2] == [Status.FAIL, Status.FAIL]
    assert emap == [None, None]


def test_valid_match_taken_from_a_later_present_vote():
    # the first "present" vote's match is bad, but another present vote gives a valid one.
    fake = FakeModelClient([
        R({"present": True, "match": 99}),
        R({"present": True, "match": 2}),
        R({"present": False, "match": None}),
    ])
    items, smap = RubricAdjudicator(fake, votes=3).judge_sections([E("B")], ["A", "B"])
    assert items[0].status == Status.PASS and smap == [2]


def test_string_booleans_fail_closed():
    # bool("false") is True -- a stringly "false"/"no"/"true" must never count as a positive vote.
    fake = FakeModelClient([R({"present": "false", "match": 1}, {"present": "true", "match": 2})])
    items, smap = RubricAdjudicator(fake, votes=1).judge_sections([E("A"), E("B")], ["A", "B"])
    assert [it.status for it in items] == [Status.FAIL, Status.FAIL]
    assert smap == [None, None]

    entries = _entries("Locations")
    fake2 = FakeModelClient([R({"present": "false", "match": 1})])
    items2, emap = RubricAdjudicator(fake2, votes=1).judge_presence("locations", [E("Alpha")], entries)
    assert items2[0].status == Status.FAIL and emap == [None]

    aerin_entry = _entries("Characters")[0]
    fake3 = FakeModelClient([R({"pc_stated": "no", "asterisk": 1})])
    by_id = {it.id: it for it in RubricAdjudicator(fake3, votes=1).judge_pcs([(E("Aerin"), aerin_entry)])}
    assert by_id["rubric.pc-stated.aerin"].status == Status.FAIL
    assert by_id["rubric.pc-marked.aerin"].status == Status.FAIL

    alpha = entries[0]
    fake4 = FakeModelClient([R({"supported": "no", "span": "Alpha the fortress"})])
    item = RubricAdjudicator(fake4, votes=1).judge_support(
        E("Alpha"), "locations", alpha, [(1, "Alpha the fortress")])
    assert item.status == Status.FAIL


def test_support_span_companion_ignores_stringly_true_vote():
    # the span must come from a vote that REALLY said supported: true, not a "false" string.
    alpha = _entries("Locations")[0]
    fake = FakeModelClient([
        R({"supported": "false", "span": "wrong span"}),
        R({"supported": True, "span": "Alpha the fortress"}),
        R({"supported": True, "span": "Alpha the fortress"}),
    ])
    item = RubricAdjudicator(fake, votes=3).judge_support(
        E("Alpha"), "locations", alpha, [(1, "Alpha the fortress")])
    assert item.status == Status.PASS and item.evidence.found == "Alpha the fortress"
