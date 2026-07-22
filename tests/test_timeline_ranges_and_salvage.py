"""Timeline robustness fixes (run-1.13, notes #2 and #3).

Two real failures: (1) reign year-ranges landed in Undated/Could-Not-Place because the
placement call died 3x on a duplicate-gap validation error with no subset-salvage; (2) a
"years ago" system with raw offset magnitudes rendered the timeline in reverse. This file
covers the code-level fixes: duplicate gaps now COALESCE (not fail), placement has a
subset-salvage, and an unresolved "years ago" system is dropped to relative placement.
(The extractor prompt change that captures a reign range into date_text is a prompt edit,
validated in the live e2e, not here.) Offline, no API.
"""

import json

from agents.reconciler import (
    Reconciler, UNDATED, _coalesce_placements, _valid_placement_subset,
    _validate_placement_decision, _build_spine_and_gaps, _UNRESOLVED_OFFSET_SYSTEM_RE,
)
from models.lore import HistoryEvent
from models.reconcile import PlacementDecision, GapPlacement


# --- fake client + helpers (self-contained) --------------------------------- #
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

    @property
    def call_count(self):
        return len(self.messages.calls)


def make_reconciler(responses):
    return Reconciler(client=FakeClient(responses))


def hev(name, date_text=None):
    return HistoryEvent(name=name, description=f"{name} happened.", scope="world",
                        date_text=date_text)


def _date_json(*triples):
    return json.dumps({"dated": [{"index": i, "system": s, "parts": p} for i, s, p in triples]})


def _pd(*pairs):
    return PlacementDecision(placements=[GapPlacement(gap=g, events=e) for g, e in pairs])


# --- coalescing duplicate gaps ---------------------------------------------- #
def test_coalesce_merges_repeated_gap():
    assert _coalesce_placements(_pd(("AR#0", [1]), ("AR#0", [2]), ("AR#1", [3])).placements) == [
        ("AR#0", [1, 2]), ("AR#1", [3])]


def test_coalesce_dedups_within_gap():
    assert _coalesce_placements(_pd(("AR#0", [1, 2]), ("AR#0", [2, 3])).placements) == [
        ("AR#0", [1, 2, 3])]


def test_validate_allows_duplicate_gap_but_flags_double_placement():
    evs = [hev("M", "1350"), hev("R"), hev("S")]
    _, gaps = _build_spine_and_gaps([(0, "AR", [1350])])
    # same gap twice, different events -> benign (coalesces)
    assert _validate_placement_decision(_pd(("AR#0", [1]), ("AR#0", [2])), evs, {0}, gaps) == []
    # same event in two DIFFERENT gaps -> still a problem
    assert _validate_placement_decision(_pd(("AR#0", [1]), ("AR#1", [1])), evs, {0}, gaps)


# --- subset salvage --------------------------------------------------------- #
def test_valid_placement_subset_keeps_good_drops_bad():
    evs = [hev("M", "1350"), hev("R"), hev("S")]
    _, gaps = _build_spine_and_gaps([(0, "AR", [1350])])
    kept = _valid_placement_subset(
        _pd(("AR#0", [1]), ("ZZ#9", [2]), ("AR#1", [0])), evs, {0}, gaps)
    # AR#0:[1] kept; ZZ#9 unknown gap dropped; AR#1:[0] is a dated event -> dropped
    assert kept == {"AR#0": [1]}


# --- through order_history -------------------------------------------------- #
def test_order_history_duplicate_gap_coalesces_not_retries():
    # The exact run failure: two placement objects on the SAME gap. Now benign -> both
    # relatives placed after the dated marker, and NO wasted retries.
    events = [hev("Dated", "1300"), hev("R1"), hev("R2")]
    rec = make_reconciler([
        _date_json((0, "AR", [1300])),
        json.dumps({"placements": [{"gap": "AR#1", "events": [1]},
                                   {"gap": "AR#1", "events": [2]}]}),
    ])
    out = rec.order_history(events)
    by = {e.name: e for e in out}
    assert by["Dated"].chronological_position == 0
    assert by["R1"].chronological_position == 1
    assert by["R2"].chronological_position == 2
    assert rec.client.call_count == 2                    # 1 date + 1 placement, no retries


def test_order_history_placement_salvage_after_persistent_bad_gap():
    # Placement keeps a good AR#1:[1] but also an unknown gap 3x -> after 3 failed
    # validations, subset-salvage keeps the good one instead of Could-Not-Place-ing all.
    events = [hev("Dated", "1300"), hev("R1")]
    bad = json.dumps({"placements": [{"gap": "AR#1", "events": [1]},
                                     {"gap": "NOPE#9", "events": []}]})
    rec = make_reconciler([_date_json((0, "AR", [1300])), bad, bad, bad])
    out = rec.order_history(events)
    by = {e.name: e for e in out}
    assert by["R1"].chronological_position == 1          # salvaged, not stranded
    assert by["R1"].calendar_system == "AR"


def test_order_history_years_ago_system_dropped_to_relative(caplog):
    # An unresolved "years ago" system (raw magnitudes sort backwards). The guard drops it
    # off the dated spine; Call 2 then orders it on the undated timeline instead.
    events = [hev("Old", "200 years ago")]
    rec = make_reconciler([
        _date_json((0, "years ago", [200])),
        json.dumps({"placements": [{"gap": f"{UNDATED}#0", "events": [0]}]}),
    ])
    with caplog.at_level("WARNING"):
        out = rec.order_history(events)
    assert out[0].calendar_system is None                # NOT dated on a reversed spine
    assert any("unresolved present-offset" in r.message for r in caplog.records)


def test_unresolved_offset_regex():
    assert _UNRESOLVED_OFFSET_SYSTEM_RE.search("years ago")
    assert _UNRESOLVED_OFFSET_SYSTEM_RE.search("400 years ago")
    assert _UNRESOLVED_OFFSET_SYSTEM_RE.search("before present")
    assert not _UNRESOLVED_OFFSET_SYSTEM_RE.search("AR years")
    assert not _UNRESOLVED_OFFSET_SYSTEM_RE.search("Elder Scrolls eras")
