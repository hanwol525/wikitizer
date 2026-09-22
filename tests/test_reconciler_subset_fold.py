"""Reconciler: typographic canonical-fold (Fix 1) + uniqueness-gated subset floor (Fix 2).

Two deterministic, under-merge-safe reconciler changes surfaced by the v2 run logs:

  * **Fix 1 (CJ bug).** `_validate_decision` / `_resolve_canonical` / `_valid_merge_subset`
    compared the LLM's canonical against member names with raw `.strip().lower()`, which does
    NOT fold curly vs straight quotes. Opus emitted a STRAIGHT-quote canonical
    (`Clara Jane "CJ" Callustoe`) for a member stored with CURLY quotes, so a valid merge was
    rejected 3x and CJ fell back to the bare "CJ" heading. The fix folds both sides through
    `_name_key` (the same fold the deterministic floors use) for MATCHING only, still returning
    the verbatim stored string.

  * **Fix 2 (Lake cluster).** A new `_merge_unique_subset_names` floor collapses a name-fragment
    into its containing entity ("Mundi"/"The Lake" -> "Lake Mundi") when the fragment's tokens are
    a proper subset of EXACTLY ONE same-type entity -- a structural signal Opus won't assert from
    facts alone. Zero or >=2 supersets -> skip (the uniqueness gate blocks "Krieger"/"Emperor"/
    "Dwarves"). Short names, HistoryEvents, and Character player_name clashes never fold.

Offline, no API. Mirrors the FakeClient / hand-built-decision style of the other floor tests.
"""

import json

from agents.reconciler import (
    Reconciler,
    _combine_group,
    _merge_unique_subset_names,
    _resolve_canonical,
    _validate_decision,
)
from models.lore import Alias, Character, HistoryEvent, Location, Scope
from models.reconcile import MergeGroup, ReconcileDecision


# --- self-contained fake client -------------------------------------------- #
class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _Resp(spec)


class FakeClient:
    def __init__(self, responses):
        self.messages = _Messages(responses)


def loc(name, aliases=None):
    return Location(name=name,
                    aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


def ch(name, player=None, is_pc=False):
    return Character(name=name, is_pc=is_pc, player_name=player)


def hev(name):
    return HistoryEvent(name=name, description="An event.", scope=Scope.WORLD)


NO_MERGE = json.dumps({"merges": [], "possible_duplicates": []})

# The exact CJ pair: a member stored with CURLY double quotes, an LLM canonical with STRAIGHT.
CURLY = "Clara Jane “CJ” Callustoe"
STRAIGHT = 'Clara Jane "CJ" Callustoe'


# === Fix 1: typographic canonical fold ===================================== #
def test_validate_accepts_straight_canonical_for_curly_member():
    # The CJ regression: raw .strip().lower() rejected this 3x; the _name_key fold matches.
    entries = [loc(CURLY), loc("CJ of the North Reach")]
    d = ReconcileDecision(merges=[MergeGroup(members=[0, 1], canonical=STRAIGHT)])
    assert _validate_decision(d, entries) == []


def test_validate_accepts_curly_apostrophe_canonical():
    # The same class of bug with a curly apostrophe vs a straight one.
    entries = [loc("O’Brien"), loc("The Brien Clan of Gol")]
    d = ReconcileDecision(merges=[MergeGroup(members=[0, 1], canonical="O'Brien")])
    assert _validate_decision(d, entries) == []


def test_validate_still_rejects_invented_canonical():
    # Fix 1 must NOT over-loosen: a canonical with genuinely different tokens is still rejected.
    entries = [loc("Lake Mundi"), loc("The Great Well")]
    d = ReconcileDecision(merges=[MergeGroup(members=[0, 1], canonical="Atlantis")])
    assert _validate_decision(d, entries)  # non-empty == problems


def test_resolve_canonical_snaps_straight_to_curly_verbatim():
    a = loc(CURLY)
    b = loc("The Great Well")
    assert _resolve_canonical([a, b], STRAIGHT) == CURLY  # verbatim curly name, not the LLM string


def test_combine_does_not_demote_real_curly_name():
    a = loc(CURLY)
    b = loc("CJ of the North Reach")
    merged = _combine_group([a, b], STRAIGHT)  # LLM handed back the straight-quote canonical
    assert merged.name == CURLY                                    # verbatim member name is heading
    assert CURLY not in [al.text for al in merged.aliases]         # real name NOT demoted to alias


# === Fix 2: uniqueness-gated subset floor ================================== #
def test_folds_unique_subset_heading_and_alias():
    out = _merge_unique_subset_names([loc("Lake Mundi"), loc("Mundi"), loc("Castle")], "Location")
    assert sorted(e.name for e in out) == ["Castle", "Lake Mundi"]  # Mundi folded, Castle untouched
    lake = next(e for e in out if e.name == "Lake Mundi")
    assert "Mundi" in [a.text for a in lake.aliases]               # fragment became an alias


def test_does_not_fold_multi_superset():
    # "Krieger" {krieger} is a subset of TWO Krieger entities -> uniqueness gate blocks it.
    out = _merge_unique_subset_names(
        [loc("Krieger"), loc("Krieger Imperium"), loc("Krieger Family")], "Location")
    assert len(out) == 3


def test_does_not_fold_short_name():
    # "Sam" is a unique subset of "Sam Wakestrider", but len <= SHORT_NAME_LEN -> never fold.
    out = _merge_unique_subset_names([loc("Sam"), loc("Sam Wakestrider")], "Location")
    assert len(out) == 2


def test_does_not_fold_across_player_clash():
    # "Aerin" c "Aerin Wakestrider" uniquely, but two different players -> _combine_group vetoes.
    a = ch("Aerin", player="Alice", is_pc=True)
    b = ch("Aerin Wakestrider", player="Bob", is_pc=True)
    out = _merge_unique_subset_names([a, b], "Character")
    assert len(out) == 2


def test_folds_character_subset_without_player_clash():
    # Same fragment/superset shape, no clashing player -> it folds; the fuller name wins the head.
    a = ch("Ferridus", is_pc=False, player=None)
    b = ch("Emperor Ferridus Krieger", is_pc=False, player=None)
    out = _merge_unique_subset_names([a, b], "Character")
    assert len(out) == 1
    assert out[0].name == "Emperor Ferridus Krieger"
    assert "Ferridus" in [al.text for al in out[0].aliases]


def test_skips_history_events():
    # Event names are model-generated labels -> the floor never runs on HistoryEvent.
    out = _merge_unique_subset_names([hev("War"), hev("War of the North")], "History")
    assert len(out) == 2


def test_zero_superset_left_alone():
    out = _merge_unique_subset_names([loc("Riverton"), loc("Gol")], "Location")
    assert sorted(e.name for e in out) == ["Gol", "Riverton"]


def test_reconcile_e2e_collapses_fragment_the_llm_declined():
    # The LLM returns an EMPTY decision (Opus declined to assert "Mundi" == "Lake Mundi");
    # the floor still collapses it end-to-end through reconcile().
    rec = Reconciler(client=FakeClient([NO_MERGE]))
    out = rec.reconcile([loc("Lake Mundi"), loc("Mundi"), loc("Castle")])
    assert sorted(e.name for e in out) == ["Castle", "Lake Mundi"]
    lake = next(e for e in out if e.name == "Lake Mundi")
    assert "Mundi" in [a.text for a in lake.aliases]
