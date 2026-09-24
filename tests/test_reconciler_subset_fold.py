"""Reconciler: typographic canonical-fold (Fix 1) + uniqueness-gated subset floor (Fix 2).

Two deterministic, under-merge-safe reconciler changes surfaced by the v2 run logs:

  * **Fix 1 (CJ bug).** `_validate_decision` / `_resolve_canonical` / `_valid_merge_subset`
    compared the LLM's canonical against member names with raw `.strip().lower()`, which does
    NOT fold curly vs straight quotes. Opus emitted a STRAIGHT-quote canonical
    (`Clara Jane "CJ" Callustoe`) for a member stored with CURLY quotes, so a valid merge was
    rejected 3x and CJ fell back to the bare "CJ" heading. The fix folds both sides through
    `_name_key` (the same fold the deterministic floors use) for MATCHING only, still returning
    the verbatim stored string.

  * **Fix 2 (Lake cluster).** A `_merge_unique_subset_names` floor collapses a name-fragment
    into its containing entity ("Mundi" -> "Lake Mundi") when the fragment's tokens are a proper
    subset of EXACTLY ONE same-type entity's name or alias, AND evidence links them (the fragment
    is already an alias of the superset, or they share a quote), AND the LLM didn't decline the
    pair. Token shape alone never folds ("Free Islands"/"Dwarven Free Islands", "Elves"/"High
    Elves", Organization "Krieger"/"Krieger Imperium"). Short names, HistoryEvents, and Character
    player_name clashes never fold.

Offline, no API. Mirrors the FakeClient / hand-built-decision style of the other floor tests.
"""

import json

from agents.reconciler import (
    Reconciler,
    _combine_group,
    _declined_name_pairs,
    _merge_unique_subset_names,
    _resolve_canonical,
    _validate_decision,
)
from models.lore import (Alias, Character, HistoryEvent, Location, Organization,
                         PeopleAndCultures, Quote, Scope)
from models.reconcile import MergeGroup, PossibleDuplicate, ReconcileDecision


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


def _aliases(aliases):
    return [Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])]


def loc(name, aliases=None, quotes=None):
    return Location(name=name, aliases=_aliases(aliases), supporting_quotes=list(quotes or []))


def ch(name, player=None, is_pc=False, aliases=None):
    return Character(name=name, is_pc=is_pc, player_name=player, aliases=_aliases(aliases))


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


# === Fix 2: evidence-gated, uniqueness-gated subset floor ================== #
def test_folds_unique_subset_when_fragment_is_already_an_alias():
    # Evidence #1: the fragment's name is already one of the superset's aliases.
    out = _merge_unique_subset_names(
        [loc("Lake Mundi", aliases=["Mundi"]), loc("Mundi"), loc("Castle")], "Location")
    assert sorted(e.name for e in out) == ["Castle", "Lake Mundi"]  # Mundi folded, Castle untouched
    lake = next(e for e in out if e.name == "Lake Mundi")
    assert "Mundi" in [a.text for a in lake.aliases]               # fragment became an alias


def test_folds_unique_subset_when_they_share_a_quote():
    # Evidence #2: the same chat line backs both entries.
    q = Quote(text="We sailed to Lake Mundi.", speaker="Matt", source_file="g.txt")
    out = _merge_unique_subset_names(
        [loc("Lake Mundi", quotes=[q]), loc("Mundi", quotes=[q])], "Location")
    assert [e.name for e in out] == ["Lake Mundi"]
    assert "Mundi" in [a.text for a in out[0].aliases]


def test_does_not_fold_on_token_shape_alone():
    # No alias, no shared quote -> a unique token subset is NOT identity.
    out = _merge_unique_subset_names([loc("Lake Mundi"), loc("Mundi"), loc("Castle")], "Location")
    assert len(out) == 3


def test_free_islands_not_folded():
    out = _merge_unique_subset_names(
        [loc("Free Islands"), loc("Dwarven Free Islands")], "Location")
    assert sorted(e.name for e in out) == ["Dwarven Free Islands", "Free Islands"]


def test_elves_not_folded_into_high_elves():
    out = _merge_unique_subset_names(
        [PeopleAndCultures(name="Elves"), PeopleAndCultures(name="High Elves")], "People")
    assert len(out) == 2


def test_organization_kind_trap_not_folded():
    # The house vs the empire named after it -- the KIND trap the prompt forbids merging.
    out = _merge_unique_subset_names(
        [Organization(name="Krieger"), Organization(name="Krieger Imperium")], "Organization")
    assert len(out) == 2


def test_uniqueness_counts_aliases():
    # "White Tower" already lives on as an alias of "Citadel" -> "Tower" has TWO containing
    # entities, so it's ambiguous even though only one NAME contains it (and even though
    # Black Tower carries "Tower" as an alias).
    out = _merge_unique_subset_names(
        [loc("Citadel", aliases=["White Tower"]), loc("Black Tower", aliases=["Tower"]),
         loc("Tower")], "Location")
    assert len(out) == 3


def test_superset_via_alias_can_be_the_target():
    # A fragment contained only by the superset's ALIAS still finds that entity.
    out = _merge_unique_subset_names(
        [loc("The Citadel", aliases=["White Tower", "Tower"]), loc("Tower")], "Location")
    assert [e.name for e in out] == ["The Citadel"]


def test_does_not_fold_pair_the_llm_declined():
    # Evidence present, but the LLM was shown the pair and kept them apart -> respected.
    out = _merge_unique_subset_names(
        [loc("Lake Mundi", aliases=["Mundi"]), loc("Mundi")], "Location",
        declined_pairs={frozenset(("mundi", "lake mundi"))})
    assert len(out) == 2


def test_does_not_fold_multi_superset():
    # "Krieger" {krieger} is a subset of TWO Krieger entities -> uniqueness gate blocks it,
    # even with alias evidence on one of them.
    out = _merge_unique_subset_names(
        [loc("Krieger"), loc("Krieger Imperium", aliases=["Krieger"]), loc("Krieger Family")],
        "Location")
    assert len(out) == 3


def test_does_not_fold_short_name():
    # "Sam" is a unique subset of "Sam Wakestrider", but len <= SHORT_NAME_LEN -> never fold.
    out = _merge_unique_subset_names(
        [loc("Sam"), loc("Sam Wakestrider", aliases=["Sam"])], "Location")
    assert len(out) == 2


def test_does_not_fold_across_player_clash():
    # "Aerin" c "Aerin Wakestrider" with alias evidence, but two different players -> veto.
    a = ch("Aerin", player="Alice", is_pc=True)
    b = ch("Aerin Wakestrider", player="Bob", is_pc=True, aliases=["Aerin"])
    out = _merge_unique_subset_names([a, b], "Character")
    assert len(out) == 2


def test_folds_character_subset_without_player_clash():
    # Fragment/superset shape + alias evidence, no clashing player -> folds; fuller name heads.
    a = ch("Ferridus")
    b = ch("Emperor Ferridus Krieger", aliases=["Ferridus"])
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


def test_declined_pairs_from_candidates_and_possible_duplicates():
    entries = [loc("Lake Mundi"), loc("Mundi"), loc("Castle"), loc("Castel")]
    d = ReconcileDecision(merges=[], possible_duplicates=[
        PossibleDuplicate(members=[2, 3], note="maybe")])
    pairs = _declined_name_pairs(entries, d)
    assert frozenset(("lake mundi", "mundi")) in pairs   # shown as a candidate, not merged
    assert frozenset(("castle", "castel")) in pairs      # parked as a possible duplicate


def test_declined_pairs_excludes_pairs_the_llm_merged():
    entries = [loc("Lake Mundi"), loc("Mundi")]
    d = ReconcileDecision(merges=[MergeGroup(members=[0, 1], canonical="Lake Mundi")])
    assert _declined_name_pairs(entries, d) == set()


def test_reconcile_e2e_respects_fragment_the_llm_declined():
    # The pair is surfaced to the LLM as a candidate and it returns an EMPTY decision
    # (keep separate). Even with alias evidence the floor must not override that call.
    rec = Reconciler(client=FakeClient([NO_MERGE]))
    out = rec.reconcile([loc("Lake Mundi", aliases=["Mundi"]), loc("Mundi"), loc("Castle")])
    assert sorted(e.name for e in out) == ["Castle", "Lake Mundi", "Mundi"]


def test_reconcile_e2e_folds_unflagged_fragment_with_evidence():
    # A fragment contained only via an ALIAS isn't surfaced by _candidate_pairs (it scans
    # names), so the LLM never judged it; with shared-quote evidence the floor folds it.
    q = Quote(text="The Tower loomed over us.", speaker="Matt", source_file="g.txt")
    rec = Reconciler(client=FakeClient([NO_MERGE]))
    out = rec.reconcile([loc("The Citadel", aliases=["White Tower"], quotes=[q]),
                         loc("Tower", quotes=[q]), loc("Riverton")])
    assert sorted(e.name for e in out) == ["Riverton", "The Citadel"]
