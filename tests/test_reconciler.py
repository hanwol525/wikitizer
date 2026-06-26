"""Unit tests for agents/reconciler.py (Phase 4.1a: the reconciler MERGE step).

These exercise the DETERMINISTIC half -- the pure-Python combiner, the scalar
resolvers, the short-name veto, the decision validator, and the `_apply` loop --
with hand-built fake entries and EXACT assertions. No network, no API key: they
run under plain `pytest`. The single live-API test (the LLM actually deciding the
typo-vs-sibling case) lives in test_reconciler_integration.py and is gated behind
the `integration` marker.

`_apply` is driven directly with a hand-built `ReconcileDecision`, so the whole
assemble path is covered with zero network; the Reconciler instance is built via
`__new__` (skipping `__init__`/the Anthropic client, which `_apply` never needs).
"""

import logging

import pytest
from pydantic import ValidationError

from models.lore import Location, Character, HistoryEvent, Quote, Scope
from models.reconcile import ReconcileDecision, MergeGroup, DetailConflict, PossibleDuplicate
from agents.reconciler import (
    Reconciler, _combine_group, _resolve_scope, _resolve_player_name,
    _short_name_veto, _validate_decision, _dedup_quotes, _VetoMerge,
    _resolve_canonical, _extract_json_object, REVIEW_PREFIX, _resolve_date_text,
)


def q(text, speaker="dm", source="dndgroup.txt"):
    return Quote(text=text, speaker=speaker, source_file=source)


def loc(name, aliases=None, details=None, quotes=None):
    return Location(name=name, aliases=aliases or [], details=details or [],
                    supporting_quotes=quotes or [])


# --- combine: aliases ------------------------------------------------------
def test_combine_unions_aliases_and_keeps_losing_name():
    a = loc("Lake Mundi", aliases=["The Pond"])
    b = loc("The Great Well", aliases=["The Well"])
    merged = _combine_group([a, b], "Lake Mundi")
    assert merged.name == "Lake Mundi"
    # losing name + all aliases present; canonical not in its own alias list
    assert set(merged.aliases) == {"The Great Well", "The Pond", "The Well"}
    assert "Lake Mundi" not in merged.aliases


# --- combine: details (exact-dupe only, order preserved) -------------------
def test_combine_dedups_details_exact_only_and_preserves_order():
    a = loc("X", details=["fed by a spring", "said to be bottomless"])
    b = loc("Y", details=["  fed by a spring ", "sacred to the Kriega"])  # whitespace dupe
    merged = _combine_group([a, b], "X")
    assert merged.details == ["fed by a spring", "said to be bottomless", "sacred to the Kriega"]


# --- combine: quotes (triple dedup, distinct speakers survive) -------------
def test_combine_dedups_quotes_on_triple_keeps_distinct_speakers():
    a = loc("X", quotes=[q("sacred to the Kriega", speaker="player_a")])
    b = loc("Y", quotes=[q("sacred to the Kriega", speaker="player_a"),   # exact dupe -> dropped
                          q("sacred to the Kriega", speaker="player_b")])  # diff speaker -> kept
    merged = _combine_group([a, b], "X")
    assert len(merged.supporting_quotes) == 2


# --- scalars: is_pc -------------------------------------------------------
def test_is_pc_true_wins():
    a = Character(name="CJ", is_pc=False)
    b = Character(name="CJ", is_pc=True)
    merged = _combine_group([a, b], "CJ")
    assert merged.is_pc is True


# --- scalars: scope (broadest wins) ---------------------------------------
def test_scope_broadest_wins():
    assert _resolve_scope([Scope.PERSONAL, Scope.REGIONAL]) == Scope.REGIONAL
    assert _resolve_scope([Scope.REGIONAL, Scope.WORLD, Scope.PERSONAL]) == Scope.WORLD


def test_history_merge_concatenates_descriptions_and_takes_broadest_scope():
    a = HistoryEvent(name="E1", description="The Empire fell.", scope=Scope.REGIONAL)
    b = HistoryEvent(name="E2", description="The dynasty collapsed.", scope=Scope.WORLD)
    merged = _combine_group([a, b], "E1")
    assert "The Empire fell." in merged.description
    assert "The dynasty collapsed." in merged.description
    assert merged.scope == Scope.WORLD
    assert merged.chronological_position is None  # untouched until 4.1b


# --- History merge carries date_text forward --------------------------------

def test_history_merge_date_beats_none():
    a = HistoryEvent(name="E1", description="The Empire fell.", scope="world",
                     date_text="342 AR")
    b = HistoryEvent(name="E2", description="The Empire fell.", scope="world")  # no date
    merged = _combine_group([a, b], "E1")
    assert merged.date_text == "342 AR"  # a stated date beats None


def test_history_merge_date_clash_keeps_first():
    a = HistoryEvent(name="E1", description="The Empire fell.", scope="world",
                     date_text="342 AR")
    b = HistoryEvent(name="E2", description="The Empire fell.", scope="world",
                     date_text="343 AR")  # a different stated date
    merged = _combine_group([a, b], "E1")
    assert merged.date_text == "342 AR"  # keeps first; the other survives in quotes/description


def test_resolve_date_text_all_none_is_none():
    a = HistoryEvent(name="E1", description="x", scope="world")
    b = HistoryEvent(name="E2", description="y", scope="world")
    assert _resolve_date_text([a, b]) is None


def test_history_merge_date_picks_stated_when_first_is_none():
    # The discriminating case: the FIRST member has no date, the SECOND states one.
    # The contract is "first STATED date wins" (the None-filter in
    # _resolve_date_text), NOT a naive "members[0].date_text" -- so the second
    # member's date must come through, not None.
    a = HistoryEvent(name="E1", description="The Empire fell.", scope="world")  # no date
    b = HistoryEvent(name="E2", description="The Empire fell.", scope="world",
                     date_text="343 AR")
    merged = _combine_group([a, b], "E1")
    assert merged.date_text == "343 AR"


def test_history_merge_date_clash_logs_at_debug_not_review(caplog):
    # A date clash is a QUIET auto-resolution: it must log at DEBUG and must NOT
    # carry the [REVIEW] prefix. [REVIEW] is reserved for the human review queue;
    # a clash is non-destructive (the losing date survives in the quotes), so
    # escalating it to a loud flag would wrongly pollute Phase 5.3's review.txt.
    a = HistoryEvent(name="E1", description="x", scope="world", date_text="342 AR")
    b = HistoryEvent(name="E2", description="y", scope="world", date_text="343 AR")
    with caplog.at_level(logging.DEBUG, logger="agents.reconciler"):
        _combine_group([a, b], "E1")
    assert any("date_text clash" in r.getMessage() for r in caplog.records)
    assert not any(REVIEW_PREFIX in r.getMessage() for r in caplog.records)


# --- scalars: player_name -------------------------------------------------
def test_player_name_name_beats_none():
    a = Character(name="CJ", player_name=None)
    b = Character(name="CJ", player_name="Hannah")
    merged = _combine_group([a, b], "CJ")
    assert merged.player_name == "Hannah"


def test_player_name_value_clash_vetoes():
    a = Character(name="The Wanderer", player_name="Hannah")
    b = Character(name="The Wanderer", player_name="Conrad")
    with pytest.raises(_VetoMerge):
        _combine_group([a, b], "The Wanderer")


# --- short-name veto ------------------------------------------------------
def test_short_name_veto_blocks_unstated():
    # CJ / DJ, no alias link between them -> veto
    assert _short_name_veto([loc("CJ"), loc("DJ")]) is True


def test_short_name_veto_allows_stated_alias():
    # one short name listed as the other's alias -> stated -> no veto
    assert _short_name_veto([loc("CJ", aliases=["DJ"]), loc("DJ")]) is False


def test_short_name_veto_ignores_long_names():
    # no short name in play -> rule doesn't apply
    assert _short_name_veto([loc("Maltaav"), loc("Maltraav")]) is False


# --- decision validation --------------------------------------------------
def _decision(members, canonical):
    return ReconcileDecision(merges=[MergeGroup(members=members, canonical=canonical)])


def test_validate_rejects_out_of_range_index():
    entries = [loc("A"), loc("B")]
    assert _validate_decision(_decision([0, 5], "A"), entries)  # non-empty == problems


def test_validate_rejects_index_in_two_groups():
    entries = [loc("A"), loc("B"), loc("C")]
    d = ReconcileDecision(merges=[
        MergeGroup(members=[0, 1], canonical="A"),
        MergeGroup(members=[1, 2], canonical="B"),  # index 1 reused
    ])
    assert _validate_decision(d, entries)


def test_validate_rejects_invented_canonical():
    entries = [loc("Lake Mundi"), loc("The Great Well")]
    assert _validate_decision(_decision([0, 1], "Atlantis"), entries)  # not a member name/alias


def test_validate_accepts_canonical_from_an_alias():
    entries = [loc("The Great Well", aliases=["Lake Mundi"]), loc("The Pond")]
    assert _validate_decision(_decision([0, 1], "Lake Mundi"), entries) == []  # clean


def test_validate_passes_clean_decision():
    entries = [loc("Lake Mundi"), loc("The Great Well")]
    assert _validate_decision(_decision([0, 1], "Lake Mundi"), entries) == []


# --- _apply loop (deterministic; no LLM) ----------------------------------
# Drive _apply directly with a hand-built decision so the whole assemble path is
# tested with zero network. Construct a Reconciler without triggering any client
# the way the noise-filter tests do (mirror that pattern); _apply itself needs no
# Claude call.
def test_apply_passes_through_singletons_and_merges():
    entries = [loc("Lake Mundi"), loc("Castle"), loc("The Great Well")]
    decision = _decision([0, 2], "Lake Mundi")  # merge 0+2, leave 1 alone
    rec = Reconciler.__new__(Reconciler)  # skip __init__/client; _apply needs none
    result = rec._apply(decision, entries, "Location")
    names = sorted(e.name for e in result)
    assert names == ["Castle", "Lake Mundi"]  # 3 entries -> 2 (one merge + one singleton)


def test_apply_vetoed_short_name_merge_keeps_members_separate():
    entries = [Character(name="CJ"), Character(name="DJ")]
    decision = _decision([0, 1], "CJ")  # LLM proposed merging short names
    rec = Reconciler.__new__(Reconciler)
    result = rec._apply(decision, entries, "Character")
    assert sorted(e.name for e in result) == ["CJ", "DJ"]  # veto -> both survive


# ===========================================================================
# Additions after the Phase 4.1a adversarial review: regression tests for the
# three confirmed bugs (empty merge group -> crash; recased canonical ->
# invented heading; self-alias defeats the short-name veto) plus deterministic
# coverage for the reconcile() orchestration the live test alone used to guard
# (retry / validate-retry / total-failure / prose-wrapped JSON recovery) and the
# [REVIEW]-log paths that ARE the product (conflicts, possible_duplicates).
# ===========================================================================


# --- fake client (mirrors test_noise_filter.py to stay self-contained) ------
class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, blocks):
        self.content = blocks


class FakeMessages:
    """``create`` consumes one spec per call; once a single spec remains it is
    returned for every further call (a one-element list = 'always the same')."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        blocks = [FakeTextBlock(spec)] if isinstance(spec, str) else spec
        return FakeResponse(blocks)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)

    @property
    def call_count(self):
        return len(self.messages.calls)


def make_reconciler(responses, **kwargs):
    return Reconciler(client=FakeClient(responses), **kwargs)


# A clean, valid decision merging the two-entry typo case below.
GOOD_MERGE = '{"merges": [{"members": [0, 1], "canonical": "Maltraav"}], "possible_duplicates": []}'


def _typo_pair():
    return [loc("Maltaav"), loc("Maltraav")]


# === BUG 1: empty / single-member merge group must not crash ================
def test_validate_rejects_empty_members_group():
    entries = [loc("A"), loc("B")]
    d = ReconcileDecision(merges=[MergeGroup(members=[], canonical="A")])
    assert _validate_decision(d, entries)  # flagged as a problem, not clean


def test_validate_rejects_single_member_group():
    entries = [loc("A"), loc("B")]
    d = ReconcileDecision(merges=[MergeGroup(members=[0], canonical="A")])
    assert _validate_decision(d, entries)


def test_apply_empty_members_group_does_not_crash():
    # The original bug: members=[] passed validation, then _combine_group([])
    # did members[0] -> IndexError out of reconcile(). The _apply guard must now
    # skip it and pass everything through as singletons.
    entries = [loc("A"), loc("B")]
    d = ReconcileDecision(merges=[MergeGroup(members=[], canonical="A")])
    rec = Reconciler.__new__(Reconciler)
    result = rec._apply(d, entries, "Location")  # must NOT raise
    assert sorted(e.name for e in result) == ["A", "B"]


def test_reconcile_empty_members_decision_does_not_crash():
    # End-to-end: a bad empty-members decision is flagged by validation, retried
    # 3x, and finally returned unmerged -- never an uncaught IndexError.
    bad = '{"merges": [{"members": [], "canonical": "Maltraav"}], "possible_duplicates": []}'
    rec = make_reconciler([bad])
    result = rec.reconcile(_typo_pair())
    assert [e.name for e in result] == ["Maltaav", "Maltraav"]  # unmerged passthrough


# === BUG 2: a recased/padded canonical must snap to the verbatim member name =
def test_resolve_canonical_prefers_name_then_alias_else_passthrough():
    a = loc("Lake Mundi", aliases=["The Pond"])
    b = loc("The Great Well")
    assert _resolve_canonical([a, b], "lake mundi") == "Lake Mundi"   # name match (recased)
    assert _resolve_canonical([a, b], "  the pond ") == "The Pond"    # alias match (padded)
    assert _resolve_canonical([a, b], "Atlantis") == "Atlantis"       # no match -> unchanged


def test_combine_snaps_recased_canonical_and_does_not_demote_real_name():
    a = loc("Lake Mundi", aliases=["The Pond"])
    b = loc("The Great Well")
    merged = _combine_group([a, b], "  lake mundi ")  # LLM recased + padded the canonical
    assert merged.name == "Lake Mundi"            # the verbatim member name, not the LLM string
    assert "Lake Mundi" not in merged.aliases     # the real name was NOT demoted to an alias
    assert set(merged.aliases) == {"The Pond", "The Great Well"}


# === BUG 3: a self-alias is not a cross-member stated link ==================
def test_short_name_veto_self_alias_is_not_a_stated_link():
    # A short name listing ITSELF in its own aliases must not satisfy the
    # stated-alias rule -- otherwise CJ(aliases=['CJ']) + DJ would wrongly merge.
    assert _short_name_veto([Character(name="CJ", aliases=["CJ"]), Character(name="DJ")]) is True
    assert _short_name_veto([Character(name="AB", aliases=["AB"]),
                             Character(name="CD", aliases=["CD"])]) is True


# === BUG 4 (live-test root cause): parse robustly around fences/preambles ====
def test_extract_json_object_handles_prose_and_strings_and_misses():
    assert _extract_json_object('prefix {"a": 1} suffix') == '{"a": 1}'
    # braces inside a string literal must not throw off the depth count
    assert _extract_json_object('x {"a": "}{"} y') == '{"a": "}{"}'
    assert _extract_json_object("no object here") is None
    assert _extract_json_object("") is None


def test_reconcile_recovers_from_prose_wrapped_json():
    # Sonnet's real failure mode: a conversational preamble around the JSON, which
    # strip_code_fences can't peel. Must still recover the embedded object + merge.
    preamble = "Looking at these two entries, they're the same place. " + GOOD_MERGE
    rec = make_reconciler([preamble])
    result = rec.reconcile(_typo_pair())
    assert len(result) == 1
    assert result[0].name == "Maltraav"


def test_reconcile_recovers_from_fenced_json():
    fenced = "```json\n" + GOOD_MERGE + "\n```"
    rec = make_reconciler([fenced])
    result = rec.reconcile(_typo_pair())
    assert len(result) == 1


# === reconcile() orchestration (retry / validate-retry / failure / short) ===
def test_reconcile_short_circuits_under_two_entries_with_no_call():
    client = FakeClient([GOOD_MERGE])
    rec = Reconciler(client=client)
    one = [loc("A")]
    assert rec.reconcile([]) == []
    result = rec.reconcile(one)
    assert len(result) == 1 and result[0] is one[0]  # passed straight through
    assert client.call_count == 0  # no API call for < 2 entries


def test_reconcile_clean_merge_collapses_two_to_one():
    rec = make_reconciler([GOOD_MERGE])
    result = rec.reconcile(_typo_pair())
    assert len(result) == 1
    assert result[0].name == "Maltraav"
    assert "Maltaav" in result[0].aliases  # losing name preserved as an alias


def test_reconcile_retries_bad_json_then_succeeds():
    client = FakeClient(["not json", "still not json", GOOD_MERGE])
    rec = Reconciler(client=client)
    result = rec.reconcile(_typo_pair())
    assert len(result) == 1
    assert client.call_count == 3  # two parse failures, then success on attempt 3


def test_reconcile_retries_structurally_invalid_decision():
    # idx 9 is out of range -> _validate_decision returns problems -> retry.
    bad = '{"merges": [{"members": [0, 9], "canonical": "Maltraav"}], "possible_duplicates": []}'
    client = FakeClient([bad, GOOD_MERGE])
    rec = Reconciler(client=client)
    result = rec.reconcile(_typo_pair())
    assert len(result) == 1
    assert client.call_count == 2


def test_reconcile_total_failure_returns_entries_unmerged_and_logs(caplog):
    client = FakeClient(["never valid json"])
    rec = Reconciler(client=client)
    entries = _typo_pair()
    with caplog.at_level(logging.ERROR, logger="agents.reconciler"):
        result = rec.reconcile(entries)
    assert [e.name for e in result] == ["Maltaav", "Maltraav"]  # unmerged passthrough
    assert client.call_count == 3  # exhausted all 3 attempts
    assert any("no usable decision after 3 attempts" in r.getMessage()
               for r in caplog.records if r.levelno == logging.ERROR)


# === the [REVIEW] log lines ARE the product on these paths ==================
def test_apply_logs_conflict_with_review_prefix_and_keeps_both_details(caplog):
    a = loc("Riverton", details=["ruled by the Kriega"])
    b = loc("Lakeside Keep", details=["ruled by the Maltraav"])
    conflict = DetailConflict(detail_a="ruled by the Kriega", source_a="a.txt",
                              detail_b="ruled by the Maltraav", source_b="b.txt",
                              note="two different rulers named")
    decision = ReconcileDecision(merges=[
        MergeGroup(members=[0, 1], canonical="Riverton", conflicts=[conflict])
    ])
    rec = Reconciler.__new__(Reconciler)
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        result = rec._apply(decision, [a, b], "Location")
    assert len(result) == 1
    # both sides of the contradiction are KEPT verbatim, never dropped
    assert "ruled by the Kriega" in result[0].details
    assert "ruled by the Maltraav" in result[0].details
    assert any(REVIEW_PREFIX in r.getMessage() and "contradiction" in r.getMessage()
               for r in caplog.records)


def test_apply_logs_one_review_line_per_conflict(caplog):
    # Guards the restored `for c in group.conflicts:` loop against a subtle
    # re-regression that collapses it to a single log: TWO conflicts must produce
    # TWO contradiction lines (one per conflict), not one summary line. This is the
    # behavior a botched auto-fix once silently dropped, so it's worth pinning hard.
    a = loc("Riverton", details=["ruled by the Kriega", "founded in spring"])
    b = loc("Lakeside Keep", details=["ruled by the Maltraav", "founded in autumn"])
    conflicts = [
        DetailConflict(detail_a="ruled by the Kriega", source_a="a.txt",
                       detail_b="ruled by the Maltraav", source_b="b.txt", note="rulers differ"),
        DetailConflict(detail_a="founded in spring", source_a="a.txt",
                       detail_b="founded in autumn", source_b="b.txt", note="seasons differ"),
    ]
    decision = ReconcileDecision(merges=[
        MergeGroup(members=[0, 1], canonical="Riverton", conflicts=conflicts)
    ])
    rec = Reconciler.__new__(Reconciler)
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        rec._apply(decision, [a, b], "Location")
    contradiction_lines = [r for r in caplog.records if "contradiction" in r.getMessage()]
    assert len(contradiction_lines) == 2  # one per conflict, never collapsed to one
    joined = " ".join(r.getMessage() for r in contradiction_lines)
    assert "rulers differ" in joined and "seasons differ" in joined


def test_apply_logs_possible_duplicate_with_review_prefix_without_merging(caplog):
    entries = [loc("Riverton"), loc("Riverside")]
    decision = ReconcileDecision(
        possible_duplicates=[PossibleDuplicate(members=[0, 1], note="sound alike, unconfirmed")]
    )
    rec = Reconciler.__new__(Reconciler)
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        result = rec._apply(decision, entries, "Location")
    assert sorted(e.name for e in result) == ["Riverside", "Riverton"]  # NEITHER merged
    assert any(REVIEW_PREFIX in r.getMessage() and "possible duplicate" in r.getMessage()
               for r in caplog.records)


def test_apply_player_name_veto_logs_review_prefix_and_keeps_separate(caplog):
    a = Character(name="The Wanderer", player_name="Hannah")
    b = Character(name="The Wanderer", player_name="Conrad")  # different real human
    decision = _decision([0, 1], "The Wanderer")
    rec = Reconciler.__new__(Reconciler)
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        result = rec._apply(decision, [a, b], "Character")
    assert len(result) == 2  # veto -> both survive separately
    assert any(REVIEW_PREFIX in r.getMessage() and "player_name clash" in r.getMessage()
               for r in caplog.records)


# === multi-group decisions and 3+ member groups (consumed-set bookkeeping) ==
def test_apply_handles_multiple_merge_groups():
    entries = [loc("Riverton"), loc("Castle Black"), loc("Rivertown"),
               loc("Black Castle"), loc("Lonely Tower")]
    decision = ReconcileDecision(merges=[
        MergeGroup(members=[0, 2], canonical="Riverton"),
        MergeGroup(members=[1, 3], canonical="Castle Black"),
    ])
    rec = Reconciler.__new__(Reconciler)
    result = rec._apply(decision, entries, "Location")
    assert sorted(e.name for e in result) == ["Castle Black", "Lonely Tower", "Riverton"]  # 5 -> 3


def test_combine_three_members_unions_all_aliases_and_dedups_quotes():
    a = loc("Riverton", aliases=["The River"], quotes=[q("at Riverton")])
    b = loc("Rivertown", quotes=[q("at Riverton")])                       # exact-dupe quote
    c = loc("River City", aliases=["Rivertown"],
            quotes=[q("by the water", speaker="player_b")])
    merged = _combine_group([a, b, c], "Riverton")
    assert merged.name == "Riverton"
    assert set(merged.aliases) == {"The River", "Rivertown", "River City"}
    assert "Riverton" not in merged.aliases
    assert len(merged.supporting_quotes) == 2  # the two identical "at Riverton" collapse to one


# === _dedup_quotes: the source_file leg of the triple key ===================
def test_dedup_quotes_keeps_distinct_source_files():
    quotes = [
        q("same fact", speaker="dm", source="a.txt"),
        q("same fact", speaker="dm", source="b.txt"),   # different source -> kept (more evidence)
        q("same fact", speaker="dm", source="a.txt"),   # exact triple dupe -> dropped
    ]
    out = _dedup_quotes(quotes)
    assert len(out) == 2
    assert {x.source_file for x in out} == {"a.txt", "b.txt"}
