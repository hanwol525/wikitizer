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

import json
import logging
import re

import pytest
from pydantic import ValidationError

from models.lore import (
    Location, Character, HistoryEvent, Quote, Scope, Detail, Alias,
    Organization, Item, PeopleAndCultures,
)
from models.reconcile import (
    ReconcileDecision, MergeGroup, DetailConflict, PossibleDuplicate,
    DateDecision, DatedEvent, PlacementDecision, GapPlacement,
)
from agents.reconciler import (
    Reconciler, _combine_group, _resolve_scope, _resolve_player_name,
    _short_name_veto, _validate_decision, _dedup_quotes, _dedup_details,
    _dedup_aliases, _canonical_sources, _entity_for_prompt, _VetoMerge,
    _resolve_canonical, _extract_json_object, REVIEW_PREFIX, _resolve_date_text,
    _sanity_guard_parts, _validate_date_decision, _build_spine_and_gaps,
    _validate_placement_decision, _weave_and_stamp, UNDATED,
)


def q(text, speaker="dm", source="dndgroup.txt"):
    return Quote(text=text, speaker=speaker, source_file=source)


def det(text, *source_files):
    return Detail(text=text, source_files=list(source_files))


def al(text, *source_files):
    return Alias(text=text, source_files=list(source_files))


def _aliases(aliases):
    """Accept bare strings (provenance-free) or real Aliases."""
    return [Alias(text=a, source_files=[]) if isinstance(a, str) else a
            for a in (aliases or [])]


def loc(name, aliases=None, details=None, quotes=None):
    return Location(name=name, aliases=_aliases(aliases), details=details or [],
                    supporting_quotes=quotes or [])


def hev(name, scope="world", date_text=None):
    return HistoryEvent(name=name, description=f"{name} happened.", scope=scope,
                        date_text=date_text)


# --- combine: aliases ------------------------------------------------------
def test_combine_unions_aliases_and_keeps_losing_name():
    a = loc("Lake Mundi", aliases=["The Pond"])
    b = loc("The Great Well", aliases=["The Well"])
    merged = _combine_group([a, b], "Lake Mundi")
    assert merged.name == "Lake Mundi"
    # losing name + all aliases present; canonical not in its own alias list
    assert {a.text for a in merged.aliases} == {"The Great Well", "The Pond", "The Well"}
    assert "Lake Mundi" not in [a.text for a in merged.aliases]


# --- combine: details (same-fact de-dup, sources unioned, order preserved) -
def test_combine_dedups_details_unions_sources_and_preserves_order():
    a = loc("X", details=[det("fed by a spring", "public.txt"),
                          det("said to be bottomless", "public.txt")])
    b = loc("Y", details=[det("  fed by a spring ", "secret.txt"),   # whitespace dupe, diff file
                          det("sacred to the Kriega", "secret.txt")])
    merged = _combine_group([a, b], "X")
    # visible axis unchanged: same texts, first-seen verbatim, same order
    assert [d.text for d in merged.details] == \
        ["fed by a spring", "said to be bottomless", "sacred to the Kriega"]
    # invisible axis: the shared fact remembers BOTH files (ordered, deduped)
    assert merged.details[0].source_files == ["public.txt", "secret.txt"]
    assert merged.details[1].source_files == ["public.txt"]
    assert merged.details[2].source_files == ["secret.txt"]


def test_combine_details_same_fact_same_file_keeps_one_source():
    a = loc("X", details=[det("fed by a spring", "public.txt")])
    b = loc("Y", details=[det("fed by a spring", "public.txt")])  # exact dupe, same file
    merged = _combine_group([a, b], "X")
    assert [d.text for d in merged.details] == ["fed by a spring"]
    assert merged.details[0].source_files == ["public.txt"]        # not doubled


def test_dedup_details_unions_orders_and_preserves_first_seen():
    # unit test the helper in isolation
    out = _dedup_details([det("a", "f1"), det("b", "f1"), det(" a ", "f2"), det("a", "f1")])
    assert [d.text for d in out] == ["a", "b"]
    assert out[0].source_files == ["f1", "f2"]   # f1 (1st & 4th) + f2, deduped
    assert out[1].source_files == ["f1"]


# --- merge prompt hides the new source_files tags (behaviour-neutrality) ----
def test_merge_prompt_shows_details_as_bare_strings_no_source_files():
    e = loc("X", details=[det("fed by a spring", "secret.txt")])
    data = _entity_for_prompt(e)
    assert data["details"] == ["fed by a spring"]        # reduced to bare text
    assert "source_files" not in json.dumps(data)        # tag never reaches the LLM


def test_entity_for_prompt_is_noop_for_history():
    ev = hev("The Sundering")
    data = _entity_for_prompt(ev)
    assert "details" not in data     # HistoryEvent has none; helper doesn't invent one


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
    assert merged.calendar_system is None         # also a 4.1b field; merge leaves it None


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
    assert merged.date_text == "342 AR"  # keeps first stated date; the other is dropped from date_text


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
    # A date clash is a QUIET auto-resolution (like is_pc/scope): it must log at
    # DEBUG and must NOT carry the [REVIEW] prefix. [REVIEW] is reserved for the
    # human review queue; a clash is auto-resolved by keeping the first stated
    # date, so escalating it to a loud flag would wrongly pollute Phase 5.3's
    # review.txt.
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
    assert "Lake Mundi" not in [a.text for a in merged.aliases]  # real name NOT demoted to an alias
    assert {a.text for a in merged.aliases} == {"The Pond", "The Great Well"}


# === BUG 3: a self-alias is not a cross-member stated link ==================
def test_short_name_veto_self_alias_is_not_a_stated_link():
    # A short name listing ITSELF in its own aliases must not satisfy the
    # stated-alias rule -- otherwise CJ(aliases=['CJ']) + DJ would wrongly merge.
    assert _short_name_veto([Character(name="CJ", aliases=[al("CJ")]), Character(name="DJ")]) is True
    assert _short_name_veto([Character(name="AB", aliases=[al("AB")]),
                             Character(name="CD", aliases=[al("CD")])]) is True


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
    assert "Maltaav" in [a.text for a in result[0].aliases]  # losing name preserved as an alias


def test_reconcile_prompt_never_carries_source_files_end_to_end():
    # Belt-and-suspenders for the behaviour-neutrality invariant: drive a real
    # reconcile() over entities whose details are tagged with source files, and
    # confirm neither the tag key nor the file names ever reach the prompt the
    # client was handed -- so a future prompt-builder change can't silently leak
    # provenance to the LLM and shift its merge decisions.
    a = loc("Maltaav", details=[det("a coastal city", "secret.txt")])
    b = loc("Maltraav", details=[det("a city on the coast", "public.txt")])
    rec = make_reconciler([GOOD_MERGE])
    rec.reconcile([a, b])
    sent_prompt = rec.client.messages.calls[0]["messages"][0]["content"]
    assert "source_files" not in sent_prompt
    assert "secret.txt" not in sent_prompt and "public.txt" not in sent_prompt


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
    a = loc("Riverton", details=[det("ruled by the Kriega")])
    b = loc("Lakeside Keep", details=[det("ruled by the Maltraav")])
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
    assert "ruled by the Kriega" in [d.text for d in result[0].details]
    assert "ruled by the Maltraav" in [d.text for d in result[0].details]
    assert any(REVIEW_PREFIX in r.getMessage() and "contradiction" in r.getMessage()
               for r in caplog.records)


def test_apply_logs_one_review_line_per_conflict(caplog):
    # Guards the restored `for c in group.conflicts:` loop against a subtle
    # re-regression that collapses it to a single log: TWO conflicts must produce
    # TWO contradiction lines (one per conflict), not one summary line. This is the
    # behavior a botched auto-fix once silently dropped, so it's worth pinning hard.
    a = loc("Riverton", details=[det("ruled by the Kriega"), det("founded in spring")])
    b = loc("Lakeside Keep", details=[det("ruled by the Maltraav"), det("founded in autumn")])
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
    assert {a.text for a in merged.aliases} == {"The River", "Rivertown", "River City"}
    assert "Riverton" not in [a.text for a in merged.aliases]
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


# ===========================================================================
# Phase 4.1b -- the pure-Python timeline engine (no network). The spine/gaps,
# the parts guard, the two structural validators, and the weave/stamp. This is
# the deterministic core, so it is over-covered on purpose.
# ===========================================================================


# --- 3a. the parts guard ----------------------------------------------------
def test_parts_guard_numeric_match_passes():
    assert _sanity_guard_parts([342], "342 AR") is True


def test_parts_guard_numeric_mismatch_fails():
    assert _sanity_guard_parts([343], "342 AR") is False          # clear transcription error


def test_parts_guard_word_only_date_is_trusted():
    assert _sanity_guard_parts([3], "the Third Age") is True       # no digits -> can't verify -> trust


def test_parts_guard_partial_overlap_passes():
    assert _sanity_guard_parts([4, 200], "4th Era 200") is True
    assert _sanity_guard_parts([4, 201], "4th Era 200") is True    # 4 still grounds it -> minor slip ok


def test_parts_guard_fully_unmoored_numeric_fails():
    assert _sanity_guard_parts([4, 200], "year 342") is False      # shares none of the stated numbers (342)


def test_parts_guard_none_date_text_is_trusted():
    # an extractor could hand a None date_text through; no digits -> can't verify -> trust
    assert _sanity_guard_parts([3], None) is True


# --- 3b. _validate_date_decision --------------------------------------------
def _dd(*triples):  # (index, system, parts) -> DateDecision
    return DateDecision(dated=[DatedEvent(index=i, system=s, parts=p) for i, s, p in triples])


def test_validate_date_rejects_out_of_range_index():
    assert _validate_date_decision(_dd((5, "AR", [1])), [hev("A", date_text="1 AR")])


def test_validate_date_rejects_event_without_date_text():
    # index 0 is a real event but has no date_text -> can't extract a date from it
    assert _validate_date_decision(_dd((0, "AR", [1])), [hev("A")])


def test_validate_date_rejects_empty_or_negative_parts():
    # empty parts -> still rejected; a NEGATIVE part -> still rejected.
    assert _validate_date_decision(_dd((0, "AR", [])), [hev("A", date_text="1 AR")])
    assert _validate_date_decision(_dd((0, "AR", [-1])), [hev("A", date_text="-1 AR")])


def test_validate_date_accepts_year_zero():
    # year 0 is a REAL in-world date ("Ferridus Krieger [0 to 50]"); parts:[0] must be
    # accepted, not rejected -- rejecting it used to undate the whole timeline.
    assert _validate_date_decision(_dd((0, "AR", [0])), [hev("A", date_text="0 AR")]) == []


def test_validate_date_rejects_duplicate_index():
    evs = [hev("A", date_text="1 AR")]
    assert _validate_date_decision(_dd((0, "AR", [1]), (0, "AR", [2])), evs)


def test_validate_date_rejects_empty_system_label():
    assert _validate_date_decision(_dd((0, "  ", [1])), [hev("A", date_text="1 AR")])


def test_validate_date_passes_clean():
    evs = [hev("A", date_text="342 AR"), hev("B", date_text="400 AR")]
    assert _validate_date_decision(_dd((0, "AR", [342]), (1, "AR", [400])), evs) == []


# --- 3c. _build_spine_and_gaps ----------------------------------------------
def test_spine_sorts_within_system_and_numbers_gaps():
    # two AR events out of order -> sorted; m=2 groups -> 3 gaps
    spines, gaps = _build_spine_and_gaps([(0, "AR", [400]), (1, "AR", [342])])
    assert spines["AR"] == [[1], [0]]                      # 342 (idx1) before 400 (idx0)
    assert set(g for g in gaps if g.startswith("AR#")) == {"AR#0", "AR#1", "AR#2"}


def test_spine_collapses_ties_into_one_rank_group():
    # identical parts -> one rank-group -> only 2 gaps (no phantom gap between ties)
    spines, gaps = _build_spine_and_gaps([(0, "AR", [342]), (1, "AR", [342])])
    assert spines["AR"] == [[0, 1]]                        # both in one group
    assert {g for g in gaps if g.startswith("AR#")} == {"AR#0", "AR#1"}


def test_spine_tie_group_preserves_input_order():
    # equal parts -> one group, members in the order they appeared in `dated`
    spines, _ = _build_spine_and_gaps([(2, "AR", [342]), (5, "AR", [342])])
    assert spines["AR"] == [[2, 5]]


def test_spine_multi_part_tuple_sorts_era_then_year():
    # [3, 999] sorts before [4, 1] element-by-element, with no calendar arithmetic
    spines, _ = _build_spine_and_gaps([(0, "Eras", [4, 1]), (1, "Eras", [3, 999])])
    assert spines["Eras"] == [[1], [0]]                    # era 3 yr 999 before era 4 yr 1


def test_spine_always_includes_undated_pseudo_system():
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1])])
    assert UNDATED in spines and spines[UNDATED] == []
    assert f"{UNDATED}#0" in gaps


def test_spine_multi_system_namespaces_gaps():
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1]), (1, "Eras", [4, 200])])
    assert "AR#0" in gaps and "Eras#0" in gaps
    assert gaps["AR#0"][0] == "AR" and gaps["Eras#0"][0] == "Eras"


def test_spine_dateless_input_has_only_undated_one_gap():
    spines, gaps = _build_spine_and_gaps([])
    assert list(spines) == [UNDATED] and spines[UNDATED] == []
    assert list(gaps) == [f"{UNDATED}#0"]


# --- 3d. _weave_and_stamp (the heart of it) ---------------------------------
def test_weave_inserts_relative_between_dated_markers():
    # spine: Founding(0)=1300, Sundering(1)=1350; relative Aldward(2) in the middle gap
    events = [hev("Founding", date_text="1300"), hev("Sundering", date_text="1350"),
              hev("Aldward")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1300]), (1, "AR", [1350])])
    out = _weave_and_stamp(events, spines, {"AR#1": [2]}, gaps)
    by_name = {e.name: e for e in out}
    assert by_name["Founding"].chronological_position == 0
    assert by_name["Aldward"].chronological_position == 1   # woven into the gap
    assert by_name["Sundering"].chronological_position == 2
    assert all(by_name[n].calendar_system == "AR" for n in ("Founding", "Aldward", "Sundering"))


def test_weave_relative_before_first_marker():
    events = [hev("Founding", date_text="1300"), hev("Before")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1300])])
    out = _weave_and_stamp(events, spines, {"AR#0": [1]}, gaps)   # gap 0 = before everything
    by_name = {e.name: e for e in out}
    assert by_name["Before"].chronological_position == 0
    assert by_name["Founding"].chronological_position == 1


def test_weave_tied_markers_then_relative_in_following_gap():
    # two events share a date (one rank-group), a relative sits in the gap AFTER them.
    # Tied markers get adjacent positions; the relative follows.
    events = [hev("Founding", date_text="1300"), hev("Charter", date_text="1300"),
              hev("After")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1300]), (1, "AR", [1300])])
    assert spines["AR"] == [[0, 1]]                          # collapsed tie -> gaps AR#0, AR#1
    out = _weave_and_stamp(events, spines, {"AR#1": [2]}, gaps)
    by_name = {e.name: e for e in out}
    assert by_name["Founding"].chronological_position == 0
    assert by_name["Charter"].chronological_position == 1
    assert by_name["After"].chronological_position == 2
    assert by_name["After"].calendar_system == "AR"


def test_weave_within_gap_order_is_preserved():
    events = [hev("Sundering", date_text="1350"), hev("X"), hev("Y")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1350])])
    out = _weave_and_stamp(events, spines, {"AR#0": [2, 1]}, gaps)  # Y then X, before Sundering
    by_name = {e.name: e for e in out}
    assert by_name["Y"].chronological_position == 0
    assert by_name["X"].chronological_position == 1
    assert by_name["Sundering"].chronological_position == 2


def test_weave_unplaced_event_is_could_not_place():
    events = [hev("Sundering", date_text="1350"), hev("Orphan")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1350])])
    out = _weave_and_stamp(events, spines, {}, gaps)            # Orphan placed nowhere
    orphan = next(e for e in out if e.name == "Orphan")
    assert orphan.calendar_system is None and orphan.chronological_position is None


def test_weave_dateless_campaign_all_on_none_timeline():
    events = [hev("A"), hev("B"), hev("C")]
    spines, gaps = _build_spine_and_gaps([])                    # no dates
    out = _weave_and_stamp(events, spines, {f"{UNDATED}#0": [0, 1, 2]}, gaps)
    for i, e in enumerate(out):
        assert e.calendar_system is None                       # the None timeline, not "(undated)"
        assert e.chronological_position == i                   # ordered 0,1,2


def test_weave_positions_number_per_system():
    events = [hev("A", date_text="1 AR"), hev("B", date_text="1 ES")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1]), (1, "ES", [1])])
    out = _weave_and_stamp(events, spines, {}, gaps)
    # each system's single event starts at position 0
    assert all(e.chronological_position == 0 for e in out)
    assert {e.calendar_system for e in out} == {"AR", "ES"}


def test_weave_undated_events_in_mixed_campaign_get_none_system_with_positions():
    # dated AR spine + an unanchored undated pair on the (undated) timeline
    events = [hev("Dated", date_text="1 AR"), hev("Aldward"), hev("Borren")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1])])
    out = _weave_and_stamp(events, spines, {f"{UNDATED}#0": [1, 2]}, gaps)
    by_name = {e.name: e for e in out}
    assert by_name["Dated"].calendar_system == "AR"
    assert by_name["Aldward"].calendar_system is None and by_name["Aldward"].chronological_position == 0
    assert by_name["Borren"].calendar_system is None and by_name["Borren"].chronological_position == 1


def test_weave_unknown_gap_id_falls_to_could_not_place():
    events = [hev("Sundering", date_text="1350"), hev("Lost")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1350])])
    out = _weave_and_stamp(events, spines, {"AR#99": [1]}, gaps)  # gap 99 doesn't exist
    lost = next(e for e in out if e.name == "Lost")
    assert lost.chronological_position is None


def test_weave_does_not_mutate_inputs():
    events = [hev("A", date_text="1 AR")]
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1])])
    _weave_and_stamp(events, spines, {}, gaps)
    assert events[0].chronological_position is None and events[0].calendar_system is None


def test_weave_distinguishes_undated_placed_from_could_not_place():
    """Locks the render-time split that 4.4 depends on. Inside the
    calendar_system=None world, an undated-but-relatively-placed event and a
    truly-floating event MUST stay distinguishable -- and the only thing that
    separates them is chronological_position (an int = we placed it; None = we
    couldn't). A plain `x in output` membership check would pass while proving
    none of that, so we assert on the position marker itself.

    Three events, one per render category:
      index 0 -> dated event in a real system      -> (real system, int)
      index 1 -> undated but placed into a gap      -> (None, int)
      index 2 -> floating (placed nowhere at all)   -> (None, None)
    """
    # Build inputs the same way the sibling weave tests do: real HistoryEvents
    # (the dated one carries a date_text), and spines/gaps produced by
    # _build_spine_and_gaps instead of hand-rolled -- so we exercise the exact
    # gap-id format ("(undated)#0") the real pipeline emits, not a stand-in.
    events = [
        hev("The Founding", date_text="400"),  # 0: dated
        hev("A Quiet Season"),                  # 1: undated but placed
        hev("A Rumor"),                         # 2: floating
    ]
    # One dated marker (event 0) -> _build_spine_and_gaps hands back its "Crown
    # Reckoning" spine PLUS the UNDATED pseudo-spine and its gap id.
    spines, gaps = _build_spine_and_gaps([(0, "Crown Reckoning", [400])])

    # Drop event 1 into the undated spine's first (and only) gap. Event 2 appears
    # in NO placement -> it falls through to the (None, None) default. "Could Not
    # Place" is produced by omission, not by a flag.
    out = _weave_and_stamp(events, spines, {f"{UNDATED}#0": [1]}, gaps)
    dated, placed, floating = out[0], out[1], out[2]

    # dated -> its own system's dated timeline
    assert dated.calendar_system == "Crown Reckoning"
    assert dated.chronological_position is not None  # it's 0 -- falsy, hence `is not None`

    # undated but placed -> ## Undated Events: None system, but a real int position.
    # Position is 0 on purpose (the *falsy* int), so this also documents that a
    # placed event can legitimately sit at 0 -- consumers MUST use `is not None`,
    # never truthiness.
    assert placed.calendar_system is None
    assert placed.chronological_position == 0

    # floating -> ## Could Not Place: both fields None.
    assert floating.calendar_system is None
    assert floating.chronological_position is None

    # Thesis: `placed` and `floating` BOTH have calendar_system=None, so that field
    # alone can't tell them apart -- only chronological_position does. It's a real
    # int for `placed` (vs None for `floating`, asserted just above), which is
    # exactly the "different kinds" distinction a naive membership check misses.
    assert isinstance(placed.chronological_position, int)


# --- 3e. _validate_placement_decision ---------------------------------------
def _setup_gaps():
    # one AR marker -> gaps AR#0, AR#1, plus (undated)#0; dated index is {0}
    spines, gaps = _build_spine_and_gaps([(0, "AR", [1350])])
    return gaps


def _pd(*pairs):  # (gap, [events]) -> PlacementDecision
    return PlacementDecision(placements=[GapPlacement(gap=g, events=e) for g, e in pairs])


def test_validate_placement_rejects_unknown_gap():
    evs = [hev("M", date_text="1350"), hev("R")]
    assert _validate_placement_decision(_pd(("AR#9", [1])), evs, {0}, _setup_gaps())


def test_validate_placement_rejects_dated_event():
    evs = [hev("M", date_text="1350"), hev("R")]
    assert _validate_placement_decision(_pd(("AR#0", [0])), evs, {0}, _setup_gaps())  # 0 is dated


def test_validate_placement_rejects_double_placement():
    evs = [hev("M", date_text="1350"), hev("R")]
    assert _validate_placement_decision(_pd(("AR#0", [1]), ("AR#1", [1])), evs, {0}, _setup_gaps())


def test_validate_placement_rejects_duplicate_gap():
    evs = [hev("M", date_text="1350"), hev("R"), hev("S")]
    assert _validate_placement_decision(_pd(("AR#0", [1]), ("AR#0", [2])), evs, {0}, _setup_gaps())


def test_validate_placement_rejects_out_of_range_event():
    evs = [hev("M", date_text="1350"), hev("R")]
    assert _validate_placement_decision(_pd(("AR#0", [9])), evs, {0}, _setup_gaps())


def test_validate_placement_passes_clean():
    evs = [hev("M", date_text="1350"), hev("R")]
    assert _validate_placement_decision(_pd(("AR#0", [1])), evs, {0}, _setup_gaps()) == []


# ===========================================================================
# Phase 4.1b Brief B: order_history orchestration (two Claude calls, no network).
# Drives order_history with a FakeClient returning canned Call 1 / Call 2 replies.
# ===========================================================================

def _date_json(*triples):  # (index, system, parts) -> a Call-1 reply
    return json.dumps({"dated": [{"index": i, "system": s, "parts": p} for i, s, p in triples]})


def _place_json(*pairs):   # (gap, [events]) -> a Call-2 reply
    return json.dumps({"placements": [{"gap": g, "events": e} for g, e in pairs]})


def test_order_history_empty_makes_no_calls():
    client = FakeClient([])
    rec = Reconciler(client=client)
    assert rec.order_history([]) == []
    assert client.call_count == 0


def test_order_history_dateless_skips_call_1():
    # nothing has a date_text -> Call 1 skipped; one Call 2 places everyone on (undated)
    events = [hev("A"), hev("B"), hev("C")]
    rec = make_reconciler([_place_json((f"{UNDATED}#0", [0, 1, 2]))])
    out = rec.order_history(events)
    assert rec.client.call_count == 1                      # only the placement call
    for i, e in enumerate(out):
        assert e.calendar_system is None and e.chronological_position == i


def test_order_history_all_dated_skips_call_2():
    # everything is dated -> nothing undated -> Call 2 skipped; one Call 1 only.
    events = [hev("A", date_text="1 AR"), hev("B", date_text="2 AR")]
    rec = make_reconciler([_date_json((0, "AR", [1]), (1, "AR", [2]))])
    out = rec.order_history(events)
    assert rec.client.call_count == 1                      # only the date call
    by_name = {e.name: e for e in out}
    assert (by_name["A"].calendar_system, by_name["A"].chronological_position) == ("AR", 0)
    assert (by_name["B"].calendar_system, by_name["B"].chronological_position) == ("AR", 1)


def test_order_history_dated_then_relative_weaves():
    # Founding(1300) + Sundering(1350) dated; Aldward relative, between them.
    events = [hev("Founding", date_text="1300"), hev("Sundering", date_text="1350"),
              hev("Aldward")]
    rec = make_reconciler([
        _date_json((0, "AR", [1300]), (1, "AR", [1350])),  # Call 1
        _place_json(("AR#1", [2])),                        # Call 2: Aldward in the middle gap
    ])
    out = rec.order_history(events)
    assert rec.client.call_count == 2
    by_name = {e.name: e for e in out}
    assert [by_name[n].chronological_position for n in ("Founding", "Aldward", "Sundering")] == [0, 1, 2]
    assert all(by_name[n].calendar_system == "AR" for n in ("Founding", "Aldward", "Sundering"))


def test_order_history_tie_then_relative_after():
    # two same-date markers (a tie) + a relative event in the gap AFTER them, all
    # the way through the orchestration (not just the engine).
    events = [hev("Founding", date_text="1300"), hev("Charter", date_text="1300"),
              hev("After")]
    rec = make_reconciler([
        _date_json((0, "AR", [1300]), (1, "AR", [1300])),  # tie
        _place_json(("AR#1", [2])),                        # After -> gap after the tie
    ])
    out = rec.order_history(events)
    by_name = {e.name: e for e in out}
    assert [by_name[n].chronological_position for n in ("Founding", "Charter", "After")] == [0, 1, 2]


def test_order_history_call_1_failure_degrades_to_relative():
    # Call 1 returns garbage 3x -> empty spine -> everyone treated as undated and placed by Call 2
    events = [hev("A", date_text="1300"), hev("B")]
    rec = make_reconciler(["nope", "nope", "nope",            # 3 failed Call-1 attempts
                           _place_json((f"{UNDATED}#0", [0, 1]))])  # Call 2 still runs
    out = rec.order_history(events)
    assert all(e.calendar_system is None for e in out)       # no dated spine survived
    assert {e.chronological_position for e in out} == {0, 1}
    assert rec.client.call_count == 4  # 3 Call-1 attempts (all failed) + 1 placement call


def test_order_history_call_2_failure_keeps_dated_spine():
    # Call 1 succeeds; Call 2 fails 3x -> dated events still stamped, relative -> Could Not Place
    events = [hev("Dated", date_text="1300"), hev("Relative")]
    rec = make_reconciler([_date_json((0, "AR", [1300])), "nope", "nope", "nope"])
    out = rec.order_history(events)
    by_name = {e.name: e for e in out}
    assert by_name["Dated"].calendar_system == "AR" and by_name["Dated"].chronological_position == 0
    assert by_name["Relative"].calendar_system is None and by_name["Relative"].chronological_position is None
    assert rec.client.call_count == 4  # 1 date call + 3 failed placement attempts


def test_order_history_guard_drops_unmoored_date_to_relative(caplog):
    # Call 1 dates the event with parts that share no number with "1300" -> guard drops it;
    # it then has no relative clue placement -> Could Not Place (not in the AR spine).
    events = [hev("Bad", date_text="1300")]
    rec = make_reconciler([_date_json((0, "AR", [9999])),   # 9999 unmoored from 1300
                           _place_json()])                  # nothing placed
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        out = rec.order_history(events)
    assert out[0].calendar_system is None and out[0].chronological_position is None
    assert rec.client.call_count == 2  # date call + placement call (Bad became relative)
    # prove the GUARD (not a plain date miss) drove the Could-Not-Place: its distinctive
    # [REVIEW] "unmoored" log line. Without this, a date-miss path yields the same (None, None).
    assert any(REVIEW_PREFIX in r.getMessage() and "unmoored" in r.getMessage()
               for r in caplog.records)


def test_order_history_retries_bad_json_then_succeeds():
    events = [hev("A", date_text="1300"), hev("B")]
    rec = make_reconciler(["garbage", _date_json((0, "AR", [1300])),   # Call 1: retry once
                           _place_json((f"{UNDATED}#0", [1]))])         # Call 2
    out = rec.order_history(events)
    by_name = {e.name: e for e in out}
    assert by_name["A"].calendar_system == "AR"
    assert by_name["B"].chronological_position == 0          # placed on the undated timeline
    assert rec.client.call_count == 3  # garbage + good date (retry) + placement


# --- the live-LLM message builders: the one orchestration seam a fake-client
# response can't validate (the canned reply's gap IDs are independent of what the
# builder actually emitted). Pin them directly so a malformed gap ID or a dropped
# [i] index prefix -- which would only fail in a live run -- is caught here. -----

def test_build_date_message_lists_only_dated_events_with_original_index():
    events = [hev("Founding", date_text="1300"), hev("Undated"),
              hev("Sundering", date_text="1350")]
    rec = Reconciler.__new__(Reconciler)            # builders need no client
    msg = rec._build_date_message(events)
    assert "[0]" in msg and "Founding" in msg and "1300" in msg     # dated, original index 0
    assert "[2]" in msg and "Sundering" in msg and "1350" in msg    # dated, original index 2
    assert "Undated" not in msg and "[1]" not in msg                # the undated event is omitted


def test_build_placement_message_gap_ids_match_lookup_and_undated_carry_index():
    events = [hev("Founding", date_text="1300"), hev("Sundering", date_text="1350"),
              hev("Aldward"), hev("Borren")]
    spines, gap_lookup = _build_spine_and_gaps([(0, "AR", [1300]), (1, "AR", [1350])])
    rec = Reconciler.__new__(Reconciler)
    msg = rec._build_placement_message(events, {0, 1}, spines)
    # EVERY gap ID the message offers must be a real key the engine handed out -- no
    # off-by-one in range(len(rank_groups)+1), no mislabeled system.
    gap_tokens = re.findall(r"\[gap (.+?)\]", msg)
    assert gap_tokens and all(g in gap_lookup for g in gap_tokens)
    assert {"AR#0", "AR#1", "AR#2", f"{UNDATED}#0"} == set(gap_tokens)
    # markers carry their dates so relative clues can anchor to them
    assert "Founding (1300)" in msg and "Sundering (1350)" in msg
    # each UNDATED event line carries its [i] index; dated events are NOT in the to-place list
    assert "[2] Aldward" in msg and "[3] Borren" in msg
    assert "[0] Founding" not in msg and "[1] Sundering" not in msg


def test_build_placement_message_marks_ties_on_one_line():
    # two same-date markers share ONE marker line (so "after Founding"/"after Charter"
    # point at the same gap) -- the prompt's tie convention depends on this rendering.
    events = [hev("Founding", date_text="1300"), hev("Charter", date_text="1300"),
              hev("After")]
    spines, gap_lookup = _build_spine_and_gaps([(0, "AR", [1300]), (1, "AR", [1300])])
    rec = Reconciler.__new__(Reconciler)
    msg = rec._build_placement_message(events, {0, 1}, spines)
    assert "Founding (1300), Charter (1300)" in msg          # one shared marker line
    gap_tokens = re.findall(r"\[gap (.+?)\]", msg)
    assert {"AR#0", "AR#1", f"{UNDATED}#0"} == set(gap_tokens)  # tie -> only AR#0, AR#1


# ===========================================================================
# Phase 4.6 Part 3: alias/name provenance (Alias objects + name_sources).
# ===========================================================================

def test_dedup_aliases_unions_orders_and_preserves_first_seen():
    out = _dedup_aliases([al("a", "f1"), al("b", "f1"), al(" a ", "f2"), al("a", "f1")])
    assert [x.text for x in out] == ["a", "b"]
    assert out[0].source_files == ["f1", "f2"]   # f1 (1st & 4th) + f2, deduped
    assert out[1].source_files == ["f1"]


def test_canonical_sources_from_name_from_alias_union_and_no_match():
    a = Location(name="Riverton", name_sources=["f1"], aliases=[al("The River", "f2")])
    b = Location(name="Riverton", name_sources=["f3"])          # same name, different file
    assert _canonical_sources([a, b], "Riverton") == ["f1", "f3"]  # union across members' names
    assert _canonical_sources([a], "The River") == ["f2"]          # matched an alias's sources
    assert _canonical_sources([a], "Nonexistent") == []            # no match -> []


def test_canonical_sources_is_case_sensitive():
    a = Location(name="Riverton", name_sources=["f1"])
    assert _canonical_sources([a], "RIVERTON") == []   # different case -> not this name's sources


def test_combine_group_carries_losing_name_provenance_into_alias():
    a = Location(name="Lake Mundi", name_sources=["f1"], aliases=[al("The Pond", "f1")])
    b = Location(name="The Great Well", name_sources=["f2"])
    merged = _combine_group([a, b], "Lake Mundi")
    assert merged.name == "Lake Mundi"
    assert merged.name_sources == ["f1"]                 # the canonical's provenance
    by_text = {x.text: x for x in merged.aliases}
    # the losing name "The Great Well" became an alias carrying ITS OWN source
    assert by_text["The Great Well"].source_files == ["f2"]
    assert by_text["The Pond"].source_files == ["f1"]


def test_combine_group_name_sources_union_when_two_members_assert_canonical():
    a = Location(name="Riverton", name_sources=["f1"], details=[det("x", "f1")])
    b = Location(name="Riverton", name_sources=["f2"], details=[det("y", "f2")])
    merged = _combine_group([a, b], "Riverton")
    assert merged.name_sources == ["f1", "f2"]           # both files attested the name


def test_entity_for_prompt_reduces_aliases_hides_provenance_keeps_key_order():
    e = Location(name="Lake Mundi", name_sources=["secret.txt"],
                 aliases=[al("The Pond", "secret.txt")],
                 details=[det("fed by a spring", "secret.txt")])
    data = _entity_for_prompt(e)
    assert data["aliases"] == ["The Pond"]               # aliases reduced to bare text
    assert data["details"] == ["fed by a spring"]
    blob = json.dumps(data)
    assert "source_files" not in blob and "name_sources" not in blob and "secret.txt" not in blob
    # key order unchanged (name_sources popped, others in place) -> prompt cache still hits
    assert list(data) == ["name", "aliases", "details", "supporting_quotes"]


# HARDCODED expected merge-prompt key order per type -- deliberately NOT computed from
# Model.model_fields. `model_dump` emits keys in model_fields order, so deriving the
# expected list from model_fields would be tautological (both sides move together and
# catch nothing). A hardcoded reference means a field REORDER on ANY type -- e.g.
# `details` before `aliases` on Character, which shifts the JSON bytes the reconciler
# serialises into the merge prompt and its sha256 cache key -- flips the test RED.
_PROMPT_KEY_ORDER = {
    Location:          ["name", "aliases", "details", "supporting_quotes"],
    Organization:      ["name", "aliases", "details", "supporting_quotes"],
    Item:              ["name", "aliases", "details", "supporting_quotes"],
    PeopleAndCultures: ["name", "aliases", "details", "supporting_quotes"],
    Character:         ["name", "aliases", "is_pc", "player_name", "details", "supporting_quotes"],
    HistoryEvent:      ["name", "aliases", "description", "scope", "date_text",
                        "calendar_system", "chronological_position", "supporting_quotes"],
}


@pytest.mark.parametrize("Model", list(_PROMPT_KEY_ORDER))
def test_entity_for_prompt_key_order_is_stable_all_types(Model):
    # _entity_for_prompt must yield a byte-identical merge prompt for every type (same
    # keys, same order, name_sources dropped, details/aliases reduced to bare text), so
    # the reconciler's prompt cache keeps hitting. The Location-only test above pins
    # Location against a hardcoded list; this extends that real guard to the other five.
    kwargs = dict(name="X", name_sources=["secret.txt"], aliases=[al("aka", "secret.txt")])
    if Model is HistoryEvent:
        kwargs.update(description="d", scope="world")
    else:
        kwargs.update(details=[det("f", "secret.txt")])
    data = _entity_for_prompt(Model(**kwargs))
    assert list(data) == _PROMPT_KEY_ORDER[Model]        # hardcoded ref -> catches a reorder
    assert "name_sources" not in json.dumps(data) and "source_files" not in json.dumps(data)
