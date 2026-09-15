"""Unified name normalization: the merge grouping key now agrees with slugify (run-1.15 §A/§B).

The merge code and the cross-link code used to normalize names differently -- slugify
stripped apostrophes + ASCII-folded accents, but the reconciler's _name_key kept them --
so "Scoia'tael" / "Scoiatael" split into several reconciler buckets (never merged) yet
collapsed to ONE crosslink anchor (dead links, duplicate pages). Both now delegate to
text_norm, so a name that will become the same anchor always shares a merge bucket.

§A: the deterministic identical-name + article floors fold apostrophe/accent/punctuation
variants. §B: the cross-type clustering key strips articles so a place-vs-org split by
"The" still clusters. Offline: floors are pure; cross-type uses a FakeClient. No API.
"""

import json

from agents.reconciler import Reconciler, _merge_identical_names, _merge_article_variants
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


# --- §A: identical-name floor folds punctuation/accent variants ------------- #
def test_apostrophe_variants_force_merge():
    out = _merge_identical_names([loc("Scoia'tael", "a base"), loc("Scoiatael", "elves")],
                                 "Location")
    assert len(out) == 1


def test_curly_and_straight_apostrophe_merge():
    out = _merge_identical_names([loc("Scoia’tael"), loc("Scoia'tael")], "Location")
    assert len(out) == 1


def test_accent_variants_force_merge():
    out = _merge_identical_names([loc("Théoden"), loc("Theoden")], "Location")
    assert len(out) == 1


def test_distinct_names_stay_split():
    # one letter apart but genuinely different -- the floor keys on exact folded name,
    # so it does NOT touch these (that's the LLM/candidate layer's call).
    out = _merge_identical_names([loc("Aerin"), loc("Gaerin")], "Location")
    assert len(out) == 2


def test_all_punctuation_names_do_not_force_merge():
    # Two un-normalizable names both fold to "" -- the per-entity sentinel keeps them
    # as separate pages instead of fusing two junk entities into one.
    out = _merge_identical_names([loc("!!!"), loc("???")], "Location")
    assert len(out) == 2


# --- §A: article floor still works through the unified key ------------------ #
def test_article_variant_merges_with_unified_key():
    out = _merge_article_variants([loc("The Citadel"), loc("Citadel")], "Location")
    assert len(out) == 1


def test_article_floor_leaves_content_word_difference_alone():
    # differ by a content word, not an article -> NOT an article variant.
    out = _merge_article_variants([loc("Krieger Family"), loc("Krieger Royal House")], "Location")
    assert len(out) == 2


# --- §A at the cross-type layer: apostrophe variants now cluster ------------- #
def test_cross_type_clusters_apostrophe_variants():
    by_type = {
        "locations": [loc("Scoia'tael", "a base")],
        "people": [peo("Scoiatael", "elven guerrillas")],
        "organizations": [], "characters": [], "items": [],
    }
    rec = Reconciler(client=FakeClient([_choices((0, ["people"]))]))
    out = rec.resolve_cross_type(by_type)
    assert len(rec.client.messages.calls) == 1     # apostrophe variant clustered across types
    assert [e.name for e in out["locations"]] == []
    assert [e.name for e in out["people"]] == ["Scoiatael"]


# --- §B: the cross-type clustering key strips a leading article -------------- #
def test_cross_type_clusters_article_variant_across_types():
    # "Citadel" (Location) vs "The Citadel" (People) -- a non-dual cross-type conflict.
    # Without the article strip these keyed apart and never clustered; now they do, so
    # the arbiter is consulted and the misfiled People copy is dropped.
    by_type = {
        "locations": [loc("Citadel", "a fortress")],
        "people": [peo("The Citadel", "its garrison folk")],
        "organizations": [], "characters": [], "items": [],
    }
    rec = Reconciler(client=FakeClient([_choices((0, ["locations"]))]))
    out = rec.resolve_cross_type(by_type)
    assert len(rec.client.messages.calls) == 1
    assert out["people"] == []
    assert [e.name for e in out["locations"]] == ["Citadel"]
