"""Tests for evals/gbf/pair.py -- the hybrid pairing spine. Fully offline (a FakeModelClient feeds
canned residual-pairing JSON; the exact-lock tests assert NO model call is made)."""

import json

from evals.common.parse import parse_wof
from evals.gbf.adjudicator import GbfAdjudicator
from evals.gbf.pair import PairMap, majority_index, norm_name, pair, real_entries


class FakeModelClient:
    """Pops one queued reply per ``complete`` call and records the prompts it saw."""

    model = "fake/deepseek"
    provider = "openrouter"
    temperature = 0.6
    thinking = False
    base_url = None

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else ""


def R(*objs):
    return json.dumps({"results": list(objs)})


_GOLD = """# The Lore

## Locations

### <a id="aldenburg"></a>Aldenburg

A town in the north.

## Characters

### <a id="kriggius-krieger"></a>Kriggius Krieger

The lost prince, also called Kriggy.

### <a id="old-maggie"></a>Old Maggie

A hedge witch.
"""


def _adj(*replies, votes=1):
    return GbfAdjudicator(FakeModelClient(replies), votes=votes)


def test_exact_anchor_lock_makes_no_model_call():
    gold = parse_wof(_GOLD)
    wof = parse_wof(_GOLD)             # identical anchors on both sides
    fake = FakeModelClient([])
    adj = GbfAdjudicator(fake, votes=1)
    pm = pair(wof, gold, adj)
    assert isinstance(pm, PairMap)
    assert all(m is not None for m in pm.matched)      # every gold entry locked
    assert fake.calls == []                            # certain matches never hit the model


def test_exact_name_lock_when_anchor_differs():
    gold = parse_wof(_GOLD)
    # Same names, DIFFERENT anchors -> anchor lock misses, normalized-name lock catches it.
    wof = parse_wof(_GOLD.replace('id="aldenburg"', 'id="aldenburg-2"')
                         .replace('id="old-maggie"', 'id="old-maggie-9"')
                         .replace('id="kriggius-krieger"', 'id="kriggius-krieger-x"'))
    fake = FakeModelClient([])
    pm = pair(wof, gold, GbfAdjudicator(fake, votes=1))
    assert all(m is not None for m in pm.matched)
    assert fake.calls == []


def test_residual_variant_pairs_via_model_not_false_na():
    gold = parse_wof(_GOLD)
    # WOF: Aldenburg locks by anchor; "Kriggy" is a name variant of gold "Kriggius Krieger"
    # (no anchor/name lock) -> must resolve via the residual model pass, not a false na.
    wof = parse_wof(
        "# The Lore\n\n## Locations\n\n### <a id=\"aldenburg\"></a>Aldenburg\n\nA town.\n\n"
        "## Characters\n\n### <a id=\"kriggy\"></a>Kriggy\n\nThe prince incognito.\n"
    )
    gold_es = real_entries(gold)
    kriggius_i = next(i for i, e in enumerate(gold_es) if e.name == "Kriggius Krieger")
    maggie_i = next(i for i, e in enumerate(gold_es) if e.name == "Old Maggie")
    # residual gold order is by index: [Kriggius (local 0), Old Maggie (local 1)]; one WOF residual (Kriggy).
    fake = FakeModelClient([R({"gold_index": 0, "wof_index": 0},
                              {"gold_index": 1, "wof_index": None})])
    pm = pair(wof, gold, GbfAdjudicator(fake, votes=1))
    assert len(fake.calls) == 1                          # the residual pass ran once
    assert pm.matched[kriggius_i] is not None and pm.matched[kriggius_i].name == "Kriggy"
    assert pm.matched[maggie_i] is None                 # genuinely unmatched -> None (na downstream)


def test_unmatched_when_no_wof_residual_is_none_without_model_call():
    gold = parse_wof(_GOLD)
    # WOF holds only Aldenburg -> Kriggius + Old Maggie are unmatched, but there is no WOF residual
    # to match against, so no model call is made and they stay None.
    wof = parse_wof("# The Lore\n\n## Locations\n\n### <a id=\"aldenburg\"></a>Aldenburg\n\nA town.\n")
    fake = FakeModelClient([])
    pm = pair(wof, gold, GbfAdjudicator(fake, votes=1))
    assert fake.calls == []
    matched_names = {e.name for e, m in zip(pm.gold_entries, pm.matched) if m is not None}
    assert matched_names == {"Aldenburg"}


def test_residual_call_error_sets_residual_failed():
    class Boom:
        model = "x"; provider = "openrouter"; temperature = 0.6; thinking = False; base_url = None
        def complete(self, system, user):
            raise RuntimeError("boom")

    gold = parse_wof(_GOLD)
    wof = parse_wof("# The Lore\n\n## Characters\n\n### <a id=\"kriggy\"></a>Kriggy\n\nIncognito.\n")
    pm = pair(wof, gold, GbfAdjudicator(Boom(), votes=1))
    assert pm.residual_failed is True
    assert set(pm.residual_indices)                      # the unmatched-residual gold indices


def test_residual_all_malformed_sets_residual_failed():
    # A residual pairing call that returns UNPARSEABLE output for every vote is an infra/parse failure,
    # NOT a considered "no match": it must set residual_failed (-> skipped downstream), never silently
    # leave entries unmatched as a false na on a complete run.
    gold = parse_wof(_GOLD)
    wof = parse_wof("# The Lore\n\n## Characters\n\n### <a id=\"kriggy\"></a>Kriggy\n\nIncognito.\n")
    fake = FakeModelClient(["not parseable prose, no json here"])
    pm = pair(wof, gold, GbfAdjudicator(fake, votes=1))
    assert fake.calls                                    # the residual pass was attempted
    assert pm.residual_failed is True
    assert all(m is None for i, m in enumerate(pm.matched) if i in set(pm.residual_indices))


def test_ambiguous_name_is_not_locked_and_goes_to_model():
    # Uniqueness gate: two gold entries share a normalized name, so the name key is ambiguous and must
    # NOT be deterministically locked -- it goes to the model residual instead (a regression to
    # "lock the first candidate" would make NO model call and mis-pair).
    dup_gold = ("# Gold\n\n## Locations\n\n"
                "### <a id=\"d1\"></a>Dup\n\nFirst dup.\n\n"
                "### <a id=\"d2\"></a>Dup\n\nSecond dup.\n")
    dup_wof = "# WOF\n\n## Locations\n\n### <a id=\"w1\"></a>Dup\n\nA dup.\n"
    gold = parse_wof(dup_gold)
    wof = parse_wof(dup_wof)
    fake = FakeModelClient([R({"gold_index": 0, "wof_index": None},
                              {"gold_index": 1, "wof_index": None})])
    pm = pair(wof, gold, GbfAdjudicator(fake, votes=1))
    assert len(fake.calls) == 1                          # ambiguous name went to the model, not locked
    assert all(m is None for m in pm.matched)            # model declined both -> none mis-locked


def test_norm_name_folds_case_ampersand_ws():
    assert norm_name("People & Cultures") == norm_name("people  and   cultures")
    assert norm_name("The Citadel") != norm_name("Citadel")   # articles NOT stripped


def test_majority_index_ties_and_out_of_range():
    # gold 0: 2 votes for 1, 1 for 2 -> majority 1
    # gold 1: tie 1 vs 2 -> None
    votes = [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 2}]
    # actually make gold1 a clean tie: 1,2,?(none)
    votes = [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: None}]
    out = majority_index(votes, 2)
    assert out[0] == 1
    assert out[1] is None                               # 1 vs 2 tie -> None


def test_wof_id_to_slug_maps_matched_pairs():
    gold = parse_wof(_GOLD)
    wof = parse_wof(_GOLD)
    pm = pair(wof, gold, GbfAdjudicator(FakeModelClient([]), votes=1))
    id_to_slug = pm.wof_id_to_slug()
    # each matched WOF entry's id() maps to its gold slug
    for w, slug in zip(pm.matched, pm.gold_slugs):
        if w is not None:
            assert id_to_slug[id(w)] == slug
