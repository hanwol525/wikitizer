"""Fix batch #3 / component D: the reconciler's declared-merge floor.

The user's declaration that Kriggy/Krigius/Ambrose are ONE character (all grouped under
one player in the player_map) is ground truth -- honored deterministically even if the
LLM split them. Runs for Characters only. Offline, no API.
"""

import json

from agents.reconciler import (
    Reconciler,
    _merge_declared_characters,
    declared_groups,
)
from models.lore import Alias, Character, Location


# --- fake client ------------------------------------------------------------ #
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


def ch(name, player="Sam", aliases=None):
    return Character(name=name, is_pc=True, player_name=player,
                     aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


GROUPS = declared_groups({"Sam": ["Kriggy", "Krigius Krieger", "Ambrose Chamberlain"]})
NO_MERGE = json.dumps({"merges": [], "possible_duplicates": []})


# --- _merge_declared_characters (unit) -------------------------------------- #
def test_merges_declared_group_by_name():
    out = _merge_declared_characters([ch("Kriggy"), ch("Ambrose Chamberlain"), ch("Tiberius", player=None)],
                                     "Character", GROUPS)
    assert len(out) == 2                                   # the two Sam records merged; Tiberius alone
    merged = [c for c in out if c.name == "Kriggy"][0]
    assert "ambrose chamberlain" in {a.text.lower() for a in merged.aliases}


def test_merges_by_alias_membership():
    # A record headed "The Disgraced One" whose ALIAS is a declared name still joins.
    out = _merge_declared_characters([ch("Kriggy"), ch("The Disgraced One", aliases=["Krigius Krieger"])],
                                     "Character", GROUPS)
    assert len(out) == 1


def test_canonical_is_first_declared_name_present():
    # "Krigius Krieger" and "Ambrose Chamberlain" present; "Kriggy" (first declared) is NOT,
    # so canonical falls to the earliest declared name that IS a member -> "Krigius Krieger".
    out = _merge_declared_characters([ch("Ambrose Chamberlain"), ch("Krigius Krieger")],
                                     "Character", GROUPS)
    assert out[0].name == "Krigius Krieger"


def test_skips_non_character_types():
    locs = [Location(name="Kriggy"), Location(name="Ambrose Chamberlain")]
    out = _merge_declared_characters(locs, "Location", GROUPS)
    assert len(out) == 2                                   # never merges non-Characters


def test_no_groups_is_noop():
    entries = [ch("Kriggy"), ch("Ambrose Chamberlain")]
    assert len(_merge_declared_characters(entries, "Character", [])) == 2


def test_preserves_order_and_leaves_unlisted_alone():
    out = _merge_declared_characters([ch("Tiberius", player=None), ch("Kriggy"), ch("Krigius Krieger")],
                                     "Character", GROUPS)
    assert [c.name for c in out] == ["Tiberius", "Kriggy"]  # Tiberius first (unlisted), merged Sam second


# --- end-to-end through reconcile() ----------------------------------------- #
def test_reconcile_declared_merge_when_llm_omits_it():
    # The LLM returns an empty decision (silent omission); the declared floor still
    # merges the fragments the user grouped under Sam.
    rec = Reconciler(client=FakeClient([NO_MERGE]),
                     player_map={"Sam": ["Kriggy", "Krigius Krieger", "Ambrose Chamberlain"]})
    out = rec.reconcile([ch("Kriggy"), ch("Krigius Krieger"), ch("Ambrose Chamberlain")])
    assert len(out) == 1
    assert out[0].player_name == "Sam"


def test_reconcile_without_player_map_leaves_them_split():
    rec = Reconciler(client=FakeClient([NO_MERGE]))          # no declared party
    out = rec.reconcile([ch("Kriggy"), ch("Ambrose Chamberlain")])
    assert len(out) == 2
