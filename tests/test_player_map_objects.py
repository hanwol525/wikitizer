"""Player objects: main_name + aliases (run-1.16 §B).

Each player's declared value becomes an object {"main_name", "aliases"}: recognition still
matches on main_name UNION aliases, but the heading/prose DEFAULT to main_name -- even when
main_name only ever surfaced as an alias in the chat. The old flat-list form still parses
(first element = main_name). Offline, no API except the FakeClient boundary.
"""

import json

import pytest

from player_map import (
    declared_groups, declared_groups_with_players, load_player_map, save_player_map,
)
from agents.reconciler import Reconciler
from models.lore import Character


# --- fake client ------------------------------------------------------------ #
class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class _Messages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self._responses[0])


class FakeClient:
    def __init__(self, responses): self.messages = _Messages(responses)


_NO_MERGES = json.dumps({"merges": [], "possible_duplicates": []})


# --- parsing: the three accepted shapes ------------------------------------- #
def test_object_shape_parses():
    pm = {"Sam": {"main_name": "Ambrose Chamberlain", "aliases": ["Kriggy", "Krigius Krieger"]}}
    assert declared_groups_with_players(pm) == [
        ("sam", "Ambrose Chamberlain", ["ambrose chamberlain", "kriggy", "krigius krieger"])]


def test_old_list_back_compat_first_is_main_name():
    pm = {"Sam": ["Kriggy", "Krigius Krieger"]}
    player, main, names = declared_groups_with_players(pm)[0]
    assert main == "Kriggy" and names == ["kriggy", "krigius krieger"]


def test_bare_string_is_main_name():
    assert declared_groups_with_players({"Hannah": "CJ"}) == [("hannah", "CJ", ["cj"])]


def test_object_missing_main_name_raises():
    with pytest.raises(ValueError):
        declared_groups({"Sam": {"aliases": ["Kriggy"]}})


def test_object_non_list_aliases_raises():
    with pytest.raises(ValueError):
        declared_groups({"Sam": {"main_name": "Kriggy", "aliases": "Krigius"}})


# --- save writes the canonical LIST form; round-trips through load ----------- #
def test_save_writes_list_form(tmp_path):
    path = str(tmp_path / "pm.json")
    save_player_map({"Sam": ["Ambrose Chamberlain", "Kriggy"]}, path)   # old dict form in
    on_disk = json.loads((tmp_path / "pm.json").read_text(encoding="utf-8"))
    assert on_disk == [{"player": "Sam", "main_name": "Ambrose Chamberlain",
                        "last_name": None, "aliases": ["Kriggy"], "pronouns": []}]
    # load returns the canonical entry-list (pronouns default to they/them at parse time)
    assert load_player_map(path) == [
        {"player": "Sam", "main_name": "Ambrose Chamberlain",
         "last_name": None, "aliases": ["Kriggy"], "pronouns": []}]


# --- the payoff: main_name is the heading even when it never surfaced -------- #
def test_main_name_is_heading_even_when_only_aliases_surfaced():
    # Neither extracted entry is literally named "Ambrose Chamberlain" -- it only exists as
    # the declared main_name. The declared-party merge must still title the page with it.
    pm = {"Sam": {"main_name": "Ambrose Chamberlain", "aliases": ["Kriggy", "Krigius Krieger"]}}
    entries = [Character(name="Kriggy", is_pc=True, player_name="Sam"),
               Character(name="Krigius Krieger", is_pc=True, player_name="Sam")]
    rec = Reconciler(client=FakeClient([_NO_MERGES]), player_map=pm)
    out = rec.reconcile(entries)
    assert len(out) == 1
    assert out[0].name == "Ambrose Chamberlain"        # main_name wins the heading
    aliases = {a.text.strip().lower() for a in out[0].aliases}
    assert "kriggy" in aliases and "krigius krieger" in aliases
