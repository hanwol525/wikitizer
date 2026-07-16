"""Fix batch #3 / component B: the declared-party config (player_map.py).

Pure, offline. The loader is tolerant of a MISSING file (returns {}) but strict on a
file that parses to the wrong shape (raises) -- a silent wrong shape would quietly
disable the whole feature. build_character_lookup / declared_groups feed the extractor
and the reconciler's declared-merge floor.
"""

import json

import pytest

from player_map import (
    build_character_lookup,
    declared_groups,
    load_player_map,
    save_player_map,
)


def write(tmp_path, obj):
    p = tmp_path / "player_map.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


# --- load_player_map -------------------------------------------------------- #
def test_missing_file_returns_empty():
    assert load_player_map("does/not/exist.json") == {}


def test_loads_and_normalizes(tmp_path):
    path = write(tmp_path, {"Sam": ["Kriggy", "Krigius Krieger"], "Hannah": ["CJ"]})
    assert load_player_map(path) == {"Sam": ["Kriggy", "Krigius Krieger"], "Hannah": ["CJ"]}


def test_single_string_value_coerced_to_list(tmp_path):
    path = write(tmp_path, {"Sam": "Kriggy"})
    assert load_player_map(path) == {"Sam": ["Kriggy"]}


def test_blank_names_dropped(tmp_path):
    path = write(tmp_path, {"Sam": ["Kriggy", "  ", ""]})
    assert load_player_map(path) == {"Sam": ["Kriggy"]}


def test_non_dict_top_level_raises(tmp_path):
    path = write(tmp_path, ["Sam", "Kriggy"])
    with pytest.raises(ValueError):
        load_player_map(path)


def test_bad_value_shape_raises(tmp_path):
    path = write(tmp_path, {"Sam": {"nested": "no"}})
    with pytest.raises(ValueError):
        load_player_map(path)


def test_non_string_name_raises(tmp_path):
    path = write(tmp_path, {"Sam": ["Kriggy", 5]})
    with pytest.raises(ValueError):
        load_player_map(path)


def test_blank_player_key_raises(tmp_path):
    path = write(tmp_path, {"  ": ["Kriggy"]})
    with pytest.raises(ValueError):
        load_player_map(path)


# --- build_character_lookup ------------------------------------------------- #
def test_lookup_inverts_and_normalizes():
    lookup = build_character_lookup({"Sam": ["Kriggy", "Krigius Krieger"], "Hannah": ["CJ"]})
    assert lookup == {"kriggy": "Sam", "krigius krieger": "Sam", "cj": "Hannah"}


def test_lookup_last_wins_on_conflict():
    # Same character (mis)declared under two players -> deterministic last-wins.
    lookup = build_character_lookup({"Sam": ["Kriggy"], "Bob": ["kriggy"]})
    assert lookup["kriggy"] == "Bob"


def test_lookup_empty_map():
    assert build_character_lookup({}) == {}


# --- declared_groups -------------------------------------------------------- #
def test_declared_groups_one_ordered_list_per_player():
    groups = declared_groups({"Sam": ["Kriggy", "Ambrose Chamberlain"], "Hannah": ["CJ"]})
    assert {frozenset(g) for g in groups} == {
        frozenset({"kriggy", "ambrose chamberlain"}), frozenset({"cj"})}
    # order is preserved (first-listed name first), so the merge can prefer it as heading
    sam = [g for g in groups if "kriggy" in g][0]
    assert sam == ["kriggy", "ambrose chamberlain"]


def test_declared_groups_skips_empty():
    assert declared_groups({"Sam": [], "Hannah": ["CJ"]}) == [["cj"]]


# --- save_player_map round-trip --------------------------------------------- #
def test_save_round_trip(tmp_path):
    path = str(tmp_path / "out.json")
    save_player_map({"Sam": ["Kriggy"], "Colin": "Aerin"}, path)   # value coerced on save
    assert load_player_map(path) == {"Sam": ["Kriggy"], "Colin": ["Aerin"]}
