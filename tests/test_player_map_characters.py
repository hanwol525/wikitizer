"""The structured party schema: DeclaredCharacter + last-name cross-product + pronouns
(run-1.17 §A).

Each declared character carries main_name (heading), an optional last_name (recognition-only,
cross-producted with main + aliases), aliases, and pronouns (default they/them). The old dict
forms still load. Offline, pure.
"""

import pytest

from player_map import (
    DEFAULT_PRONOUNS, build_character_lookup, build_pronoun_lookup, declared_characters,
    declared_groups, declared_groups_with_players,
)


def test_last_name_cross_product_recognition():
    cfg = [{"player": "Sam", "main_name": "Kriggy", "last_name": "Krieger",
            "aliases": ["Krigius"], "pronouns": ["he", "him"]}]
    dc = declared_characters(cfg)[0]
    assert dc.recognition == ["kriggy", "krigius", "kriggy krieger", "krigius krieger"]
    assert dc.main_name == "Kriggy" and dc.last_name == "Krieger"
    assert dc.pronouns == ["he", "him"]


def test_no_last_name_no_cross_product_and_default_pronouns():
    dc = declared_characters([{"player": "H", "main_name": "CJ"}])[0]
    assert dc.recognition == ["cj"]
    assert dc.pronouns == DEFAULT_PRONOUNS       # defaulted they/them


def test_multi_pc_one_player_two_entries():
    cfg = [{"player": "Sam", "main_name": "Kriggy"}, {"player": "Sam", "main_name": "Baldric"}]
    dcs = declared_characters(cfg)
    assert [d.main_name for d in dcs] == ["Kriggy", "Baldric"]
    assert all(d.player == "Sam" for d in dcs)


def test_lookups_include_cross_product():
    cfg = [{"player": "Sam", "main_name": "Kriggy", "last_name": "Krieger", "aliases": ["Krigius"]}]
    lk = build_character_lookup(cfg)
    assert lk["kriggy"] == "Sam" and lk["kriggy krieger"] == "Sam" and lk["krigius krieger"] == "Sam"
    pl = build_pronoun_lookup(cfg)
    assert pl["kriggy krieger"] == DEFAULT_PRONOUNS   # defaulted, reachable by full form


def test_back_compat_old_dict_forms():
    assert declared_characters({"Sam": {"main_name": "CJ", "aliases": []}})[0].main_name == "CJ"
    assert declared_characters({"Sam": ["Kriggy", "Krigius"]})[0].recognition == ["kriggy", "krigius"]
    assert declared_characters({"Sam": "Solo"})[0].main_name == "Solo"


def test_strict_shapes_raise():
    with pytest.raises(ValueError):
        declared_characters([{"player": "Sam"}])                                  # no main_name
    with pytest.raises(ValueError):
        declared_characters([{"player": "Sam", "main_name": "K", "aliases": "no"}])   # aliases not a list
    with pytest.raises(ValueError):
        declared_characters([{"player": "Sam", "main_name": "K", "pronouns": "she/her"}])  # str, not list
    with pytest.raises(ValueError):
        declared_characters(["not a dict"])                                       # bad list entry


def test_thin_wrappers_carry_cross_product():
    cfg = [{"player": "Sam", "main_name": "Kriggy", "last_name": "Krieger", "aliases": ["Krigius"]}]
    assert declared_groups(cfg) == [["kriggy", "krigius", "kriggy krieger", "krigius krieger"]]
    player, main, rec = declared_groups_with_players(cfg)[0]
    assert player == "sam" and main == "Kriggy"
    assert "kriggy krieger" in rec
