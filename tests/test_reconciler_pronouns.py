"""Declared-party pronoun stamping + multi-PC handling in the reconciler (run-1.17 §D/§A).

The declared floor stamps each merged PC's ground-truth pronouns and, because each declared
character is its own group, keeps two of one player's PCs distinct. A page named after a
player who owns several PCs can't be disambiguated -> left separate + [REVIEW]. Offline, pure.
"""

import logging

from agents.reconciler import _merge_declared_characters
from player_map import declared_characters
from models.lore import Character


def _rec_lists(cfg):
    dcs = declared_characters(cfg)
    return ([d.recognition for d in dcs], [d.player.strip().lower() for d in dcs],
            [d.main_name for d in dcs], [d.pronouns for d in dcs])


def ch(name, player=None):
    return Character(name=name, is_pc=True, player_name=player)


def _fold(cfg, entries):
    groups, keys, mains, prons = _rec_lists(cfg)
    return _merge_declared_characters(entries, "Character", groups, keys,
                                      main_names=mains, pronouns=prons)


# --- pronoun stamping ------------------------------------------------------- #
def test_declared_fold_stamps_pronouns_and_cross_product_folds():
    cfg = [{"player": "Sam", "main_name": "Kriggy", "last_name": "Krieger",
            "aliases": ["Krigius"], "pronouns": ["he", "him"]}]
    out = _fold(cfg, [ch("Kriggy Krieger"), ch("Krigius")])
    assert len(out) == 1
    assert out[0].name == "Kriggy"           # heading = main_name
    assert out[0].pronouns == ["he", "him"]  # ground-truth pronouns stamped


def test_single_member_declared_char_is_stamped_too():
    cfg = [{"player": "Conrad", "main_name": "Aerin", "pronouns": ["they", "them"]}]
    out = _fold(cfg, [ch("Aerin")])
    assert out[0].pronouns == ["they", "them"]


# --- multi-PC -------------------------------------------------------------- #
def test_two_pcs_of_one_player_stay_distinct():
    cfg = [{"player": "Sam", "main_name": "Kriggy"}, {"player": "Sam", "main_name": "Baldric"}]
    out = _fold(cfg, [ch("Kriggy"), ch("Baldric")])
    assert sorted(c.name for c in out) == ["Baldric", "Kriggy"]     # NOT merged into one


def test_player_named_page_with_ambiguous_player_is_left_separate(caplog):
    cfg = [{"player": "Sam", "main_name": "Kriggy"}, {"player": "Sam", "main_name": "Baldric"}]
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        out = _fold(cfg, [ch("Kriggy"), ch("Baldric"), ch("Sam")])
    assert "Sam" in [c.name for c in out]        # can't tell which PC -> stays its own page
    assert "can't disambiguate" in caplog.text


def test_single_pc_player_named_page_still_folds():
    cfg = [{"player": "Sam", "main_name": "Kriggy"}]
    out = _fold(cfg, [ch("Kriggy"), ch("Sam")])
    assert [c.name for c in out] == ["Kriggy"]   # a lone-PC player folds the 'Sam' page in
