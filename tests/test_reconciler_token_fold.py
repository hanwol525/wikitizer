"""The compound-name token fold (run-1.14, note #4/#6).

'Kriggy Krieger' never merged into 'Krigius Krieger' (same person): exact match needs a
literal declared name/alias, and the typo floor needs edit-distance-1. The token fold
catches it -- a name that CONTAINS a declared bare alias (>=4 chars) as a whole word folds
into that declared character. Anchored to declared aliases only, so a shared surname
('Krieger') or a short token can't trigger it. Offline, no API.
"""

from agents.reconciler import _merge_declared_characters
from player_map import declared_groups_with_players
from models.lore import Character


def ch(name, is_pc=True, player=None):
    return Character(name=name, is_pc=is_pc, player_name=player)


def _gk(pm):
    pairs = declared_groups_with_players(pm)
    return [g for _, _, g in pairs], [p for p, _, _ in pairs]


def test_compound_name_folds_via_declared_bare_alias():
    groups, keys = _gk({"Sam": ["Kriggy", "Krigius Krieger"]})
    out = _merge_declared_characters(
        [ch("Krigius Krieger", player="Sam"), ch("Kriggy Krieger")], "Character", groups, keys)
    assert len(out) == 1
    assert out[0].name == "Krigius Krieger"
    assert "kriggy krieger" in {a.text.strip().lower() for a in out[0].aliases}


def test_shared_surname_alone_does_not_fold():
    # 'Kassius Krieger' shares only the surname, which is NOT a bare declared alias.
    groups, keys = _gk({"Sam": ["Kriggy", "Krigius Krieger"]})
    out = _merge_declared_characters(
        [ch("Krigius Krieger", player="Sam"), ch("Kassius Krieger")], "Character", groups, keys)
    assert len(out) == 2


def test_multiword_declared_name_contributes_no_bare_token():
    # A multi-word declared name ("krigius krieger") gives NO bare token, so a shared
    # surname can't fold via it.
    groups, keys = _gk({"Sam": ["Krigius Krieger"]})
    out = _merge_declared_characters(
        [ch("Krigius Krieger", player="Sam"), ch("Kassius Krieger")], "Character", groups, keys)
    assert len(out) == 2


def test_short_declared_token_does_not_fold():
    # A declared bare alias < 4 chars ('CJ') can't be a token-fold key.
    groups, keys = _gk({"Conrad": ["CJ", "Clara Jane"]})
    out = _merge_declared_characters(
        [ch("CJ", player="Conrad"), ch("CJ Rutherford")], "Character", groups, keys)
    assert len(out) == 2


def test_token_fold_skipped_when_flagged_distinct():
    groups, keys = _gk({"Sam": ["Kriggy", "Krigius Krieger"]})
    out = _merge_declared_characters(
        [ch("Krigius Krieger", player="Sam"), ch("Kriggy Krieger")], "Character", groups, keys,
        possible_dup_names={"kriggy krieger"})
    assert len(out) == 2                                 # distinctness gate covers the token fold too
