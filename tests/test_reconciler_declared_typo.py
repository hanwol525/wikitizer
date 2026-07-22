"""Declared-name typo floor in the reconciler (run-1.13, note #6).

The reconciler kept "Kriggius" separate from the declared "Krigius" -- an obvious
one-letter typo -- because the declared-merge floor matched declared names EXACTLY and
all fuzzy merging was delegated to the LLM (which declined). This adds a deterministic
floor: a name one edit from a DECLARED name/alias (both >= DECLARED_TYPO_MIN_LEN chars)
folds into that declared character, kept as an alias and logged [REVIEW]. Anchored only to
ground-truth declared names, so it can't run wild. Offline, no API.
"""

from agents.reconciler import _merge_declared_characters, REVIEW_PREFIX
from player_map import declared_groups_with_players
from models.lore import Alias, Character


def ch(name, is_pc=True, player=None, aliases=None):
    return Character(name=name, is_pc=is_pc, player_name=player,
                     aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


def _gk(player_map):
    pairs = declared_groups_with_players(player_map)
    return [g for _, g in pairs], [p for p, _ in pairs]


# --- a one-edit typo of a declared name folds ------------------------------- #
def test_one_edit_typo_of_declared_name_folds():
    groups, keys = _gk({"Sam": ["Krigius Krieger", "Krigius", "Kriggy"]})
    krig = ch("Krigius Krieger", player="Sam")
    typo = ch("Kriggius")                                # one 'g' inserted into "Krigius"
    out = _merge_declared_characters([krig, typo], "Character", groups, keys)
    assert len(out) == 1
    assert out[0].name == "Krigius Krieger"
    alias_texts = {a.text.strip().lower() for a in out[0].aliases}
    assert "kriggius" in alias_texts                    # the typo is KEPT as a spelling-variant alias


def test_typo_fold_logs_review(caplog):
    groups, keys = _gk({"Sam": ["Krigius Krieger", "Krigius"]})
    with caplog.at_level("WARNING"):
        _merge_declared_characters(
            [ch("Krigius Krieger", player="Sam"), ch("Kriggius")], "Character", groups, keys)
    assert any(REVIEW_PREFIX in r.message and "misspelling" in r.message for r in caplog.records)


# --- guards: short names and distance-2 do NOT fuzzy-fold ------------------- #
def test_short_name_typo_is_not_folded():
    # Both names < DECLARED_TYPO_MIN_LEN (=4): a 1-edit gap in a short name is a
    # different name, not a typo (CJ/DJ energy). Must NOT fold.
    groups, keys = _gk({"Conrad": ["Ned"]})
    out = _merge_declared_characters(
        [ch("Ned", player="Conrad"), ch("Ted")], "Character", groups, keys)
    assert len(out) == 2


def test_distance_two_is_not_folded():
    groups, keys = _gk({"Sam": ["Maltraav"]})
    out = _merge_declared_characters(
        [ch("Maltraav", player="Sam"), ch("Maltroov")], "Character", groups, keys)
    assert len(out) == 2                                 # edit distance 2 -> only distance 1 folds


def test_unrelated_name_is_not_folded():
    groups, keys = _gk({"Sam": ["Krigius"]})
    out = _merge_declared_characters(
        [ch("Krigius", player="Sam"), ch("Skjoldr")], "Character", groups, keys)
    assert len(out) == 2
