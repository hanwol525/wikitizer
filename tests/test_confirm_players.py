"""Fix batch #3 / component E: the interactive --confirm-players builder.

`confirm_player_map` is pure except for injected input_fn/print_fn, so we drive it with
scripted answers and no real stdin. It confirms/corrects each discovered PC's player and
returns an updated {player: [names]} map merged with the existing config.
"""

from main import confirm_player_map
from models.lore import Alias, Character


def pc(name, player=None, aliases=None):
    return Character(name=name, is_pc=True, player_name=player,
                     aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


def scripted(answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def silent(*a, **k):
    pass


# --- behavior --------------------------------------------------------------- #
def test_enter_keeps_current_player():
    out = confirm_player_map([pc("Kriggy", player="Sam")], {}, input_fn=scripted([""]), print_fn=silent)
    assert out == {"Sam": ["Kriggy"]}


def test_typed_name_sets_player():
    out = confirm_player_map([pc("Kriggy", player=None)], {}, input_fn=scripted(["Sam"]), print_fn=silent)
    assert out == {"Sam": ["Kriggy"]}


def test_dash_skips_the_character():
    out = confirm_player_map([pc("Kriggy", player="Sam")], {}, input_fn=scripted(["-"]), print_fn=silent)
    assert out == {}


def test_blank_with_no_current_is_skipped():
    out = confirm_player_map([pc("Ghost", player=None)], {}, input_fn=scripted([""]), print_fn=silent)
    assert out == {}


def test_includes_aliases_under_the_player():
    out = confirm_player_map([pc("Kriggy", player="Sam", aliases=["Krigius Krieger"])], {},
                             input_fn=scripted([""]), print_fn=silent)
    assert out == {"Sam": ["Kriggy", "Krigius Krieger"]}


def test_merges_with_existing_config():
    out = confirm_player_map([pc("CJ", player="Hannah")], {"Sam": ["Kriggy"]},
                             input_fn=scripted([""]), print_fn=silent)
    assert out == {"Sam": ["Kriggy"], "Hannah": ["CJ"]}


def test_reassignment_moves_name_off_old_player():
    # Existing config wrongly has CJ under Conrad; the user corrects it to Hannah.
    out = confirm_player_map([pc("CJ", player="Conrad")], {"Conrad": ["CJ", "Skjoldr"]},
                             input_fn=scripted(["Hannah"]), print_fn=silent)
    assert out == {"Conrad": ["Skjoldr"], "Hannah": ["CJ"]}


def test_empty_pcs_returns_existing_unchanged():
    existing = {"Sam": ["Kriggy"]}
    out = confirm_player_map([], existing, input_fn=scripted([]), print_fn=silent)
    assert out == existing
