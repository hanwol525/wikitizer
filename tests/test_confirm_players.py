"""Fix batch #3 / component E: the interactive --confirm-players builder.

`confirm_player_map` is pure except for injected input_fn/print_fn, so we drive it with
scripted answers and no real stdin. It confirms/corrects each discovered PC's player and
returns the updated declared party in the CANONICAL LIST form (one entry per character),
merged with the existing config.

The list shape is the point of the rewrite: the old {player: [names]} return was lossy on
every round-trip -- a player's second PC became an alias of the first, and last_name /
pronouns were dropped for EVERY entry (and the loader then defaults pronouns to they/them,
which the prose pass enforces on the next run). Several tests below lock that down.
"""

from main import confirm_player_map
from models.lore import Alias, Character


def pc(name, player=None, aliases=None, pronouns=None):
    return Character(name=name, is_pc=True, player_name=player,
                     pronouns=list(pronouns or []),
                     aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


def scripted(answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def silent(*a, **k):
    pass


def entry(player, main_name, last_name=None, aliases=None, pronouns=None):
    """One canonical entry, with the defaults the loader/normalizer fills in."""
    return {"player": player, "main_name": main_name, "last_name": last_name,
            "aliases": list(aliases or []), "pronouns": list(pronouns or ["they", "them"])}


# --- behavior --------------------------------------------------------------- #
def test_enter_keeps_current_player():
    out = confirm_player_map([pc("Kriggy", player="Sam")], [], input_fn=scripted([""]),
                             print_fn=silent)
    assert out == [entry("Sam", "Kriggy", pronouns=[])]


def test_typed_name_sets_player():
    out = confirm_player_map([pc("Kriggy", player=None)], [], input_fn=scripted(["Sam"]),
                             print_fn=silent)
    assert out == [entry("Sam", "Kriggy", pronouns=[])]


def test_dash_skips_the_character():
    out = confirm_player_map([pc("Kriggy", player="Sam")], [], input_fn=scripted(["-"]),
                             print_fn=silent)
    assert out == []


def test_blank_with_no_current_is_skipped():
    out = confirm_player_map([pc("Ghost", player=None)], [], input_fn=scripted([""]),
                             print_fn=silent)
    assert out == []


def test_includes_aliases_under_the_character():
    out = confirm_player_map([pc("Kriggy", player="Sam", aliases=["Krigius Krieger"])], [],
                             input_fn=scripted([""]), print_fn=silent)
    assert out == [entry("Sam", "Kriggy", aliases=["Krigius Krieger"], pronouns=[])]


def test_a_new_character_keeps_its_extraction_time_pronouns():
    out = confirm_player_map([pc("Kriggy", player="Sam", pronouns=["he", "him"])], [],
                             input_fn=scripted([""]), print_fn=silent)
    assert out[0]["pronouns"] == ["he", "him"]


def test_merges_with_existing_config():
    existing = [entry("Sam", "Kriggy")]
    out = confirm_player_map([pc("CJ", player="Hannah")], existing,
                             input_fn=scripted([""]), print_fn=silent)
    assert out == [entry("Sam", "Kriggy"), entry("Hannah", "CJ", pronouns=[])]


# --- the round-trip losses this rewrite fixes -------------------------------- #
def test_last_name_and_pronouns_survive_the_round_trip():
    # The headline bug: confirming a character used to drop its last_name AND pronouns.
    existing = [entry("Sam", "Kriggy", last_name="Krieger", aliases=["Krigius"],
                      pronouns=["he", "him"])]
    out = confirm_player_map([pc("Kriggy", player="Sam")], existing,
                             input_fn=scripted([""]), print_fn=silent)
    assert out == existing


def test_two_pcs_for_one_player_stay_two_characters():
    # The other round-trip loss: {player: [names]} made the second PC an alias of the
    # first, so the next run merged two distinct characters into one page.
    existing = [entry("Sam", "Kriggy", pronouns=["he", "him"]),
                entry("Sam", "Ambrose", pronouns=["she", "her"])]
    out = confirm_player_map([pc("Kriggy", player="Sam"), pc("Ambrose", player="Sam")],
                             existing, input_fn=scripted(["", ""]), print_fn=silent)
    assert out == existing
    assert [e["main_name"] for e in out] == ["Kriggy", "Ambrose"]


def test_matches_an_existing_entry_by_its_last_name_form():
    # The PC was reconciled under its full "Kriggy Krieger" name; it must land on the
    # declared Kriggy entry (recognition cross-product) rather than minting a new one.
    existing = [entry("Sam", "Kriggy", last_name="Krieger", pronouns=["he", "him"])]
    out = confirm_player_map([pc("Kriggy Krieger", player="Sam")], existing,
                             input_fn=scripted([""]), print_fn=silent)
    assert len(out) == 1
    assert out[0]["main_name"] == "Kriggy"                 # declared heading preserved
    assert out[0]["pronouns"] == ["he", "him"]
    assert out[0]["aliases"] == []                         # already a recognition form


# --- reassignment ------------------------------------------------------------ #
def test_reassignment_changes_only_the_player():
    existing = [entry("Conrad", "CJ", last_name="Cloudspire", pronouns=["she", "her"]),
                entry("Conrad", "Skjoldr")]
    out = confirm_player_map([pc("CJ", player="Conrad")], existing,
                             input_fn=scripted(["Hannah"]), print_fn=silent)
    assert out == [entry("Hannah", "CJ", last_name="Cloudspire", pronouns=["she", "her"]),
                   entry("Conrad", "Skjoldr")]             # the other PC is untouched


def test_an_alias_match_reassigns_the_owning_character():
    # The config declares CJ as an alias of Skjoldr, so a discovered "CJ" IS that
    # character (recognition semantics, same as the extractor/reconciler use): the
    # answer moves the whole entry rather than minting a rival CJ under Hannah.
    existing = [entry("Conrad", "Skjoldr", aliases=["CJ"], pronouns=["he", "him"])]
    out = confirm_player_map([pc("CJ", player="Hannah")], existing,
                             input_fn=scripted([""]), print_fn=silent)
    assert out == [entry("Hannah", "Skjoldr", aliases=["CJ"], pronouns=["he", "him"])]


def test_a_duplicate_entry_collapses_and_donates_its_last_name():
    # The same character hand-declared twice under two players: confirming it keeps ONE
    # entry, and the survivor inherits the last_name it lacked (a recognition form that
    # would otherwise be silently lost).
    existing = [entry("Sam", "Kriggy"),
                entry("Conrad", "Kriggy", last_name="Krieger")]
    out = confirm_player_map([pc("Kriggy", player="Sam")], existing,
                             input_fn=scripted([""]), print_fn=silent)
    assert out == [entry("Sam", "Kriggy", last_name="Krieger")]


def test_empty_pcs_returns_existing_unchanged():
    existing = [entry("Sam", "Kriggy", last_name="Krieger", pronouns=["he", "him"])]
    out = confirm_player_map([], existing, input_fn=scripted([]), print_fn=silent)
    assert out == existing


def test_accepts_an_old_dict_shaped_config():
    # Back-compat: the old {player: [names]} config still loads (one character per player).
    out = confirm_player_map([pc("CJ", player="Hannah")], {"Sam": ["Kriggy", "Krigius"]},
                             input_fn=scripted([""]), print_fn=silent)
    assert out == [entry("Sam", "Kriggy", aliases=["Krigius"]),
                   entry("Hannah", "CJ", pronouns=[])]
