"""C1: deterministic de-conflation (pure, no LLM).

Player-name tokens are replaced with the character's canonical name BEFORE the prose
LLM runs, so the LLM never has to reason about who-is-who (which is what made it swap
characters). `build_deconflation_map` has two collision guards that keep a shared or
ambiguous player name from swapping two characters. Offline, pure.
"""

from agents.prose_agent import (
    build_deconflation_map,
    deconflate_entities,
    deconflate_events,
)
from models.lore import Alias, Character, Detail, HistoryEvent, Location, Quote, Scope


def det(text):
    return Detail(text=text, source_files=["g.txt"])


def pc(name, player_name, aliases=None):
    return Character(name=name, is_pc=True, player_name=player_name,
                     aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


# ======================================================================= #
# build_deconflation_map + collision guards
# ======================================================================= #
def test_map_is_pc_with_player_name_only():
    chars = [pc("Kriggy Krieger", "Sam"),
             Character(name="Barliman", is_pc=False, player_name=None),   # NPC -> out
             Character(name="Nameless", is_pc=True, player_name=None),    # no player -> out
             Character(name="Blank", is_pc=True, player_name="   ")]      # blank -> out
    assert build_deconflation_map(chars) == {"sam": "Kriggy Krieger"}


def test_collision_shared_player_name_is_dropped():
    # Two character records tagged with the SAME player -> ambiguous -> key dropped
    # (rather than silently mapping "sam" to whichever sorts last and swapping them).
    chars = [pc("Kriggy Krieger", "Sam"), pc("Skjoldr", "Sam")]
    assert build_deconflation_map(chars) == {}


def test_collision_player_name_equals_another_characters_name_is_dropped():
    # Player "Alex" plays character "Rogar"; another character is literally named "Alex".
    # De-conflating "alex" would rewrite that other character's name -> drop the key.
    chars = [pc("Rogar", "Alex"), Character(name="Alex", is_pc=False)]
    assert build_deconflation_map(chars) == {}


def test_player_name_equal_to_own_target_is_kept():
    # If the ONLY name matching the key is the character it maps to, that's fine.
    chars = [pc("Sam the Bold", "Sam")]
    assert build_deconflation_map(chars) == {"sam": "Sam the Bold"}


# ======================================================================= #
# deconflate_entities / deconflate_events
# ======================================================================= #
def test_deconflate_entities_rewrites_detail_text_only():
    q = Quote(text="Sam did a thing", speaker="M", source_file="g.txt")
    e = Location(name="Samsville",   # entity name is NOT rewritten
                 details=[det("Sam founded it"), det("A quiet town")],
                 supporting_quotes=[q])
    out = deconflate_entities([e], {"sam": "Kriggy"})
    assert [d.text for d in out[0].details] == ["Kriggy founded it", "A quiet town"]
    assert out[0].name == "Samsville"                       # name untouched
    assert out[0].supporting_quotes[0].text == "Sam did a thing"   # quote verbatim
    assert e.details[0].text == "Sam founded it"            # original not mutated


def test_deconflate_events_rewrites_description_and_title():
    ev = HistoryEvent(name="Sam's Departure", description="Sam left home.", scope=Scope.WORLD)
    out = deconflate_events([ev], {"sam": "Kriggy Krieger"})
    assert out[0].name == "Kriggy Krieger's Departure"      # possessive de-conflated
    assert out[0].description == "Kriggy Krieger left home."
    assert ev.name == "Sam's Departure"                     # original untouched


def test_word_boundary_does_not_fire_inside_a_longer_word():
    e = Location(name="X", details=[det("Sammy waved and Samuel bowed")])
    out = deconflate_entities([e], {"sam": "Kriggy"})
    assert out[0].details[0].text == "Sammy waved and Samuel bowed"   # unchanged


def test_deconflation_is_case_insensitive():
    e = Location(name="X", details=[det("SAM and sam and Sam")])
    out = deconflate_entities([e], {"sam": "Kriggy"})
    assert out[0].details[0].text == "Kriggy and Kriggy and Kriggy"


def test_empty_map_is_passthrough():
    e = Location(name="X", details=[det("Sam did it")])
    out = deconflate_entities([e], {})
    assert out[0].details[0].text == "Sam did it"
    ev = HistoryEvent(name="Sam's Deed", description="Sam.", scope=Scope.WORLD)
    assert deconflate_events([ev], {})[0].name == "Sam's Deed"


def test_longer_player_name_wins_over_shorter_prefix():
    # Both "sam" and "sam smith" are keys; "Sam Smith" must map as a whole, not "Kriggy Smith".
    e = Location(name="X", details=[det("Sam Smith arrived")])
    out = deconflate_entities([e], {"sam": "Kriggy", "sam smith": "Lord Kriggy"})
    assert out[0].details[0].text == "Lord Kriggy arrived"
