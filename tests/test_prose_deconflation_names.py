"""Player-name de-conflation in entity NAMES + the "X's character" collapse (run-1.14, #1/#2).

A synthesized label leaked a player name into a heading ("Conrad's companion" should be
CJ's); de-conflation only rewrote detail text, never the name/aliases. And the blind token
replace left a redundant possessive ("Sam's character" -> "Krigius Krieger's character").
Now de-conflation rewrites name + aliases too, and collapses "<Character>'s character" ->
"<Character>". Offline, pure.
"""

from agents.prose_agent import deconflate_entities, deconflate_events
from models.lore import Alias, Detail, Location, HistoryEvent


def loc(name, aliases=None, details=None):
    return Location(
        name=name,
        aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])],
        details=[Detail(text=d, source_files=["g.txt"]) for d in (details or [])],
    )


# --- §4: entity name + alias de-conflation ---------------------------------- #
def test_entity_name_deconflated():
    out = deconflate_entities([loc("Conrad's companion", details=["guards the prince"])],
                              {"conrad": "CJ"})
    assert out[0].name == "CJ's companion"


def test_entity_alias_deconflated():
    out = deconflate_entities([loc("The Crew", aliases=["Conrad's crew"], details=["a band"])],
                              {"conrad": "CJ"})
    assert out[0].aliases[0].text == "CJ's crew"


def test_non_player_name_untouched():
    out = deconflate_entities([loc("The Citadel", details=["a fortress"])], {"conrad": "CJ"})
    assert out[0].name == "The Citadel"


# --- §5: "X's character" collapse ------------------------------------------- #
def test_xs_character_collapses_in_detail():
    out = deconflate_entities(
        [loc("X", details=["Sam's character participated in the battle"])],
        {"sam": "Krigius Krieger"})
    assert out[0].details[0].text == "Krigius Krieger participated in the battle"


def test_xs_pc_collapses():
    out = deconflate_entities([loc("X", details=["Sam's PC ran away"])], {"sam": "Krigius Krieger"})
    assert out[0].details[0].text == "Krigius Krieger ran away"


def test_event_name_and_description_deconflated_and_collapsed():
    ev = HistoryEvent(name="Sam's Flight", description="Sam's character ran away.", scope="world")
    out = deconflate_events([ev], {"sam": "Krigius Krieger"})
    assert out[0].name == "Krigius Krieger's Flight"          # possessive title kept (not "'s character")
    assert out[0].description == "Krigius Krieger ran away."  # redundant "'s character" collapsed


def test_plural_characters_not_collapsed():
    # "characters" (plural) is real prose, not the redundant artifact -> left alone.
    out = deconflate_entities(
        [loc("X", details=["Krigius Krieger's characters were many"])],
        {"sam": "Krigius Krieger"})
    assert "Krigius Krieger's characters" in out[0].details[0].text
