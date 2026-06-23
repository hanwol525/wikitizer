import pytest
from pydantic import ValidationError
from models.lore import Quote, Location, Character, HistoryEvent, OtherDetail


def test_quote_builds_with_all_fields():
    q = Quote(
        text="The royal family is human yeah",
        speaker="Matt",
        source_file="royalty.txt",
    )
    assert q.speaker == "Matt"


def test_location_with_quotes():
    loc = Location(
        name="Lake Mundi",
        aliases=["The Great Well", "The Pond"],
        details=["A massive central lake divided into three rings."],
        supporting_quotes=[
            Quote(text="The Great Well. The Pond. Lake Mundi.",
                  speaker="Matt", source_file="dndgroup.txt")
        ],
    )
    assert loc.name == "Lake Mundi"
    assert len(loc.aliases) == 2
    assert loc.supporting_quotes[0].speaker == "Matt"


def test_character_defaults():
    npc = Character(name="Wizard Strong")
    assert npc.is_pc is False
    assert npc.player_name is None
    assert npc.aliases == []
    assert npc.details == []
    assert npc.supporting_quotes == []


def test_character_with_aliases_round_trips():
    kriggy = Character(name="Kriggy", aliases=["Kriggy Krieger"])
    assert kriggy.aliases == ["Kriggy Krieger"]


def test_two_characters_dont_share_an_aliases_list():
    a = Character(name="Kriggy")
    b = Character(name="Tiberius")
    a.aliases.append("Kriggy Krieger")
    assert b.aliases == []


def test_player_character():
    pc = Character(name="CJ", is_pc=True, player_name="Hannah")
    assert pc.is_pc is True
    assert pc.player_name == "Hannah"


def test_history_event_unplaceable():
    event = HistoryEvent(
        description="The Aldward family rose to prominence.",
        scope="personal",
    )
    assert event.chronological_position is None


def test_two_locations_dont_share_a_list():
    a = Location(name="Eglon")
    b = Location(name="Aprus")
    a.aliases.append("capital city")
    assert b.aliases == []


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Quote(text="incomplete quote with no speaker attached", source_file="dndgroup.txt")


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        Location(name="Kriega", supporting_quotes="not a list of quotes")