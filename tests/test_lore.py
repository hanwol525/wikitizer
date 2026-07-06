import pytest
from pydantic import ValidationError
from models.lore import (
    Quote, Detail, Location, Character, HistoryEvent, Scope,
    Organization, Item, PeopleAndCultures,
)


def det(text, *source_files):
    return Detail(text=text, source_files=list(source_files))


# --- Detail: the fact-grain provenance twin of Quote -------------------------

def test_detail_defaults_to_no_sources():
    assert Detail(text="x").source_files == []


def test_detail_round_trips_through_json_preserving_source_order():
    d = Detail(text="x", source_files=["a.txt", "b.txt"])
    again = Detail.model_validate_json(d.model_dump_json())
    assert again == d
    assert again.source_files == ["a.txt", "b.txt"]   # order preserved


def test_two_details_dont_share_a_source_files_list():
    # Same default_factory guard the other models get: independent list objects.
    a = Detail(text="x")
    b = Detail(text="x")
    a.source_files.append("a.txt")
    assert b.source_files == []


def test_entity_details_reject_bare_strings():
    # The retype bites: `details` must be Detail objects now, not plain strings.
    with pytest.raises(ValidationError):
        Location(name="Lake Mundi", details=["a bare string"])


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
        details=[det("A massive central lake divided into three rings.")],
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
        name="Rise of the Aldward Family",
        description="The Aldward family rose to prominence.",
        scope="personal",
    )
    assert event.chronological_position is None


def test_history_event_with_name_and_aliases_round_trips():
    event = HistoryEvent(
        name="The Maltraav-Kriega War",
        aliases=["the Border War"],
        description="A brutal war between Maltraav and Kriega.",
        scope="regional",
    )
    assert event.name == "The Maltraav-Kriega War"
    assert event.aliases == ["the Border War"]


def test_history_event_aliases_default_empty():
    event = HistoryEvent(name="x", description="y", scope="world")
    assert event.aliases == []


def test_history_event_scope_enum_rejects_off_menu_value():
    with pytest.raises(ValidationError):
        HistoryEvent(name="x", description="y", scope="banana")


def test_history_event_scope_string_coerces_to_enum_member():
    event = HistoryEvent(name="x", description="y", scope="personal")
    assert event.scope is Scope.PERSONAL


def test_history_event_date_text_defaults_none():
    event = HistoryEvent(name="x", description="y", scope="world")
    assert event.date_text is None


def test_history_event_date_text_round_trips():
    event = HistoryEvent(name="The Sundering", description="A cataclysm.",
                         scope="world", date_text="342 AR")
    assert event.date_text == "342 AR"


def test_history_event_calendar_system_defaults_none():
    event = HistoryEvent(name="x", description="y", scope="world")
    assert event.calendar_system is None


def test_history_event_calendar_system_round_trips():
    event = HistoryEvent(name="The Sundering", description="A cataclysm.",
                         scope="world", calendar_system="AR years")
    assert event.calendar_system == "AR years"


def test_history_event_requires_name():
    # name is now a required field (no default) -> omitting it must raise, so a
    # regression that gave it a default wouldn't slip past.
    with pytest.raises(ValidationError):
        HistoryEvent(description="y", scope="world")


def test_scope_member_equals_its_text():
    # The (str, Enum) mix-in: a member compares equal to its string value, which
    # is what lets renderer code keep doing `event.scope == "world"`.
    assert Scope.WORLD == "world"


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


# --- typed lore categories (Organization / Item / PeopleAndCultures) ---------
# All three are exact Location-shaped clones, so a parametrized sweep keeps these
# DRY rather than nine near-identical functions.

@pytest.mark.parametrize("Model", [Organization, Item, PeopleAndCultures])
def test_typed_lore_defaults(Model):
    obj = Model(name="Test")
    assert obj.aliases == [] and obj.details == [] and obj.supporting_quotes == []


@pytest.mark.parametrize("Model", [Organization, Item, PeopleAndCultures])
def test_typed_lore_round_trips(Model):
    obj = Model(name="Test", aliases=["alt"], details=[det("a fact")])
    assert obj.name == "Test" and obj.aliases == ["alt"]


@pytest.mark.parametrize("Model", [Organization, Item, PeopleAndCultures])
def test_typed_lore_no_shared_aliases_list(Model):
    a, b = Model(name="A"), Model(name="B")
    a.aliases.append("x")
    assert b.aliases == []


@pytest.mark.parametrize("Model", [Organization, Item, PeopleAndCultures])
def test_typed_lore_requires_name(Model):
    with pytest.raises(ValidationError):
        Model()   # name has no default -> omitting it must raise