"""Tests for agents/characters_extractor.py (Phase 3.4: the CharactersExtractor).

Every test injects a fake Anthropic client, so the suite makes ZERO real API
calls and needs no API key. The fake-client classes are copied from the other
agent test files to keep this file self-contained (the established per-phase
pattern).

The truly inherited machinery -- batching, the verbatim-quote drop, the bool /
out-of-range source_id guards (all in BaseExtractor._resolve_quote), empty input,
default model/max_tokens -- is already covered by test_base_extractor.py /
test_locations_extractor.py, so this file focuses on what is NEW to characters:
the roster threading, the player_name roster check, the name-collision flag, and
the is_pc fallback. One exception: the detail/quote loop is re-implemented per
subclass (the _build_entry template-method seam, NOT inherited), so its guards
get one consolidated character-specific test here.

Heads-up: with verify_quotes=True by default, any detail expected to SURVIVE must
quote text that actually appears in its source message, or it is (correctly)
dropped -- so the fake message content is lined up with the fake response quotes.
"""

import json
import logging
from datetime import datetime

from agents.characters_extractor import CharactersExtractor
from models.lore import Character, Quote
from models.message import Message


# --- fake client (copied from the other agent test files) -------------------

class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, blocks):
        self.content = blocks


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        blocks = [FakeTextBlock(spec)] if isinstance(spec, str) else spec
        return FakeResponse(blocks)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)

    @property
    def call_count(self):
        return len(self.messages.calls)


# --- helpers ----------------------------------------------------------------

def make_message(content, sender="Matt", source_file="dndgroup.txt"):
    return Message(
        sender=sender,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        content=content,
        source_file=source_file,
    )


def make_agent(responses, player_names, **kwargs):
    return CharactersExtractor(
        client=FakeClient(responses), player_names=player_names, **kwargs
    )


# --- happy path / smoke -----------------------------------------------------

def test_happy_path_pc_and_npc():
    messages = [
        make_message("Sam plays Kriggy right?"),
        make_message("yeah Kriggy's the disgraced son of a noble house, Battlemaster with 18 AC"),
        make_message("Emperor Tiberius rules the Krieger Imperium with an iron fist"),
        make_message("Kriggy is Tiberius's younger brother"),
    ]
    response = json.dumps([
        {
            "name": "Kriggy",
            "aliases": [],
            "is_pc": True,
            "player_name": "Sam",
            "details": [
                {"detail": "The disgraced son of a noble house",
                 "quote": "Kriggy's the disgraced son of a noble house", "source_id": 1},
                {"detail": "Younger brother of Emperor Tiberius",
                 "quote": "Kriggy is Tiberius's younger brother", "source_id": 3},
            ],
        },
        {
            "name": "Emperor Tiberius",
            "aliases": [],
            "is_pc": False,
            "player_name": None,
            "details": [
                {"detail": "Rules the Krieger Imperium with an iron fist",
                 "quote": "Emperor Tiberius rules the Krieger Imperium with an iron fist",
                 "source_id": 2},
            ],
        },
    ])
    agent = make_agent([response], player_names={"Sam"})

    result = agent.extract(messages)

    assert [c.name for c in result] == ["Kriggy", "Emperor Tiberius"]
    assert all(isinstance(c, Character) for c in result)

    kriggy, tiberius = result
    assert kriggy.is_pc is True
    assert kriggy.player_name == "Sam"
    assert [a.text for a in kriggy.aliases] == []
    assert [d.text for d in kriggy.details] == [
        "The disgraced son of a noble house",
        "Younger brother of Emperor Tiberius",
    ]
    # each fact is tagged with the file of the message it cited (details[0] cited
    # message 1, details[1] cited message 3) -- dormant provenance for exclude-sources
    assert kriggy.details[0].source_files == [messages[1].source_file]
    assert kriggy.details[1].source_files == [messages[3].source_file]
    # the PC's supporting quote carries speaker/source_file from its message
    q0 = kriggy.supporting_quotes[0]
    assert isinstance(q0, Quote)
    assert q0.text == "Kriggy's the disgraced son of a noble house"
    assert q0.speaker == "Matt"           # from the Message, not Claude
    assert q0.source_file == "dndgroup.txt"

    assert tiberius.is_pc is False
    assert tiberius.player_name is None


# --- aliases ----------------------------------------------------------------

def test_aliases_are_captured():
    messages = [make_message("Kriggy is short for Kriggy Krieger")]
    response = json.dumps([
        {"name": "Kriggy", "aliases": ["Kriggy Krieger"], "is_pc": False,
         "player_name": None, "details": []},
    ])
    agent = make_agent([response], player_names={"Sam"})

    result = agent.extract(messages)

    assert len(result) == 1
    assert [a.text for a in result[0].aliases] == ["Kriggy Krieger"]


def test_aliases_are_defended():
    messages = [make_message("whatever")]
    # non-string alias entries filtered out...
    response_filtered = json.dumps([
        {"name": "Kriggy", "aliases": ["Kriggy Krieger", 5, None], "is_pc": False,
         "player_name": None, "details": []},
    ])
    agent = make_agent([response_filtered], player_names={"Sam"})
    assert [a.text for a in agent.extract(messages)[0].aliases] == ["Kriggy Krieger"]

    # ...and a non-list aliases becomes [] rather than crashing.
    response_nonlist = json.dumps([
        {"name": "Kriggy", "aliases": "notalist", "is_pc": False,
         "player_name": None, "details": []},
    ])
    agent = make_agent([response_nonlist], player_names={"Sam"})
    assert [a.text for a in agent.extract(messages)[0].aliases] == []


# --- player_name roster check -----------------------------------------------

def test_off_roster_player_name_nulled_with_warning(caplog):
    messages = [make_message("whatever")]
    response = json.dumps([
        {"name": "Bob", "aliases": [], "is_pc": True,
         "player_name": "Tiberius", "details": []},  # Tiberius is not a real person
    ])
    agent = make_agent([response], player_names={"Sam", "Hannah"})

    with caplog.at_level(logging.WARNING, logger="agents.characters_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].player_name is None      # nulled
    assert result[0].is_pc is True            # is_pc left as Claude set it
    assert any(
        "not a known real person" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_roster_check_is_case_insensitive_and_does_not_rewrite():
    messages = [make_message("whatever")]
    response = json.dumps([
        {"name": "Kriggy", "aliases": [], "is_pc": True,
         "player_name": "sam", "details": []},  # lowercase form of roster "Sam"
    ])
    agent = make_agent([response], player_names={"Sam"})

    result = agent.extract(messages)

    assert result[0].player_name == "sam"     # kept, and NOT canonicalized to "Sam"


def test_empty_roster_disables_player_name_check():
    messages = [make_message("whatever")]
    response = json.dumps([
        {"name": "Kriggy", "aliases": [], "is_pc": True,
         "player_name": "Sam", "details": []},
    ])
    agent = make_agent([response], player_names=set())

    result = agent.extract(messages)

    assert result[0].player_name == "Sam"     # no roster -> nothing to reject against


def test_whitespace_only_roster_member_behaves_like_empty_roster():
    # A roster whose only member is whitespace must behave like an empty roster
    # (check disabled) -- not build a truthy {""} lookup that nulls every real
    # player_name. The blank also must not leak into the prompt roster.
    messages = [make_message("whatever")]
    response = json.dumps([
        {"name": "Kriggy", "aliases": [], "is_pc": True, "player_name": "Sam", "details": []},
    ])
    agent = make_agent([response], player_names={"   "})

    result = agent.extract(messages)

    assert result[0].player_name == "Sam"          # NOT nulled
    assert "__PLAYER_ROSTER__" not in agent.system_prompt   # token still filled (with empty)


# --- name-collision flag ----------------------------------------------------

def test_name_collision_flagged_but_kept(caplog):
    messages = [make_message("lol Ryan named his character Hannah as a joke")]
    response = json.dumps([
        {"name": "Hannah", "aliases": [], "is_pc": True,
         "player_name": "Ryan", "details": []},
    ])
    agent = make_agent([response], player_names={"Sam", "Hannah", "Ryan"})

    with caplog.at_level(logging.WARNING, logger="agents.characters_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].name == "Hannah"         # KEPT despite name-collision
    assert result[0].is_pc is True            # left as Claude set it
    assert result[0].player_name == "Ryan"    # Ryan is a real person -> valid
    assert any(
        "matches a real person" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
    # the OFF-roster nulling warning must NOT fire -- Ryan is a valid real person
    assert not any(
        "not a known real person" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_name_collision_when_character_shares_its_own_players_name(caplog):
    # The load-bearing collision case the flag exists for: a character whose name
    # AND player_name are both the same in-roster real person. The player_name
    # roster check PASSES (Sam is real, so it's not nulled), so the name-collision
    # flag is the ONLY thing that surfaces this possible player/character mix-up.
    messages = [make_message("whatever")]
    response = json.dumps([
        {"name": "Sam", "aliases": [], "is_pc": True, "player_name": "Sam", "details": []},
    ])
    agent = make_agent([response], player_names={"Sam"})

    with caplog.at_level(logging.WARNING, logger="agents.characters_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].name == "Sam"
    assert result[0].player_name == "Sam"     # passes the roster check, NOT nulled
    assert any(
        "matches a real person" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


# --- is_pc / player_name fallbacks ------------------------------------------

def test_is_pc_falls_back_to_false_when_missing_or_non_bool():
    messages = [make_message("whatever")]
    # omitted is_pc
    omitted = json.dumps([
        {"name": "Kriggy", "aliases": [], "player_name": None, "details": []},
    ])
    agent = make_agent([omitted], player_names={"Sam"})
    assert agent.extract(messages)[0].is_pc is False

    # non-bool is_pc ("yes")
    junk = json.dumps([
        {"name": "Kriggy", "aliases": [], "is_pc": "yes", "player_name": None, "details": []},
    ])
    agent = make_agent([junk], player_names={"Sam"})
    assert agent.extract(messages)[0].is_pc is False


def test_missing_or_blank_player_name_becomes_none():
    messages = [make_message("whatever")]
    response = json.dumps([
        {"name": "Kriggy", "aliases": [], "is_pc": False, "details": []},  # player_name omitted
    ])
    agent = make_agent([response], player_names={"Sam"})
    assert agent.extract(messages)[0].player_name is None


# --- detail/quote loop guards (re-implemented per subclass, not inherited) ---

def test_detail_loop_guards_drop_malformed_details_without_crashing(caplog):
    # The detail/quote loop is re-implemented in CharactersExtractor._build_entry
    # (it's the template-method seam, NOT inherited), so its guards need a
    # character-specific test even though Locations covers the equivalent logic.
    messages = [make_message("Kriggy is the disgraced son of a noble house")]

    # a non-list 'details' is coerced to [] -> character kept with no details.
    nonlist = json.dumps([
        {"name": "Tiberius", "aliases": [], "is_pc": False, "player_name": None, "details": 5},
    ])
    agent = make_agent([nonlist], player_names={"Sam"})
    with caplog.at_level(logging.WARNING, logger="agents.characters_extractor"):
        res = agent.extract(messages)
    assert res[0].name == "Tiberius"
    assert [d.text for d in res[0].details] == []
    assert any("non-list" in r.getMessage() for r in caplog.records)

    # malformed details mixed with a valid one: a non-dict detail and a non-str
    # 'detail' are skipped; only the well-formed (verbatim) detail survives.
    mixed = json.dumps([
        {"name": "Kriggy", "aliases": [], "is_pc": False, "player_name": None, "details": [
            "not a dict",
            {"detail": 123, "quote": "Kriggy is the disgraced son of a noble house", "source_id": 0},
            {"detail": "The disgraced son of a noble house",
             "quote": "Kriggy is the disgraced son of a noble house", "source_id": 0},
        ]},
    ])
    agent = make_agent([mixed], player_names={"Sam"})
    res = agent.extract(messages)
    assert [d.text for d in res[0].details] == ["The disgraced son of a noble house"]


# --- roster threading into the prompt ---------------------------------------

def test_roster_lands_in_prompt_sorted_and_token_replaced():
    agent = CharactersExtractor(client=FakeClient(["[]"]), player_names={"Sam", "Hannah"})
    assert "__PLAYER_ROSTER__" not in agent.system_prompt   # token was filled
    assert "Hannah, Sam" in agent.system_prompt             # sorted, comma-joined


def test_replace_preserves_literal_json_braces_in_prompt():
    # Documents WHY we use .replace() not .format(): the prompt's JSON example is
    # full of literal { } that .format() would try to treat as fill-in slots.
    agent = CharactersExtractor(client=FakeClient(["[]"]), player_names={"Sam"})
    assert '{"name":' in agent.system_prompt
