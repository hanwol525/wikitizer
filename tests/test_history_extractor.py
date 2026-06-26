"""Tests for agents/history_extractor.py (Phase 3.5: the HistoryExtractor).

Every test injects a fake Anthropic client, so the suite makes ZERO real API
calls and needs no API key. The fake-client classes are copied from the other
agent test files to keep this file self-contained (the established per-phase
pattern).

The inherited machinery -- batching, the verbatim-quote drop, the bool /
out-of-range source_id guards (all in BaseExtractor._resolve_quote), empty input,
default model/max_tokens -- is already covered by test_base_extractor.py /
test_locations_extractor.py, so this file focuses on what is NEW/different for
history: the FLAT quotes loop (no details), the Scope coercion, the name
fallback, the missing-description drop, and the empty-quotes keep-and-flag.

Heads-up: with verify_quotes=True by default, any quote expected to SURVIVE must
appear verbatim in its source message, so the fake message content is lined up
with the fake response quotes.
"""

import json
import logging
from datetime import datetime

from agents.history_extractor import HistoryExtractor, VALID_SCOPES
from models.lore import HistoryEvent, Quote, Scope
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


def make_agent(responses, **kwargs):
    return HistoryExtractor(client=FakeClient(responses), **kwargs)


# --- happy path / smoke -----------------------------------------------------

def test_happy_path_three_events():
    messages = [
        make_message("The Krieger Imperium was founded after the old kingdoms collapsed"),
        make_message("Maltraav and Kriega fought a brutal war"),
        make_message("that war happened before the Empire fell"),
        make_message("what's everyone rolling for initiative"),
        make_message("The Aldward family rose to prominence around then too"),
        make_message("everyone just calls it the Border War"),
    ]
    response = json.dumps([
        {
            "name": "Founding of the Krieger Imperium",
            "aliases": [],
            "description": "The Krieger Imperium was founded after the old kingdoms collapsed.",
            "scope": "world",
            "quotes": [
                {"quote": "The Krieger Imperium was founded after the old kingdoms collapsed",
                 "source_id": 0},
            ],
        },
        {
            "name": "The Maltraav-Kriega War",
            "aliases": ["the Border War"],
            "description": "A brutal war fought between Maltraav and Kriega, which took place "
                           "before the Empire fell.",
            "scope": "regional",
            "quotes": [
                {"quote": "Maltraav and Kriega fought a brutal war", "source_id": 1},
                {"quote": "that war happened before the Empire fell", "source_id": 2},
            ],
        },
        {
            "name": "Rise of the Aldward Family",
            "aliases": [],
            "description": "The Aldward family rose to prominence.",
            "scope": "personal",
            "quotes": [
                {"quote": "The Aldward family rose to prominence", "source_id": 4},
            ],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert [e.name for e in result] == [
        "Founding of the Krieger Imperium",
        "The Maltraav-Kriega War",
        "Rise of the Aldward Family",
    ]
    assert all(isinstance(e, HistoryEvent) for e in result)
    assert [e.scope for e in result] == [Scope.WORLD, Scope.REGIONAL, Scope.PERSONAL]
    assert result[1].aliases == ["the Border War"]
    assert result[1].description.startswith("A brutal war fought between Maltraav and Kriega")
    # chronological_position is never set by the extractor
    assert all(e.chronological_position is None for e in result)
    # a supporting quote carries metadata pulled from its message
    q = result[0].supporting_quotes[0]
    assert isinstance(q, Quote)
    assert q.speaker == "Matt"
    assert q.source_file == "dndgroup.txt"


def test_flat_quotes_structure_both_land_with_metadata():
    # An event with two quotes from two different messages -> both land in
    # supporting_quotes (proves the flat quotes loop, not a details loop).
    messages = [
        make_message("Maltraav and Kriega fought a brutal war", sender="Sam"),
        make_message("that war happened before the Empire fell", sender="Hannah"),
    ]
    response = json.dumps([
        {
            "name": "The War",
            "aliases": [],
            "description": "A brutal war that happened before the Empire fell.",
            "scope": "regional",
            "quotes": [
                {"quote": "Maltraav and Kriega fought a brutal war", "source_id": 0},
                {"quote": "that war happened before the Empire fell", "source_id": 1},
            ],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    quotes = result[0].supporting_quotes
    assert [q.text for q in quotes] == [
        "Maltraav and Kriega fought a brutal war",
        "that war happened before the Empire fell",
    ]
    assert [q.speaker for q in quotes] == ["Sam", "Hannah"]


# --- scope coercion ---------------------------------------------------------

def test_scope_off_list_defaults_to_world_with_warning(caplog):
    messages = [make_message("The Empire fell in a great cataclysm")]
    response = json.dumps([
        {"name": "The Cataclysm", "aliases": [],
         "description": "The Empire fell in a great cataclysm.",
         "scope": "wrold",  # typo, off-list
         "quotes": [{"quote": "The Empire fell in a great cataclysm", "source_id": 0}]},
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        result = agent.extract(messages)

    assert result[0].scope == Scope.WORLD
    assert result[0].scope == "world"   # the (str, Enum) behavior
    # the warning fires AND names the offending raw value
    assert any(
        ("off-list" in r.getMessage() or "defaulting to 'world'" in r.getMessage())
        and "wrold" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_scope_is_case_and_space_forgiving_without_warning(caplog):
    messages = [make_message("Two countries went to war")]
    response = json.dumps([
        {"name": "The War", "aliases": [],
         "description": "Two countries went to war.",
         "scope": " Regional ",  # mixed case + surrounding spaces
         "quotes": [{"quote": "Two countries went to war", "source_id": 0}]},
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        result = agent.extract(messages)

    assert result[0].scope == Scope.REGIONAL
    # it normalized, it did NOT default -> no scope warning
    assert not any(
        "off-list" in r.getMessage() or "defaulting to 'world'" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_valid_scope_passes_through():
    messages = [make_message("A noble died quietly")]
    response = json.dumps([
        {"name": "A Death", "aliases": [],
         "description": "A noble died quietly.", "scope": "personal",
         "quotes": [{"quote": "A noble died quietly", "source_id": 0}]},
    ])
    agent = make_agent([response])
    assert agent.extract(messages)[0].scope is Scope.PERSONAL


# --- name fallback / description drop ---------------------------------------

def test_missing_name_falls_back_to_short_description(caplog):
    messages = [make_message("The Krieger Imperium was founded after the old kingdoms collapsed")]
    response = json.dumps([
        {"aliases": [],   # no name
         "description": "The Krieger Imperium was founded after the old kingdoms collapsed.",
         "scope": "world",
         "quotes": [{"quote": "The Krieger Imperium was founded after the old kingdoms collapsed",
                     "source_id": 0}]},
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    # first-sentence (split on ".") short form of the description, trimmed to <=80
    assert result[0].name == "The Krieger Imperium was founded after the old kingdoms collapsed"
    assert any("no usable name" in r.getMessage() for r in caplog.records)


def test_missing_description_drops_event(caplog):
    messages = [make_message("something happened")]
    response = json.dumps([
        {"name": "Nameless Event", "aliases": [], "scope": "world",  # no description
         "quotes": [{"quote": "something happened", "source_id": 0}]},
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        result = agent.extract(messages)

    assert result == []
    assert any("no usable description" in r.getMessage() for r in caplog.records)


# --- empty quotes: keep + flag ----------------------------------------------

def test_empty_quotes_event_kept_and_flagged(caplog):
    # The only quote is non-verbatim -> dropped by the verbatim check, leaving the
    # event with zero quotes. History KEEPS such an event (missing lore is worse)
    # but flags it loudly, because its name is model-generated.
    messages = [make_message("The Empire rose long ago")]
    response = json.dumps([
        {"name": "Rise of the Empire", "aliases": [],
         "description": "The Empire rose to power.", "scope": "world",
         "quotes": [{"quote": "totally invented text not in any message", "source_id": 0}]},
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].name == "Rise of the Empire"
    assert result[0].supporting_quotes == []
    assert any("no surviving verbatim quotes" in r.getMessage() for r in caplog.records)


# --- aliases captured + defended --------------------------------------------

def test_aliases_captured_and_defended():
    messages = [make_message("the war happened")]

    captured = json.dumps([
        {"name": "The War", "aliases": ["the Border War"],
         "description": "A war happened.", "scope": "regional",
         "quotes": [{"quote": "the war happened", "source_id": 0}]},
    ])
    assert make_agent([captured]).extract(messages)[0].aliases == ["the Border War"]

    filtered = json.dumps([
        {"name": "The War", "aliases": ["x", 5, None],
         "description": "A war happened.", "scope": "regional",
         "quotes": [{"quote": "the war happened", "source_id": 0}]},
    ])
    assert make_agent([filtered]).extract(messages)[0].aliases == ["x"]

    nonlist = json.dumps([
        {"name": "The War", "aliases": "notalist",
         "description": "A war happened.", "scope": "regional",
         "quotes": [{"quote": "the war happened", "source_id": 0}]},
    ])
    assert make_agent([nonlist]).extract(messages)[0].aliases == []


# --- chronological_position is never read through ---------------------------

def test_chronological_position_ignored_if_present_in_response():
    messages = [make_message("the war happened")]
    response = json.dumps([
        {"name": "The War", "aliases": [], "description": "A war happened.",
         "scope": "regional", "chronological_position": 3,  # extractor must ignore this
         "quotes": [{"quote": "the war happened", "source_id": 0}]},
    ])
    agent = make_agent([response])
    assert agent.extract(messages)[0].chronological_position is None


# --- scope: missing / non-string --------------------------------------------

def test_missing_or_non_string_scope_defaults_to_world(caplog):
    # The off-list path above used a *string* ("wrold"); this drives the distinct
    # `else None` branch -- an omitted scope and a non-string scope both default
    # to "world" with the warning.
    messages = [make_message("the empire rose")]
    omitted = json.dumps([
        {"name": "Rise", "aliases": [], "description": "The empire rose.",
         "quotes": [{"quote": "the empire rose", "source_id": 0}]},  # no scope key
    ])
    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        res = make_agent([omitted]).extract(messages)
    assert res[0].scope == Scope.WORLD
    assert any("off-list" in r.getMessage() for r in caplog.records)

    nonstr = json.dumps([
        {"name": "Rise", "aliases": [], "description": "The empire rose.", "scope": 7,
         "quotes": [{"quote": "the empire rose", "source_id": 0}]},
    ])
    assert make_agent([nonstr]).extract(messages)[0].scope == Scope.WORLD


# --- name fallback edges ----------------------------------------------------

def test_name_fallback_truncates_a_period_less_description_to_80(caplog):
    # No period -> split(".")[0] is the whole string -> truncated to 80 chars.
    long_desc = "The empire expanded relentlessly across the continent for many generations without pause"
    assert len(long_desc) > 80 and "." not in long_desc
    messages = [make_message(long_desc)]
    response = json.dumps([
        {"aliases": [], "description": long_desc, "scope": "world",   # no name
         "quotes": [{"quote": "The empire expanded relentlessly", "source_id": 0}]},
    ])
    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        res = make_agent([response]).extract(messages)
    assert res[0].name == long_desc[:80]
    assert len(res[0].name) == 80


def test_name_fallback_uses_full_description_when_first_segment_empty():
    # A leading '.' makes split(".")[0] == "" (falsy) -> the `or description.strip()`
    # arm fires, so the name is the whole (trimmed) description, not "".
    messages = [make_message("A war happened long ago")]
    response = json.dumps([
        {"aliases": [], "description": ".A war happened long ago", "scope": "regional",  # no name
         "quotes": [{"quote": "A war happened long ago", "source_id": 0}]},
    ])
    res = make_agent([response]).extract(messages)
    assert res[0].name == ".A war happened long ago"


# --- malformed quotes structure ---------------------------------------------

def test_non_list_quotes_coerced_to_empty_kept_and_flagged(caplog):
    messages = [make_message("the empire rose")]
    response = json.dumps([
        {"name": "Rise", "aliases": [], "description": "The empire rose.",
         "scope": "world", "quotes": "notalist"},
    ])
    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        res = make_agent([response]).extract(messages)
    assert len(res) == 1
    assert res[0].supporting_quotes == []
    assert any("non-list 'quotes'" in r.getMessage() for r in caplog.records)


def test_non_dict_quote_entry_skipped_good_sibling_kept(caplog):
    messages = [make_message("the empire rose")]
    response = json.dumps([
        {"name": "Rise", "aliases": [], "description": "The empire rose.", "scope": "world",
         "quotes": [
             "just a string, not an object",
             {"quote": "the empire rose", "source_id": 0},
         ]},
    ])
    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        res = make_agent([response]).extract(messages)
    assert len(res) == 1
    assert [q.text for q in res[0].supporting_quotes] == ["the empire rose"]
    assert any("is not an object" in r.getMessage() for r in caplog.records)


def test_literal_empty_quotes_list_kept_and_flagged(caplog):
    # The other empty-quotes test reaches [] via the verbatim drop; this drives it
    # via a literally-empty quotes list from Claude. Both converge on keep+flag.
    messages = [make_message("the empire rose")]
    response = json.dumps([
        {"name": "Rise", "aliases": [], "description": "The empire rose.",
         "scope": "world", "quotes": []},
    ])
    with caplog.at_level(logging.WARNING, logger="agents.history_extractor"):
        res = make_agent([response]).extract(messages)
    assert len(res) == 1
    assert res[0].supporting_quotes == []
    assert any("no surviving verbatim quotes" in r.getMessage() for r in caplog.records)


# --- date_text: verbatim capture, null when absent --------------------------

def test_date_text_captured_verbatim():
    messages = [make_message("the Sundering of 342 AR shattered the land")]
    response = json.dumps([
        {"name": "The Sundering", "aliases": [], "description": "A cataclysm.",
         "scope": "world", "date_text": "342 AR",
         "quotes": [{"quote": "the Sundering of 342 AR", "source_id": 0}]},
    ])
    event = make_agent([response]).extract(messages)[0]
    assert event.date_text == "342 AR"  # exact, not normalized to a number


def test_date_text_absent_defaults_to_none():
    messages = [make_message("the war happened")]
    response = json.dumps([
        {"name": "The War", "aliases": [], "description": "A war happened.",
         "scope": "regional",
         "quotes": [{"quote": "the war happened", "source_id": 0}]},
    ])
    assert make_agent([response]).extract(messages)[0].date_text is None


def test_date_text_non_string_or_blank_treated_as_none():
    messages = [make_message("the war happened")]
    numeric = json.dumps([
        {"name": "The War", "aliases": [], "description": "A war happened.",
         "scope": "regional", "date_text": 342,  # not a string -> None
         "quotes": [{"quote": "the war happened", "source_id": 0}]},
    ])
    assert make_agent([numeric]).extract(messages)[0].date_text is None
    blank = json.dumps([
        {"name": "The War", "aliases": [], "description": "A war happened.",
         "scope": "regional", "date_text": "   ",  # blank -> None
         "quotes": [{"quote": "the war happened", "source_id": 0}]},
    ])
    assert make_agent([blank]).extract(messages)[0].date_text is None


def test_date_text_whitespace_trimmed():
    messages = [make_message("the Sundering of 342 AR")]
    response = json.dumps([
        {"name": "The Sundering", "aliases": [], "description": "A cataclysm.",
         "scope": "world", "date_text": "  342 AR  ",
         "quotes": [{"quote": "the Sundering of 342 AR", "source_id": 0}]},
    ])
    assert make_agent([response]).extract(messages)[0].date_text == "342 AR"


def test_date_text_null_with_relative_clue_stays_none_and_keeps_description():
    # The boundary intent hinges on: a RELATIVE clue is NOT a date. The mocked layer
    # can't prove the model's judgment, but it CAN pin the extractor contract -- when
    # Claude returns date_text=null alongside a description carrying the relative
    # clue, the extractor stores None and leaves the clue untouched in the prose.
    messages = [make_message("the Krieger Imperium was founded after the old kingdoms collapsed")]
    response = json.dumps([
        {"name": "Founding of the Imperium", "aliases": [],
         "description": "The Krieger Imperium was founded after the old kingdoms collapsed.",
         "scope": "world", "date_text": None,
         "quotes": [{"quote": "the Krieger Imperium was founded after the old kingdoms collapsed",
                     "source_id": 0}]},
    ])
    event = make_agent([response]).extract(messages)[0]
    assert event.date_text is None
    assert "after the old kingdoms collapsed" in event.description  # clue stays in the prose


# --- module-level sanity ----------------------------------------------------

def test_valid_scopes_constant_built_from_enum():
    assert VALID_SCOPES == {"world", "regional", "personal"}
