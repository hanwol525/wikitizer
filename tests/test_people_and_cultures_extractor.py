"""Tests for agents/people_and_cultures_extractor.py (the PeopleAndCulturesExtractor).

Mirrors test_locations_extractor.py / test_organization_extractor.py: every test
injects a self-contained fake Anthropic client, so the suite makes ZERO real API
calls and needs no API key.

PeopleAndCulturesExtractor is a Locations-shaped extractor with the same
missing-name fallback as Organization/Item (details processed BEFORE the name): a
no-name entry is KEPT when it has a usable detail (name = a short form of that
detail) and only DROPPED when it has neither a name nor any surviving detail.
Both halves get their own test below.
"""

import json
import logging
from datetime import datetime

import pytest

from agents.people_and_cultures_extractor import PeopleAndCulturesExtractor
from models.lore import PeopleAndCultures, Quote
from models.message import Message


# --- fake client (copied from test_locations_extractor.py) ------------------

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

def make_message(content, sender="Matt", source_file="group.txt"):
    return Message(
        sender=sender,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        content=content,
        source_file=source_file,
    )


def make_agent(responses, **kwargs):
    return PeopleAndCulturesExtractor(client=FakeClient(responses), **kwargs)


# --- happy path -------------------------------------------------------------

def test_extract_happy_path_single_batch():
    messages = [
        make_message("The Krieg are a seafaring people from the northern coasts"),
        make_message("Krieg raiders are feared for their longships"),
        make_message("direwolves hunt in packs across the northern tundra"),
    ]
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [
                {
                    "detail": "A seafaring people from the northern coasts",
                    "quote": "The Krieg are a seafaring people from the northern coasts",
                    "source_id": 0,
                },
                {
                    "detail": "Their raiders are feared for their longships",
                    "quote": "Krieg raiders are feared for their longships",
                    "source_id": 1,
                },
            ],
        },
        {
            "name": "Direwolves",
            "aliases": [],
            "details": [{
                "detail": "Hunt in packs across the northern tundra",
                "quote": "direwolves hunt in packs across the northern tundra",
                "source_id": 2,
            }],
        },
    ])
    client = FakeClient([response])
    agent = PeopleAndCulturesExtractor(client=client)

    result = agent.extract(messages)

    assert client.call_count == 1
    assert [p.name for p in result] == ["The Krieg", "Direwolves"]
    assert [a.text for a in result[0].aliases] == []
    assert [d.text for d in result[0].details] == [
        "A seafaring people from the northern coasts",
        "Their raiders are feared for their longships",
    ]
    assert len(result[0].supporting_quotes) == 2
    assert [d.text for d in result[1].details] == ["Hunt in packs across the northern tundra"]
    assert len(result[1].supporting_quotes) == 1
    assert all(isinstance(p, PeopleAndCultures) for p in result)


# --- quote-metadata attachment ----------------------------------------------

def test_quote_metadata_pulled_from_message_not_claude():
    quote = "The hill clans keep to the high passes and trade in furs"
    messages = [make_message(quote, sender="Matt", source_file="dndgroup.txt")]
    response = json.dumps([
        {
            "name": "The hill clans",
            "aliases": [],
            "details": [{"detail": "Keep to the high passes", "quote": quote, "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    q = result[0].supporting_quotes[0]
    assert isinstance(q, Quote)
    assert q.text == quote                    # exactly Claude's quote string
    assert q.speaker == "Matt"                # from the Message, not Claude
    assert q.source_file == "dndgroup.txt"    # from the Message, not Claude


# --- batching ---------------------------------------------------------------

def test_extract_batches_with_ceil_calls_and_resets_ids():
    messages = [make_message(f"msg {i}") for i in range(5)]
    client = FakeClient(["[]"])
    agent = PeopleAndCulturesExtractor(client=client, batch_size=2)

    agent.extract(messages)

    assert client.call_count == 3
    first_payload = json.loads(client.messages.calls[0]["messages"][0]["content"])
    second_payload = json.loads(client.messages.calls[1]["messages"][0]["content"])
    third_payload = json.loads(client.messages.calls[2]["messages"][0]["content"])
    assert [o["id"] for o in first_payload] == [0, 1]
    assert [o["content"] for o in first_payload] == ["msg 0", "msg 1"]
    assert [o["id"] for o in second_payload] == [0, 1]   # reset, not [2, 3]
    assert [o["content"] for o in second_payload] == ["msg 2", "msg 3"]
    assert [o["id"] for o in third_payload] == [0]
    assert third_payload[0]["content"] == "msg 4"


def test_payload_preserves_non_ascii_content_via_ensure_ascii_false():
    content = "The Krieg — “the northern people” \U0001f6f6"  # — “ ” 🛶
    messages = [make_message(content)]
    client = FakeClient(["[]"])
    agent = PeopleAndCulturesExtractor(client=client)
    agent.extract(messages)

    sent = client.messages.calls[0]["messages"][0]["content"]
    assert content in sent
    assert "\\u201c" not in sent
    assert json.loads(sent)[0]["content"] == content


# --- verbatim verification --------------------------------------------------

def test_verbatim_check_drops_non_matching_quote_keeps_siblings(caplog):
    messages = [make_message("The Krieg sail the cold seas. They raid the southern coasts.")]
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [
                {"detail": "Sail the cold seas", "quote": "The Krieg sail the cold seas", "source_id": 0},
                {"detail": "Fabricated", "quote": "The Krieg worship dragons", "source_id": 0},
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert [d.text for d in result[0].details] == ["Sail the cold seas"]
    assert len(result[0].supporting_quotes) == 1
    assert any(
        "verbatim" in r.getMessage() and "id 0" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_verbatim_check_tolerates_multiline_message():
    messages = [make_message("The Krieg are a seafaring\npeople from the\nnorthern coasts")]
    flat_quote = "The Krieg are a seafaring people from the northern coasts"
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [{"detail": "A seafaring people", "quote": flat_quote, "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    assert [d.text for d in result[0].details] == ["A seafaring people"]
    assert result[0].supporting_quotes[0].text == flat_quote


def test_verify_quotes_false_keeps_non_matching_quote():
    messages = [make_message("The Krieg are real")]
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [{
                "detail": "Invented fact",
                "quote": "totally invented text not in the message",
                "source_id": 0,
            }],
        },
    ])
    agent = make_agent([response], verify_quotes=False)

    result = agent.extract(messages)

    assert len(result) == 1
    assert [d.text for d in result[0].details] == ["Invented fact"]
    assert result[0].supporting_quotes[0].text == "totally invented text not in the message"


def test_verbatim_check_drops_empty_quote(caplog):
    messages = [make_message("The Krieg sail the cold seas")]
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [
                {"detail": "Sail the cold seas", "quote": "The Krieg sail the cold seas", "source_id": 0},
                {"detail": "Fabricated", "quote": "", "source_id": 0},  # blank -> dropped
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert [d.text for d in result[0].details] == ["Sail the cold seas"]
    assert all(q.text for q in result[0].supporting_quotes)


def test_identical_quotes_deduped_but_details_kept():
    messages = [make_message("The Krieg are seafarers")]
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [
                {"detail": "Seafarers", "quote": "The Krieg are seafarers", "source_id": 0},
                {"detail": "People of the sea", "quote": "The Krieg are seafarers", "source_id": 0},
            ],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    assert [d.text for d in result[0].details] == ["Seafarers", "People of the sea"]   # details NOT deduped
    # each fact is tagged with the file of the message it cited (dormant provenance)
    assert result[0].details[0].source_files == [messages[0].source_file]
    assert result[0].details[1].source_files == [messages[0].source_file]
    assert len(result[0].supporting_quotes) == 1                      # identical quote deduped


# --- missing name: the divergence from Locations ----------------------------

def test_no_name_with_no_details_dropped_sibling_kept(caplog):
    # No name AND no surviving detail -> nothing to build a page around -> dropped.
    messages = [make_message("The hill clans trade in furs")]
    response = json.dumps([
        {"aliases": [], "details": []},  # no name, no details -> dropped
        {
            "name": "The hill clans",
            "aliases": [],
            "details": [{
                "detail": "Trade in furs",
                "quote": "The hill clans trade in furs",
                "source_id": 0,
            }],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert [p.name for p in result] == ["The hill clans"]
    assert any(
        "no usable name and no surviving details" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_no_name_with_detail_kept_name_from_first_detail(caplog):
    # No name but a usable detail -> KEPT, name = a short form of the first detail
    # (trimmed to 80 chars). The History-style keep-don't-drop move.
    messages = [make_message("the northern tribes roam the tundra following the herds")]
    long_detail = (
        "A loose grouping of nomadic northern tribes who roam the open tundra all year "
        "round, following the great herds"
    )
    assert len(long_detail) > 80   # so the [:80] trim is actually exercised
    response = json.dumps([
        {
            "aliases": [],   # no name
            "details": [{
                "detail": long_detail,
                "quote": "the northern tribes roam the tundra following the herds",
                "source_id": 0,
            }],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].name == long_detail[:80]
    assert len(result[0].name) == 80
    assert [d.text for d in result[0].details] == [long_detail]
    assert any(
        "using a short form of its first detail" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


# --- malformed responses ----------------------------------------------------

def test_non_dict_entry_ignored_siblings_kept(caplog):
    messages = [make_message("The Krieg are seafarers")]
    response = json.dumps([
        "just a string, not an object",
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [{"detail": "Seafarers", "quote": "The Krieg are seafarers", "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert [p.name for p in result] == ["The Krieg"]
    assert any(
        "not an object" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_non_list_details_coerced_to_empty_no_crash(caplog):
    messages = [make_message("The Krieg exist")]
    response = '[{"name": "The Krieg", "aliases": [], "details": 5}]'
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert [p.name for p in result] == ["The Krieg"]
    assert [d.text for d in result[0].details] == []
    assert result[0].supporting_quotes == []
    assert any(
        "non-list" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_detail_with_non_string_detail_or_quote_dropped(caplog):
    messages = [make_message("The Krieg are seafarers")]
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [
                {"detail": "Seafarers", "quote": "The Krieg are seafarers", "source_id": 0},
                {"detail": 123, "quote": "The Krieg are seafarers", "source_id": 0},   # non-str detail
                {"quote": "The Krieg are seafarers", "source_id": 0},                   # missing detail
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert [d.text for d in result[0].details] == ["Seafarers"]
    assert any(
        "missing a string" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_aliases_sanitizers_drop_non_list_and_non_string_elements():
    messages = [make_message("The Krieg exist"), make_message("The clans exist")]
    response = json.dumps([
        {"name": "The Krieg", "aliases": "Bob", "details": []},                       # non-list -> []
        {"name": "The hill clans", "aliases": ["Highlanders", 7, None, "The Clans"], "details": []},
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    by_name = {p.name: p for p in result}
    assert [a.text for a in by_name["The Krieg"].aliases] == []                              # not ['B', 'o', 'b']
    assert [a.text for a in by_name["The hill clans"].aliases] == ["Highlanders", "The Clans"]


def test_out_of_range_source_id_dropped_other_details_kept(caplog):
    messages = [make_message("The Krieg are fierce")]
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [
                {"detail": "Fierce", "quote": "The Krieg are fierce", "source_id": 0},    # valid
                {"detail": "Out of range", "quote": "The Krieg are fierce", "source_id": 5}, # bad id
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert [d.text for d in result[0].details] == ["Fierce"]
    assert any(
        "out of range" in r.getMessage() and "5" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_bool_source_id_dropped_never_indexed_as_0_or_1(caplog):
    messages = [make_message("The Krieg exist"), make_message("Direwolves exist")]
    response = json.dumps([
        {
            "name": "The Krieg",
            "aliases": [],
            "details": [{"detail": "Mislabeled", "quote": "Direwolves exist", "source_id": True}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].name == "The Krieg"
    assert [d.text for d in result[0].details] == []
    assert result[0].supporting_quotes == []
    assert any(
        "non-integer source_id" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_claude_json_error_on_batch1_valid_batch2(caplog):
    messages = [make_message("garbage in"), make_message("The Krieg are a people")]
    valid = json.dumps([
        {"name": "The Krieg", "aliases": [], "details": [
            {"detail": "A people", "quote": "The Krieg are a people", "source_id": 0}
        ]},
    ])
    client = FakeClient(["not json at all", valid])
    agent = PeopleAndCulturesExtractor(client=client, batch_size=1, max_json_retries=1)

    with caplog.at_level(logging.ERROR, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert [p.name for p in result] == ["The Krieg"]
    assert any(
        "failed to return valid JSON" in r.getMessage()
        for r in caplog.records if r.levelno == logging.ERROR
    )


def test_non_list_response_skips_batch_with_warning(caplog):
    messages = [make_message("The Krieg are fierce")]
    response = '{"name": "The Krieg"}'  # an object, not the expected array
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.people_and_cultures_extractor"):
        result = agent.extract(messages)

    assert result == []
    assert any(
        "expected a JSON array" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


# --- empty input ------------------------------------------------------------

def test_empty_input_returns_empty_and_makes_no_calls():
    client = FakeClient(["[]"])
    agent = PeopleAndCulturesExtractor(client=client)

    assert agent.extract([]) == []
    assert client.call_count == 0


# --- defaults / construction (inherits BaseExtractor unchanged) -------------

def test_defaults_are_sonnet_and_big_max_tokens():
    agent = PeopleAndCulturesExtractor(client=FakeClient(["[]"]))
    assert agent.model == "claude-sonnet-4-6"
    assert agent.temperature == 0.2
    assert agent.max_tokens == 8192
    assert agent.batch_size == 20
    assert agent.verify_quotes is True


def test_defaults_reach_the_request():
    messages = [make_message("a")]
    client = FakeClient(["[]"])
    agent = PeopleAndCulturesExtractor(client=client)
    agent.extract(messages)

    sent = client.messages.calls[0]
    assert sent["model"] == "claude-sonnet-4-6"
    assert sent["max_tokens"] == 8192
    assert sent["temperature"] == 0.2


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        PeopleAndCulturesExtractor(client=FakeClient(["[]"]), batch_size=0)
