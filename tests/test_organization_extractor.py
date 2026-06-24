"""Tests for agents/organization_extractor.py (the OrganizationExtractor).

Mirrors test_locations_extractor.py: every test injects a self-contained fake
Anthropic client, so the suite makes ZERO real API calls and needs no API key.

OrganizationExtractor is a Locations-shaped extractor, so most cases port over
directly. The ONE divergence is the missing-name handling: Organization processes
details BEFORE the name and falls back to a short form of the first surviving
detail (History-style), so a no-name entry is KEPT when it has a usable detail
and only DROPPED when it has neither a name nor any surviving detail. Both halves
get their own test below.
"""

import json
import logging
from datetime import datetime

import pytest

from agents.organization_extractor import OrganizationExtractor
from models.lore import Organization, Quote
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
    return OrganizationExtractor(client=FakeClient(responses), **kwargs)


# --- happy path -------------------------------------------------------------

def test_extract_happy_path_single_batch():
    messages = [
        make_message(
            "Almost every country southeast of the Cloud Mountains is under the "
            "control of the Krieger Imperium"
        ),
        make_message("The Imperium is ruled by Emperor Tiberius and his war council"),
        make_message("Tansy's Adventuring Agency takes contracts out of Crown's Nest"),
    ]
    response = json.dumps([
        {
            "name": "Krieger Imperium",
            "aliases": ["the Imperium"],
            "details": [
                {
                    "detail": "Controls almost every country southeast of the Cloud Mountains",
                    "quote": (
                        "Almost every country southeast of the Cloud Mountains is under the "
                        "control of the Krieger Imperium"
                    ),
                    "source_id": 0,
                },
                {
                    "detail": "Ruled by Emperor Tiberius and his war council",
                    "quote": "The Imperium is ruled by Emperor Tiberius and his war council",
                    "source_id": 1,
                },
            ],
        },
        {
            "name": "Tansy's Adventuring Agency",
            "aliases": [],
            "details": [{
                "detail": "Takes contracts out of Crown's Nest",
                "quote": "Tansy's Adventuring Agency takes contracts out of Crown's Nest",
                "source_id": 2,
            }],
        },
    ])
    client = FakeClient([response])
    agent = OrganizationExtractor(client=client)

    result = agent.extract(messages)

    assert client.call_count == 1
    assert [o.name for o in result] == ["Krieger Imperium", "Tansy's Adventuring Agency"]
    assert result[0].aliases == ["the Imperium"]
    assert result[0].details == [
        "Controls almost every country southeast of the Cloud Mountains",
        "Ruled by Emperor Tiberius and his war council",
    ]
    assert len(result[0].supporting_quotes) == 2
    assert result[1].aliases == []
    assert result[1].details == ["Takes contracts out of Crown's Nest"]
    assert len(result[1].supporting_quotes) == 1
    assert all(isinstance(o, Organization) for o in result)


# --- quote-metadata attachment ----------------------------------------------

def test_quote_metadata_pulled_from_message_not_claude():
    quote = "The Order of the Dawn guards the eastern passes"
    messages = [make_message(quote, sender="Matt", source_file="dndgroup.txt")]
    response = json.dumps([
        {
            "name": "Order of the Dawn",
            "aliases": [],
            "details": [{"detail": "Guards the eastern passes", "quote": quote, "source_id": 0}],
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
    agent = OrganizationExtractor(client=client, batch_size=2)

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
    content = "The Imperium — “the Empire” \U0001f3f0"  # — “ ” 🏰
    messages = [make_message(content)]
    client = FakeClient(["[]"])
    agent = OrganizationExtractor(client=client)
    agent.extract(messages)

    sent = client.messages.calls[0]["messages"][0]["content"]
    assert content in sent
    assert "\\u201c" not in sent
    assert json.loads(sent)[0]["content"] == content


# --- verbatim verification --------------------------------------------------

def test_verbatim_check_drops_non_matching_quote_keeps_siblings(caplog):
    messages = [make_message("The guild was founded long ago. It has many members.")]
    response = json.dumps([
        {
            "name": "The Guild",
            "aliases": [],
            "details": [
                {"detail": "Founded long ago", "quote": "The guild was founded long ago", "source_id": 0},
                {"detail": "Fabricated", "quote": "The guild rules the world", "source_id": 0},
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Founded long ago"]
    assert len(result[0].supporting_quotes) == 1
    assert any(
        "verbatim" in r.getMessage() and "id 0" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_verbatim_check_tolerates_multiline_message():
    messages = [make_message("The Order of the Dawn\nguards the eastern\npasses of the realm")]
    flat_quote = "The Order of the Dawn guards the eastern passes of the realm"
    response = json.dumps([
        {
            "name": "Order of the Dawn",
            "aliases": [],
            "details": [{"detail": "Guards the passes", "quote": flat_quote, "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Guards the passes"]
    assert result[0].supporting_quotes[0].text == flat_quote


def test_verify_quotes_false_keeps_non_matching_quote():
    messages = [make_message("The Guild is real")]
    response = json.dumps([
        {
            "name": "The Guild",
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
    assert result[0].details == ["Invented fact"]
    assert result[0].supporting_quotes[0].text == "totally invented text not in the message"


def test_verbatim_check_drops_empty_quote(caplog):
    messages = [make_message("The Guild takes contracts")]
    response = json.dumps([
        {
            "name": "The Guild",
            "aliases": [],
            "details": [
                {"detail": "Takes contracts", "quote": "The Guild takes contracts", "source_id": 0},
                {"detail": "Fabricated", "quote": "", "source_id": 0},  # blank -> dropped
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Takes contracts"]
    assert all(q.text for q in result[0].supporting_quotes)


def test_identical_quotes_deduped_but_details_kept():
    messages = [make_message("The Imperium rules the south")]
    response = json.dumps([
        {
            "name": "The Imperium",
            "aliases": [],
            "details": [
                {"detail": "Rules the south", "quote": "The Imperium rules the south", "source_id": 0},
                {"detail": "Southern power", "quote": "The Imperium rules the south", "source_id": 0},
            ],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Rules the south", "Southern power"]   # details NOT deduped
    assert len(result[0].supporting_quotes) == 1                         # identical quote deduped


# --- missing name: the divergence from Locations ----------------------------

def test_no_name_with_no_details_dropped_sibling_kept(caplog):
    # No name AND no surviving detail -> nothing to build a page around -> dropped.
    messages = [make_message("Tansy's Adventuring Agency takes contracts")]
    response = json.dumps([
        {"aliases": [], "details": []},  # no name, no details -> dropped
        {
            "name": "Tansy's Adventuring Agency",
            "aliases": [],
            "details": [{
                "detail": "Takes contracts",
                "quote": "Tansy's Adventuring Agency takes contracts",
                "source_id": 0,
            }],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert [o.name for o in result] == ["Tansy's Adventuring Agency"]
    assert any(
        "no usable name and no surviving details" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_no_name_with_detail_kept_name_from_first_detail(caplog):
    # No name but a usable detail -> KEPT, name = a short form of the first detail
    # (trimmed to 80 chars). The History-style keep-don't-drop move.
    messages = [make_message("the council that rules Gol just raised taxes again")]
    long_detail = (
        "The ruling council of Gol that governs the city, sets its laws, and recently "
        "raised the taxes on every resident"
    )
    assert len(long_detail) > 80   # so the [:80] trim is actually exercised
    response = json.dumps([
        {
            "aliases": [],   # no name
            "details": [{
                "detail": long_detail,
                "quote": "the council that rules Gol just raised taxes again",
                "source_id": 0,
            }],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].name == long_detail[:80]
    assert len(result[0].name) == 80
    assert result[0].details == [long_detail]
    assert any(
        "using a short form of its first detail" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


# --- malformed responses ----------------------------------------------------

def test_non_dict_entry_ignored_siblings_kept(caplog):
    messages = [make_message("The Guild takes contracts")]
    response = json.dumps([
        "just a string, not an object",
        {
            "name": "The Guild",
            "aliases": [],
            "details": [{"detail": "Takes contracts", "quote": "The Guild takes contracts", "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert [o.name for o in result] == ["The Guild"]
    assert any(
        "not an object" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_non_list_details_coerced_to_empty_no_crash(caplog):
    messages = [make_message("The Guild exists")]
    response = '[{"name": "The Guild", "aliases": [], "details": 5}]'
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert [o.name for o in result] == ["The Guild"]
    assert result[0].details == []
    assert result[0].supporting_quotes == []
    assert any(
        "non-list" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_detail_with_non_string_detail_or_quote_dropped(caplog):
    messages = [make_message("The Guild takes contracts")]
    response = json.dumps([
        {
            "name": "The Guild",
            "aliases": [],
            "details": [
                {"detail": "Takes contracts", "quote": "The Guild takes contracts", "source_id": 0},
                {"detail": 123, "quote": "The Guild takes contracts", "source_id": 0},   # non-str detail
                {"quote": "The Guild takes contracts", "source_id": 0},                   # missing detail
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Takes contracts"]
    assert any(
        "missing a string" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_aliases_sanitizers_drop_non_list_and_non_string_elements():
    messages = [make_message("The Guild exists"), make_message("The Order exists")]
    response = json.dumps([
        {"name": "The Guild", "aliases": "Bob", "details": []},                   # non-list -> []
        {"name": "The Order", "aliases": ["The Dawn", 7, None, "Dawn Order"], "details": []},
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    by_name = {o.name: o for o in result}
    assert by_name["The Guild"].aliases == []                       # not ['B', 'o', 'b']
    assert by_name["The Order"].aliases == ["The Dawn", "Dawn Order"]


def test_out_of_range_source_id_dropped_other_details_kept(caplog):
    messages = [make_message("The Guild is powerful")]
    response = json.dumps([
        {
            "name": "The Guild",
            "aliases": [],
            "details": [
                {"detail": "Powerful", "quote": "The Guild is powerful", "source_id": 0},    # valid
                {"detail": "Out of range", "quote": "The Guild is powerful", "source_id": 5}, # bad id
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Powerful"]
    assert any(
        "out of range" in r.getMessage() and "5" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_bool_source_id_dropped_never_indexed_as_0_or_1(caplog):
    messages = [make_message("The Guild exists"), make_message("The Order exists")]
    response = json.dumps([
        {
            "name": "The Guild",
            "aliases": [],
            "details": [{"detail": "Mislabeled", "quote": "The Order exists", "source_id": True}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].name == "The Guild"
    assert result[0].details == []
    assert result[0].supporting_quotes == []
    assert any(
        "non-integer source_id" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_claude_json_error_on_batch1_valid_batch2(caplog):
    messages = [make_message("garbage in"), make_message("The Guild is a guild")]
    valid = json.dumps([
        {"name": "The Guild", "aliases": [], "details": [
            {"detail": "A guild", "quote": "The Guild is a guild", "source_id": 0}
        ]},
    ])
    client = FakeClient(["not json at all", valid])
    agent = OrganizationExtractor(client=client, batch_size=1, max_json_retries=1)

    with caplog.at_level(logging.ERROR, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert [o.name for o in result] == ["The Guild"]
    assert any(
        "failed to return valid JSON" in r.getMessage()
        for r in caplog.records if r.levelno == logging.ERROR
    )


def test_non_list_response_skips_batch_with_warning(caplog):
    messages = [make_message("The Guild is powerful")]
    response = '{"name": "The Guild"}'  # an object, not the expected array
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.organization_extractor"):
        result = agent.extract(messages)

    assert result == []
    assert any(
        "expected a JSON array" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


# --- empty input ------------------------------------------------------------

def test_empty_input_returns_empty_and_makes_no_calls():
    client = FakeClient(["[]"])
    agent = OrganizationExtractor(client=client)

    assert agent.extract([]) == []
    assert client.call_count == 0


# --- defaults / construction (inherits BaseExtractor unchanged) -------------

def test_defaults_are_sonnet_and_big_max_tokens():
    agent = OrganizationExtractor(client=FakeClient(["[]"]))
    assert agent.model == "claude-sonnet-4-6"
    assert agent.temperature == 0.2
    assert agent.max_tokens == 8192
    assert agent.batch_size == 20
    assert agent.verify_quotes is True


def test_defaults_reach_the_request():
    messages = [make_message("a")]
    client = FakeClient(["[]"])
    agent = OrganizationExtractor(client=client)
    agent.extract(messages)

    sent = client.messages.calls[0]
    assert sent["model"] == "claude-sonnet-4-6"
    assert sent["max_tokens"] == 8192
    assert sent["temperature"] == 0.2


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        OrganizationExtractor(client=FakeClient(["[]"]), batch_size=0)
