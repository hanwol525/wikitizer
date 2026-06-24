"""Tests for agents/item_extractor.py (the ItemExtractor).

Mirrors test_locations_extractor.py / test_organization_extractor.py: every test
injects a self-contained fake Anthropic client, so the suite makes ZERO real API
calls and needs no API key.

ItemExtractor is a Locations-shaped extractor with the same missing-name fallback
as OrganizationExtractor (details processed BEFORE the name): a no-name entry is
KEPT when it has a usable detail (name = a short form of that detail) and only
DROPPED when it has neither a name nor any surviving detail. Both halves get
their own test below.
"""

import json
import logging
from datetime import datetime

import pytest

from agents.item_extractor import ItemExtractor
from models.lore import Item, Quote
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
    return ItemExtractor(client=FakeClient(responses), **kwargs)


# --- happy path -------------------------------------------------------------

def test_extract_happy_path_single_batch():
    messages = [
        make_message("The Amulet of Destiny is the only thing that can seal the rift"),
        make_message("they say the Amulet was forged before the Maltraav-Kriega War"),
        make_message("Kriggy's family sword has been passed down for seven generations"),
    ]
    response = json.dumps([
        {
            "name": "Amulet of Destiny",
            "aliases": ["the Amulet"],
            "details": [
                {
                    "detail": "The only thing that can seal the rift",
                    "quote": "The Amulet of Destiny is the only thing that can seal the rift",
                    "source_id": 0,
                },
                {
                    "detail": "Said to have been forged before the Maltraav-Kriega War",
                    "quote": "they say the Amulet was forged before the Maltraav-Kriega War",
                    "source_id": 1,
                },
            ],
        },
        {
            "name": "Kriggy's family sword",
            "aliases": [],
            "details": [{
                "detail": "Has been passed down for seven generations",
                "quote": "Kriggy's family sword has been passed down for seven generations",
                "source_id": 2,
            }],
        },
    ])
    client = FakeClient([response])
    agent = ItemExtractor(client=client)

    result = agent.extract(messages)

    assert client.call_count == 1
    assert [it.name for it in result] == ["Amulet of Destiny", "Kriggy's family sword"]
    assert result[0].aliases == ["the Amulet"]
    assert result[0].details == [
        "The only thing that can seal the rift",
        "Said to have been forged before the Maltraav-Kriega War",
    ]
    assert len(result[0].supporting_quotes) == 2
    assert result[1].aliases == []
    assert result[1].details == ["Has been passed down for seven generations"]
    assert len(result[1].supporting_quotes) == 1
    assert all(isinstance(it, Item) for it in result)


# --- quote-metadata attachment ----------------------------------------------

def test_quote_metadata_pulled_from_message_not_claude():
    quote = "Frostbite is a sword that never melts the ice it touches"
    messages = [make_message(quote, sender="Matt", source_file="dndgroup.txt")]
    response = json.dumps([
        {
            "name": "Frostbite",
            "aliases": [],
            "details": [{"detail": "A sword that never melts ice", "quote": quote, "source_id": 0}],
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
    agent = ItemExtractor(client=client, batch_size=2)

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
    content = "The Amulet — “of Destiny” \U0001f48e"  # — “ ” 💎
    messages = [make_message(content)]
    client = FakeClient(["[]"])
    agent = ItemExtractor(client=client)
    agent.extract(messages)

    sent = client.messages.calls[0]["messages"][0]["content"]
    assert content in sent
    assert "\\u201c" not in sent
    assert json.loads(sent)[0]["content"] == content


# --- verbatim verification --------------------------------------------------

def test_verbatim_check_drops_non_matching_quote_keeps_siblings(caplog):
    messages = [make_message("The blade was forged in dragonfire. It glows in the dark.")]
    response = json.dumps([
        {
            "name": "The Blade",
            "aliases": [],
            "details": [
                {"detail": "Forged in dragonfire", "quote": "The blade was forged in dragonfire", "source_id": 0},
                {"detail": "Fabricated", "quote": "The blade can talk", "source_id": 0},
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Forged in dragonfire"]
    assert len(result[0].supporting_quotes) == 1
    assert any(
        "verbatim" in r.getMessage() and "id 0" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_verbatim_check_tolerates_multiline_message():
    messages = [make_message("The Amulet of Destiny\nis the only thing\nthat can seal the rift")]
    flat_quote = "The Amulet of Destiny is the only thing that can seal the rift"
    response = json.dumps([
        {
            "name": "Amulet of Destiny",
            "aliases": [],
            "details": [{"detail": "Seals the rift", "quote": flat_quote, "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Seals the rift"]
    assert result[0].supporting_quotes[0].text == flat_quote


def test_verify_quotes_false_keeps_non_matching_quote():
    messages = [make_message("Frostbite is real")]
    response = json.dumps([
        {
            "name": "Frostbite",
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
    messages = [make_message("The Amulet seals the rift")]
    response = json.dumps([
        {
            "name": "The Amulet",
            "aliases": [],
            "details": [
                {"detail": "Seals the rift", "quote": "The Amulet seals the rift", "source_id": 0},
                {"detail": "Fabricated", "quote": "", "source_id": 0},  # blank -> dropped
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Seals the rift"]
    assert all(q.text for q in result[0].supporting_quotes)


def test_identical_quotes_deduped_but_details_kept():
    messages = [make_message("The Amulet is ancient")]
    response = json.dumps([
        {
            "name": "The Amulet",
            "aliases": [],
            "details": [
                {"detail": "Ancient", "quote": "The Amulet is ancient", "source_id": 0},
                {"detail": "Very old", "quote": "The Amulet is ancient", "source_id": 0},
            ],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Ancient", "Very old"]   # details NOT deduped
    assert len(result[0].supporting_quotes) == 1           # identical quote deduped


# --- missing name: the divergence from Locations ----------------------------

def test_no_name_with_no_details_dropped_sibling_kept(caplog):
    # No name AND no surviving detail -> nothing to build a page around -> dropped.
    messages = [make_message("Frostbite never melts the ice")]
    response = json.dumps([
        {"aliases": [], "details": []},  # no name, no details -> dropped
        {
            "name": "Frostbite",
            "aliases": [],
            "details": [{
                "detail": "Never melts ice",
                "quote": "Frostbite never melts the ice",
                "source_id": 0,
            }],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert [it.name for it in result] == ["Frostbite"]
    assert any(
        "no usable name and no surviving details" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_no_name_with_detail_kept_name_from_first_detail(caplog):
    # No name but a usable detail -> KEPT, name = a short form of the first detail
    # (trimmed to 80 chars). The History-style keep-don't-drop move.
    messages = [make_message("Kriggy's family sword has been passed down for seven generations")]
    long_detail = (
        "An ancestral blade belonging to Kriggy that has been carefully passed down "
        "through his family for seven full generations"
    )
    assert len(long_detail) > 80   # so the [:80] trim is actually exercised
    response = json.dumps([
        {
            "aliases": [],   # no name
            "details": [{
                "detail": long_detail,
                "quote": "Kriggy's family sword has been passed down for seven generations",
                "source_id": 0,
            }],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
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
    messages = [make_message("Frostbite is a sword")]
    response = json.dumps([
        "just a string, not an object",
        {
            "name": "Frostbite",
            "aliases": [],
            "details": [{"detail": "A sword", "quote": "Frostbite is a sword", "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert [it.name for it in result] == ["Frostbite"]
    assert any(
        "not an object" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_non_list_details_coerced_to_empty_no_crash(caplog):
    messages = [make_message("Frostbite exists")]
    response = '[{"name": "Frostbite", "aliases": [], "details": 5}]'
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert [it.name for it in result] == ["Frostbite"]
    assert result[0].details == []
    assert result[0].supporting_quotes == []
    assert any(
        "non-list" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_detail_with_non_string_detail_or_quote_dropped(caplog):
    messages = [make_message("Frostbite is a sword")]
    response = json.dumps([
        {
            "name": "Frostbite",
            "aliases": [],
            "details": [
                {"detail": "A sword", "quote": "Frostbite is a sword", "source_id": 0},
                {"detail": 123, "quote": "Frostbite is a sword", "source_id": 0},   # non-str detail
                {"quote": "Frostbite is a sword", "source_id": 0},                   # missing detail
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["A sword"]
    assert any(
        "missing a string" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_aliases_sanitizers_drop_non_list_and_non_string_elements():
    messages = [make_message("Frostbite exists"), make_message("The Amulet exists")]
    response = json.dumps([
        {"name": "Frostbite", "aliases": "Bob", "details": []},                      # non-list -> []
        {"name": "The Amulet", "aliases": ["Destiny", 7, None, "The Seal"], "details": []},
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    by_name = {it.name: it for it in result}
    assert by_name["Frostbite"].aliases == []                       # not ['B', 'o', 'b']
    assert by_name["The Amulet"].aliases == ["Destiny", "The Seal"]


def test_out_of_range_source_id_dropped_other_details_kept(caplog):
    messages = [make_message("The Amulet is powerful")]
    response = json.dumps([
        {
            "name": "The Amulet",
            "aliases": [],
            "details": [
                {"detail": "Powerful", "quote": "The Amulet is powerful", "source_id": 0},    # valid
                {"detail": "Out of range", "quote": "The Amulet is powerful", "source_id": 5}, # bad id
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["Powerful"]
    assert any(
        "out of range" in r.getMessage() and "5" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_bool_source_id_dropped_never_indexed_as_0_or_1(caplog):
    messages = [make_message("The Amulet exists"), make_message("Frostbite exists")]
    response = json.dumps([
        {
            "name": "The Amulet",
            "aliases": [],
            "details": [{"detail": "Mislabeled", "quote": "Frostbite exists", "source_id": True}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].name == "The Amulet"
    assert result[0].details == []
    assert result[0].supporting_quotes == []
    assert any(
        "non-integer source_id" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_claude_json_error_on_batch1_valid_batch2(caplog):
    messages = [make_message("garbage in"), make_message("Frostbite is a sword")]
    valid = json.dumps([
        {"name": "Frostbite", "aliases": [], "details": [
            {"detail": "A sword", "quote": "Frostbite is a sword", "source_id": 0}
        ]},
    ])
    client = FakeClient(["not json at all", valid])
    agent = ItemExtractor(client=client, batch_size=1, max_json_retries=1)

    with caplog.at_level(logging.ERROR, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert [it.name for it in result] == ["Frostbite"]
    assert any(
        "failed to return valid JSON" in r.getMessage()
        for r in caplog.records if r.levelno == logging.ERROR
    )


def test_non_list_response_skips_batch_with_warning(caplog):
    messages = [make_message("The Amulet is powerful")]
    response = '{"name": "The Amulet"}'  # an object, not the expected array
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.item_extractor"):
        result = agent.extract(messages)

    assert result == []
    assert any(
        "expected a JSON array" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )


# --- empty input ------------------------------------------------------------

def test_empty_input_returns_empty_and_makes_no_calls():
    client = FakeClient(["[]"])
    agent = ItemExtractor(client=client)

    assert agent.extract([]) == []
    assert client.call_count == 0


# --- defaults / construction (inherits BaseExtractor unchanged) -------------

def test_defaults_are_sonnet_and_big_max_tokens():
    agent = ItemExtractor(client=FakeClient(["[]"]))
    assert agent.model == "claude-sonnet-4-6"
    assert agent.temperature == 0.2
    assert agent.max_tokens == 8192
    assert agent.batch_size == 20
    assert agent.verify_quotes is True


def test_defaults_reach_the_request():
    messages = [make_message("a")]
    client = FakeClient(["[]"])
    agent = ItemExtractor(client=client)
    agent.extract(messages)

    sent = client.messages.calls[0]
    assert sent["model"] == "claude-sonnet-4-6"
    assert sent["max_tokens"] == 8192
    assert sent["temperature"] == 0.2


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        ItemExtractor(client=FakeClient(["[]"]), batch_size=0)
