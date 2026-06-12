"""Tests for agents/noise_filter.py (Phase 3.2: the NoiseFilterAgent classifier).

Like test_base_agent.py, every test injects a fake Anthropic client, so the
suite makes ZERO real API calls and needs no API key. The fake mimics the real
response shape (an object with a ``.content`` list of text blocks) and records
the kwargs of every ``messages.create`` call, so we can assert both how many
batches we sent and the request params (model/temperature) we sent them with.

The fake-client classes are copied from test_base_agent.py to keep this file
self-contained. (Optional future cleanup, not now: hoist them into the empty
conftest.py to share with test_base_agent.py.)

Coverage:
  * happy path, single batch: correct (Message, label) pairs in order
  * batching: ceil(5/2) == 3 calls, batch-local ids, labels in original order
  * id rejoin: out-of-order ids map back onto the right messages
  * missing id in the response -> ambiguous + warning
  * invalid label -> ambiguous + warning
  * unknown/extra id -> ignored + warning, real messages still labeled
  * non-list response -> whole batch ambiguous + warning
  * ClaudeJSONError -> whole batch ambiguous + error logged
  * empty input -> [] with zero API calls
  * defaults: Haiku model + temperature 0.0 (and they reach the request)
  * select_for_extraction: keeps lore + ambiguous, drops the rest, order kept
"""

import logging
from datetime import datetime

from agents.noise_filter import (
    NoiseFilterAgent,
    select_for_extraction,
    VALID_LABELS,
)
from models.message import Message


# --- fake client (copied from test_base_agent.py) ---------------------------

class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, blocks):
        self.content = blocks


class FakeMessages:
    """``messages`` namespace exposing ``create``.

    ``responses`` is a list of specs consumed one per call; once a single spec
    remains it is returned for every further call (so a one-element list models
    "always returns the same thing"). A spec is either a ``str`` (-> one text
    block) or an explicit list of block objects.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # recorded kwargs, one entry per create() call

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
    """A minimal valid Message; only ``content`` matters to the classifier."""
    return Message(
        sender=sender,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        content=content,
        source_file=source_file,
    )


def make_agent(responses, **kwargs):
    return NoiseFilterAgent(client=FakeClient(responses), **kwargs)


# --- happy path -------------------------------------------------------------

def test_classify_happy_path_single_batch():
    messages = [
        make_message("Lake Mundi sits at the center of the world"),
        make_message("lol"),
        make_message("what's your AC"),
    ]
    response = (
        '[{"id": 0, "label": "lore"}, '
        '{"id": 1, "label": "noise"}, '
        '{"id": 2, "label": "mechanic"}]'
    )
    client = FakeClient([response])
    agent = NoiseFilterAgent(client=client)

    result = agent.classify(messages)

    assert client.call_count == 1
    assert result == [
        (messages[0], "lore"),
        (messages[1], "noise"),
        (messages[2], "mechanic"),
    ]
    # the original Message objects flow through untouched (identity preserved)
    assert all(result[i][0] is messages[i] for i in range(3))


# --- batching ---------------------------------------------------------------

def test_classify_batches_with_ceil_calls_and_preserves_order():
    messages = [make_message(f"msg {i}") for i in range(5)]
    # batch_size=2 over 5 -> ceil(5/2) == 3 batches, with batch-local ids
    # [0,1], [0,1], [0].
    responses = [
        '[{"id": 0, "label": "lore"}, {"id": 1, "label": "noise"}]',
        '[{"id": 0, "label": "mechanic"}, {"id": 1, "label": "ambiguous"}]',
        '[{"id": 0, "label": "lore"}]',
    ]
    client = FakeClient(responses)
    agent = NoiseFilterAgent(client=client, batch_size=2)

    result = agent.classify(messages)

    assert client.call_count == 3
    assert [label for _, label in result] == [
        "lore", "noise", "mechanic", "ambiguous", "lore",
    ]
    assert [msg for msg, _ in result] == messages  # original order, all present


def test_classify_sends_batch_local_ids():
    # Each batch's payload must start its ids at 0, independent of position in
    # the overall message list -- so nothing can drift across batch boundaries.
    import json

    messages = [make_message(f"msg {i}") for i in range(3)]
    responses = [
        '[{"id": 0, "label": "lore"}, {"id": 1, "label": "noise"}]',
        '[{"id": 0, "label": "lore"}]',
    ]
    client = FakeClient(responses)
    agent = NoiseFilterAgent(client=client, batch_size=2)
    agent.classify(messages)

    first_payload = json.loads(client.messages.calls[0]["messages"][0]["content"])
    second_payload = json.loads(client.messages.calls[1]["messages"][0]["content"])
    assert [o["id"] for o in first_payload] == [0, 1]
    assert [o["content"] for o in first_payload] == ["msg 0", "msg 1"]  # id->content binding
    assert [o["id"] for o in second_payload] == [0]  # reset, not [2]
    assert second_payload[0]["content"] == "msg 2"  # batch 2 carries the right message


def test_payload_preserves_non_ascii_content_via_ensure_ascii_false():
    # The brief mandates json.dumps(..., ensure_ascii=False): chat content has
    # curly quotes, em-dashes, and emoji, and we want the literal characters in
    # the payload, not \uXXXX escapes. If ensure_ascii regressed to the json
    # default (True), every other test would still pass green (ASCII content is
    # byte-identical either way), so pin the non-ASCII case explicitly here.
    import json

    content = "Loved “Lake Mundi” — the Great Well \U0001f30a"  # “ ” — 🌊
    messages = [make_message(content)]
    client = FakeClient(['[{"id": 0, "label": "lore"}]'])
    agent = NoiseFilterAgent(client=client)
    agent.classify(messages)

    sent = client.messages.calls[0]["messages"][0]["content"]
    assert content in sent          # literal characters survive into the request...
    assert "\\u201c" not in sent    # ...and were NOT \u-escaped (ensure_ascii=False)
    assert json.loads(sent)[0]["content"] == content  # and round-trips cleanly


# --- id rejoin --------------------------------------------------------------

def test_classify_rejoins_out_of_order_ids():
    messages = [
        make_message("zero"),
        make_message("one"),
        make_message("two"),
    ]
    # Claude returns the labels in a scrambled order; we must map by id, not by
    # position in the response.
    response = (
        '[{"id": 2, "label": "mechanic"}, '
        '{"id": 0, "label": "lore"}, '
        '{"id": 1, "label": "noise"}]'
    )
    agent = make_agent([response])

    result = agent.classify(messages)

    assert result == [
        (messages[0], "lore"),
        (messages[1], "noise"),
        (messages[2], "mechanic"),
    ]


# --- degraded responses -----------------------------------------------------

def test_missing_id_falls_back_to_ambiguous_with_warning(caplog):
    messages = [make_message("a"), make_message("b"), make_message("c")]
    # id 1 is absent from the response.
    response = '[{"id": 0, "label": "lore"}, {"id": 2, "label": "noise"}]'
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.noise_filter"):
        result = agent.classify(messages)

    assert [label for _, label in result] == ["lore", "ambiguous", "noise"]
    assert any(
        "missing id 1" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_invalid_label_coerced_to_ambiguous_with_warning(caplog):
    messages = [make_message("a")]
    response = '[{"id": 0, "label": "garbage"}]'
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.noise_filter"):
        result = agent.classify(messages)

    assert result == [(messages[0], "ambiguous")]
    assert any(
        "invalid label" in r.getMessage() and "garbage" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_unknown_extra_id_ignored_with_warning(caplog):
    messages = [make_message("a"), make_message("b")]
    # id 99 was never sent; it must be ignored, and the real two still labeled.
    response = (
        '[{"id": 0, "label": "lore"}, '
        '{"id": 1, "label": "noise"}, '
        '{"id": 99, "label": "lore"}]'
    )
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.noise_filter"):
        result = agent.classify(messages)

    assert result == [(messages[0], "lore"), (messages[1], "noise")]
    assert any(
        "id 99" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_non_integer_id_is_ignored_never_mislabels_noise(caplog):
    # A JSON float or bool id compares numerically equal to an int (1.0 == 1,
    # False == 0) and hashes the same, so without a type guard it would slip the
    # range check AND collide with a real local id in the {id: label} dict --
    # silently labeling a message 'noise' and dropping its lore forever. Both
    # forms must be ignored, leaving the safe 'ambiguous' fallback. The crucial
    # assertion is that 'noise' never appears.
    messages = [make_message("a"), make_message("b")]
    for bad_response in (
        '[{"id": 1.0, "label": "noise"}]',    # float id would collide with int 1
        '[{"id": false, "label": "noise"}]',  # bool id would collide with int 0
    ):
        agent = make_agent([bad_response])
        with caplog.at_level(logging.WARNING, logger="agents.noise_filter"):
            result = agent.classify(messages)
        labels = [label for _, label in result]
        assert labels == ["ambiguous", "ambiguous"]
        assert "noise" not in labels  # the one unacceptable outcome
        assert any(
            "non-integer id" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )
        caplog.clear()


def test_list_of_non_dict_entries_ignored_then_ambiguous(caplog):
    # A list whose entries aren't objects (a stray scalar or null) must not
    # crash; each bad entry is ignored and every message falls back to ambiguous.
    messages = [make_message("a"), make_message("b")]
    response = '["lore", 42, null]'
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.noise_filter"):
        result = agent.classify(messages)

    assert [label for _, label in result] == ["ambiguous", "ambiguous"]
    assert any(
        "not an object" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_non_list_response_makes_whole_batch_ambiguous_with_warning(caplog):
    messages = [make_message("a"), make_message("b")]
    # A JSON object instead of the expected array.
    response = '{"id": 0, "label": "lore"}'
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.noise_filter"):
        result = agent.classify(messages)

    assert [label for _, label in result] == ["ambiguous", "ambiguous"]
    assert any(
        "expected a JSON array" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_claude_json_error_makes_whole_batch_ambiguous_with_error(caplog):
    messages = [make_message("a"), make_message("b")]
    # Non-JSON text; with max_json_retries=1, call_claude_json raises after one
    # try and _classify_batch must catch it.
    client = FakeClient(["not json at all"])
    agent = NoiseFilterAgent(client=client, max_json_retries=1)

    with caplog.at_level(logging.ERROR, logger="agents.noise_filter"):
        result = agent.classify(messages)

    assert [label for _, label in result] == ["ambiguous", "ambiguous"]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("failed to return valid JSON" in r.getMessage() for r in errors)


# --- empty input ------------------------------------------------------------

def test_empty_input_returns_empty_and_makes_no_calls():
    client = FakeClient(['[{"id": 0, "label": "lore"}]'])
    agent = NoiseFilterAgent(client=client)

    assert agent.classify([]) == []
    assert client.call_count == 0  # short-circuit, no API call


# --- defaults ---------------------------------------------------------------

def test_defaults_are_haiku_and_zero_temperature():
    agent = NoiseFilterAgent(client=FakeClient(["[]"]))
    assert agent.model == "claude-haiku-4-5-20251001"
    assert agent.temperature == 0.0


def test_defaults_reach_the_request():
    messages = [make_message("a")]
    client = FakeClient(['[{"id": 0, "label": "lore"}]'])
    agent = NoiseFilterAgent(client=client)
    agent.classify(messages)

    sent = client.messages.calls[0]
    assert sent["model"] == "claude-haiku-4-5-20251001"
    assert sent["temperature"] == 0.0


def test_explicit_overrides_still_pass_through_kwargs():
    # batch_size is consumed by NoiseFilterAgent; other kwargs reach BaseAgent.
    agent = NoiseFilterAgent(
        client=FakeClient(["[]"]),
        batch_size=10,
        model="claude-sonnet-4-6",
        temperature=0.5,
        max_json_retries=2,
    )
    assert agent.batch_size == 10
    assert agent.model == "claude-sonnet-4-6"  # explicit beats the setdefault
    assert agent.temperature == 0.5
    assert agent.max_json_retries == 2


# --- select_for_extraction --------------------------------------------------

def test_select_for_extraction_keeps_lore_and_ambiguous_in_order():
    m_lore = make_message("lore msg")
    m_mech = make_message("mechanic msg")
    m_noise = make_message("noise msg")
    m_amb = make_message("ambiguous msg")
    classified = [
        (m_lore, "lore"),
        (m_mech, "mechanic"),
        (m_noise, "noise"),
        (m_amb, "ambiguous"),
    ]

    assert select_for_extraction(classified) == [m_lore, m_amb]


def test_select_for_extraction_empty():
    assert select_for_extraction([]) == []


# --- module-level sanity ----------------------------------------------------

def test_valid_labels_constant():
    assert VALID_LABELS == {"lore", "mechanic", "noise", "ambiguous"}
