"""Tests for agents/locations_extractor.py (Phase 3.3: the LocationsExtractor).

Like test_base_agent.py / test_noise_filter.py, every test injects a fake
Anthropic client, so the suite makes ZERO real API calls and needs no API key.
The fake mimics the real response shape (an object with a ``.content`` list of
text blocks) and records the kwargs of every ``messages.create`` call, so we can
assert both how many batches we sent and the request params we sent them with.

The fake-client classes are copied from test_base_agent.py / test_noise_filter.py
to keep this file self-contained, matching the established per-phase pattern.

The pure verbatim helpers (``_normalize_for_match`` / ``_quote_is_verbatim``)
moved up to ``agents/base_extractor.py`` in the Step 2 lift, so their direct unit
tests live in test_base_extractor.py; here we exercise them end-to-end through
the agent (the verbatim-drop / multi-line-tolerate / verify-off cases below).

Coverage:
  * happy path, single batch: two locations with correct name/aliases/details.
  * quote-metadata attachment: speaker + source_file pulled from the Message,
    never from Claude.
  * batching: ceil(5/2) == 3 calls, batch-local ids reset per batch, id->content
    binding right in batch 2.
  * ensure_ascii=False: curly quotes / em-dash / emoji survive as literal chars.
  * verbatim check (on): drops a non-matching quote (+warning), tolerates a
    multi-line message matched by a flat quote.
  * verify_quotes=False: keeps the non-matching quote (flag truly disables it).
  * malformed entry (missing name) skipped; non-dict entry ignored; sibling kept.
  * unknown source_id (out of range) -> detail dropped, other details kept.
  * bool source_id -> dropped, never silently indexed as 0/1.
  * ClaudeJSONError on batch 1, valid batch 2 -> batch 2 survives, error logged.
  * empty input -> [] with zero API calls.
  * defaults reach the request: Sonnet model + max_tokens 8192.
"""

import json
import logging
from datetime import datetime

import pytest

from agents.locations_extractor import LocationsExtractor
from models.lore import Location, Quote
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
    """A minimal valid Message; only ``content`` (and, for attachment tests, the
    sender/source_file metadata) matters to the extractor."""
    return Message(
        sender=sender,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        content=content,
        source_file=source_file,
    )


def make_agent(responses, **kwargs):
    return LocationsExtractor(client=FakeClient(responses), **kwargs)


# --- happy path -------------------------------------------------------------

def test_extract_happy_path_single_batch():
    messages = [
        make_message("The Great Well. The Pond. Lake Mundi."),
        make_message("Lake Mundi is a massive central lake divided into three rings"),
        make_message(
            "Almost every country southeast of the Cloud Mountains is under the "
            "control of the Krieger Imperium"
        ),
    ]
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": ["The Great Well", "The Pond"],
            "details": [{
                "detail": "A massive central lake divided into three rings",
                "quote": "Lake Mundi is a massive central lake divided into three rings",
                "source_id": 1,
            }],
        },
        {
            "name": "Cloud Mountains",
            "aliases": [],
            "details": [{
                "detail": "Countries to their southeast are controlled by the Krieger Imperium",
                "quote": (
                    "Almost every country southeast of the Cloud Mountains is under the "
                    "control of the Krieger Imperium"
                ),
                "source_id": 2,
            }],
        },
    ])
    client = FakeClient([response])
    agent = LocationsExtractor(client=client)

    result = agent.extract(messages)

    assert client.call_count == 1
    assert [loc.name for loc in result] == ["Lake Mundi", "Cloud Mountains"]
    assert result[0].aliases == ["The Great Well", "The Pond"]
    assert result[0].details == ["A massive central lake divided into three rings"]
    assert result[1].aliases == []
    assert result[1].details == [
        "Countries to their southeast are controlled by the Krieger Imperium"
    ]
    # one supporting quote each, matching its source message verbatim -> kept
    assert len(result[0].supporting_quotes) == 1
    assert len(result[1].supporting_quotes) == 1
    assert all(isinstance(loc, Location) for loc in result)


# --- quote-metadata attachment (the centerpiece) ----------------------------

def test_quote_metadata_pulled_from_message_not_claude():
    quote = "Lake Mundi is a massive central lake divided into three rings"
    messages = [make_message(quote, sender="Matt", source_file="dndgroup.txt")]
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": [],
            "details": [{
                "detail": "A massive central lake divided into three rings",
                "quote": quote,
                "source_id": 0,
            }],
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
    # batch_size=2 over 5 -> ceil(5/2) == 3 batches; responses can be empty (we're
    # asserting the requests we SENT, not the locations we got back).
    client = FakeClient(["[]"])
    agent = LocationsExtractor(client=client, batch_size=2)

    agent.extract(messages)

    assert client.call_count == 3
    first_payload = json.loads(client.messages.calls[0]["messages"][0]["content"])
    second_payload = json.loads(client.messages.calls[1]["messages"][0]["content"])
    third_payload = json.loads(client.messages.calls[2]["messages"][0]["content"])
    assert [o["id"] for o in first_payload] == [0, 1]
    assert [o["content"] for o in first_payload] == ["msg 0", "msg 1"]
    # batch 2's ids RESET to [0, 1] (not [2, 3]) and carry the right messages.
    assert [o["id"] for o in second_payload] == [0, 1]
    assert [o["content"] for o in second_payload] == ["msg 2", "msg 3"]
    assert [o["id"] for o in third_payload] == [0]  # last batch has one message
    assert third_payload[0]["content"] == "msg 4"


def test_payload_preserves_non_ascii_content_via_ensure_ascii_false():
    # Chat content has curly quotes, em-dashes, and emoji; the brief mandates
    # json.dumps(..., ensure_ascii=False) so the literal characters reach the
    # request, not \uXXXX escapes. ASCII-only content would pass either way, so
    # pin the non-ASCII case explicitly.
    content = "Lake Mundi — the Great Well “The Pond” \U0001f30a"  # — “ ” 🌊
    messages = [make_message(content)]
    client = FakeClient(["[]"])
    agent = LocationsExtractor(client=client)
    agent.extract(messages)

    sent = client.messages.calls[0]["messages"][0]["content"]
    assert content in sent             # literal characters survive...
    assert "\\u201c" not in sent       # ...and were NOT \u-escaped
    assert json.loads(sent)[0]["content"] == content  # round-trips cleanly


# --- verbatim verification --------------------------------------------------

def test_verbatim_check_drops_non_matching_quote_keeps_siblings(caplog):
    messages = [make_message("Lake Mundi is a massive central lake. It has three rings.")]
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": [],
            "details": [
                {  # appears verbatim -> kept
                    "detail": "A massive central lake",
                    "quote": "Lake Mundi is a massive central lake",
                    "source_id": 0,
                },
                {  # NOT in the message -> dropped
                    "detail": "Home of dragons",
                    "quote": "Lake Mundi is full of dragons",
                    "source_id": 0,
                },
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["A massive central lake"]   # only the good one
    assert len(result[0].supporting_quotes) == 1
    assert any(
        "verbatim" in r.getMessage() and "id 0" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_verbatim_check_tolerates_multiline_message():
    # The cited message spans several lines; Claude's flat one-line quote matches
    # after normalizing -> the detail is KEPT (no false alarm).
    messages = [make_message("Lake Mundi is a massive\ncentral lake divided\ninto three rings")]
    flat_quote = "Lake Mundi is a massive central lake divided into three rings"
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": [],
            "details": [{"detail": "A central lake", "quote": flat_quote, "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["A central lake"]
    assert result[0].supporting_quotes[0].text == flat_quote  # stored as given


def test_verify_quotes_false_keeps_non_matching_quote():
    messages = [make_message("Lake Mundi is real")]
    response = json.dumps([
        {
            "name": "Lake Mundi",
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
    assert result[0].details == ["Invented fact"]            # kept, check disabled
    assert result[0].supporting_quotes[0].text == "totally invented text not in the message"


def test_verbatim_check_drops_empty_quote(caplog):
    # An empty (or blank) quote normalizes to "" -- a substring of everything --
    # so without the guard a fabricated detail with a blank quote would survive.
    # It must be dropped (verify on) just like any other non-matching quote.
    messages = [make_message("Lake Mundi is huge")]
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": [],
            "details": [
                {"detail": "It is huge", "quote": "Lake Mundi is huge", "source_id": 0},  # kept
                {"detail": "Fabricated", "quote": "", "source_id": 0},                      # blank -> dropped
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["It is huge"]               # the fabricated one is gone
    assert all(q.text for q in result[0].supporting_quotes)  # no empty-text quote stored


def test_identical_quotes_deduped_but_details_kept():
    # Two distinct details citing the SAME verbatim quote: both details survive,
    # but the identical Quote is stored once (dedup via 'q not in quotes_out').
    messages = [make_message("Lake Mundi is huge")]
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": [],
            "details": [
                {"detail": "It is huge", "quote": "Lake Mundi is huge", "source_id": 0},
                {"detail": "Notably large", "quote": "Lake Mundi is huge", "source_id": 0},
            ],
        },
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["It is huge", "Notably large"]  # details NOT deduped
    assert len(result[0].supporting_quotes) == 1                  # identical quote deduped


# --- malformed responses ----------------------------------------------------

def test_missing_name_entry_skipped_siblings_kept(caplog):
    messages = [make_message("Gol is a city of spires")]
    response = json.dumps([
        {"aliases": [], "details": []},  # no name -> skipped
        {
            "name": "Gol",
            "aliases": [],
            "details": [{"detail": "A city of spires", "quote": "Gol is a city of spires", "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert [loc.name for loc in result] == ["Gol"]
    # Anchor on the distinctive phrase, not the bare 4-letter substring "name"
    # (which also appears in unrelated warnings like the validation one).
    assert any(
        "no usable name" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_non_dict_entry_ignored_siblings_kept(caplog):
    messages = [make_message("Gol is a city of spires")]
    response = json.dumps([
        "just a string, not an object",
        {
            "name": "Gol",
            "aliases": [],
            "details": [{"detail": "A city", "quote": "Gol is a city of spires", "source_id": 0}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert [loc.name for loc in result] == ["Gol"]
    assert any(
        "not an object" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_non_list_details_coerced_to_empty_no_crash(caplog):
    # A non-list 'details' (here an int) would crash `for d in raw_details`; the
    # never-crash rule coerces it to empty and emits the named location with no
    # details rather than killing the batch.
    messages = [make_message("Gol is a city")]
    response = '[{"name": "Gol", "aliases": [], "details": 5}]'
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert [loc.name for loc in result] == ["Gol"]
    assert result[0].details == []
    assert result[0].supporting_quotes == []
    assert any(
        "non-list" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_detail_with_non_string_detail_or_quote_dropped(caplog):
    # A detail dict whose 'detail'/'quote' is missing or non-string is dropped at
    # the detail level (a distinct branch from the non-dict-detail guard); the
    # valid sibling detail survives.
    messages = [make_message("Lake Mundi is huge")]
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": [],
            "details": [
                {"detail": "It is huge", "quote": "Lake Mundi is huge", "source_id": 0},  # valid
                {"detail": 123, "quote": "Lake Mundi is huge", "source_id": 0},            # non-str detail
                {"quote": "Lake Mundi is huge", "source_id": 0},                            # missing detail
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["It is huge"]   # only the well-formed detail
    assert any(
        "missing a string" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_aliases_sanitizers_drop_non_list_and_non_string_elements():
    # Two defensive guards on aliases: a stray non-list value -> [] (no one-char
    # splat), and non-string elements filtered out of a real list.
    messages = [make_message("Gol is a city"), make_message("Eglon is a fortress")]
    response = json.dumps([
        {"name": "Gol", "aliases": "Bob", "details": []},                  # non-list -> []
        {"name": "Eglon", "aliases": ["The Keep", 7, None, "Old Eglon"], "details": []},  # filter
    ])
    agent = make_agent([response])

    result = agent.extract(messages)

    by_name = {loc.name: loc for loc in result}
    assert by_name["Gol"].aliases == []                          # not ['B', 'o', 'b']
    assert by_name["Eglon"].aliases == ["The Keep", "Old Eglon"]  # ints/None dropped


def test_out_of_range_source_id_dropped_other_details_kept(caplog):
    messages = [make_message("Lake Mundi is huge")]
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": [],
            "details": [
                {"detail": "It is huge", "quote": "Lake Mundi is huge", "source_id": 0},   # valid
                {"detail": "Out of range", "quote": "Lake Mundi is huge", "source_id": 5}, # bad id
            ],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    assert result[0].details == ["It is huge"]   # the in-range detail survives
    assert any(
        "out of range" in r.getMessage() and "5" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_bool_source_id_dropped_never_indexed_as_0_or_1(caplog):
    # Two messages so batch[1] exists: a bool source_id of `true` would index as
    # batch[1] (== messages[1]) and, because its quote happens to match message 1,
    # would silently attach the WRONG message's speaker. The type guard must drop
    # it instead.
    messages = [make_message("Lake Mundi exists"), make_message("Cloud Mountains exist")]
    response = json.dumps([
        {
            "name": "Lake Mundi",
            "aliases": [],
            "details": [{"detail": "Mislabeled", "quote": "Cloud Mountains exist", "source_id": True}],
        },
    ])
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert len(result) == 1
    # Named but no surviving detail -> empty details/quotes, NOT a wrong attachment.
    assert result[0].name == "Lake Mundi"
    assert result[0].details == []
    assert result[0].supporting_quotes == []
    assert any(
        "non-integer source_id" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_claude_json_error_on_batch1_valid_batch2(caplog):
    # batch_size=1 over 2 messages: batch 1 returns junk (raises ClaudeJSONError
    # with max_json_retries=1), batch 2 returns a valid location. The run must
    # survive: batch 2's location comes back, batch 1 logged at error.
    messages = [make_message("garbage in"), make_message("Gol is a city")]
    valid = json.dumps([
        {"name": "Gol", "aliases": [], "details": [
            {"detail": "A city", "quote": "Gol is a city", "source_id": 0}
        ]},
    ])
    client = FakeClient(["not json at all", valid])
    agent = LocationsExtractor(client=client, batch_size=1, max_json_retries=1)

    with caplog.at_level(logging.ERROR, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert [loc.name for loc in result] == ["Gol"]
    assert any(
        "failed to return valid JSON" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR
    )


def test_non_list_response_skips_batch_with_warning(caplog):
    messages = [make_message("Lake Mundi is huge")]
    response = '{"name": "Lake Mundi"}'  # an object, not the expected array
    agent = make_agent([response])

    with caplog.at_level(logging.WARNING, logger="agents.locations_extractor"):
        result = agent.extract(messages)

    assert result == []
    assert any(
        "expected a JSON array" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


# --- empty input ------------------------------------------------------------

def test_empty_input_returns_empty_and_makes_no_calls():
    client = FakeClient(["[]"])
    agent = LocationsExtractor(client=client)

    assert agent.extract([]) == []
    assert client.call_count == 0


# --- defaults / construction ------------------------------------------------

def test_defaults_are_sonnet_and_big_max_tokens():
    agent = LocationsExtractor(client=FakeClient(["[]"]))
    assert agent.model == "claude-sonnet-4-6"   # NOT overridden to Haiku
    assert agent.temperature == 0.2             # NOT overridden to 0
    assert agent.max_tokens == 8192
    assert agent.batch_size == 20
    assert agent.verify_quotes is True


def test_defaults_reach_the_request():
    messages = [make_message("a")]
    client = FakeClient(["[]"])
    agent = LocationsExtractor(client=client)
    agent.extract(messages)

    sent = client.messages.calls[0]
    assert sent["model"] == "claude-sonnet-4-6"
    assert sent["max_tokens"] == 8192
    assert sent["temperature"] == 0.2   # NOT overridden to 0 like the noise filter


def test_explicit_max_tokens_beats_setdefault():
    agent = LocationsExtractor(client=FakeClient(["[]"]), max_tokens=2048)
    assert agent.max_tokens == 2048  # explicit wins over the 8192 setdefault


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        LocationsExtractor(client=FakeClient(["[]"]), batch_size=0)
