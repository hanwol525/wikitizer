"""Tests for agents/base_extractor.py (Phase 3.3 Step 2: the BaseExtractor spine).

Two groups:
  * the pure verbatim helpers ``_normalize_for_match`` / ``_quote_is_verbatim``
    (no client needed -- called directly), lifted here from the Step 1 locations
    tests when the helpers moved up into base_extractor.py.
  * the base seam itself: ``_resolve_quote`` attaches the right message metadata
    and honors ``verify_quotes``, and rejects a bool / out-of-range source_id
    with ``None``; and instantiating ``BaseExtractor`` and calling its
    ``_build_entry`` stub raises ``NotImplementedError`` (the base isn't meant to
    run alone).

A tiny fake client is injected only so ``BaseExtractor`` can be constructed
without touching the network; ``_resolve_quote`` never calls Claude, so these
make zero API calls.
"""

import logging
from datetime import datetime

import pytest

from agents.base_extractor import (
    BaseExtractor,
    _normalize_for_match,
    _quote_is_verbatim,
)
from models.lore import Quote
from models.message import Message


# --- minimal fake client (only needed to construct BaseExtractor) -----------

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


def make_message(content, sender="Matt", source_file="group.txt"):
    return Message(
        sender=sender,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        content=content,
        source_file=source_file,
    )


def make_base_extractor(**kwargs):
    return BaseExtractor(client=FakeClient(["[]"]), **kwargs)


# --- pure verbatim helpers (moved from test_locations_extractor.py) ---------

def test_quote_is_verbatim_exact_match():
    msg = "Lake Mundi is a massive central lake divided into three rings"
    assert _quote_is_verbatim(msg, msg) is True


def test_quote_is_verbatim_tolerates_multiline_message():
    # The message has real newlines; Claude's quote is one flat line. Same words,
    # different whitespace -> must still match once whitespace is squeezed.
    message = "Lake Mundi is a massive\ncentral lake divided\ninto three rings"
    flat_quote = "Lake Mundi is a massive central lake divided into three rings"
    assert _quote_is_verbatim(flat_quote, message) is True


def test_quote_is_verbatim_folds_curly_vs_straight_punctuation():
    curly_msg = "He said “Lake Mundi” is the world’s heart"
    straight_quote = "He said \"Lake Mundi\" is the world's heart"
    # straight quote vs curly message...
    assert _quote_is_verbatim(straight_quote, curly_msg) is True
    # ...and the reverse: curly quote vs straight message.
    assert _quote_is_verbatim(curly_msg, straight_quote) is True


def test_quote_is_verbatim_false_on_reword():
    message = "Lake Mundi is a massive central lake"
    reworded = "Lake Mundi is a huge central lake"  # "huge" != "massive"
    assert _quote_is_verbatim(reworded, message) is False


def test_quote_is_verbatim_false_on_invented_text():
    message = "Lake Mundi is a massive central lake"
    invented = "Dragons rule the skies above Gol"
    assert _quote_is_verbatim(invented, message) is False


def test_quote_is_verbatim_false_on_empty_or_blank_quote():
    # "" normalizes to "" which is a substring of everything; an empty/blank quote
    # must be treated as NOT verbatim, or a fabricated detail with a blank quote
    # would defeat the anti-hallucination guard and reach the wiki.
    message = "Lake Mundi is a massive central lake"
    assert _quote_is_verbatim("", message) is False
    assert _quote_is_verbatim("   ", message) is False
    assert _quote_is_verbatim("\n\t", message) is False


def test_normalize_for_match_squeezes_and_folds():
    # Collapse whitespace, fold curly, strip ends, but do NOT lowercase.
    assert _normalize_for_match("  a\n\t b  ") == "a b"
    assert _normalize_for_match("don’t") == "don't"
    assert _normalize_for_match("Lake") == "Lake"  # case preserved


# --- _resolve_quote: metadata attachment + verify flag ----------------------

def test_resolve_quote_attaches_message_metadata():
    batch = [make_message("Lake Mundi is huge", sender="Matt", source_file="dndgroup.txt")]
    agent = make_base_extractor()

    q = agent._resolve_quote("Lake Mundi is huge", 0, batch)

    assert isinstance(q, Quote)
    assert q.text == "Lake Mundi is huge"
    assert q.speaker == "Matt"            # from the Message, not the caller
    assert q.source_file == "dndgroup.txt"


def test_resolve_quote_verify_on_rejects_non_matching_quote(caplog):
    batch = [make_message("Lake Mundi is huge")]
    agent = make_base_extractor(verify_quotes=True)

    with caplog.at_level(logging.WARNING, logger="agents.base_extractor"):
        q = agent._resolve_quote("not in the message", 0, batch)

    assert q is None
    assert any("verbatim" in r.getMessage() for r in caplog.records)


def test_resolve_quote_verify_off_keeps_non_matching_quote():
    batch = [make_message("Lake Mundi is huge")]
    agent = make_base_extractor(verify_quotes=False)

    q = agent._resolve_quote("not in the message", 0, batch)

    assert isinstance(q, Quote)
    assert q.text == "not in the message"  # check disabled -> kept as-is


def test_resolve_quote_rejects_bool_source_id(caplog):
    # batch[1] exists, so an unguarded bool `True` would index batch[1]; the type
    # guard must return None instead of silently attaching the wrong message.
    batch = [make_message("zero"), make_message("one")]
    agent = make_base_extractor(verify_quotes=False)

    with caplog.at_level(logging.WARNING, logger="agents.base_extractor"):
        q = agent._resolve_quote("one", True, batch)

    assert q is None
    assert any("non-integer source_id" in r.getMessage() for r in caplog.records)


def test_resolve_quote_rejects_out_of_range_source_id(caplog):
    batch = [make_message("zero")]
    agent = make_base_extractor(verify_quotes=False)

    with caplog.at_level(logging.WARNING, logger="agents.base_extractor"):
        q = agent._resolve_quote("zero", 5, batch)

    assert q is None
    assert any("out of range" in r.getMessage() for r in caplog.records)


# --- the _build_entry seam must be filled in by subclasses -------------------

def test_base_build_entry_raises_not_implemented():
    agent = make_base_extractor()
    with pytest.raises(NotImplementedError):
        agent._build_entry({"name": "Lake Mundi"}, [make_message("Lake Mundi is huge")])


# --- construction defaults carried by the base ------------------------------

def test_base_defaults_max_tokens_and_named_params():
    agent = make_base_extractor()
    assert agent.max_tokens == 8192     # setdefault carried by BaseExtractor
    assert agent.model == "claude-sonnet-4-6"  # NOT overridden to Haiku
    assert agent.temperature == 0.2
    assert agent.batch_size == 20
    assert agent.verify_quotes is True


def test_base_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        make_base_extractor(batch_size=0)
