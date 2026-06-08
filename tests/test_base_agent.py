"""Tests for agents/base.py (Phase 3.1: the shared BaseAgent class).

Every test injects a fake Anthropic client, so the suite makes ZERO real API
calls and needs no API key. The fake mimics the real response shape -- an object
with a ``.content`` list of blocks, each block carrying ``.type`` and (for text
blocks) ``.text`` -- and records the kwargs of every ``messages.create`` call so
we can assert both the request we sent and how many times we sent it.

Coverage:
  * strip_code_fences: no fence, ```json fence, bare ``` fence, surrounding
    whitespace -- plus the don't-mangle edge cases (unclosed fence, inline
    backticks).
  * call_claude: returns the text; concatenates multiple text blocks; ignores
    non-text blocks; sends the right request params.
  * call_claude_json: parses a dict and a list; strips a fence then parses;
    retries-then-succeeds (proving the retry fired via call count); and raises
    after exactly max_json_retries attempts with the last raw text attached.
  * the dev-only on-disk cache: a repeat call is served from disk, not the API.
"""

import json
import logging

import pytest

from agents.base import BaseAgent, ClaudeJSONError, strip_code_fences


# --- fake client ------------------------------------------------------------
# Shaped exactly like the bits of the real SDK response BaseAgent touches.

class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeNonTextBlock:
    """A non-text block (e.g. thinking/tool_use). Deliberately has NO .text, so
    if BaseAgent ever forgot to filter on type=="text" it would crash here."""

    def __init__(self, block_type="thinking"):
        self.type = block_type


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


def make_agent(responses, **kwargs):
    return BaseAgent(client=FakeClient(responses), **kwargs)


# --- strip_code_fences ------------------------------------------------------

def test_strip_fences_plain_json_unchanged():
    text = '{"name": "Lake Mundi"}'
    assert strip_code_fences(text) == text


def test_strip_fences_json_language_hint():
    assert strip_code_fences('```json\n{"name": "Lake Mundi"}\n```') == '{"name": "Lake Mundi"}'


def test_strip_fences_bare_fence():
    assert strip_code_fences('```\n{"name": "Lake Mundi"}\n```') == '{"name": "Lake Mundi"}'


def test_strip_fences_with_surrounding_whitespace():
    fenced = '  \n```json\n{"name": "Lake Mundi"}\n```\n  '
    assert strip_code_fences(fenced) == '{"name": "Lake Mundi"}'


def test_strip_fences_unclosed_fence_left_unchanged():
    # No trailing fence line -> not a clean wrapper; don't mangle it.
    text = '```json\n{"name": "Lake Mundi"}'
    assert strip_code_fences(text) == text


def test_strip_fences_inline_backticks_left_unchanged():
    # Backticks mid-string (not a leading fence) are ordinary content.
    text = 'the spell is `fireball`'
    assert strip_code_fences(text) == text


# --- call_claude ------------------------------------------------------------

def test_call_claude_returns_text():
    agent = make_agent(["the royal family is human"])
    assert agent.call_claude("sys", "usr") == "the royal family is human"


def test_call_claude_concatenates_multiple_text_blocks():
    agent = make_agent([[FakeTextBlock("Lake "), FakeTextBlock("Mundi")]])
    assert agent.call_claude("sys", "usr") == "Lake Mundi"


def test_call_claude_ignores_non_text_blocks():
    blocks = [FakeTextBlock("Lake "), FakeNonTextBlock(), FakeTextBlock("Mundi")]
    agent = make_agent([blocks])
    assert agent.call_claude("sys", "usr") == "Lake Mundi"


def test_call_claude_sends_expected_request_params():
    client = FakeClient(["ok"])
    agent = BaseAgent(client=client, model="claude-sonnet-4-6", temperature=0.2, max_tokens=4096)
    agent.call_claude("you are a lore extractor", "extract from: Lake Mundi is huge")

    assert client.call_count == 1
    sent = client.messages.calls[0]
    assert sent["model"] == "claude-sonnet-4-6"
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 4096
    assert sent["system"] == "you are a lore extractor"
    assert sent["messages"] == [
        {"role": "user", "content": "extract from: Lake Mundi is huge"}
    ]


# --- call_claude_json -------------------------------------------------------

def test_call_claude_json_parses_dict():
    agent = make_agent(['{"name": "Lake Mundi", "details": ["central lake"]}'])
    assert agent.call_claude_json("sys", "usr") == {
        "name": "Lake Mundi",
        "details": ["central lake"],
    }


def test_call_claude_json_parses_list():
    # The extractors return JSON lists, so a top-level array must parse too.
    agent = make_agent(['[{"name": "Lake Mundi"}, {"name": "Gol"}]'])
    assert agent.call_claude_json("sys", "usr") == [
        {"name": "Lake Mundi"},
        {"name": "Gol"},
    ]


def test_call_claude_json_strips_fence_then_parses():
    agent = make_agent(['```json\n{"name": "Lake Mundi"}\n```'])
    assert agent.call_claude_json("sys", "usr") == {"name": "Lake Mundi"}


def test_call_claude_json_retries_then_succeeds():
    # Junk first, valid JSON second: the result is correct AND the client was
    # called exactly twice, which is only possible if the retry fired.
    client = FakeClient(["not json at all", '{"ok": true}'])
    agent = BaseAgent(client=client)
    assert agent.call_claude_json("sys", "usr") == {"ok": True}
    assert client.call_count == 2


def test_call_claude_json_raises_after_exhausting_retries():
    client = FakeClient(["still not json"])  # one spec -> returned every call
    agent = BaseAgent(client=client, max_json_retries=3)
    with pytest.raises(ClaudeJSONError) as excinfo:
        agent.call_claude_json("sys", "usr")
    # Tried exactly max_json_retries times...
    assert client.call_count == 3
    # ...and the raised error carries the last raw response for debugging.
    assert excinfo.value.raw == "still not json"
    assert "still not json" in str(excinfo.value)


def test_call_claude_json_respects_custom_retry_count():
    client = FakeClient(["nope"])
    agent = BaseAgent(client=client, max_json_retries=2)
    with pytest.raises(ClaudeJSONError):
        agent.call_claude_json("sys", "usr")
    assert client.call_count == 2


def test_parse_retry_logs_a_warning_per_attempt(caplog):
    # Brief items 3 & 4: each parse-retry must emit a warning *via logging* (not
    # print). One WARNING record per failed attempt -> max_json_retries total
    # when every attempt fails. Guards against the warning being dropped or
    # silently downgraded to print().
    client = FakeClient(["still not json"])
    agent = BaseAgent(client=client, max_json_retries=3)
    with caplog.at_level(logging.WARNING, logger="agents.base"):
        with pytest.raises(ClaudeJSONError):
            agent.call_claude_json("sys", "usr")
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "invalid JSON" in r.getMessage()
    ]
    assert len(warnings) == 3


# --- dev-only response cache (lives in call_claude_json) --------------------

def test_cache_serves_repeat_call_from_disk(tmp_path):
    # With one spec, the fake would keep "answering"; the cache should mean the
    # second identical call never reaches the client at all.
    client = FakeClient(['{"cached": true}'])
    agent = BaseAgent(client=client, cache=True, cache_dir=str(tmp_path / ".llm_cache"))

    first = agent.call_claude_json("sys", "usr")
    second = agent.call_claude_json("sys", "usr")

    assert first == second == {"cached": True}
    assert client.call_count == 1  # second call was served from disk


def test_cache_distinguishes_different_prompts(tmp_path):
    client = FakeClient(['{"a": 1}', '{"b": 2}'])
    agent = BaseAgent(client=client, cache=True, cache_dir=str(tmp_path / ".llm_cache"))

    assert agent.call_claude_json("sys", "first") == {"a": 1}
    assert agent.call_claude_json("sys", "second") == {"b": 2}  # different key -> new call
    assert client.call_count == 2


def test_cache_off_by_default(tmp_path):
    # Default agent (no cache flag) hits the client every time.
    client = FakeClient(['{"x": 1}'])
    agent = BaseAgent(client=client)
    agent.call_claude_json("sys", "usr")
    agent.call_claude_json("sys", "usr")
    assert client.call_count == 2


def test_cache_does_not_poison_the_json_retry(tmp_path):
    # Regression guard for the cache+retry interaction: a malformed first
    # response must NOT be cached, or the retry would just replay it from disk
    # and never re-ask. With the cache ON, junk-then-valid must still succeed by
    # genuinely calling Claude twice.
    client = FakeClient(["not json", '{"ok": true}'])
    agent = BaseAgent(client=client, cache=True, cache_dir=str(tmp_path / ".llm_cache"))

    assert agent.call_claude_json("sys", "usr") == {"ok": True}
    assert client.call_count == 2  # retry actually re-asked, wasn't served the junk
