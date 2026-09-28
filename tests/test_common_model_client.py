"""Tests for evals/common/model_client.py -- the thin OpenAI-compat adapter. Offline.

The client construction is network-free (openai.OpenAI is lazy), so we can build a real one
and swap its inner `_client` for a fake to exercise the FIX #6 response_format fallback
without any network. build_model_client's env logic is tested directly.
"""

import httpx
import pytest

from evals.common.model_client import (
    OpenAICompatModelClient,
    build_model_client,
    strip_think,
)


# --- strip_think ------------------------------------------------------------ #

def test_strip_think_removes_leading_block():
    assert strip_think("<think>reasoning...</think>\n{\"x\":1}") == '{"x":1}'
    assert strip_think("  <think>a\nb</think>  hello") == "hello"


def test_strip_think_leaves_plain_text():
    assert strip_think('{"verdicts": []}') == '{"verdicts": []}'
    assert strip_think("") == ""


# --- build_model_client (env logic) ---------------------------------------- #

def test_build_returns_none_without_credentials():
    assert build_model_client({}) is None
    assert build_model_client({"LLM_OPENAI_BASE_URL": "https://x"}) is None   # key missing
    assert build_model_client({"LLM_OPENAI_API_KEY": "k"}) is None            # base_url missing


def test_build_reads_env_defaults_and_overrides():
    c = build_model_client({"LLM_OPENAI_BASE_URL": "https://x/api/v1",
                            "LLM_OPENAI_API_KEY": "k"})
    assert c is not None
    assert c.model == "qwen/qwen3-8b" and c.temperature == 0.6 and c.thinking is False
    c2 = build_model_client({"LLM_OPENAI_BASE_URL": "https://x/api/v1",
                             "LLM_OPENAI_API_KEY": "k",
                             "WIKITIZER_FC_MODEL": "meta/other", "WIKITIZER_FC_TEMPERATURE": "0.3"})
    assert c2.model == "meta/other" and c2.temperature == 0.3


def test_build_tolerates_bad_temperature():
    c = build_model_client({"LLM_OPENAI_BASE_URL": "https://x", "LLM_OPENAI_API_KEY": "k",
                            "WIKITIZER_FC_TEMPERATURE": "not-a-number"})
    assert c.temperature == 0.6                       # falls back to the default


# --- complete: content read + FIX #6 fallback ------------------------------- #

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeCompletions:
    """Records each create() call; raises BadRequestError whenever response_format is sent."""

    def __init__(self, openai_module, content="the answer"):
        self._openai = openai_module
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            resp = httpx.Response(400, request=httpx.Request("POST", "https://x/api"))
            raise self._openai.BadRequestError("response_format unsupported",
                                               response=resp, body=None)
        return _Resp(self._content)


class _FakeInner:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


def _client_with_fake(content="the answer"):
    c = OpenAICompatModelClient(model="qwen/qwen3-8b", temperature=0.6,
                                base_url="https://x/api/v1", api_key="k")
    fake = _FakeCompletions(c._openai, content=content)
    c._client = _FakeInner(fake)
    return c, fake


def test_complete_falls_back_when_response_format_rejected():
    # FIX #6: first attempt (with response_format) 400s -> retry once WITHOUT it.
    c, fake = _client_with_fake()
    out = c.complete("system", "user")
    assert out == "the answer"
    assert len(fake.calls) == 2
    assert "response_format" in fake.calls[0] and "response_format" not in fake.calls[1]


def test_complete_strips_think_from_content():
    c, _ = _client_with_fake(content="<think>hmm</think>{\"verdicts\":[]}")
    assert c.complete("s", "u") == '{"verdicts":[]}'


def test_complete_sends_thinking_off_extra_body():
    c, fake = _client_with_fake()
    c.complete("s", "u")
    # Both attempts carry the thinking-off extra_body.
    assert fake.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
