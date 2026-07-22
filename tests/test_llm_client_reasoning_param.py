"""The reasoning-disable knob for the OpenAI-compat adapter (agents/llm_client.py).

Offline. GLM-5.2 over OpenRouter reasons for thousands of tokens by default, which blows the
token budget mid-thought (truncation -> retries) and leaks chain-of-thought into the response.
`WIKITIZER_OPENAI_REASONING` lets us send OpenRouter's reasoning-disable payload as `extra_body`
on every GLM call. These pin the env->payload mapping and the adapter wiring, and lock the
backward-compat guarantee that an UNSET knob sends exactly today's request (no `extra_body`).
"""

import pytest

from agents.llm_client import AnthropicCompatClient, _reasoning_extra_body


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("WIKITIZER_OPENAI_REASONING", raising=False)


# --- _reasoning_extra_body: env -> payload ---------------------------------- #

def test_unset_is_none():
    assert _reasoning_extra_body() is None


def test_empty_is_none(monkeypatch):
    monkeypatch.setenv("WIKITIZER_OPENAI_REASONING", "   ")
    assert _reasoning_extra_body() is None


@pytest.mark.parametrize("val", ["off", "OFF", "false", "0", "no", "disable", "disabled"])
def test_off_disables_thinking_both_ways(monkeypatch, val):
    monkeypatch.setenv("WIKITIZER_OPENAI_REASONING", val)
    assert _reasoning_extra_body() == {
        "reasoning": {"enabled": False},
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.parametrize("val", ["low", "medium", "high", "HIGH"])
def test_effort_levels(monkeypatch, val):
    monkeypatch.setenv("WIKITIZER_OPENAI_REASONING", val)
    assert _reasoning_extra_body() == {"reasoning": {"effort": val.lower()}}


def test_raw_json_is_used_verbatim(monkeypatch):
    monkeypatch.setenv("WIKITIZER_OPENAI_REASONING", '{"chat_template_kwargs":{"enable_thinking":false}}')
    assert _reasoning_extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_malformed_json_is_ignored_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("WIKITIZER_OPENAI_REASONING", '{not valid json')
    import logging
    with caplog.at_level(logging.WARNING):
        assert _reasoning_extra_body() is None
    assert any("did not parse" in r.getMessage() for r in caplog.records)


def test_unrecognized_value_is_ignored_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("WIKITIZER_OPENAI_REASONING", "banana")
    import logging
    with caplog.at_level(logging.WARNING):
        assert _reasoning_extra_body() is None
    assert any("not recognized" in r.getMessage() for r in caplog.records)


# --- adapter wiring: extra_body reaches chat.completions.create ------------- #

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Completion("ok")


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class FakeOpenAI:
    def __init__(self):
        self.chat = _Chat()


def _call(extra_body):
    inner = FakeOpenAI()
    client = AnthropicCompatClient(inner, extra_body=extra_body)
    client.messages.create(model="z-ai/glm-5.2", max_tokens=100, temperature=0.2,
                           system=[], messages=[{"role": "user", "content": "hi"}])
    return inner.chat.completions.calls[0]


def test_extra_body_is_forwarded_when_set():
    payload = {"reasoning": {"enabled": False}}
    sent = _call(payload)
    assert sent["extra_body"] == payload


def test_extra_body_absent_when_none():
    # Backward-compat: the None path must send exactly today's request, no extra_body key.
    sent = _call(None)
    assert "extra_body" not in sent
