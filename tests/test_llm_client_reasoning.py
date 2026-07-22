"""Robust response read in the OpenAI-compat adapter (agents/llm_client.py).

Offline. GLM-5.2 (and reasoning models over OpenRouter) can leave `message.content` empty and
carry the answer in a `reasoning_content` / `reasoning` field, or exhaust the token budget on
reasoning (`finish_reason="length"`, `content=None`). The old adapter read only
`content or ""`, silently yielding "" -> a 3-attempt JSON retry storm. `_extract_text` now falls
back to the reasoning fields and, when it's STILL empty, logs the `finish_reason` so a real
truncation is visible. These tests pin all four paths without any network.
"""

import logging

from agents.llm_client import AnthropicCompatClient, _extract_text


# --- fakes mirroring an openai-SDK completion ------------------------------- #

class _Msg:
    def __init__(self, content=None, reasoning_content=None, reasoning=None, model_extra=None):
        self.content = content
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content
        if reasoning is not None:
            self.reasoning = reasoning
        # provider extras that aren't declared attributes live here on real SDK objects
        self.model_extra = model_extra


class _Choice:
    def __init__(self, message, finish_reason=None):
        self.message = message
        self.finish_reason = finish_reason


class _Completion:
    def __init__(self, choices):
        self.choices = choices


class _Completions:
    def __init__(self, completion):
        self._completion = completion
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._completion


class _Chat:
    def __init__(self, completion):
        self.completions = _Completions(completion)


class _FakeOpenAI:
    def __init__(self, completion):
        self.chat = _Chat(completion)


def _through_adapter(choice):
    """Drive one choice through the real AnthropicCompatClient and return the joined text
    exactly as BaseAgent.call_claude would read it."""
    client = AnthropicCompatClient(_FakeOpenAI(_Completion([choice])))
    resp = client.messages.create(model="z-ai/glm-5.2", max_tokens=100, temperature=0.2,
                                  system=[], messages=[{"role": "user", "content": "hi"}])
    return "".join(b.text for b in resp.content if b.type == "text")


# --- the four paths --------------------------------------------------------- #

def test_content_present_is_used_and_reasoning_ignored():
    choice = _Choice(_Msg(content="the answer", reasoning="chain of thought"))
    assert _through_adapter(choice) == "the answer"


def test_empty_content_falls_back_to_reasoning_content():
    choice = _Choice(_Msg(content=None, reasoning_content="answer via reasoning_content"))
    assert _through_adapter(choice) == "answer via reasoning_content"


def test_empty_content_falls_back_to_reasoning():
    choice = _Choice(_Msg(content="", reasoning="answer via reasoning"))
    assert _through_adapter(choice) == "answer via reasoning"


def test_reasoning_in_model_extra_is_found():
    # provider extra not exposed as a declared attribute -> read from model_extra
    choice = _Choice(_Msg(content=None, model_extra={"reasoning": "extra answer"}))
    assert _through_adapter(choice) == "extra answer"


def test_all_empty_returns_empty_and_logs_finish_reason(caplog):
    choice = _Choice(_Msg(content=None), finish_reason="length")
    with caplog.at_level(logging.WARNING):
        assert _through_adapter(choice) == ""
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("finish_reason" in m and "length" in m for m in warned)


def test_no_choices_is_empty_string(caplog):
    class _Empty:
        choices = []
    with caplog.at_level(logging.WARNING):
        assert _extract_text(_Empty()) == ""
    assert any("no choices" in r.getMessage() for r in caplog.records)
