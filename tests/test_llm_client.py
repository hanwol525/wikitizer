"""The Anthropic-shaped wrapper over an OpenAI-compatible endpoint (agents/llm_client.py).

Offline: a fake inner "OpenAI" client is injected, so no network and no `openai` package
behavior is exercised. We verify the translation both ways -- the Anthropic-format request
(system block-list + user turns, cache_control dropped) becomes OpenAI chat messages, and the
OpenAI response (`.choices[0].message.content`) is wrapped back into the Anthropic
`.content[].text` shape base.py reads.
"""

from agents.llm_client import AnthropicCompatClient, _flatten_system


# --- fake OpenAI inner client ---------------------------------------------- #
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Completion(self._content)


class _Chat:
    def __init__(self, content):
        self.completions = _Completions(content)


class FakeOpenAI:
    def __init__(self, content="OUT"):
        self.chat = _Chat(content)


ANTHROPIC_SYSTEM = [{"type": "text", "text": "SYS PROMPT",
                    "cache_control": {"type": "ephemeral"}}]
USER = [{"role": "user", "content": "hello there"}]


def call(inner, **over):
    client = AnthropicCompatClient(inner)
    kwargs = dict(model="z-ai/glm-5.2", max_tokens=8192, temperature=0.2,
                  system=ANTHROPIC_SYSTEM, messages=USER)
    kwargs.update(over)
    return client.messages.create(**kwargs)


# --- response shape (what call_claude reads) -------------------------------- #
def test_response_is_wrapped_into_anthropic_text_block():
    resp = call(FakeOpenAI("the answer"))
    assert resp.content[0].type == "text"
    assert resp.content[0].text == "the answer"
    # base.py's exact join must yield the same string
    assert "".join(b.text for b in resp.content if b.type == "text") == "the answer"


def test_none_content_becomes_empty_string():
    resp = call(FakeOpenAI(None))
    assert resp.content[0].text == ""


# --- request translation ---------------------------------------------------- #
def test_request_flattens_system_and_preserves_user_turn():
    inner = FakeOpenAI()
    call(inner)
    sent = inner.chat.completions.calls[0]
    assert sent["model"] == "z-ai/glm-5.2"
    assert sent["max_tokens"] == 8192
    assert sent["temperature"] == 0.2
    # system block-list -> one OpenAI system message; user turn preserved after it
    assert sent["messages"] == [
        {"role": "system", "content": "SYS PROMPT"},
        {"role": "user", "content": "hello there"},
    ]


def test_cache_control_is_dropped():
    inner = FakeOpenAI()
    call(inner)
    sent = inner.chat.completions.calls[0]
    # no Anthropic-only field leaks into the OpenAI payload
    assert "cache_control" not in repr(sent["messages"])


def test_empty_system_omits_the_system_message():
    inner = FakeOpenAI()
    call(inner, system=[])
    sent = inner.chat.completions.calls[0]
    assert sent["messages"] == [{"role": "user", "content": "hello there"}]


# --- _flatten_system helper ------------------------------------------------- #
def test_flatten_system_variants():
    assert _flatten_system([{"type": "text", "text": "A"},
                            {"type": "text", "text": "B"}]) == "AB"
    assert _flatten_system("plain string") == "plain string"
    assert _flatten_system(None) == ""
    assert _flatten_system([{"type": "image", "text": "ignored"}]) == ""   # non-text block skipped
