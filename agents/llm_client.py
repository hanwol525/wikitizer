"""Pluggable LLM backend: run the agents on Anthropic (default) OR any OpenAI-compatible
endpoint (OpenRouter, Fireworks, ...) for a cheaper model like GLM-5.2 -- chosen per agent
by config, with Anthropic as the untouched default.

The whole app talks to the model through ONE method, ``BaseAgent.call_claude``, which calls
``self.client.messages.create(... system=[{...cache_control...}] ...)`` and reads
``response.content[i].text``. That is the Anthropic client surface. To swap providers WITHOUT
rewriting ``call_claude`` (and without breaking the ~700 tests whose fakes sit exactly at the
``messages.create`` boundary), we present that same tiny surface here and translate to the
OpenAI Chat Completions shape *below* it: ``AnthropicCompatClient`` exposes
``.messages.create(...) -> .content[].text`` but drives an OpenAI-compatible client inside.

Model + provider are one env var per role: ``WIKITIZER_<ROLE>_MODEL``. A value prefixed
``openrouter:`` or ``openai:`` routes to the OpenAI-compat endpoint (``LLM_OPENAI_BASE_URL`` /
``LLM_OPENAI_API_KEY``) with the bare slug; no prefix -> Anthropic. Unset -> the current
Anthropic defaults, so a run with no config behaves exactly as before.
"""

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Default per-request read timeout (seconds). Bounds a STALLED request so it fails and the
# SDK's max_retries recover it in minutes, rather than hanging on the SDK's 600s default read
# (a single stuck request otherwise freezes the whole ThreadPoolExecutor stage). Set well above
# the ~230s a full 8192-token non-streaming reply can take, so a legit slow call isn't killed;
# the connect timeout stays short. Overridable via WIKITIZER_LLM_TIMEOUT for a one-off tune.
DEFAULT_LLM_TIMEOUT_SECONDS = 300.0


# Single source of truth for the per-role default models (imported by base.py / noise_filter
# so the literals can't drift). "Big" = careful reasoning + verbatim quotes; "cheap" = the
# high-volume classifier.
DEFAULT_BIG_MODEL = "claude-sonnet-4-6"
DEFAULT_CHEAP_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MODELS = {
    "NOISE": DEFAULT_CHEAP_MODEL,
    "PROSE": DEFAULT_BIG_MODEL,
    "EXTRACT": DEFAULT_BIG_MODEL,
    "RECONCILE": DEFAULT_BIG_MODEL,
}

# A model value starting with one of these routes to the OpenAI-compatible endpoint.
_OPENAI_PREFIXES = ("openrouter:", "openai:")


# --------------------------------------------------------------------------- #
# The Anthropic-shaped wrapper over an OpenAI-compatible endpoint.
# --------------------------------------------------------------------------- #

class _TextBlock:
    """One Anthropic-style text content block (what call_claude reads: .type / .text)."""

    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    """Mimics an Anthropic Messages response: a .content list of typed blocks."""

    def __init__(self, text):
        self.content = [_TextBlock(text)]


def _flatten_system(system):
    """Anthropic passes the system prompt as a list of content blocks (with cache_control);
    OpenAI wants a single system string. Concatenate the text of every text block and drop
    the cache_control (no OpenAI equivalent). Tolerates a bare string too."""
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        return ""
    parts = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def _message_field(message, name):
    """Read ``name`` off an openai-SDK message object OR its ``model_extra``. Provider extras
    like ``reasoning`` / ``reasoning_content`` (OpenRouter reasoning models) aren't declared
    fields on the SDK's message model, so they may live in ``model_extra`` rather than as a
    plain attribute -- check both."""
    val = getattr(message, name, None)
    if val is not None:
        return val
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    return None


def _extract_text(resp):
    """Pull the assistant's text out of an OpenAI-compatible completion, robustly.

    The happy path is ``choices[0].message.content``. But GLM-5.2 (and other reasoning models
    over OpenRouter) can leave ``content`` empty and put the answer -- or, when the token budget
    is spent on reasoning, only the chain-of-thought -- in a non-standard ``reasoning_content`` /
    ``reasoning`` field. Reading only ``content`` then silently yields ``""`` and burns the full
    3-attempt JSON retry loop upstream (the ``char 0`` warning storm). So: try ``content``, then
    those reasoning fields; and when it's STILL empty, log the ``finish_reason`` so a genuine
    ``length`` truncation is VISIBLE instead of an invisible empty-string retry.
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        logger.warning("LLM response had no choices; treating as empty.")
        return ""
    message = choices[0].message
    for field in ("content", "reasoning_content", "reasoning"):
        val = _message_field(message, field)
        if isinstance(val, str) and val.strip():
            return val
    finish = getattr(choices[0], "finish_reason", None)
    logger.warning(
        "LLM response had empty content (finish_reason=%r); returning empty string -- this "
        "triggers the JSON re-ask upstream. finish_reason='length' means the token budget was "
        "exhausted (likely on reasoning), which a robust read alone can't recover.",
        finish,
    )
    return ""


class _Messages:
    """The `.messages` namespace: a single `.create(...)` matching the Anthropic call site."""

    def __init__(self, openai_client, extra_body=None):
        self._openai = openai_client
        # Optional OpenAI ``extra_body`` (e.g. a reasoning-disable payload) sent on every call.
        # None -> the request is byte-identical to before this knob existed.
        self._extra_body = extra_body

    def create(self, *, model, max_tokens, temperature, system, messages):
        # Anthropic system-block-list -> one OpenAI system message; user turns pass through
        # (the {"role","content"} shape is already OpenAI-compatible).
        oai_messages = []
        system_text = _flatten_system(system)
        if system_text:
            oai_messages.append({"role": "system", "content": system_text})
        for m in messages:
            oai_messages.append({"role": m["role"], "content": m["content"]})
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                      messages=oai_messages)
        if self._extra_body:                       # only when set, so the None path is unchanged
            kwargs["extra_body"] = self._extra_body
        resp = self._openai.chat.completions.create(**kwargs)
        # Read the text robustly (content, else a reasoning field; log finish_reason if empty)
        # and wrap it back into the Anthropic block shape so the caller's
        # `"".join(b.text for b in resp.content if b.type == "text")` is unchanged.
        text = _extract_text(resp)
        return _Response(text)


class AnthropicCompatClient:
    """Presents the slice of the ``anthropic.Anthropic`` surface that ``BaseAgent`` uses
    (``.messages.create(...) -> .content[].text``), backed by an OpenAI-compatible client.
    Inject the inner client for tests; production builds a real ``openai.OpenAI``.
    ``extra_body`` (optional) is forwarded on every call -- used to disable GLM's reasoning."""

    def __init__(self, openai_client, extra_body=None):
        self.messages = _Messages(openai_client, extra_body)


# --------------------------------------------------------------------------- #
# Building clients + resolving per-role backends.
# --------------------------------------------------------------------------- #

def _parse_model_spec(raw):
    """('anthropic'|'openai', bare_model) from a WIKITIZER_<ROLE>_MODEL value. A prefix of
    ``openrouter:``/``openai:`` selects the OpenAI-compat path with the remaining slug; no
    prefix -> Anthropic with the value verbatim."""
    for prefix in _OPENAI_PREFIXES:
        if raw.startswith(prefix):
            return "openai", raw[len(prefix):].strip()
    return "anthropic", raw


def _reasoning_extra_body():
    """Build the OpenAI ``extra_body`` that controls a reasoning model's thinking, from env
    ``WIKITIZER_OPENAI_REASONING``. Applies ONLY to the OpenAI-compat path (GLM etc.), never
    to Anthropic roles.

    GLM-5.2 over OpenRouter reasons for thousands of tokens by default, which blows the token
    budget mid-thought (truncation -> retries) and leaks chain-of-thought into the response. This
    knob turns that off. Values:
      - unset/empty                -> None (no extra_body; behavior unchanged)
      - off/false/0/no/disable(d)  -> disable thinking, sending BOTH the OpenRouter unified param
                                      AND the provider-native chat_template toggle (the one the
                                      Wafer provider named), for the best odds one is honored
      - low/medium/high            -> cap reasoning effort
      - a value starting with '{'  -> parsed as raw JSON and used verbatim (escape hatch, e.g.
                                      just chat_template_kwargs if a provider 400s on `reasoning`)
    A malformed JSON value logs a warning and returns None -- a config typo must never crash a run.
    """
    raw = os.environ.get("WIKITIZER_OPENAI_REASONING", "").strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "WIKITIZER_OPENAI_REASONING looks like JSON but did not parse (%s); ignoring it.",
                exc,
            )
            return None
    low = raw.lower()
    if low in ("off", "false", "0", "no", "disable", "disabled"):
        return {"reasoning": {"enabled": False},
                "chat_template_kwargs": {"enable_thinking": False}}
    if low in ("low", "medium", "high"):
        return {"reasoning": {"effort": low}}
    logger.warning(
        "WIKITIZER_OPENAI_REASONING=%r is not recognized (use off / low|medium|high / a JSON "
        "object); ignoring it.", raw,
    )
    return None


def _request_timeout():
    """The per-request timeout for both SDK clients: a generous read (default 300s, or
    WIKITIZER_LLM_TIMEOUT) with a short connect, so a stalled request fails fast and retries
    instead of hanging on the SDK's 600s default. A bad env value falls back to the default."""
    try:
        read = float(os.environ.get("WIKITIZER_LLM_TIMEOUT", "") or DEFAULT_LLM_TIMEOUT_SECONDS)
    except ValueError:
        logger.warning("WIKITIZER_LLM_TIMEOUT is not a number; using the %ss default.",
                       DEFAULT_LLM_TIMEOUT_SECONDS)
        read = DEFAULT_LLM_TIMEOUT_SECONDS
    return httpx.Timeout(read, connect=10.0)


def build_llm_client(provider):
    """Build a real client for a provider. SDKs are imported lazily so this module (and the
    translation logic) import without either SDK installed, and an Anthropic-only user never
    needs ``openai`` present."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(max_retries=3, timeout=_request_timeout())
    if provider == "openai":
        import openai
        base_url = os.environ.get("LLM_OPENAI_BASE_URL")
        api_key = os.environ.get("LLM_OPENAI_API_KEY")
        if not base_url:
            raise ValueError(
                "An agent is routed to an OpenAI-compatible model (openrouter:/openai: prefix) "
                "but LLM_OPENAI_BASE_URL is not set. Add it (and LLM_OPENAI_API_KEY) to .env."
            )
        return AnthropicCompatClient(
            openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=3,
                          timeout=_request_timeout()),
            extra_body=_reasoning_extra_body(),
        )
    raise ValueError(f"Unknown LLM provider: {provider!r}")


class LLMBackendResolver:
    """Resolves ``(client, model)`` per agent role from env, caching one client per distinct
    provider so same-provider roles share a connection. An ``override_client`` (a test fake or
    an explicitly-injected client) short-circuits provider selection and is used for every
    role -- which is how injecting one fake into the Orchestrator still wires all agents."""

    def __init__(self, override_client=None):
        self._override = override_client
        self._cache = {}   # provider -> client

    def resolve(self, role):
        raw = os.environ.get(f"WIKITIZER_{role}_MODEL", DEFAULT_MODELS[role])
        provider, model = _parse_model_spec(raw)
        if self._override is not None:
            return self._override, model
        client = self._cache.get(provider)
        if client is None:
            client = build_llm_client(provider)
            self._cache[provider] = client
        return client, model
