"""Per-role backend resolution (agents/llm_client.py): which provider + model each agent
gets from env, and that same-provider roles share one cached client.

Offline. The default path (no env) must resolve to the current Anthropic models so a run
with no config is unchanged; an `openrouter:`/`openai:` prefix routes a role elsewhere.
"""

import pytest

from agents.llm_client import (
    AnthropicCompatClient,
    DEFAULT_BIG_MODEL,
    DEFAULT_CHEAP_MODEL,
    LLMBackendResolver,
    _parse_model_spec,
    build_llm_client,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # Hermetic: drop any real WIKITIZER_*/LLM_* the dev may have set.
    for var in ("WIKITIZER_NOISE_MODEL", "WIKITIZER_PROSE_MODEL", "WIKITIZER_EXTRACT_MODEL",
                "WIKITIZER_RECONCILE_MODEL", "LLM_OPENAI_BASE_URL", "LLM_OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --- _parse_model_spec ------------------------------------------------------ #
def test_no_prefix_is_anthropic():
    assert _parse_model_spec("claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")


def test_openrouter_prefix_routes_to_openai_with_bare_slug():
    assert _parse_model_spec("openrouter:z-ai/glm-5.2") == ("openai", "z-ai/glm-5.2")


def test_openai_prefix_too():
    assert _parse_model_spec("openai:some/model") == ("openai", "some/model")


# --- LLMBackendResolver: override (the injected-fake path) ------------------- #
def test_override_client_used_for_every_role_with_default_models():
    fake = object()
    r = LLMBackendResolver(override_client=fake)
    assert r.resolve("NOISE") == (fake, DEFAULT_CHEAP_MODEL)
    assert r.resolve("EXTRACT") == (fake, DEFAULT_BIG_MODEL)
    assert r.resolve("PROSE") == (fake, DEFAULT_BIG_MODEL)


def test_override_still_honors_env_model_choice(monkeypatch):
    monkeypatch.setenv("WIKITIZER_NOISE_MODEL", "openrouter:z-ai/glm-5.2")
    fake = object()
    r = LLMBackendResolver(override_client=fake)
    client, model = r.resolve("NOISE")
    assert client is fake and model == "z-ai/glm-5.2"   # model parsed even when client is overridden


# --- LLMBackendResolver: real per-provider clients + caching ---------------- #
def test_default_resolves_to_a_cached_anthropic_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")   # SDK reads it; no network at construction
    r = LLMBackendResolver()
    c1, m1 = r.resolve("EXTRACT")
    c2, m2 = r.resolve("RECONCILE")
    assert m1 == DEFAULT_BIG_MODEL and m2 == DEFAULT_BIG_MODEL
    assert c1 is c2                                        # both Anthropic -> one cached client


def test_openai_role_builds_a_compat_client(monkeypatch):
    monkeypatch.setenv("LLM_OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WIKITIZER_PROSE_MODEL", "openrouter:z-ai/glm-5.2")
    r = LLMBackendResolver()
    client, model = r.resolve("PROSE")
    assert isinstance(client, AnthropicCompatClient)
    assert model == "z-ai/glm-5.2"


# --- build_llm_client error path -------------------------------------------- #
def test_openai_without_base_url_raises():
    with pytest.raises(ValueError):
        build_llm_client("openai")     # LLM_OPENAI_BASE_URL unset (clean_env)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_llm_client("bananas")
