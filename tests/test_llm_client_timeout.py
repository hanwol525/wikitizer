"""The bounded per-request timeout on the real SDK clients (agents/llm_client.py).

A stalled request must fail fast and let max_retries recover it, rather than hang on the
SDK's 600s default read (which froze a whole run). Offline: constructing a client makes no
network call (the SDK reads the key lazily), so we can assert the timeout it was built with.
"""

import logging

import httpx
import pytest

from agents.llm_client import (
    build_llm_client,
    _request_timeout,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("WIKITIZER_LLM_TIMEOUT", raising=False)


# --- _request_timeout ------------------------------------------------------- #
def test_default_timeout_value():
    t = _request_timeout()
    assert isinstance(t, httpx.Timeout)
    assert t.read == DEFAULT_LLM_TIMEOUT_SECONDS   # generous read, above a full 8192-tok reply
    assert t.connect == 10.0                       # short connect


def test_env_override_honored(monkeypatch):
    monkeypatch.setenv("WIKITIZER_LLM_TIMEOUT", "120")
    assert _request_timeout().read == 120.0


def test_blank_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WIKITIZER_LLM_TIMEOUT", "")
    assert _request_timeout().read == DEFAULT_LLM_TIMEOUT_SECONDS


def test_bad_env_falls_back_to_default_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("WIKITIZER_LLM_TIMEOUT", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="agents.llm_client"):
        t = _request_timeout()
    assert t.read == DEFAULT_LLM_TIMEOUT_SECONDS
    assert any("WIKITIZER_LLM_TIMEOUT" in r.getMessage() for r in caplog.records)


# --- the real anthropic client carries the bounded timeout ------------------ #
def test_anthropic_client_built_with_bounded_timeout(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")   # lazy; no network at construction
    client = build_llm_client("anthropic")
    assert client.timeout.read == DEFAULT_LLM_TIMEOUT_SECONDS
    assert client.timeout.connect == 10.0                 # NOT the SDK's 600s default read


def test_anthropic_client_honors_env_timeout(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("WIKITIZER_LLM_TIMEOUT", "90")
    assert build_llm_client("anthropic").timeout.read == 90.0
