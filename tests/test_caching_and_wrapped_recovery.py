"""Tests for the prompt-caching + wrapped-batch-recovery follow-ups.

Covers:
  * Prompt caching: `call_claude` (and therefore the extractor path through
    `call_claude_json`) now sends the system prompt as a content block carrying
    ``cache_control: {"type": "ephemeral"}``.
  * Wrapped-batch recovery: json_repair sometimes recovers a batch into a nested
    shape like ``["", [{...}, {...}]]``; `_extract_batch` (and the noise filter's
    parse loop) now flatten one level so those entries aren't dropped. The wrapped
    shape is fed as VALID JSON (a list holding a string + a nested list), which
    exercises the flatten deterministically without depending on json_repair.
  * `--cache` wiring: `Orchestrator(cache=...)` threads the disk-cache flag to every
    agent.

Additive, offline (no API key), per the repo convention.
"""

from datetime import datetime

import pytest

from agents.base import BaseAgent
from agents.locations_extractor import LocationsExtractor
from agents.noise_filter import NoiseFilterAgent
from models.message import Message
from orchestrator import Orchestrator, PipelineConfig


# --------------------------------------------------------------------------- #
# fakes (record create() kwargs; return canned text)
# --------------------------------------------------------------------------- #
class _Block:
    type = "text"
    def __init__(self, text): self.text = text

class _Resp:
    def __init__(self, text): self.content = [_Block(text)]

class _Messages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _Resp(spec)

class FakeClient:
    def __init__(self, responses): self.messages = _Messages(responses)
    @property
    def call_count(self): return len(self.messages.calls)


def msg(content, sender="dm", source_file="log.txt"):
    return Message(sender=sender, timestamp=datetime(2024, 1, 1, 12, 0, 0),
                   content=content, source_file=source_file)


# ======================================================================= #
# Prompt caching -- the system prompt is an ephemeral cache_control block
# ======================================================================= #
def test_call_claude_marks_system_prompt_for_caching():
    agent = BaseAgent(client=FakeClient(["ok"]))
    agent.call_claude("SYSTEM RULES", "user text")
    sent = agent.client.messages.calls[0]
    assert sent["system"] == [
        {"type": "text", "text": "SYSTEM RULES", "cache_control": {"type": "ephemeral"}}
    ]
    # the user message is the varying, uncached part -- unchanged
    assert sent["messages"] == [{"role": "user", "content": "user text"}]


def test_extractor_path_also_marks_system_prompt_for_caching():
    # extract -> _extract_batch -> call_claude_json -> call_claude -> create
    agent = LocationsExtractor(client=FakeClient(["[]"]))
    agent.extract([msg("Gol is a continent")])
    sent = agent.client.messages.calls[0]
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["system"][0]["text"] == LocationsExtractor.system_prompt  # full prompt, unchanged


# ======================================================================= #
# Wrapped-batch recovery -- one-level flatten in _extract_batch
# ======================================================================= #
def test_extract_batch_recovers_wrapped_nested_entries():
    # The exact real-run shape: a stray "" plus the real entries nested one level.
    # Valid JSON, so it parses to ["", [ {...}, {...} ]] and exercises the flatten.
    wrapped = ('["", [{"name": "Sam", "aliases": [], "details": []}, '
               '{"name": "Emperor Tiberius Krieger", "aliases": [], "details": []}]]')
    agent = LocationsExtractor(client=FakeClient([wrapped]))
    result = agent.extract([msg("Sam is the brother of the Emperor")])
    names = sorted(e.name for e in result)
    assert names == ["Emperor Tiberius Krieger", "Sam"]  # both recovered, not dropped


def test_extract_batch_still_skips_genuine_junk_but_keeps_siblings(caplog):
    # A bare string element is NOT a list -> still skipped-and-logged; the dict survives.
    import logging
    resp = '["just a string, not an object", {"name": "Gol", "aliases": [], "details": []}]'
    agent = LocationsExtractor(client=FakeClient([resp]))
    with caplog.at_level(logging.WARNING, logger="agents.base_extractor"):
        result = agent.extract([msg("Gol is a continent")])
    assert [e.name for e in result] == ["Gol"]
    assert any("is not an object" in r.getMessage() for r in caplog.records)


# ======================================================================= #
# Wrapped-batch recovery -- one-level flatten in the noise filter
# ======================================================================= #
def test_noise_filter_recovers_wrapped_nested_rows():
    wrapped = '[[{"id": 0, "label": "lore"}, {"id": 1, "label": "noise"}]]'
    agent = NoiseFilterAgent(client=FakeClient([wrapped]))
    out = agent.classify([msg("Lake Mundi is huge"), msg("what's everyone's AC")])
    labels = [label for _, label in out]
    assert labels == ["lore", "noise"]  # rows applied, not dumped to 'ambiguous'


# ======================================================================= #
# --cache wiring -- Orchestrator threads the flag to every agent
# ======================================================================= #
def _config():
    return PipelineConfig(
        speaker_map={"+1555": "Sam", "exporter": "Hannah"},
        crosslink_words={"require_article": [], "never_link": []},
    )


def test_orchestrator_cache_flag_enables_the_disk_cache_on_every_agent(monkeypatch):
    monkeypatch.delenv("WIKITIZER_LLM_CACHE", raising=False)  # isolate from the env toggle
    orch = Orchestrator(client=FakeClient(["[]"]), cache=True)
    noise_filter, extractors, reconciler, prose = orch._build_agents(_config())
    agents = [noise_filter, reconciler, prose, *extractors.values()]
    assert all(a._cache_enabled for a in agents)


def test_orchestrator_cache_defaults_off(monkeypatch):
    monkeypatch.delenv("WIKITIZER_LLM_CACHE", raising=False)
    orch = Orchestrator(client=FakeClient(["[]"]))  # cache defaults to False
    noise_filter, extractors, reconciler, prose = orch._build_agents(_config())
    agents = [noise_filter, reconciler, prose, *extractors.values()]
    assert not any(a._cache_enabled for a in agents)
