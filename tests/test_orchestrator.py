"""Tests for orchestrator.py (Phase 4.5: the pipeline orchestrator).

These exercise the orchestrator's OWN logic -- stage sequencing, the parallel
extractor fan-out, and the error policy (which stages degrade vs the one hard
stop) -- NOT the agents, which have their own tests. The seam is exactly the one
the brief calls for: subclass `Orchestrator` and override `_build_agents` to hand
back small STUB agents whose classify/extract/reconcile/order_history return
canned typed objects (or raise, to drive each error path). That keeps the whole
suite offline: no network, no API key, no `integration` marker.

Two isolation seams are used on purpose:
  * `_build_agents` is overridden (the brief's seam) to swap in stub agents.
  * The pure parse+filter step lives directly in `run()` (not in an agent), so a
    `monkeypatch` fixture replaces `orchestrator.parse_chat_log` /
    `orchestrator.filter_reactions` with canned message output -- fully isolating
    the orchestrator from the Phase 2 parser's file format. ONE additive test
    (`test_run_over_a_real_parsed_log`) skips that monkeypatch and feeds a real
    minimal log file instead, to lock down the real parse wiring (arg order).

The stubs deliberately MIRROR the real agents' empty-input short-circuit (empty
in -> empty out), so the same default stubs drive both the happy path (non-empty
input -> canned entities) and the empty-input test (no input -> empty wiki).
"""

import logging

import pytest

import orchestrator
from orchestrator import Orchestrator, PipelineConfig
from models.lore import (
    Character,
    HistoryEvent,
    Item,
    Location,
    Organization,
    PeopleAndCultures,
    Quote,
    Scope,
)
from tests.fixtures.case import build_message


# --- canned lore + tiny builders ------------------------------------------- #

# A real Quote (real models/lore.py constructor) so the happy path mints a real
# footnote. speaker/source are arbitrary -- the renderer just needs the triple.
Q = Quote(text="A river town.", speaker="Alice", source_file="group.txt")


def hev(name, description):
    """A minimal HistoryEvent: `scope` is required; the timeline fields default to
    None (i.e. 'Could Not Place' shaped until a stub order_history stamps them)."""
    return HistoryEvent(name=name, description=description, scope=Scope.WORLD)


def _config(**overrides):
    """A PipelineConfig with everything pre-loaded (plain dicts, nothing on disk).
    speaker_map VALUES feed the characters extractor's roster inside _build_agents."""
    base = dict(
        speaker_map={"+15551230000": "Alice", "exporter": "Bob"},
        crosslink_words={"require_article": [], "never_link": []},
    )
    base.update(overrides)
    return PipelineConfig(**base)


# --- stub agents ------------------------------------------------------------ #
# Each stub is the smallest object with the method run() calls. They mirror the
# real agents' empty-input short-circuit so empty input flows through as empty.

class _StubNoise:
    """Stands in for NoiseFilterAgent. classify() labels every message 'lore' so
    select_for_extraction keeps them all; raise_exc drives the hard-stop path."""

    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.seen = None

    def classify(self, messages):
        self.seen = list(messages)
        if self.raise_exc is not None:
            raise self.raise_exc
        return [(m, "lore") for m in messages]   # [] in -> [] out


class _StubExtractor:
    """Stands in for one extractor. Returns its canned list for non-empty input,
    [] for empty input (mirroring BaseExtractor's short-circuit); raise_exc drives
    the degrade path. `called` records that the fan-out actually submitted it."""

    def __init__(self, result=None, raise_exc=None):
        self.result = list(result or [])
        self.raise_exc = raise_exc
        self.called = False

    def extract(self, messages):
        self.called = True
        if self.raise_exc is not None:
            raise self.raise_exc
        if not messages:
            return []
        return list(self.result)


class _StubReconciler:
    """Stands in for Reconciler. reconcile() passes entries through unchanged (the
    real one dedups; passthrough is enough to test wiring). reconcile_raise_on is a
    class -> reconcile raises for that entity type (degrade path). order_raise ->
    order_history raises. Otherwise order_history stamps a single dated timeline so
    History renders under '## History'."""

    def __init__(self, reconcile_raise_on=None, order_raise=False):
        self.reconcile_raise_on = reconcile_raise_on
        self.order_raise = order_raise
        self.order_called = False

    def reconcile(self, entries):
        if (entries and self.reconcile_raise_on is not None
                and isinstance(entries[0], self.reconcile_raise_on)):
            raise RuntimeError("boom: reconcile")
        return list(entries)

    def order_history(self, events):
        self.order_called = True
        if self.order_raise:
            raise RuntimeError("boom: order_history")
        # Stamp a consistent calendar_system + ascending position, as if the real
        # timeline pass had placed them -> renders as a dated '## History' section.
        return [e.model_copy(update={"calendar_system": "AR years",
                                     "chronological_position": i})
                for i, e in enumerate(events)]


def _extractors(**overrides):
    """The six stub extractors, keyed EXACTLY like the orchestrator's real dict.
    Defaults return one canned entity per type; override any key per-test."""
    exts = {
        "locations": _StubExtractor([Location(name="Riverton",
                                              details=["A river town."],
                                              supporting_quotes=[Q])]),
        "characters": _StubExtractor([Character(name="Gimli",
                                               details=["A dwarf who visited Riverton."])]),
        "history": _StubExtractor([hev("The Founding", "Riverton was founded.")]),
        "organizations": _StubExtractor([Organization(name="The Rivermen")]),
        "items": _StubExtractor([Item(name="The Amulet")]),
        "people": _StubExtractor([PeopleAndCultures(name="The Krieg")]),
    }
    exts.update(overrides)
    return exts


class _StubbedOrchestrator(Orchestrator):
    """Orchestrator with `_build_agents` overridden to return the stub agents, so
    run()'s sequencing + error policy run with zero real Claude calls."""

    def __init__(self, noise, extractors, reconciler):
        # A sentinel client: _build_agents is overridden, so it's never used.
        super().__init__(client=object())
        self._noise = noise
        self._extractors = extractors
        self._reconciler = reconciler

    def _build_agents(self, config):
        return self._noise, self._extractors, self._reconciler


@pytest.fixture
def patched_parse(monkeypatch):
    """Replace the pure parse+filter step so run() gets a canned, non-empty message
    list without any real file. Isolates the orchestrator from the parser format."""
    def fake_parse(filepath, speaker_map):
        return [build_message("Alice", "Riverton sits on the river Mund.", "group.txt")]
    monkeypatch.setattr(orchestrator, "parse_chat_log", fake_parse)
    monkeypatch.setattr(orchestrator, "filter_reactions", lambda msgs: list(msgs))


# --- 1. happy path ---------------------------------------------------------- #

def test_happy_path_renders_all_wired_sections(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(), _StubReconciler())
    out = orch.run(["dummy.txt"], _config())

    assert out                                   # non-empty wiki
    assert "## Locations" in out
    assert "## History" in out
    assert "## Characters" in out
    assert "Riverton" in out
    # footnotes look sane: a marker in the body + the definitions block.
    assert "[^1]" in out and "## Footnotes" in out
    # cross-links look sane: a bare entity mention became an internal link.
    assert "[Riverton](#riverton)" in out


# --- 2. extractor degradation ----------------------------------------------- #

def test_extractor_failure_drops_only_its_section(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    exts = _extractors(locations=_StubExtractor(raise_exc=RuntimeError("boom: extract")))
    orch = _StubbedOrchestrator(_StubNoise(), exts, _StubReconciler())
    out = orch.run(["dummy.txt"], _config())

    assert "## Locations" not in out             # the dead extractor's section is gone
    assert "## Characters" in out                # siblings still render
    assert "Gimli" in out
    assert "[REVIEW]" in caplog.text and "locations" in caplog.text


# --- 3. per-type reconcile degradation -------------------------------------- #

def test_reconcile_failure_uses_unreconciled_entries(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    orch = _StubbedOrchestrator(
        _StubNoise(), _extractors(), _StubReconciler(reconcile_raise_on=Location))
    out = orch.run(["dummy.txt"], _config())

    # the un-reconciled locations are used, so Riverton still renders...
    assert "## Locations" in out and "Riverton" in out
    # ...and the degrade is flagged for review.
    assert "[REVIEW]" in caplog.text and "reconcile" in caplog.text


# --- 4. order_history degradation ------------------------------------------- #

def test_order_history_failure_renders_could_not_place(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    orch = _StubbedOrchestrator(
        _StubNoise(), _extractors(), _StubReconciler(order_raise=True))
    out = orch.run(["dummy.txt"], _config())

    # events kept their unset timeline fields -> they render under Could Not Place
    # (which, when it's the whole History section, uses the explanatory note).
    assert "## History" in out
    assert "The Founding" in out
    assert "couldn't be placed" in out
    assert "[REVIEW]" in caplog.text and "order_history" in caplog.text


# --- 5. noise-filter hard-stop ---------------------------------------------- #

def test_noise_filter_failure_aborts_run(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    orch = _StubbedOrchestrator(
        _StubNoise(raise_exc=RuntimeError("boom: classify")),
        _extractors(), _StubReconciler())

    with pytest.raises(RuntimeError):
        orch.run(["dummy.txt"], _config())

    # the abort logs at error with NO [REVIEW] prefix (a crash is unmissable itself).
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("Noise filter" in r.getMessage() for r in errors)
    assert "[REVIEW]" not in caplog.text


# --- 6. empty input --------------------------------------------------------- #

def test_empty_input_returns_empty_and_warns(caplog):
    caplog.set_level(logging.INFO)
    # files=[] -> no parse at all -> no messages; stubs mirror empty -> empty wiki.
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(), _StubReconciler())
    out = orch.run([], _config())

    assert out == ""
    assert "wiki will be empty" in caplog.text


# --- 7. all six extractors run ---------------------------------------------- #

def test_all_six_extractors_run(patched_parse):
    exts = _extractors()
    orch = _StubbedOrchestrator(_StubNoise(), exts, _StubReconciler())
    orch.run(["dummy.txt"], _config())

    # the fan-out submitted every extractor (the pool joined before run() returned).
    assert all(e.called for e in exts.values())
    assert set(exts) == {"locations", "characters", "history",
                         "organizations", "items", "people"}


# --- 8. shared-client wiring ------------------------------------------------ #

def test_shared_client_threads_into_every_agent():
    # Call the REAL _build_agents and confirm one injected client reaches them all.
    fake = object()
    orch = Orchestrator(client=fake)
    noise, extractors, reconciler = orch._build_agents(_config())

    assert orch.client is fake
    assert noise.client is fake
    assert reconciler.client is fake
    assert len(extractors) == 6
    for ext in extractors.values():
        assert ext.client is fake


# --- additive: real parse wiring (no monkeypatch) --------------------------- #

def test_run_over_a_real_parsed_log(tmp_path, caplog):
    """Additive hardening beyond the brief's 8 groups: skip the parse monkeypatch
    and feed a REAL minimal group log, so parse_chat_log's arg order and the
    filter_reactions wiring are exercised for real (a swapped-arg regression the
    monkeypatched tests would mask). Stubs still handle the agent layer."""
    caplog.set_level(logging.INFO)
    log = tmp_path / "group.txt"
    # Minimal but valid group export: participant header (leads with a comma), the
    # dashes row, a leading lone timestamp to seed the first message cleanly, one
    # body line, then a complete phone footer that names the sender.
    log.write_text(
        ",+15551230000\n"
        "----------------------------------------\n"
        "01/01/2024 12:00:00\n"
        "Riverton sits on the river Mund.\n"
        "+15551230000 01/01/2024 12:00:05\n",
        encoding="utf-8",
    )
    noise = _StubNoise()
    orch = _StubbedOrchestrator(noise, _extractors(), _StubReconciler())
    out = orch.run([str(log)], _config())

    # a real message reached the (stub) noise filter, and the wiki rendered.
    assert noise.seen and len(noise.seen) == 1
    assert "Riverton sits on the river Mund." in noise.seen[0].content
    assert "## Locations" in out and "Riverton" in out
