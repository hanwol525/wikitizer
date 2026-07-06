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
from pathlib import Path

import pytest

import orchestrator
from orchestrator import Orchestrator, PipelineConfig, WikiOutput, _history_pipeline
from models.lore import (
    Character,
    Detail,
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


def det(text, *source_files):
    return Detail(text=text, source_files=list(source_files))


def q(text, source="group.txt"):
    return Quote(text=text, speaker="M", source_file=source)


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
                                              details=[det("A river town.")],
                                              supporting_quotes=[Q])]),
        "characters": _StubExtractor([Character(name="Gimli",
                                               details=[det("A dwarf who visited Riverton.")])]),
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


@pytest.fixture
def patched_parse_multi(monkeypatch):
    """Like patched_parse, but tags each message with its file's BARE name, so a
    multi-file `files` list yields multi-source messages (what exclusion needs)."""
    def fake_parse(filepath, speaker_map):
        name = Path(filepath).name
        return [build_message("Alice", f"Content from {name}.", name)]
    monkeypatch.setattr(orchestrator, "parse_chat_log", fake_parse)
    monkeypatch.setattr(orchestrator, "filter_reactions", lambda msgs: list(msgs))


# --- 1. happy path ---------------------------------------------------------- #

def test_happy_path_renders_all_wired_sections(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(), _StubReconciler())
    out = orch.run(["dummy.txt"], _config()).full

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
    out = orch.run(["dummy.txt"], _config()).full

    assert "## Locations" not in out             # the dead extractor's section is gone
    assert "## Characters" in out                # siblings still render
    assert "Gimli" in out
    assert "[REVIEW]" in caplog.text and "locations" in caplog.text


# --- 3. per-type reconcile degradation -------------------------------------- #

def test_reconcile_failure_uses_unreconciled_entries(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    orch = _StubbedOrchestrator(
        _StubNoise(), _extractors(), _StubReconciler(reconcile_raise_on=Location))
    out = orch.run(["dummy.txt"], _config()).full

    # the un-reconciled locations are used, so Riverton still renders...
    assert "## Locations" in out and "Riverton" in out
    # ...and the degrade is flagged for review.
    assert "[REVIEW]" in caplog.text and "reconcile" in caplog.text


# --- 4. order_history degradation ------------------------------------------- #

def test_order_history_failure_renders_could_not_place(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    orch = _StubbedOrchestrator(
        _StubNoise(), _extractors(), _StubReconciler(order_raise=True))
    out = orch.run(["dummy.txt"], _config()).full

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
    out = orch.run([], _config()).full

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
    out = orch.run([str(log)], _config()).full

    # a real message reached the (stub) noise filter, and the wiki rendered.
    assert noise.seen and len(noise.seen) == 1
    assert "Riverton sits on the river Mund." in noise.seen[0].content
    assert "## Locations" in out and "Riverton" in out


# --- 4.6 Part 2: _history_pipeline (the restricted doc's second History pass) #
# Same degrade-don't-crash policy as the main run, exercised with the existing
# _StubExtractor / _StubReconciler flags.

def test_history_pipeline_happy_path_extracts_reconciles_orders():
    ev = hev("The Founding", "It began.")
    out = _history_pipeline([build_message("A", "x", "group.txt")],
                            _StubExtractor([ev]), _StubReconciler())
    assert [e.name for e in out] == ["The Founding"]
    # the stub order_history stamps a dated timeline -> events came back ordered
    assert out[0].calendar_system == "AR years" and out[0].chronological_position == 0


def test_history_pipeline_extract_failure_returns_empty(caplog):
    caplog.set_level(logging.ERROR)
    out = _history_pipeline([build_message("A", "x", "g.txt")],
                            _StubExtractor(raise_exc=RuntimeError("boom")), _StubReconciler())
    assert out == []
    assert "[REVIEW]" in caplog.text and "extract failed" in caplog.text


def test_history_pipeline_reconcile_failure_passes_raw_through():
    ev = hev("E", "d")
    rec = _StubReconciler(reconcile_raise_on=HistoryEvent)
    out = _history_pipeline([build_message("A", "x", "g.txt")], _StubExtractor([ev]), rec)
    # reconcile raised -> raw events used; order_history still stamps them
    assert [e.name for e in out] == ["E"]
    assert out[0].calendar_system == "AR years"


def test_history_pipeline_order_failure_returns_unordered():
    ev = hev("E", "d")
    rec = _StubReconciler(order_raise=True)
    out = _history_pipeline([build_message("A", "x", "g.txt")], _StubExtractor([ev]), rec)
    # order_history raised -> the un-ordered (reconciled) events come back
    assert [e.name for e in out] == ["E"]
    assert out[0].chronological_position is None   # never stamped


# --- 4.6 Part 2: the restricted doc end-to-end through run() ----------------- #

class _HistoryStub:
    """A history extractor whose extract() returns a PublicEvent always, plus a
    SecretEvent ONLY when it sees a message sourced from 'secret.txt'. Models the
    leak-proof-by-construction path: the restricted extraction never sees the secret
    message, so SecretEvent never forms. Mirrors the empty-in -> empty-out contract."""

    def __init__(self):
        self.called = False

    def extract(self, messages):
        self.called = True
        if not messages:
            return []
        events = [hev("PublicEvent", "A public event.")]
        if any(m.source_file == "secret.txt" for m in messages):
            events.append(hev("SecretEvent", "A secret event."))
        return events


def test_run_without_exclusions_has_no_restricted_doc(patched_parse):
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(), _StubReconciler())
    result = orch.run(["dummy.txt"], _config())
    assert isinstance(result, WikiOutput)
    assert result.full                        # non-empty full doc
    assert result.restricted is None          # no exclusions -> no restricted doc (not "")


def test_run_with_exclusions_produces_both_docs(patched_parse_multi):
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(), _StubReconciler())
    result = orch.run(["logs/group.txt", "logs/secret.txt"], _config(),
                      exclude_sources=["secret.txt"])
    assert result.full is not None and result.restricted is not None


def test_restricted_doc_omits_excluded_entity_content(patched_parse_multi):
    # One public entity, one whose fact AND quote are both secret-only.
    public = Location(name="Publicton", details=[det("A public town.", "group.txt")],
                      supporting_quotes=[q("Publicton exists", "group.txt")])
    secret = Location(name="Secretville", details=[det("A hidden fort.", "secret.txt")],
                      supporting_quotes=[q("Secretville exists", "secret.txt")])
    exts = _extractors(locations=_StubExtractor([public, secret]))
    orch = _StubbedOrchestrator(_StubNoise(), exts, _StubReconciler())
    result = orch.run(["logs/group.txt", "logs/secret.txt"], _config(),
                      exclude_sources=["secret.txt"])
    assert "Secretville" in result.full and "Publicton" in result.full   # full has both
    assert "Secretville" not in result.restricted   # secret-only entity carved out
    assert "Publicton" in result.restricted         # public entity survives


def test_restricted_history_reruns_over_the_subset(patched_parse_multi):
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(history=_HistoryStub()),
                                _StubReconciler())
    result = orch.run(["logs/group.txt", "logs/secret.txt"], _config(),
                      exclude_sources=["secret.txt"])
    # full run saw the secret message -> SecretEvent formed; the restricted extract
    # never saw it -> SecretEvent can't form (leak-proof by construction).
    assert "SecretEvent" in result.full and "PublicEvent" in result.full
    assert "SecretEvent" not in result.restricted
    assert "PublicEvent" in result.restricted


def test_bad_exclude_name_aborts_before_any_agent():
    # No parse fixture on purpose: validation must raise BEFORE any parse/agent work.
    noise = _StubNoise()
    orch = _StubbedOrchestrator(noise, _extractors(), _StubReconciler())
    with pytest.raises(ValueError):
        orch.run(["logs/group.txt"], _config(), exclude_sources=["ghost.txt"])
    assert noise.seen is None   # classify never ran -> aborted before touching any agent


def test_restricted_carves_secret_fact_and_quote_from_a_surviving_entity(patched_parse_multi):
    # The carving that DOES work, at the orchestrator level: a SURVIVING mixed-content
    # entity keeps its public fact+quote but has its secret-sourced fact AND secret
    # quote stripped from the restricted doc (both are source-tagged, so provenance
    # carving reaches them). Uses a non-Locations type on purpose (the review noted
    # only Locations was covered end-to-end).
    mixed = Character(
        name="Gandalf",
        details=[det("A wandering wizard.", "group.txt"),
                 det("Is secretly the Maia Olorin.", "secret.txt")],
        supporting_quotes=[q("Gandalf wanders the wilds", "group.txt"),
                           q("Gandalf is the Maia Olorin", "secret.txt")],
    )
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(characters=_StubExtractor([mixed])),
                                _StubReconciler())
    result = orch.run(["logs/group.txt", "logs/secret.txt"], _config(),
                      exclude_sources=["secret.txt"])
    # full doc carries everything
    assert "A wandering wizard." in result.full and "secretly the Maia Olorin" in result.full
    assert "Gandalf is the Maia Olorin" in result.full          # secret quote in full footnotes
    # restricted: entity survives on the public fact; secret FACT + secret QUOTE carved out
    assert "Gandalf" in result.restricted
    assert "A wandering wizard." in result.restricted
    assert "secretly the Maia Olorin" not in result.restricted  # secret fact text gone
    assert "Gandalf is the Maia Olorin" not in result.restricted  # secret quote text gone


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN LIMITATION (4.6 Part 2, deferred): entity name/aliases carry no source "
           "provenance, so a secret-derived NAME on a surviving entity is not scrubbed and "
           "leaks into the restricted doc's heading/anchor. Fix deferred (re-extract entities "
           "like History, or add name/alias provenance). See exclusion.py 'KNOWN LIMITATION'. "
           "This test asserts the DESIRED leak-free behaviour, so it xfails today and flips to "
           "passing (strict -> forces removing this marker) once the gap is closed.",
)
def test_restricted_doc_leaks_secret_derived_entity_name_KNOWN_GAP(patched_parse_multi):
    # A surviving entity whose NAME came only from the excluded file (here the secret
    # true-name "Blackspire Keep") but which keeps a public fact -> the entity survives
    # and its name/anchor currently leak into the restricted doc. Desired: absent.
    leaky = Location(
        name="Blackspire Keep",                                     # DM-file-only true name
        details=[det("An old fort on the hill.", "group.txt")],     # public -> entity survives
        supporting_quotes=[q("the old fort on the hill", "group.txt")],
    )
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(locations=_StubExtractor([leaky])),
                                _StubReconciler())
    result = orch.run(["logs/group.txt", "logs/secret.txt"], _config(),
                      exclude_sources=["secret.txt"])
    assert "Blackspire Keep" not in result.restricted   # DESIRED (fails today -> xfail)
