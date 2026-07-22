"""Parallel per-type reconcile in Orchestrator.run() (orchestrator.py).

Offline, reusing test_orchestrator.py's stub harness (plain stub classes + helpers,
not fixtures -- those are re-declared here). The six per-type reconciles are
independent single LLM calls holding no shared mutable state, so run() fans them
over the same thread pool as the extractors. These tests prove:
  * they actually run CONCURRENTLY (a 6-party barrier only releases when all six
    reconciles are simultaneously in-flight -- impossible if the loop were serial),
  * the parallel path yields BYTE-IDENTICAL output vs. a fully-sequential run, and
  * a per-type failure still DEGRADES to the un-reconciled list without failing run.
"""

import logging
import threading

import pytest

import orchestrator
from models.lore import Location
from tests.test_orchestrator import (
    _StubbedOrchestrator, _StubNoise, _StubReconciler, _extractors, _config,
)
from tests.fixtures.case import build_message


@pytest.fixture
def patched_parse(monkeypatch):
    """Canned, non-empty parse output so run() reaches the reconcile stage without
    touching a real file (same shape as test_orchestrator.py's own fixture)."""
    def fake_parse(filepath, speaker_map, input_format="auto"):
        return [build_message("Alice", "Riverton sits on the river Mund.", "group.txt")]
    monkeypatch.setattr(orchestrator, "parse_messages", fake_parse)
    monkeypatch.setattr(orchestrator, "filter_reactions", lambda msgs: list(msgs))


class _BarrierReconciler:
    """Concurrency probe. Every reconcile() call blocks on a barrier, so a call can
    only return once ALL `parties` types are simultaneously inside reconcile(). If
    run() reconciled sequentially, only one would ever be live: max_concurrent would
    stay 1 and the barrier would time out (surfacing as a degrade, still failing the
    max_concurrent assertion). order_history stamps a dated timeline like the real stub."""

    def __init__(self, parties):
        self.barrier = threading.Barrier(parties, timeout=10)
        self._lock = threading.Lock()
        self._live = 0
        self.max_concurrent = 0

    def reconcile(self, entries):
        with self._lock:
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
        try:
            self.barrier.wait()
        finally:
            with self._lock:
                self._live -= 1
        return list(entries)

    def order_history(self, events, current_year=None):
        return [e.model_copy(update={"calendar_system": "AR years",
                                     "chronological_position": i})
                for i, e in enumerate(events)]


def test_reconcile_runs_all_six_types_concurrently(patched_parse):
    # _extractors() has exactly six types; default max_workers=6 gives the pool room
    # for all six. A serial loop could never satisfy a 6-party barrier.
    rec = _BarrierReconciler(parties=6)
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(), rec)
    out = orch.run(["dummy.txt"], _config()).full

    assert out                                   # a wiki was produced
    assert rec.max_concurrent == 6               # all six reconciles overlapped


def test_parallel_reconcile_output_matches_sequential(patched_parse):
    # max_workers=1 (everything serial) vs 6 (fanned out) must render identical wikis.
    seq = _StubbedOrchestrator(_StubNoise(), _extractors(), _StubReconciler())
    par = _StubbedOrchestrator(_StubNoise(), _extractors(), _StubReconciler())

    seq_out = seq.run(["dummy.txt"], _config(max_workers=1)).full
    par_out = par.run(["dummy.txt"], _config(max_workers=6)).full

    assert seq_out == par_out


def test_reconcile_degrades_per_type_without_failing_run(patched_parse, caplog):
    caplog.set_level(logging.INFO)
    orch = _StubbedOrchestrator(_StubNoise(), _extractors(),
                                _StubReconciler(reconcile_raise_on=Location))
    out = orch.run(["dummy.txt"], _config()).full

    # The failed type fell back to un-reconciled entries (passthrough stub), so its
    # section still renders, and a [REVIEW] line names it -- degrade, not hard-stop.
    assert "## Locations" in out
    assert any("reconcile failed" in r.getMessage() and "locations" in r.getMessage().lower()
               for r in caplog.records)
