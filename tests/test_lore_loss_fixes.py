"""Tests for the lore-loss / merge-error fixes found from a full end-to-end run.

Covers (see the plan file for the bug catalog):
  A+B  base.loads_tolerant + end-to-end extractor recovery of malformed JSON
       (unescaped inner quotes; a conversational preamble) -- these used to drop
       the WHOLE 20-message batch.
  C    year 0 accepted by the timeline date validator (in test_reconciler.py).
  D    per-group / per-event validation degradation (_valid_merge_subset,
       _valid_dated_subset, and the reconcile() salvage path).
  H    merged-description bullets no longer spill a second paragraph out of the
       timeline list (_flatten + render_history).

Additive, offline (no API key), per the repo convention -- new file, existing
tests untouched.
"""

import json
import logging
from datetime import datetime

import pytest

from agents.base import loads_tolerant
from agents.locations_extractor import LocationsExtractor
from agents.reconciler import (
    Reconciler, _valid_merge_subset, _valid_dated_subset, REVIEW_PREFIX,
)
from models.lore import Location, HistoryEvent, Scope, Alias, Detail
from models.message import Message
from models.reconcile import (
    ReconcileDecision, MergeGroup, DateDecision, DatedEvent,
)
from renderer.markdown import render_history, _flatten
from renderer.crosslink import build_crosslink_map
from renderer.footnotes import FootnoteRegistry


# --------------------------------------------------------------------------- #
# tiny fakes / builders (self-contained, mirror the other test modules)
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

def loc(name, aliases=None):
    return Location(name=name, aliases=[Alias(text=a, source_files=[]) for a in (aliases or [])])

def hev(name, description="An event happened.", date_text=None):
    return HistoryEvent(name=name, description=description, scope=Scope.WORLD, date_text=date_text)


# ======================================================================= #
# A+B -- loads_tolerant (pure)
# ======================================================================= #
def test_loads_tolerant_passes_wellformed_through_unchanged():
    assert loads_tolerant('[{"a": 1}]') == [{"a": 1}]
    assert loads_tolerant('{"x": [1, 2]}') == {"x": [1, 2]}
    # a markdown fence still works (strip_code_fences path)
    assert loads_tolerant('```json\n[1, 2]\n```') == [1, 2]


def test_loads_tolerant_recovers_unescaped_inner_quotes():
    # Bug A: Claude straightened the source's curly quotes to " inside the string
    # and didn't escape them -> plain json.loads dies. The inner text must survive.
    raw = '[{"name": "X", "quote": "The other "countries" are provinces"}]'
    out = loads_tolerant(raw)
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["quote"] == 'The other "countries" are provinces'


def test_loads_tolerant_recovers_after_conversational_preamble():
    # Bug B: reasoning before the array.
    raw = 'Looking at these messages, here is what I found:\n\n[{"name": "X"}]'
    out = loads_tolerant(raw)
    assert out == [{"name": "X"}]


def test_loads_tolerant_raises_on_pure_prose_and_non_container():
    # Pure prose repairs to "" -> not a dict/list -> raises, so the retry loop keeps
    # its retries (repair never fabricates a spurious empty batch).
    for bad in ["not json at all", "", "   ", "42", '"a bare string"']:
        with pytest.raises(json.JSONDecodeError):
            loads_tolerant(bad)


# ======================================================================= #
# A+B -- end to end through the extractor: the whole 20-batch is recovered
# ======================================================================= #
def test_extractor_recovers_batch_with_unescaped_quotes_and_keeps_the_fact():
    # The message carries CURLY quotes (as the export does); Claude returns the quote
    # with STRAIGHT unescaped quotes (invalid JSON). json_repair recovers it, and the
    # verbatim check passes (curly<->straight fold), so the fact survives instead of
    # the whole batch being dropped.
    message = msg('The other “countries” are now referred to as provinces')
    raw = ('[{"name": "The Empire", "aliases": [], "details": ['
           '{"detail": "Composed of provinces", '
           '"quote": "The other "countries" are now referred to as provinces", '
           '"source_id": 0}]}]')
    agent = LocationsExtractor(client=FakeClient([raw]))
    result = agent.extract([message])
    assert len(result) == 1
    assert result[0].name == "The Empire"
    assert [d.text for d in result[0].details] == ["Composed of provinces"]
    assert len(result[0].supporting_quotes) == 1


def test_extractor_recovers_batch_after_a_preamble():
    message = msg("Gol is a continent")
    raw = ('Looking at these messages, I found one location.\n\n'
           '[{"name": "Gol", "aliases": [], "details": ['
           '{"detail": "A continent", "quote": "Gol is a continent", "source_id": 0}]}]')
    agent = LocationsExtractor(client=FakeClient([raw]))
    result = agent.extract([message])
    assert len(result) == 1 and result[0].name == "Gol"
    assert [d.text for d in result[0].details] == ["A continent"]


# ======================================================================= #
# D -- reconciler per-group salvage
# ======================================================================= #
def test_valid_merge_subset_keeps_good_group_drops_bad(caplog):
    entries = [loc("Maltaav"), loc("Maltraav"), loc("Zeta")]
    decision = ReconcileDecision(merges=[
        MergeGroup(members=[0, 1], canonical="Maltraav"),   # good
        MergeGroup(members=[0, 9], canonical="Maltraav"),   # 0 reused + 9 out of range
    ])
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        out = _valid_merge_subset(decision, entries, "Location")
    assert len(out.merges) == 1
    assert out.merges[0].members == [0, 1]
    assert any(REVIEW_PREFIX in r.getMessage() and "dropping invalid merge group" in r.getMessage()
               for r in caplog.records)


def test_valid_merge_subset_drops_lt2_and_invalid_canonical():
    entries = [loc("A"), loc("B")]
    decision = ReconcileDecision(merges=[
        MergeGroup(members=[0], canonical="A"),               # < 2 members
        MergeGroup(members=[0, 1], canonical="Nonexistent"),  # canonical not a member name
    ])
    assert _valid_merge_subset(decision, entries, "Location").merges == []


def test_reconcile_salvages_valid_merge_when_no_fully_clean_decision(caplog):
    # 3x a decision that's structurally invalid (a reused/out-of-range index) but
    # carries ONE perfectly good merge. Old behaviour: discard EVERYTHING unmerged.
    # New: salvage the good merge.
    entries = [loc("Maltaav"), loc("Maltraav"), loc("Zeta")]
    bad = ('{"merges": [{"members": [0, 1], "canonical": "Maltraav", "conflicts": []},'
           '{"members": [0, 9], "canonical": "Maltraav", "conflicts": []}],'
           '"possible_duplicates": []}')
    rec = Reconciler(client=FakeClient([bad]))
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        result = rec.reconcile(entries)
    assert rec.client.call_count == 3                       # tried for a clean decision
    names = sorted(e.name for e in result)
    assert names == ["Maltraav", "Zeta"]                    # good merge applied, not dropped
    merged = next(e for e in result if e.name == "Maltraav")
    assert "Maltaav" in [a.text for a in merged.aliases]
    assert any("salvageable merge group" in r.getMessage() for r in caplog.records)


def test_reconcile_still_passes_through_when_nothing_ever_parsed(caplog):
    # Guard: genuinely-unparseable output (no salvageable decision) still returns
    # everything unmerged with the loud error -- the salvage must not mask this.
    entries = [loc("Maltaav"), loc("Maltraav")]
    rec = Reconciler(client=FakeClient(["not json at all"]))
    with caplog.at_level(logging.ERROR, logger="agents.reconciler"):
        result = rec.reconcile(entries)
    assert sorted(e.name for e in result) == ["Maltaav", "Maltraav"]
    assert any("no usable decision after 3 attempts" in r.getMessage() for r in caplog.records)


# ======================================================================= #
# D -- timeline per-event salvage
# ======================================================================= #
def test_valid_dated_subset_keeps_valid_and_drops_bad(caplog):
    events = [hev("A", date_text="5 AR"), hev("B", date_text="10 AR"), hev("C")]  # C has no date
    decision = DateDecision(dated=[
        DatedEvent(index=0, system="AR", parts=[5]),   # valid
        DatedEvent(index=9, system="AR", parts=[1]),   # out of range
        DatedEvent(index=2, system="AR", parts=[1]),   # event C has no date_text
    ])
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        kept = _valid_dated_subset(decision, events)
    assert [d.index for d in kept] == [0]
    assert any("dropping invalid dated entry" in r.getMessage() for r in caplog.records)


def test_valid_dated_subset_accepts_year_zero():
    events = [hev("Founding", date_text="0 AR")]
    kept = _valid_dated_subset(DateDecision(dated=[DatedEvent(index=0, system="AR", parts=[0])]),
                               events)
    assert [d.index for d in kept] == [0]  # year 0 survives salvage too


# ======================================================================= #
# H -- merged-description bullets stay one list item
# ======================================================================= #
def test_flatten_collapses_blank_lines():
    assert _flatten("First para.\n\nSecond para.") == "First para. Second para."
    assert _flatten("151 to\n200") == "151 to 200"


def test_render_history_keeps_merged_description_in_one_bullet():
    # A merged HistoryEvent's description is "\n\n".join(...) -- it must NOT spill a
    # second, un-bulleted paragraph out of the timeline list.
    events = [hev("Maltraav-Kriega War",
                  description="War erupted around 200 years ago.\n\n"
                              "The Imperium later grew confident and expanded.")]
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    lines = out.splitlines()
    bullets = [ln for ln in lines if ln.startswith("- ")]
    assert len(bullets) == 1
    # the whole (formerly two-paragraph) description is on the ONE bullet line...
    assert ("War erupted around 200 years ago. The Imperium later grew confident and expanded."
            in bullets[0])
    # ...and the second sentence never spills out as its own non-bullet line.
    assert not any(ln.strip().startswith("The Imperium later grew confident") and not ln.startswith("- ")
                   for ln in lines)
