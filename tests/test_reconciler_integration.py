"""Phase 4.1a + 4.1b -- LIVE / integration tests for the reconciler.

Phase 4.1a (MERGE step): one real Anthropic API call proving the headline case
end-to-end: the LLM must MERGE a one-letter typo of the same place
("Maltaav"/"Maltraav") yet must NOT merge two genuinely different people whose
names differ by one letter when the text states they're siblings ("CJ"/"DJ").

Phase 4.1b (the timeline pass, ``order_history``): two live tests proving Sonnet
output flows through the two-call pipeline -- dated events sort + a relative clue
weaves in, and two distinct calendars stay on separate timelines. (Here we CAN
assert by exact name: these events are hand-built, not extractor output, so their
names are stable.)

The deterministic Python (combiner, vetoes, validators, the timeline engine) is
covered exhaustively in test_reconciler.py; these tests only prove the LLM halves
make the right calls.

Gating mirrors test_extractors_integration.py: marked ``integration`` (deselected
by default -- a plain ``pytest`` skips it), and ``skipif``-ed when
``ANTHROPIC_API_KEY`` is absent. The repo-root ``conftest.py`` loads ``.env``
first, so a key sitting only in ``.env`` is visible before the skip is evaluated.
"""

import os
from pathlib import Path

import pytest

from models.lore import Location, Character, HistoryEvent, Quote, Detail
from agents.reconciler import Reconciler

try:
    from tests.eyeball import format_entities
except ImportError:  # eyeball is committed, but stay robust to a partial checkout
    format_entities = None


def det(text, *source_files):
    return Detail(text=text, source_files=list(source_files))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="live reconciler test needs ANTHROPIC_API_KEY",
    ),
]

OUTPUT_DIR = Path(__file__).parent / "output"


def _eyeball(title, entities):
    """Optional human-eyeball report, matching the extractor integration pattern.
    Best-effort: never let a reporting hiccup fail the actual assertion."""
    if format_entities is None:
        return
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "reconciler_review.txt", "a", encoding="utf-8") as fh:
            fh.write(format_entities(entities, title))
    except Exception:  # noqa: BLE001 -- reporting is non-critical
        pass


@pytest.mark.integration
def test_reconciler_merges_typo_but_not_one_letter_siblings():
    rec = Reconciler()

    # Typo case: the SAME coastal city, one letter off, with NOTHING that marks
    # them as two distinct places -> MERGE. (The details are deliberately
    # compatible. An earlier version gave one "a coastal city" and the other
    # "ruled by an imperium"; a correctly-conservative model read city-vs-imperium
    # as a real distinctness signal and routed them to possible_duplicates instead
    # of merging -- so the fixture now matches its own "nothing distinguishes them"
    # premise, leaving only the spelling typo. Two source_files still exercise the
    # cross-source quote merge.)
    maltaav = Location(name="Maltaav", details=[det("a coastal trading city", "session1.txt")],
                       supporting_quotes=[Quote(text="we sailed into Maltaav",
                                                speaker="dm", source_file="session1.txt")])
    maltraav = Location(name="Maltraav", details=[det("a city on the coast", "session2.txt")],
                        supporting_quotes=[Quote(text="back at the docks in Maltraav",
                                                 speaker="dm", source_file="session2.txt")])
    merged_places = rec.reconcile([maltaav, maltraav])
    _eyeball("reconciler — Maltaav/Maltraav (expect MERGE)", merged_places)
    assert len(merged_places) == 1  # collapsed to one

    # Sibling case: one letter off but explicitly different people -> DO NOT MERGE.
    cj = Character(name="CJ", is_pc=True, player_name="Hannah",
                   details=[det("has a sister named DJ", "dndgroup.txt")],
                   supporting_quotes=[Quote(text="CJ's sister is DJ",
                                            speaker="player_a", source_file="dndgroup.txt")])
    dj = Character(name="DJ", details=[det("CJ's sister", "dndgroup.txt")],
                   supporting_quotes=[Quote(text="DJ, CJ's sister, joined us",
                                            speaker="player_a", source_file="dndgroup.txt")])
    result = rec.reconcile([cj, dj])
    _eyeball("reconciler — CJ/DJ siblings (expect NO merge)", result)
    names = sorted(e.name for e in result)
    assert names == ["CJ", "DJ"]  # both survive as separate entries


# --- Phase 4.1b: order_history (the timeline pass) --------------------------

@pytest.mark.integration
def test_order_history_sorts_dates_and_places_relative_clue():
    rec = Reconciler()
    events = [
        HistoryEvent(name="The Sundering", description="A cataclysm in 342 AR.",
                     scope="world", date_text="342 AR"),
        HistoryEvent(name="The Founding", description="The kingdom was founded in 100 AR.",
                     scope="world", date_text="100 AR"),
        HistoryEvent(name="The Reckoning",
                     description="A war that broke out shortly before the Sundering.",
                     scope="regional"),  # undated, relative clue -> before The Sundering
    ]
    out = rec.order_history(events)
    pos = {e.name: e.chronological_position for e in out}
    cal = {e.name: e.calendar_system for e in out}
    # dated events sorted by date; the relative event lands before the Sundering
    assert pos["The Founding"] < pos["The Reckoning"] < pos["The Sundering"]
    assert cal["The Founding"] == cal["The Sundering"]      # same AR timeline
    assert cal["The Reckoning"] == cal["The Sundering"]     # woven onto that timeline


@pytest.mark.integration
def test_order_history_multi_system_keeps_timelines_separate():
    rec = Reconciler()
    events = [
        HistoryEvent(name="AR Event", description="Happened in 500 AR.", scope="world",
                     date_text="500 AR"),
        HistoryEvent(name="Era Event", description="Happened in the 4th Era, year 200.",
                     scope="world", date_text="4th Era 200"),
    ]
    out = rec.order_history(events)
    cal = {e.name: e.calendar_system for e in out}
    # two different calendars -> two different system labels (not lumped together)
    assert cal["AR Event"] is not None and cal["Era Event"] is not None
    assert cal["AR Event"] != cal["Era Event"]
