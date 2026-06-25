"""Phase 4.1a -- LIVE / integration test for the reconciler MERGE step.

One real Anthropic API call proving the headline case end-to-end: the LLM must
MERGE a one-letter typo of the same place ("Maltaav"/"Maltraav") yet must NOT
merge two genuinely different people whose names differ by one letter when the
text states they're siblings ("CJ"/"DJ"). The deterministic Python (combiner,
vetoes, validator) is covered exhaustively in test_reconciler.py; this test is
only here to prove the LLM half makes the right call on the contrasting pair.

Gating mirrors test_extractors_integration.py: marked ``integration`` (deselected
by default -- a plain ``pytest`` skips it), and ``skipif``-ed when
``ANTHROPIC_API_KEY`` is absent. The repo-root ``conftest.py`` loads ``.env``
first, so a key sitting only in ``.env`` is visible before the skip is evaluated.
"""

import os
from pathlib import Path

import pytest

from models.lore import Location, Character, Quote
from agents.reconciler import Reconciler

try:
    from tests.eyeball import format_entities
except ImportError:  # eyeball is committed, but stay robust to a partial checkout
    format_entities = None


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
    maltaav = Location(name="Maltaav", details=["a coastal trading city"],
                       supporting_quotes=[Quote(text="we sailed into Maltaav",
                                                speaker="dm", source_file="session1.txt")])
    maltraav = Location(name="Maltraav", details=["a city on the coast"],
                        supporting_quotes=[Quote(text="back at the docks in Maltraav",
                                                 speaker="dm", source_file="session2.txt")])
    merged_places = rec.reconcile([maltaav, maltraav])
    _eyeball("reconciler — Maltaav/Maltraav (expect MERGE)", merged_places)
    assert len(merged_places) == 1  # collapsed to one

    # Sibling case: one letter off but explicitly different people -> DO NOT MERGE.
    cj = Character(name="CJ", is_pc=True, player_name="Hannah",
                   details=["has a sister named DJ"],
                   supporting_quotes=[Quote(text="CJ's sister is DJ",
                                            speaker="player_a", source_file="dndgroup.txt")])
    dj = Character(name="DJ", details=["CJ's sister"],
                   supporting_quotes=[Quote(text="DJ, CJ's sister, joined us",
                                            speaker="player_a", source_file="dndgroup.txt")])
    result = rec.reconcile([cj, dj])
    _eyeball("reconciler — CJ/DJ siblings (expect NO merge)", result)
    names = sorted(e.name for e in result)
    assert names == ["CJ", "DJ"]  # both survive as separate entries
