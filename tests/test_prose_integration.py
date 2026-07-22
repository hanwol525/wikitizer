"""Phase C -- LIVE / integration tests for the ProseAgent copyedit.

The copyedit REWRITE is LLM behavior (de-dup, no fact-loss, no invented relationships),
so it can only be checked against real Claude. De-conflation itself is now deterministic
(covered offline in test_prose_deconflation.py); here we run it first, then let the LLM
copyedit the de-conflated text. Marked `integration`, deselected by default; opt in with
`pytest -m integration`. Skips when ANTHROPIC_API_KEY is absent. Assertions are LOOSE.
"""

import os

import pytest

from agents.prose_agent import (
    ProseAgent, build_deconflation_map, deconflate_events)
from models.lore import Character, Detail, HistoryEvent, Location, Scope


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="live prose tests need ANTHROPIC_API_KEY",
    ),
]


def _det(text):
    return Detail(text=text, source_files=["g.txt"])


def test_prose_copyedits_a_deconflated_event_without_losing_facts():
    # A merged event whose concatenated description names the same person as the PLAYER
    # ("Sam") and the CHARACTER ("Kriggy Krieger"). De-conflation (deterministic) rewrites
    # "Sam" -> the character; then the LLM copyedit must collapse the redundancy into ONE
    # account, keep the non-player name (Conrad), and invent nothing.
    chars = [Character(name="Kriggy Krieger", is_pc=True, player_name="Sam")]
    deconf = build_deconflation_map(chars)
    ev = HistoryEvent(
        name="Sam's Departure",
        description=("After his military failures, Sam ran away to prove himself.\n\n"
                     "Kriggy Krieger was removed from duty and departed on an adventure.\n\n"
                     "Sam, a royal figure, left the Imperium with his bodyguard Conrad."),
        scope=Scope.WORLD,
    )
    deconflated = deconflate_events([ev], deconf)
    # the title was de-conflated deterministically already:
    assert "Kriggy" in deconflated[0].name and "sam" not in deconflated[0].name.lower()

    agent = ProseAgent()
    out = agent.polish_events(deconflated)
    prose = out[0].prose
    assert prose, "prose should be set"
    assert "Kriggy" in prose                 # the character name is used
    assert "Conrad" in prose                 # a non-player name is preserved (not dropped)
    assert "sam" not in prose.lower()        # no bare player name left after de-conflation


def test_prose_dedups_repeated_facts_without_dropping_distinct_ones():
    e = Location(
        name="Gol",
        details=[_det("One of the largest continents"),
                 _det("A very large continent"),          # restatement of the above
                 _det("Home to dwarven strongholds")],    # a distinct fact
    )
    agent = ProseAgent()
    out = agent.polish_entities([e])
    prose = out[0].prose
    assert prose
    assert "dwarven" in prose.lower()        # the distinct fact survives
    assert "continent" in prose.lower()      # the (deduped) size fact survives
