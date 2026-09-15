"""Roster-aware subject-bleed [REVIEW] flag (run-1.16 §C).

The Characters extractor can attach a fact to the wrong character (CJ's fact landing on
Aerin's page). No code can verify a free-form sentence's subject without deleting legit
relationship facts, so this is LOG-ONLY: it flags a detail whose LEADING subject is a
different extracted character ("CJ doesn't...") and leaves a mid-sentence mention ("cousin
of Skjoldr") alone. Pure, offline, no API.
"""

import logging

from agents.reconciler import flag_subject_bleed
from models.lore import Alias, Character, Detail


def ch(name, *facts, aliases=None):
    return Character(
        name=name,
        aliases=[Alias(text=a, source_files=[]) for a in (aliases or [])],
        details=[Detail(text=f, source_files=["g.txt"]) for f in facts],
    )


def _flagged(characters, caplog):
    with caplog.at_level(logging.WARNING, logger="agents.reconciler"):
        flag_subject_bleed(characters)
    return "subject-bleed" in caplog.text


def test_leading_other_character_is_flagged(caplog):
    roster = [ch("CJ"), ch("Aerin", "CJ doesn't think she is lame")]
    assert _flagged(roster, caplog)


def test_leading_possessive_other_character_is_flagged(caplog):
    roster = [ch("CJ"), ch("Aerin", "CJ's whole opinion is that Aerin is fine")]
    assert _flagged(roster, caplog)


def test_mid_sentence_relationship_is_not_flagged(caplog):
    # Skjoldr appears mid-sentence as a legit relationship object, not the subject.
    roster = [ch("Skjoldr"), ch("Aerin", "The cousin of Skjoldr, a lake-elf guide")]
    assert not _flagged(roster, caplog)


def test_own_name_leading_is_not_flagged(caplog):
    roster = [ch("CJ"), ch("Aerin Wakestrider", "Aerin Wakestrider is a lake elf")]
    assert not _flagged(roster, caplog)


def test_non_character_leading_word_is_not_flagged(caplog):
    roster = [ch("CJ"), ch("Aerin", "She navigates dangerous storms on the lake")]
    assert not _flagged(roster, caplog)


def test_alias_of_another_character_leading_is_flagged(caplog):
    # The leading name is an ALIAS of a different character -> still a bleed.
    roster = [ch("Clara Jane", aliases=["CJ"]), ch("Aerin", "CJ commandeered the vessel")]
    assert _flagged(roster, caplog)
