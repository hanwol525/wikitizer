"""C2b: surfacing spelling-close candidate pairs to the reconciler LLM.

At scale the model silently omits merges it should make (e.g. "Maltaav"/"Maltraav").
`_candidate_pairs` computes an advisory list of near-duplicate names and
`_build_user_message` appends it to the USER message so the model must adjudicate each
pair. It NEVER auto-merges (CJ/DJ siblings stay the model's call), and the SYSTEM prompt
is untouched (prompt cache intact). Offline, no API.
"""

from agents.reconciler import (
    Reconciler,
    RECONCILER_SYSTEM_PROMPT,
    _candidate_pairs,
    _edit_distance,
)
from models.lore import Location


class _Fake:
    class messages:
        @staticmethod
        def create(**kwargs):
            raise AssertionError("no API call expected in these tests")


def loc(name):
    return Location(name=name)


# --- _edit_distance -------------------------------------------------------- #
def test_edit_distance_values():
    assert _edit_distance("maltaav", "maltraav") == 1     # one insertion
    assert _edit_distance("abc", "abc") == 0
    assert _edit_distance("kitten", "sitting") == 3


# --- _candidate_pairs ------------------------------------------------------ #
def test_finds_one_letter_off_and_article_descriptor_pairs():
    entries = [loc("Maltaav"), loc("Maltraav"), loc("Gol"),
               loc("The Imperium"), loc("Krieger Imperium")]
    pairs = _candidate_pairs(entries)
    idx_pairs = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in idx_pairs        # Maltaav / Maltraav -> one letter off
    assert (3, 4) in idx_pairs        # The Imperium / Krieger Imperium -> plus extra words
    # Gol is far from everything -> never paired
    assert not any(2 in (i, j) for i, j, _ in pairs)


def test_short_one_letter_names_are_not_surfaced():
    # A one-letter gap in a 2-char name (CJ/DJ) is a DIFFERENT name, not a typo.
    pairs = _candidate_pairs([loc("CJ"), loc("DJ")])
    assert pairs == []


def test_identical_names_are_not_candidates():
    # Identical names are handled by the deterministic floor, not surfaced here.
    pairs = _candidate_pairs([loc("Gol"), loc("gol")])
    assert pairs == []


def test_candidate_limit_is_respected():
    entries = [loc(f"Name{i:04d}") for i in range(200)]   # many near-identical
    pairs = _candidate_pairs(entries, limit=10)
    assert len(pairs) == 10


# --- _build_user_message injects the block; system prompt untouched -------- #
def test_user_message_carries_the_candidate_block():
    rec = Reconciler(client=_Fake())
    msg = rec._build_user_message([loc("Maltaav"), loc("Maltraav")])
    assert "POSSIBLE DUPLICATES TO ADJUDICATE" in msg
    assert '[0] "Maltaav" vs [1] "Maltraav"' in msg


def test_system_prompt_is_unchanged_by_the_candidate_feature():
    rec = Reconciler(client=_Fake())
    assert rec.system_prompt == RECONCILER_SYSTEM_PROMPT
    assert "POSSIBLE DUPLICATES TO ADJUDICATE" not in RECONCILER_SYSTEM_PROMPT


def test_no_candidate_block_when_no_similar_names():
    rec = Reconciler(client=_Fake())
    msg = rec._build_user_message([loc("Gol"), loc("Eglon")])
    assert "POSSIBLE DUPLICATES TO ADJUDICATE" not in msg
