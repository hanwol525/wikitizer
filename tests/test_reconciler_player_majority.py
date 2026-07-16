"""Fix B: player_name resolution is majority-wins on LLM evidence-based merges.

The reconciler used to VETO any character merge whose members carried two different
player_names -- so ONE mislabeled record ("Conrad" alongside three "Sam"s) blocked a
correct 5-way merge the LLM had already found. Now, on an LLM (evidence-based) merge,
a strict plurality wins and only a genuine tie still vetoes; the deterministic
identical-name FLOOR keeps the hard veto (two same-named PCs with different players
are two people). Offline, no API.
"""

import json

import pytest

from agents.reconciler import (
    Reconciler,
    _VetoMerge,
    _combine_group,
    _resolve_player_name,
)
from models.lore import Character


# --- fake client (same shape as the other reconciler tests) ----------------- #
class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _Resp(spec)


class FakeClient:
    def __init__(self, responses):
        self.messages = _Messages(responses)


def pc(name, player_name):
    return Character(name=name, is_pc=True, player_name=player_name)


# ======================================================================= #
# _resolve_player_name (unit)
# ======================================================================= #
def test_name_vs_none_returns_the_name_in_both_modes():
    members = [pc("Kriggy", "Sam"), pc("Kriggius", None)]
    assert _resolve_player_name(members, allow_majority=False) == "Sam"
    assert _resolve_player_name(members, allow_majority=True) == "Sam"


def test_all_none_returns_none():
    assert _resolve_player_name([pc("A", None), pc("B", None)], allow_majority=True) is None


def test_majority_wins_when_allowed():
    members = [pc("Kriggy", "Sam"), pc("Kriggius", "Sam"), pc("Ambrose", "Conrad")]
    assert _resolve_player_name(members, allow_majority=True) == "Sam"


def test_majority_counts_case_insensitively_but_returns_verbatim():
    members = [pc("A", "Sam"), pc("B", "sam"), pc("C", "Conrad")]
    assert _resolve_player_name(members, allow_majority=True) == "Sam"


def test_tie_still_vetoes_even_when_majority_allowed():
    members = [pc("Kriggy", "Sam"), pc("Kriggius", "Conrad")]
    with pytest.raises(_VetoMerge):
        _resolve_player_name(members, allow_majority=True)


def test_floor_vetoes_any_clash_even_a_lopsided_one():
    # allow_majority=False (the identical-name floor path): a 3-vs-1 split STILL vetoes.
    members = [pc("Sam", "Sam"), pc("Sam", "Sam"), pc("Sam", "Sam"), pc("Sam", "Conrad")]
    with pytest.raises(_VetoMerge):
        _resolve_player_name(members, allow_majority=False)


# ======================================================================= #
# _combine_group threads allow_majority through
# ======================================================================= #
def test_combine_group_majority_merges_the_mislabeled_record():
    members = [pc("Kriggy Krieger", "Sam"), pc("Kriggius", "Sam"), pc("Ambrose", "Conrad")]
    merged = _combine_group(members, "Kriggy Krieger", allow_majority=True)
    assert merged.player_name == "Sam"
    assert merged.name == "Kriggy Krieger"


def test_combine_group_default_still_vetoes_a_clash():
    members = [pc("Kriggy", "Sam"), pc("Kriggius", "Conrad")]
    with pytest.raises(_VetoMerge):
        _combine_group(members, "Kriggy")            # default allow_majority=False


# ======================================================================= #
# end-to-end through reconcile() -- the Kriggy scenario from the run
# ======================================================================= #
def _merge_decision(members, canonical):
    return json.dumps({"merges": [{"members": members, "canonical": canonical, "conflicts": []}],
                       "possible_duplicates": []})


def test_reconcile_llm_merge_survives_one_bad_player_name():
    entries = [pc("Kriggy Krieger", "Sam"), pc("Kriggius", "Sam"), pc("Ambrose", "Conrad")]
    rec = Reconciler(client=FakeClient([_merge_decision([0, 1, 2], "Kriggy Krieger")]))
    out = rec.reconcile(entries)
    assert len(out) == 1                              # the 5-way-style veto no longer fires
    assert out[0].player_name == "Sam"               # majority kept, "Conrad" overridden


def test_reconcile_llm_merge_with_a_true_tie_stays_separate():
    entries = [pc("Kriggy Krieger", "Sam"), pc("Kriggius", "Conrad")]
    rec = Reconciler(client=FakeClient([_merge_decision([0, 1], "Kriggy Krieger")]))
    out = rec.reconcile(entries)
    assert len(out) == 2                              # 1-vs-1 tie -> vetoed -> kept apart
