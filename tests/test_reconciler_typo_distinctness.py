"""The typo-floor distinctness gate (run-1.14, note #4 regression).

Run-1.13's declared-typo floor folded 'Gaerin' into 'Aerin' -- but the LLM had explicitly
parked them in possible_duplicates as siblings. The fuzzy folds (typo + token) now SKIP any
name the LLM flagged that way, so a deterministic floor can't override the model's stated
distinctness. An UNflagged typo (Kriggius) still folds. Offline, no API.
"""

import json

from agents.reconciler import Reconciler, _merge_declared_characters
from player_map import declared_groups_with_players
from models.lore import Alias, Character


# --- fake client ------------------------------------------------------------ #
class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Msgs:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return type("R", (), {"content": [_Block(spec)]})


class FakeClient:
    def __init__(self, responses):
        self.messages = _Msgs(responses)


def ch(name, is_pc=True, player=None):
    return Character(name=name, is_pc=is_pc, player_name=player)


def _gk(pm):
    pairs = declared_groups_with_players(pm)
    return [g for _, _, g in pairs], [p for p, _, _ in pairs]


# --- unit: the gate on _merge_declared_characters --------------------------- #
def test_flagged_sibling_is_not_typo_folded():
    groups, keys = _gk({"Hannah": ["Aerin", "Aerin Wakestrider"]})
    aerin = ch("Aerin Wakestrider", player="Hannah")
    gaerin = ch("Gaerin")                                 # one edit from declared "Aerin"

    # baseline: with no distinctness signal, the typo floor folds it
    assert len(_merge_declared_characters([aerin, gaerin], "Character", groups, keys)) == 1

    # with the LLM's possible-duplicate flag, the fuzzy fold is skipped -> kept separate
    guarded = _merge_declared_characters([aerin, gaerin], "Character", groups, keys,
                                         possible_dup_names={"gaerin"})
    assert len(guarded) == 2


def test_unflagged_typo_still_folds():
    groups, keys = _gk({"Sam": ["Krigius", "Krigius Krieger"]})
    out = _merge_declared_characters(
        [ch("Krigius Krieger", player="Sam"), ch("Kriggius")], "Character", groups, keys,
        possible_dup_names={"someone-else"})              # Kriggius NOT flagged
    assert len(out) == 1


# --- integration: reconcile() honors its own possible_duplicates ------------ #
def test_reconcile_possible_duplicate_beats_typo_floor():
    pm = {"Hannah": ["Aerin", "Aerin Wakestrider"]}
    decision = json.dumps({"merges": [],
                           "possible_duplicates": [{"members": [0, 1], "note": "siblings"}]})
    rec = Reconciler(client=FakeClient([decision]), player_map=pm)
    out = rec.reconcile([ch("Aerin Wakestrider", player="Hannah"), ch("Gaerin")])
    assert len(out) == 2
    assert "Gaerin" in {c.name for c in out}              # the sibling survived the floor
