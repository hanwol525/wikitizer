"""C2a: the deterministic identical-name merge floor.

The reconciler's single giant LLM call silently omits valid merges at scale -- even
byte-identical names it is explicitly told to merge, with no log. `_merge_identical_names`
is the deterministic net: it force-merges same-name entries AFTER the LLM decision.
Offline, no API.
"""

import json

from agents.reconciler import (
    Reconciler,
    _merge_identical_names,
    _name_key,
)
from models.lore import Character, Detail, HistoryEvent, Location, Scope


# --- self-contained fake client -------------------------------------------- #
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


def loc(name, details=None):
    return Location(name=name, details=[Detail(text=d, source_files=["g.txt"])
                                        for d in (details or [])])


def hev(name):
    return HistoryEvent(name=name, description="An event.", scope=Scope.WORLD)


NO_MERGE = json.dumps({"merges": [], "possible_duplicates": []})


# --- _name_key ------------------------------------------------------------- #
def test_name_key_folds_case_whitespace_and_curly_punct():
    assert _name_key("Taken Lands") == _name_key("taken  lands")
    assert _name_key("O'Brien") == _name_key("O’Brien")   # curly vs straight apostrophe
    assert _name_key("Gol") != _name_key("Gel")


# --- _merge_identical_names ------------------------------------------------ #
def test_merges_byte_identical_and_case_variant_names():
    out = _merge_identical_names([loc("Taken Lands"), loc("taken lands"), loc("Gol")], "Location")
    names = sorted(e.name for e in out)
    assert names == ["Gol", "Taken Lands"]   # the two Taken Lands collapsed to one


def test_skips_history_events_identical_labels_stay_separate():
    # Event names are model-generated labels; two identical labels are NOT necessarily
    # the same event, so the floor leaves them alone.
    out = _merge_identical_names([hev("The Battle"), hev("The Battle")], "History")
    assert len(out) == 2


def test_character_player_clash_vetoes_the_identical_merge():
    # Two PCs with the same in-world name but DIFFERENT players are two different people.
    a = Character(name="Sam", is_pc=True, player_name="Alice")
    b = Character(name="Sam", is_pc=True, player_name="Bob")
    out = _merge_identical_names([a, b], "Character")
    assert len(out) == 2   # veto kept them separate


def test_identical_characters_same_player_do_merge():
    a = Character(name="Sam", is_pc=True, player_name="Alice")
    b = Character(name="Sam", is_pc=True, player_name=None)   # None doesn't clash
    out = _merge_identical_names([a, b], "Character")
    assert len(out) == 1


# --- end-to-end through reconcile() ---------------------------------------- #
def test_reconcile_floor_merges_identical_names_the_llm_omitted():
    # The LLM returns an EMPTY decision (the silent-omission bug); the floor still
    # merges the two identical "Taken Lands".
    rec = Reconciler(client=FakeClient([NO_MERGE]))
    out = rec.reconcile([loc("Taken Lands"), loc("Taken Lands")])
    assert len(out) == 1
    assert out[0].name == "Taken Lands"
