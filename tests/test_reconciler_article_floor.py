"""The deterministic article-variant merge floor (output-quality batch).

`_merge_identical_names` keys the FULL name, so "The Citadel" and "Citadel" slip past it
and survive as two pages -- which then makes the cross-link pass drop their shared surface
as "claimed by 2 entities", so neither links. `_merge_article_variants` is the net: it
force-merges entries whose names are identical after stripping a leading the/a/an, AFTER
the LLM decision and the identical-name floor. It must NOT merge the descriptor KIND-trap
("Krieger" vs "Krieger Imperium"), must skip HistoryEvent, and must still honor the
Character player_name veto. Offline, no API.
"""

import json

from agents.reconciler import Reconciler, _merge_article_variants
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


# --- _merge_article_variants ----------------------------------------------- #
def test_merges_the_x_with_bare_x():
    out = _merge_article_variants([loc("The Citadel"), loc("Citadel")], "Location")
    assert len(out) == 1
    merged = out[0]
    # First-seen name is the heading; the bare form survives as an alias (nothing lost).
    assert merged.name == "The Citadel"
    alias_texts = {a.text.strip().lower() for a in merged.aliases}
    assert "citadel" in alias_texts


def test_merges_multiword_article_variant():
    out = _merge_article_variants([loc("Kraken Clan"), loc("The Kraken Clan")], "Location")
    assert len(out) == 1
    assert out[0].name == "Kraken Clan"


def test_leaves_descriptor_kind_trap_separate():
    # These differ by a CONTENT word, not an article -> different article-stripped keys,
    # so the floor never groups them (a person/family vs. the empire named after them).
    out = _merge_article_variants([loc("Krieger"), loc("Krieger Imperium")], "Location")
    assert len(out) == 2


def test_unrelated_names_untouched():
    out = _merge_article_variants([loc("Gol"), loc("The Citadel"), loc("Riverton")], "Location")
    assert sorted(e.name for e in out) == ["Gol", "Riverton", "The Citadel"]


def test_skips_history_events():
    # Event names are model-generated labels; leave article variants alone.
    out = _merge_article_variants([hev("The Battle"), hev("Battle")], "History")
    assert len(out) == 2


def test_character_player_clash_still_vetoes():
    # "The Sam" / "Sam" collapse to one key, but two different players => veto, kept apart.
    a = Character(name="The Sam", is_pc=True, player_name="Alice")
    b = Character(name="Sam", is_pc=True, player_name="Bob")
    out = _merge_article_variants([a, b], "Character")
    assert len(out) == 2


def test_character_article_variant_merges_without_clash():
    a = Character(name="The Warden", is_pc=False, player_name=None)
    b = Character(name="Warden", is_pc=False, player_name=None)
    out = _merge_article_variants([a, b], "Character")
    assert len(out) == 1


# --- end-to-end through reconcile() ---------------------------------------- #
def test_reconcile_floor_merges_article_variant_the_llm_omitted():
    # The LLM returns an EMPTY decision; the floor still collapses "The Citadel"/"Citadel".
    rec = Reconciler(client=FakeClient([NO_MERGE]))
    out = rec.reconcile([loc("The Citadel"), loc("Citadel")])
    assert len(out) == 1
