"""Player-name fold as the EXTRACTOR really emits it (run-1.13, note #4).

The existing test_reconciler_player_fold.py builds the phantom with player_name populated
(the old fold's precondition). In the REAL pipeline the characters extractor NULLS
player_name for an undeclared character, so a phantom named after a player arrives as
`player_name=None` -- which made the old is_pc AND player_name==name fold impossible to
fire. This file exercises that real shape: the loosened fold catches both a PC-style
phantom ('Sam' -> Krigius) and an NPC-style phantom ('Conrad' -> CJ), strips the player
key from the aliases, and thereby unblocks prose de-conflation. Offline, no API.
"""

import json

from agents.reconciler import Reconciler, _merge_declared_characters
from agents.prose_agent import build_deconflation_map
from player_map import declared_groups_with_players
from models.lore import Alias, Character


# --- fake client ------------------------------------------------------------ #
class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


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


NO_MERGE = json.dumps({"merges": [], "possible_duplicates": []})

# Sam plays Krigius; Conrad plays CJ. (The player_map maps a PLAYER key -> their
# character's declared names; "Sam"/"Conrad" are keys, never character values.)
PM = {"Sam": ["Krigius Krieger", "Kriggy"], "Conrad": ["CJ"]}
_pairs = declared_groups_with_players(PM)
GROUPS = [g for _, g in _pairs]
KEYS = [p for p, _ in _pairs]


def ch(name, is_pc=True, player=None, aliases=None):
    # Default player=None: how the extractor emits an UNDECLARED character.
    return Character(name=name, is_pc=is_pc, player_name=player,
                     aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


def declared_pc(name, player):
    # How the extractor emits a DECLARED character: config stamps player_name + is_pc.
    return ch(name, is_pc=True, player=player)


# --- the fold on real extractor shapes -------------------------------------- #
def test_pc_phantom_with_null_playername_folds():
    # 'Sam' phantom: is_pc=True (a prince), player_name=None (extractor nulled it).
    krig = declared_pc("Krigius Krieger", "Sam")
    phantom = ch("Sam", is_pc=True, player=None)
    out = _merge_declared_characters([krig, phantom], "Character", GROUPS, KEYS)
    assert len(out) == 1
    assert out[0].name == "Krigius Krieger"
    assert out[0].player_name == "Sam"                      # the declared player survives the merge


def test_npc_phantom_folds_too():
    # 'Conrad' phantom arrives is_pc=False (an NPC bodyguard page) -- the case the old
    # is_pc-gated fold could never catch. Ground-truth fold takes it anyway.
    cj = declared_pc("CJ", "Conrad")
    conrad = ch("Conrad", is_pc=False, player=None)
    out = _merge_declared_characters([cj, conrad], "Character", GROUPS, KEYS)
    assert len(out) == 1
    assert out[0].name == "CJ"


def test_player_key_is_not_kept_as_an_alias():
    krig = declared_pc("Krigius Krieger", "Sam")
    phantom = ch("Sam", is_pc=True, player=None, aliases=["The Prince"])
    out = _merge_declared_characters([krig, phantom], "Character", GROUPS, KEYS)
    alias_texts = {a.text.strip().lower() for a in out[0].aliases}
    assert "sam" not in alias_texts                         # player key stripped
    assert "the prince" in alias_texts                      # a genuine alias survives the strip


# --- through reconcile(): the fold unblocks de-conflation ------------------- #
def test_reconcile_folds_both_and_unblocks_deconfliction():
    rec = Reconciler(client=FakeClient([NO_MERGE]), player_map=PM)
    out = rec.reconcile([
        declared_pc("Krigius Krieger", "Sam"),
        ch("Sam", is_pc=True, player=None),          # PC phantom
        declared_pc("CJ", "Conrad"),
        ch("Conrad", is_pc=False, player=None),      # NPC phantom
    ])
    assert sorted(c.name for c in out) == ["CJ", "Krigius Krieger"]

    # With the phantoms folded and their player-key names stripped from the aliases,
    # 'sam'/'conrad' are no longer in-world character names, so the de-conflation guard
    # no longer bails -- both player->character rewrites are emitted.
    dmap = build_deconflation_map(out)
    assert dmap.get("sam") == "Krigius Krieger"
    assert dmap.get("conrad") == "CJ"
