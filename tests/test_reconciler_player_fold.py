"""The player-name fold in the reconciler's declared-merge floor.

When the player_map declares Hannah=CJ, the characters extractor sometimes ALSO mints a
phantom character literally named 'Hannah' -- the player minted as their own PC. That
phantom (a) renders as a spurious page and (b) makes prose de-conflation bail (the player's
name is also an in-world character name). The fold merges such a phantom into the declared
PC and strips the player's real name from the aliases, so it never shows as 'CJ (aka
Hannah)'. Run-1.13 loosened the fold to ground-truth-wins: it fires on the NAME match alone
(the extractor nulls player_name for undeclared characters, so the old is_pc/player_name
gate was dead code in the real pipeline); only an EXPLICIT contrary player tag still vetoes.
Offline, no API.
"""

import json

from agents.reconciler import Reconciler, _merge_declared_characters
from player_map import declared_groups_with_players
from models.lore import Alias, Character, Location


# --- fake client ------------------------------------------------------------ #
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


NO_MERGE = json.dumps({"merges": [], "possible_duplicates": []})


def ch(name, is_pc=True, player="Hannah", aliases=None):
    return Character(name=name, is_pc=is_pc, player_name=player,
                     aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


def _groups_and_keys(player_map):
    pairs = declared_groups_with_players(player_map)
    return [g for _, g in pairs], [p for p, _ in pairs]


PM = {"Hannah": ["CJ"]}
GROUPS, KEYS = _groups_and_keys(PM)


# --- the fold (unit) -------------------------------------------------------- #
def test_phantom_named_like_player_folds_into_declared_pc():
    cj = ch("CJ", aliases=["Ceej"])
    phantom = ch("Hannah")                      # name == player, is_pc, player_name == 'Hannah'
    out = _merge_declared_characters([cj, phantom], "Character", GROUPS, KEYS)

    assert len(out) == 1
    merged = out[0]
    assert merged.name == "CJ"                  # declared name wins the heading
    alias_texts = {a.text.strip().lower() for a in merged.aliases}
    assert "hannah" not in alias_texts          # the player's real name is stripped, not 'CJ (aka Hannah)'
    assert "ceej" in alias_texts                # a genuine in-world alias survives the strip


def test_npc_named_like_player_now_folds_ground_truth_wins():
    # BEHAVIOR REVERSAL (run-1.13, user's explicit call): a character named like a
    # declared player folds into that player's PC regardless of is_pc/player_name, because
    # the extractor nulls player_name for undeclared characters (so the old is_pc AND
    # player_name==name guard was dead code in the real pipeline). The accepted trade-off
    # is that a coincidental NPC sharing a player's first name is also swept in (logged
    # [REVIEW]). This is what fixes the real "Conrad is a 90-yo dwarf" NPC-page leak.
    cj = ch("CJ")
    npc = ch("Hannah", is_pc=False, player=None)   # arrives as the extractor emits it
    out = _merge_declared_characters([cj, npc], "Character", GROUPS, KEYS)
    assert len(out) == 1
    assert out[0].name == "CJ"


def test_pc_named_like_player_but_tagged_another_player_is_not_folded():
    cj = ch("CJ")
    weird = ch("Hannah", player="Sam")             # named Hannah but explicitly played by Sam
    out = _merge_declared_characters([cj, weird], "Character", GROUPS, KEYS)
    assert len(out) == 2                            # an explicit CONTRARY player tag vetoes the fold


def test_fold_disabled_without_player_keys():
    # Backward-compat: the old 3-arg call (no player_keys) must NOT fold.
    out = _merge_declared_characters([ch("CJ"), ch("Hannah")], "Character", GROUPS)
    assert len(out) == 2


def test_lone_phantom_without_a_real_pc_passes_through():
    # Only the phantom exists (no 'CJ' entry) -> nothing to merge into; it passes through
    # unchanged rather than being renamed to a name no member has (documents the limit).
    out = _merge_declared_characters([ch("Hannah")], "Character", GROUPS, KEYS)
    assert [c.name for c in out] == ["Hannah"]


def test_fold_coexists_with_normal_declared_merge():
    pm = {"Sam": ["Kriggy", "Ambrose Chamberlain"], "Hannah": ["CJ"]}
    groups, keys = _groups_and_keys(pm)
    out = _merge_declared_characters(
        [ch("Kriggy", player="Sam"), ch("Ambrose Chamberlain", player="Sam"),
         ch("CJ"), ch("Hannah")],
        "Character", groups, keys)
    names = sorted(c.name for c in out)
    assert names == ["CJ", "Kriggy"]            # Sam's two merged; CJ + Hannah-phantom folded


def test_non_character_types_untouched():
    locs = [Location(name="CJ"), Location(name="Hannah")]
    assert len(_merge_declared_characters(locs, "Location", GROUPS, KEYS)) == 2


# --- end-to-end through reconcile() ----------------------------------------- #
def test_reconcile_folds_phantom_and_unblocks_deconfliction():
    rec = Reconciler(client=FakeClient([NO_MERGE]), player_map=PM)
    out = rec.reconcile([ch("CJ"), ch("Hannah")])
    assert len(out) == 1
    assert out[0].name == "CJ"
    assert out[0].player_name == "Hannah"
    # player 'Hannah' now maps to exactly one character (CJ), so build_deconflation_map
    # won't drop it as ambiguous.
    from agents.prose_agent import build_deconflation_map
    assert build_deconflation_map(out) == {"hannah": "CJ"}
