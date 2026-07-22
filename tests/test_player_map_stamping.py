"""Fix batch #3 / component C: the characters extractor stamps player_name + is_pc
authoritatively from the declared party (player_map).

Who plays a character is a fact only the group knows, so when the user has declared it
(by name OR alias), that wins over the LLM's guess. Absent map -> the LLM's
roster-validated guess stands. Offline (FakeClient, no key).
"""

import json
import logging
from datetime import datetime

from agents.characters_extractor import CharactersExtractor
from models.message import Message


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


def msg(content, sender="Matt"):
    return Message(sender=sender, timestamp=datetime(2024, 1, 1, 12, 0, 0),
                   content=content, source_file="g.txt")


def agent(response, roster, player_map=None):
    return CharactersExtractor(client=FakeClient([json.dumps(response)]),
                               player_names=roster, player_map=player_map)


def one(response, roster, messages, player_map=None):
    return agent(response, roster, player_map).extract(messages)[0]


# --- stamping --------------------------------------------------------------- #
def test_map_stamps_player_and_forces_is_pc():
    # LLM said NPC / no player; the map declares Kriggy is Sam's -> authoritative.
    resp = [{"name": "Kriggy", "aliases": [], "is_pc": False, "player_name": None,
             "details": [{"detail": "A noble", "quote": "Kriggy is a noble", "source_id": 0}]}]
    c = one(resp, {"Sam"}, [msg("Kriggy is a noble")], player_map={"Sam": ["Kriggy"]})
    assert c.player_name == "Sam"
    assert c.is_pc is True


def test_map_overrides_a_wrong_llm_guess():
    # LLM guessed Hannah (a roster member) for Kriggy; the map says Sam -> Sam wins.
    resp = [{"name": "Kriggy", "aliases": [], "is_pc": True, "player_name": "Hannah",
             "details": [{"detail": "A noble", "quote": "Kriggy is a noble", "source_id": 0}]}]
    c = one(resp, {"Sam", "Hannah"}, [msg("Kriggy is a noble")], player_map={"Sam": ["Kriggy"]})
    assert c.player_name == "Sam"


def test_map_matches_by_alias():
    # The record is headed "Ambrose Chamberlain"; the map lists it as one of Sam's aliases.
    resp = [{"name": "Ambrose Chamberlain", "aliases": [], "is_pc": False, "player_name": None,
             "details": [{"detail": "x", "quote": "Ambrose Chamberlain arrived", "source_id": 0}]}]
    c = one(resp, {"Sam"}, [msg("Ambrose Chamberlain arrived")],
            player_map={"Sam": ["Kriggy", "Ambrose Chamberlain"]})
    assert c.player_name == "Sam" and c.is_pc is True


def test_unlisted_character_drops_llm_guess_when_map_present():
    # Config-authoritative: when a player_map IS configured, an UNDECLARED character does
    # NOT keep the LLM's player_name guess. A plausible-but-wrong guess surviving here
    # (a narrator name on a character they didn't declare) fabricates a player_name clash
    # that vetoes a legitimate merge downstream and splits one character into two pages.
    resp = [{"name": "Tiberius", "aliases": [], "is_pc": True, "player_name": "Sam",
             "details": [{"detail": "x", "quote": "Tiberius rules", "source_id": 0}]}]
    c = one(resp, {"Sam"}, [msg("Tiberius rules")], player_map={"Sam": ["Kriggy"]})
    assert c.player_name is None         # dropped -- Tiberius isn't in the map; config wins
    assert c.is_pc is True               # is_pc left as the LLM's value; only player_name is dropped


def test_no_map_leaves_llm_guess_untouched():
    resp = [{"name": "Kriggy", "aliases": [], "is_pc": True, "player_name": "Sam",
             "details": [{"detail": "x", "quote": "Kriggy is a noble", "source_id": 0}]}]
    c = one(resp, {"Sam"}, [msg("Kriggy is a noble")])   # no player_map
    assert c.player_name == "Sam" and c.is_pc is True


def test_off_roster_declared_player_warns_but_is_used(caplog):
    # A declared player who isn't in the speaker-map roster is likely a typo -> warn,
    # but honor the user's config anyway.
    with caplog.at_level(logging.WARNING, logger="agents.characters_extractor"):
        resp = [{"name": "Kriggy", "aliases": [], "is_pc": False, "player_name": None,
                 "details": [{"detail": "x", "quote": "Kriggy is a noble", "source_id": 0}]}]
        c = one(resp, {"Sam"}, [msg("Kriggy is a noble")], player_map={"Samuel": ["Kriggy"]})
    assert c.player_name == "Samuel"
    assert any("not in the speaker-map roster" in r.getMessage() for r in caplog.records)
