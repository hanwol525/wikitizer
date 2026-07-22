"""Config-authoritative player_name (output-quality batch).

When a player_map IS configured, it is the sole source of truth for who plays a character:
an UNDECLARED character never keeps the LLM's player_name guess. This closes the root cause
behind the "Kriggy renders twice" bug -- a bogus guess (player_name='Conrad' on an
undeclared "Kriggy Krieger") survived the roster check and fabricated a merge veto in the
reconciler, splitting one character into two pages.

When NO player_map is configured, the legacy roster-validated-guess fallback is preserved
(the offline/synthetic path). Offline (FakeClient, no key, no network).
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


def one(response, roster, messages, player_map=None):
    agent = CharactersExtractor(client=FakeClient([json.dumps(response)]),
                                player_names=roster, player_map=player_map)
    return agent.extract(messages)[0]


# --- with a player_map configured: config is authoritative -------------------- #
def test_undeclared_char_with_map_drops_a_roster_valid_guess(caplog):
    # "Conrad" is a real person in the roster, so the roster check alone would KEEP it.
    # But the character isn't declared in the map, so the guess is dropped anyway.
    resp = [{"name": "Kriggy Krieger", "aliases": [], "is_pc": True, "player_name": "Conrad",
             "details": [{"detail": "A prince", "quote": "Kriggy Krieger is a prince",
                          "source_id": 0}]}]
    with caplog.at_level(logging.INFO, logger="agents.characters_extractor"):
        c = one(resp, {"Sam", "Conrad"}, [msg("Kriggy Krieger is a prince")],
                player_map={"Sam": ["Kriggy"]})
    assert c.player_name is None                    # the fabricated player is gone
    assert any("config is authoritative" in r.getMessage() for r in caplog.records)


def test_undeclared_char_with_map_and_no_guess_is_quiet(caplog):
    # No guess to drop -> player_name stays None and nothing is logged about it.
    resp = [{"name": "Tiberius", "aliases": [], "is_pc": False, "player_name": None,
             "details": [{"detail": "x", "quote": "Tiberius rules", "source_id": 0}]}]
    with caplog.at_level(logging.INFO, logger="agents.characters_extractor"):
        c = one(resp, {"Sam"}, [msg("Tiberius rules")], player_map={"Sam": ["Kriggy"]})
    assert c.player_name is None
    assert not any("config is authoritative" in r.getMessage() for r in caplog.records)


def test_declared_char_still_stamped_with_map():
    # The authoritative-override path is untouched: a declared character is stamped + is_pc.
    resp = [{"name": "Kriggy", "aliases": [], "is_pc": False, "player_name": None,
             "details": [{"detail": "A noble", "quote": "Kriggy is a noble", "source_id": 0}]}]
    c = one(resp, {"Sam"}, [msg("Kriggy is a noble")], player_map={"Sam": ["Kriggy"]})
    assert c.player_name == "Sam" and c.is_pc is True


def test_config_authority_matches_by_alias():
    # Declared via an alias -> still stamped; the drop branch only fires for a true miss.
    resp = [{"name": "Ambrose Chamberlain", "aliases": [], "is_pc": True, "player_name": "Conrad",
             "details": [{"detail": "x", "quote": "Ambrose Chamberlain arrived", "source_id": 0}]}]
    c = one(resp, {"Sam", "Conrad"}, [msg("Ambrose Chamberlain arrived")],
            player_map={"Sam": ["Kriggy", "Ambrose Chamberlain"]})
    assert c.player_name == "Sam"


# --- with NO player_map: legacy roster-validated guess is preserved ----------- #
def test_no_map_keeps_roster_valid_guess():
    # This locks the legacy fallback so the existing extractor test-suite stays valid.
    resp = [{"name": "Kriggy Krieger", "aliases": [], "is_pc": True, "player_name": "Conrad",
             "details": [{"detail": "x", "quote": "Kriggy Krieger is a prince", "source_id": 0}]}]
    c = one(resp, {"Sam", "Conrad"}, [msg("Kriggy Krieger is a prince")])   # no player_map
    assert c.player_name == "Conrad"                # roster-valid guess stands


def test_no_map_still_nulls_off_roster_guess():
    # The legacy roster check itself is unchanged: an off-roster guess is still nulled.
    resp = [{"name": "Kriggy", "aliases": [], "is_pc": True, "player_name": "Nobody",
             "details": [{"detail": "x", "quote": "Kriggy is a noble", "source_id": 0}]}]
    c = one(resp, {"Sam"}, [msg("Kriggy is a noble")])   # no player_map
    assert c.player_name is None
