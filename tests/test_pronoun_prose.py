"""The prose polish carries a declared character's pronouns into its payload (run-1.17 §F).

Only a Character with non-empty pronouns adds a "pronouns" key to its polish payload item
(so NPCs and non-character types stay minimal); the prompt then tells the model to use them.
We assert the wire payload, not the LLM's rewrite (that's the integration suite). Offline.
"""

import json

from agents.prose_agent import ProseAgent
from models.lore import Character, Detail, Location


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
        return _Resp(self._responses[0])


class FakeClient:
    def __init__(self, responses):
        self.messages = _Messages(responses)


def det(t):
    return Detail(text=t, source_files=["g.txt"])


def _sent_payload(agent):
    """The JSON payload string sent in the (only) user message."""
    msgs = agent.client.messages.calls[0]["messages"]
    return "".join(
        block["text"] if isinstance(block, dict) else str(block)
        for m in msgs for block in (m["content"] if isinstance(m["content"], list) else [m["content"]])
    ) if isinstance(msgs[0]["content"], list) else msgs[0]["content"]


def test_character_pronouns_go_into_the_payload():
    ch = Character(name="Aerin", pronouns=["they", "them"], details=[det("She sails the lake")])
    agent = ProseAgent(client=FakeClient([json.dumps([{"id": 0, "body": "They sail the lake."}])]))
    out = agent.polish_entities([ch])
    sent = _sent_payload(agent)
    assert '"pronouns"' in sent and "they" in sent          # pronouns threaded to the model
    assert out[0].prose == "They sail the lake."


def test_non_character_has_no_pronouns_key():
    loc = Location(name="Gol", details=[det("A vast continent")])
    agent = ProseAgent(client=FakeClient([json.dumps([{"id": 0, "body": "A vast continent."}])]))
    agent.polish_entities([loc])
    assert '"pronouns"' not in _sent_payload(agent)         # omitted for non-characters


def test_character_without_pronouns_has_no_key():
    ch = Character(name="Ned", details=[det("A quiet farmer")])   # NPC: no declared pronouns
    agent = ProseAgent(client=FakeClient([json.dumps([{"id": 0, "body": "A quiet farmer."}])]))
    agent.polish_entities([ch])
    assert '"pronouns"' not in _sent_payload(agent)
