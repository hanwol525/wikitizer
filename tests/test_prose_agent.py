"""Phase C: the ProseAgent's copyedit pass (polish_entities / polish_events).

Offline (FakeClient, no key). The copyedit REWRITE quality is LLM behavior (integration
suite); here we lock the plumbing: the batch wire protocol (ids map back, model_copy
sets `prose`, originals untouched, details/quotes preserved), and every degrade path
(bad JSON / missing id / empty body -> prose stays None so the renderer falls back).

De-conflation (now a deterministic pre-pass, not the LLM's job) is tested separately in
test_prose_deconflation.py.
"""

import json

from agents.prose_agent import ProseAgent
from models.lore import Detail, HistoryEvent, Location, Quote, Scope


# --- fakes ------------------------------------------------------------------ #
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


def det(text, *sources):
    return Detail(text=text, source_files=list(sources))


def loc(name, details=None, prose=None, quotes=None):
    return Location(name=name,
                    details=[det(d) if isinstance(d, str) else d for d in (details or [])],
                    supporting_quotes=quotes or [],
                    prose=prose)


def hev(name, description="An event happened.", prose=None):
    return HistoryEvent(name=name, description=description, scope=Scope.WORLD, prose=prose)


def _bodies_json(*pairs):
    return json.dumps([{"id": i, "body": b} for i, b in pairs])


# --- polish_entities: happy path ------------------------------------------- #
def test_polish_entities_sets_prose_and_preserves_details_and_quotes():
    q = Quote(text="Gol is huge", speaker="M", source_file="g.txt")
    e = loc("Gol", details=["A big continent", "Home to dwarves"], quotes=[q])
    agent = ProseAgent(client=FakeClient([_bodies_json((0, "A vast continent, home to dwarves."))]))
    out = agent.polish_entities([e])
    assert out[0].prose == "A vast continent, home to dwarves."
    assert [d.text for d in out[0].details] == ["A big continent", "Home to dwarves"]
    assert out[0].supporting_quotes == [q]
    assert e.prose is None            # original not mutated (the carve reads it)


def test_polish_entities_payload_has_no_deconflation_map_or_inbound():
    e = loc("Gol", details=["A big continent"])
    agent = ProseAgent(client=FakeClient([_bodies_json((0, "A continent."))]))
    agent.polish_entities([e])
    user_msg = agent.client.messages.calls[0]["messages"][0]["content"]
    assert "deconflation_map" not in user_msg   # de-conflation is deterministic upstream
    assert "inbound" not in user_msg            # back-fill removed


# --- polish_entities: degrade paths ---------------------------------------- #
def test_polish_entities_degrades_on_unparseable_json():
    e = loc("Gol", details=["A big continent"])
    agent = ProseAgent(client=FakeClient(["this is not json at all"]))
    out = agent.polish_entities([e])
    assert out[0].prose is None
    assert [d.text for d in out[0].details] == ["A big continent"]   # facts preserved


def test_polish_entities_missing_id_leaves_that_entity_unpolished():
    a = loc("A", details=["fact a"])
    b = loc("B", details=["fact b"])
    agent = ProseAgent(client=FakeClient([_bodies_json((0, "Polished A."))]))   # only id 0
    out = agent.polish_entities([a, b])
    assert out[0].prose == "Polished A."
    assert out[1].prose is None


def test_polish_entities_empty_body_leaves_prose_none():
    e = loc("Gol", details=["A big continent"])
    agent = ProseAgent(client=FakeClient([_bodies_json((0, "   "))]))   # whitespace body
    out = agent.polish_entities([e])
    assert out[0].prose is None


def test_polish_entities_empty_list_makes_no_call():
    agent = ProseAgent(client=FakeClient([_bodies_json()]))
    assert agent.polish_entities([]) == []
    assert agent.client.messages.calls == []


# --- polish_events --------------------------------------------------------- #
def test_polish_events_sets_prose_and_preserves_description_and_name():
    ev = hev("The Departure", description="X left.\n\nX left again.")
    agent = ProseAgent(client=FakeClient([_bodies_json((0, "X departed to prove himself."))]))
    out = agent.polish_events([ev])
    assert out[0].prose == "X departed to prove himself."
    assert out[0].description == "X left.\n\nX left again."   # raw kept
    assert out[0].name == "The Departure"                    # title untouched by the LLM
    assert ev.prose is None                                   # original untouched


def test_polish_events_degrades_on_bad_json():
    ev = hev("The Founding", description="It was founded.")
    agent = ProseAgent(client=FakeClient(["nope not json"]))
    out = agent.polish_events([ev])
    assert out[0].prose is None
    assert out[0].description == "It was founded."


# --- model field ----------------------------------------------------------- #
def test_prose_field_defaults_none_and_model_copy_carries_it():
    e = Location(name="X")
    assert e.prose is None
    assert e.model_copy(update={"prose": "hi"}).prose == "hi"
