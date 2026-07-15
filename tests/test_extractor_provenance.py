"""Phase 4.6 Part 3: the extractors tag ``name_sources`` and ``Alias.source_files``
from the batch's (file-pure) ``source_file`` -- the provenance the ``--exclude-sources``
carve depends on.

Each extractor's own test file already checks ``Detail.source_files``; this file pins
the NAME/ALIAS tagging, which was otherwise untested and is load-bearing: a dropped
``name_sources=[source]`` leaves every public entity with ``name_sources=[]`` -> the
carve's ``any(sf not in excluded for sf in [])`` is False -> the name is treated as
non-public -> the entity is re-headed to an alias or (the common no-alias case)
DROPPED, silently emptying the restricted doc while the suite stays green. (A confirmed
gap from the Part 3 adversarial review; proven by mutation-removing the kwarg.)

Parametrized across all six extractors via the shared FakeClient harness, so it also
exercises the real ``BaseExtractor.extract`` -> ``_build_entry`` path (the orchestrator's
end-to-end exclusion tests run through STUB extractors, never the real tagging).
"""

import json
from datetime import datetime

import pytest

from models.message import Message
from agents.locations_extractor import LocationsExtractor
from agents.characters_extractor import CharactersExtractor
from agents.history_extractor import HistoryExtractor
from agents.organization_extractor import OrganizationExtractor
from agents.item_extractor import ItemExtractor
from agents.people_and_cultures_extractor import PeopleAndCulturesExtractor


# --- fake client (same shape as the per-extractor test files) ---------------

class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, blocks):
        self.content = blocks


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _FakeResponse([_FakeTextBlock(spec)] if isinstance(spec, str) else spec)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _msg(content, source_file):
    return Message(sender="Matt", timestamp=datetime(2024, 1, 1, 12, 0, 0),
                   content=content, source_file=source_file)


# The response SHAPES differ (history has description + a flat quotes list, not
# details), so a builder per family. All carry a name AND a non-empty alias.

def _noun_response(name, alias, content):
    return json.dumps([{"name": name, "aliases": [alias],
                        "details": [{"detail": "d", "quote": content, "source_id": 0}]}])


def _char_response(name, alias, content):
    return json.dumps([{"name": name, "aliases": [alias], "is_pc": False, "player_name": None,
                        "details": [{"detail": "d", "quote": content, "source_id": 0}]}])


def _hist_response(name, alias, content):
    return json.dumps([{"name": name, "aliases": [alias], "description": "It happened.",
                        "scope": "world", "quotes": [{"quote": content, "source_id": 0}]}])


EXTRACTORS = [
    ("locations", lambda c: LocationsExtractor(client=c), _noun_response),
    ("organizations", lambda c: OrganizationExtractor(client=c), _noun_response),
    ("items", lambda c: ItemExtractor(client=c), _noun_response),
    ("people", lambda c: PeopleAndCulturesExtractor(client=c), _noun_response),
    ("characters", lambda c: CharactersExtractor(client=c, player_names={"Sam"}), _char_response),
    ("history", lambda c: HistoryExtractor(client=c), _hist_response),
]
_IDS = [e[0] for e in EXTRACTORS]


@pytest.mark.parametrize("label,make,resp", EXTRACTORS, ids=_IDS)
def test_extractor_tags_name_and_alias_provenance_from_batch_source(label, make, resp):
    content = "Lake Mundi is huge and also called The Pond"
    agent = make(_FakeClient([resp("Lake Mundi", "The Pond", content)]))
    result = agent.extract([_msg(content, "secret.txt")])
    assert len(result) == 1
    assert result[0].name_sources == ["secret.txt"]              # name tagged from the file
    assert [a.text for a in result[0].aliases] == ["The Pond"]
    assert result[0].aliases[0].source_files == ["secret.txt"]   # alias tagged from the file


@pytest.mark.parametrize("label,make,resp", EXTRACTORS, ids=_IDS)
def test_extractor_provenance_is_per_file_across_batches(label, make, resp):
    # Two files -> two file-pure batches; each entity's name/alias provenance is its
    # OWN file's. Goes red if an extractor drops name_sources=[source], reverts aliases
    # to bare strings, or batching stops being file-pure (a cross-file batch would tag
    # both entities with one file).
    ca = "Aville is nice and also called A-town"
    cb = "Bville is grand and also called B-town"
    agent = make(_FakeClient([resp("Aville", "A-town", ca), resp("Bville", "B-town", cb)]))
    result = agent.extract([_msg(ca, "a.txt"), _msg(cb, "b.txt")])
    by_name = {e.name: e for e in result}
    assert by_name["Aville"].name_sources == ["a.txt"]
    assert by_name["Aville"].aliases[0].source_files == ["a.txt"]
    assert by_name["Bville"].name_sources == ["b.txt"]
    assert by_name["Bville"].aliases[0].source_files == ["b.txt"]
