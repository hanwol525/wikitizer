"""Per-batch containment in BaseExtractor (agents/base_extractor.py).

Regression for a real production failure: a GLM run logged
`ERROR orchestrator: [REVIEW] organizations extractor failed; omitting that section.
Expecting value: line 913 column 1 (char 5016)` -- a raw `json.JSONDecodeError` (a *sibling*
of `ClaudeJSONError`, not caught by the old narrow `except ClaudeJSONError`) escaped a single
batch and deleted the ENTIRE section. `_extract_batch`'s docstring promises it "never raises";
these tests lock that contract for ANY exception, so one bad batch loses ~batch_size messages,
never the whole type.

Offline (fake client). We drive TWO batches (batch_size=1 over 2 same-file messages): batch 1's
call blows up, batch 2 returns a valid Location -> the surviving batch's entry must come through
and `extract()` must not raise.
"""

import json
import logging
from datetime import datetime

import pytest

from agents.locations_extractor import LocationsExtractor
from models.message import Message


# --- fake client whose create() can RAISE (spec is an exception) or return text ------ #

class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, specs):
        self._specs = list(specs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._specs.pop(0) if len(self._specs) > 1 else self._specs[0]
        if isinstance(spec, BaseException):
            raise spec                      # simulate an error escaping the underlying call
        return _Response([_Block(spec)])


class _Client:
    def __init__(self, specs):
        self.messages = _Messages(specs)


def _msg(content):
    return Message(sender="Matt", timestamp=datetime(2024, 1, 1, 12, 0, 0),
                   content=content, source_file="group.txt")


_GOOD_BATCH = json.dumps([{
    "name": "Riverton",
    "aliases": [],
    "details": [{"detail": "A river town", "quote": "Riverton is a river town", "source_id": 0}],
}])


@pytest.mark.parametrize("boom", [
    json.JSONDecodeError("Expecting value", "doc", 0),   # the exact sibling that escaped
    RecursionError("json_repair blew the stack"),        # deep-nesting path
    RuntimeError("provider APIError"),                   # any provider/adapter error
])
def test_one_bad_batch_is_contained_sibling_survives(boom, caplog):
    # Two messages, batch_size=1 -> two batches. Batch 1's call raises `boom`; batch 2 is good.
    messages = [_msg("Riverton is a river town"), _msg("also Riverton is a river town")]
    agent = LocationsExtractor(client=_Client([boom, _GOOD_BATCH]), batch_size=1)

    with caplog.at_level(logging.ERROR):
        result = agent.extract(messages)     # MUST NOT raise

    names = [loc.name for loc in result]
    assert "Riverton" in names               # the surviving batch's entry came through
    assert any("skipping" in r.getMessage() for r in caplog.records)  # the bad batch was logged


class _BoomOnFirstEntry(LocationsExtractor):
    """A subclass whose _build_entry raises on a sentinel entry, to prove a surprise error in
    the per-entry seam drops just THAT entry, not the whole batch."""

    def _build_entry(self, raw, batch):
        if raw.get("name") == "BOOM":
            raise RuntimeError("surprise entry failure")
        return super()._build_entry(raw, batch)


def test_bad_entry_is_skipped_batch_survives(caplog):
    messages = [_msg("Riverton is a river town")]
    response = json.dumps([
        {"name": "BOOM", "aliases": [], "details": []},                      # raises in _build_entry
        {"name": "Riverton", "aliases": [],
         "details": [{"detail": "A river town", "quote": "Riverton is a river town",
                      "source_id": 0}]},                                     # good sibling
    ])
    agent = _BoomOnFirstEntry(client=_Client([response]), batch_size=20)

    with caplog.at_level(logging.WARNING):
        result = agent.extract(messages)     # MUST NOT raise

    assert [loc.name for loc in result] == ["Riverton"]   # sibling survived, BOOM skipped
    assert any("entry build failed" in r.getMessage() for r in caplog.records)
