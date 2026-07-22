"""The dropped-detail warnings in BaseExtractor._resolve_quote now name the DETAIL
(the fact a quote was meant to support), not just the quote (agents/base_extractor.py).

Offline (fake client). When a detail-carrying extractor drops a detail -- because the
quote isn't verbatim, or cites a bad id -- the WARNING now appends `| detail: '<text>'`
so a human reading the log can judge whether the quote actually backed the claim. The
History extractor has no per-quote fact (a flat quote list), so it passes None and the
suffix is correctly ABSENT (no noisy `detail: None`).
"""

import json
import logging
from datetime import datetime

from agents.locations_extractor import LocationsExtractor
from agents.history_extractor import HistoryExtractor
from models.message import Message


# --- fake client (standard per-phase copy) ---------------------------------- #

class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response([_Block(self._response)])


class _Client:
    def __init__(self, response):
        self.messages = _Messages(response)


def _msg(content):
    return Message(sender="Matt", timestamp=datetime(2024, 1, 1, 12, 0, 0),
                   content=content, source_file="group.txt")


# --- the enriched warnings -------------------------------------------------- #

def test_non_verbatim_drop_names_the_detail(caplog):
    messages = [_msg("Lake Mundi is a massive central lake.")]
    response = json.dumps([{
        "name": "Lake Mundi",
        "aliases": [],
        "details": [{  # quote is NOT in the message -> dropped
            "detail": "Home of dragons",
            "quote": "Lake Mundi is full of dragons",
            "source_id": 0,
        }],
    }])
    agent = LocationsExtractor(client=_Client(response))

    with caplog.at_level(logging.WARNING):
        agent.extract(messages)

    drop = next(r for r in caplog.records
                if "dropping this detail" in r.getMessage() and r.levelno == logging.WARNING)
    msg = drop.getMessage()
    assert "detail: 'Home of dragons'" in msg          # the fact is now visible
    assert "Lake Mundi is full of dragons" in msg      # alongside the offending quote


def test_out_of_range_source_id_drop_names_the_detail(caplog):
    messages = [_msg("Lake Mundi is real.")]            # one message -> id 0 only
    response = json.dumps([{
        "name": "Lake Mundi",
        "aliases": [],
        "details": [{
            "detail": "Cited a nonexistent message",
            "quote": "Lake Mundi is real",
            "source_id": 9,                             # out of range
        }],
    }])
    agent = LocationsExtractor(client=_Client(response))

    with caplog.at_level(logging.WARNING):
        agent.extract(messages)

    drop = next(r for r in caplog.records
                if "dropping this detail" in r.getMessage() and r.levelno == logging.WARNING)
    assert "detail: 'Cited a nonexistent message'" in drop.getMessage()


def test_history_drop_omits_the_detail_suffix(caplog):
    # History has no per-quote fact, so it passes detail_text=None -> the suffix must
    # NOT appear (we don't want a noisy "detail: None" on every history drop).
    messages = [_msg("The Empire rose and fell.")]
    response = json.dumps([{
        "name": "The Fall",
        "aliases": [],
        "description": "The Empire fell in a cataclysm.",
        "scope": "world",
        "quotes": [{"quote": "a line that is not verbatim anywhere", "source_id": 0}],
    }])
    agent = HistoryExtractor(client=_Client(response))

    with caplog.at_level(logging.WARNING):
        agent.extract(messages)

    drops = [r.getMessage() for r in caplog.records if "dropping this detail" in r.getMessage()]
    assert drops                                        # the bad quote WAS dropped+logged
    assert all("detail:" not in m for m in drops)       # ...but with no detail suffix
