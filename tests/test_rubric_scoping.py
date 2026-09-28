"""Runner-level tests for evals/rubric/runner.py -- the corrections the review flagged:
#1 derived `complete` (a degraded run reports complete=False, never a false "full"),
#2 the normalized category->section join (a cosmetically-different rubric section name still
scopes), and the sourcing GATES (absent -> na, cites-nothing -> FAIL no-call). Fully offline.
"""

import json
from datetime import datetime, timezone

from evals.common.models import Status
from evals.rubric.loader import Rubric
from evals.rubric.runner import grade

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def R(*objs):
    return json.dumps({"results": list(objs)})


class ScriptedClient:
    model = "fake"
    provider = "openrouter"
    temperature = 0.6
    thinking = False
    base_url = None

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.replies.pop(0) if self.replies else "{}"


class FlakyClient:
    """Returns one scripted reply on the first call, then raises on every later call."""

    model = "fake"
    provider = "openrouter"
    temperature = 0.6
    thinking = False
    base_url = None

    def __init__(self, first_reply):
        self.first = first_reply
        self.n = 0

    def complete(self, system, user):
        self.n += 1
        if self.n == 1:
            return self.first
        raise RuntimeError("boom")


class RaisingClient:
    """Raises on every call -- so even the first check (judge_sections) degrades."""

    model = "fake"
    provider = "openrouter"
    temperature = 0.6
    thinking = False
    base_url = None

    def complete(self, system, user):
        raise RuntimeError("boom")


def _by_id(result):
    return {it.id: it for it in result.items}


# --- correction #1: a degraded run reports complete=False ------------------- #

def test_degraded_run_reports_incomplete():
    rubric = Rubric.model_validate({"sections": ["Locations"], "locations": ["Aldenburg"]})
    wof = ("# W\n\nsub\n\n---\n\n## Locations\n\n"
           "### <a id=\"a\"></a>Aldenburg\n\nA town.[^1]\n\n"
           "## Footnotes\n\n[^1]: `q` — B, dm.txt\n")
    # sections succeeds; presence raises -> presence items degrade to SKIPPED.
    client = FlakyClient(R({"present": True, "match": 1}))
    result = grade(wof, "w.md", rubric, votes=1, use_model=True, model_client=client, now=_NOW)
    items = _by_id(result)
    assert result.summary.complete is False                       # derived from the skipped item
    pres = items["rubric.presence.locations.aldenburg"]
    assert pres.status == Status.SKIPPED and "[REVIEW]" in pres.evidence.detail
    assert items["rubric.sections.locations"].status == Status.PASS


def test_sections_degrade_skips_presence_instead_of_failing_it():
    # If the FIRST check (judge_sections) errors, downstream presence must degrade to SKIPPED,
    # not be hard-FAILed as "section absent" (a content verdict it was never evaluated for).
    rubric = Rubric.model_validate({"sections": ["Locations"], "locations": ["Aldenburg"]})
    wof = ("# W\n\nsub\n\n---\n\n## Locations\n\n"
           "### <a id=\"a\"></a>Aldenburg\n\nA town.\n")
    result = grade(wof, "w.md", rubric, votes=1, use_model=True, model_client=RaisingClient(), now=_NOW)
    items = _by_id(result)
    assert result.summary.complete is False
    assert items["rubric.sections.locations"].status == Status.SKIPPED
    pres = items["rubric.presence.locations.aldenburg"]
    assert pres.status == Status.SKIPPED                          # NOT FAIL "section absent"
    assert "section check errored" in pres.evidence.detail
    assert "section absent" not in pres.evidence.detail
    assert items["rubric.sourcing.locations.aldenburg"].status == Status.NA


# --- correction #2: a cosmetically-different section name still scopes ------- #

def test_cosmetic_section_name_still_scopes_presence():
    # rubric names the section "People and Cultures" (no ampersand); the WOF renders it with '&'.
    rubric = Rubric.model_validate({
        "sections": [{"name": "People and Cultures"}],
        "people_and_cultures": ["The Krieg"],
    })
    wof = ("# W\n\nsub\n\n---\n\n## People & Cultures\n\n"
           "### <a id=\"krieg\"></a>The Krieg\n\nA warrior people of the north.\n")
    # call 1: sections (People and Cultures -> WOF section #1); call 2: presence (The Krieg present).
    client = ScriptedClient([R({"present": True, "match": 1}), R({"present": True, "match": 1})])
    result = grade(wof, "w.md", rubric, votes=1, use_model=True, model_client=client, now=_NOW)
    items = _by_id(result)
    pres = items["rubric.presence.people-and-cultures.the-krieg"]
    assert pres.status == Status.PASS                             # scoped correctly, NOT "section absent"
    assert "section absent" not in pres.evidence.detail


# --- sourcing gates --------------------------------------------------------- #

def test_sourcing_gates_absent_na_and_cites_nothing_fail():
    rubric = Rubric.model_validate({"sections": ["Locations"], "locations": ["Real", "Ghost"]})
    # "Real" is present but cites no footnote; "Ghost" is absent.
    wof = ("# W\n\nsub\n\n---\n\n## Locations\n\n"
           "### <a id=\"real\"></a>Real\n\nA place with no citation.\n")
    client = ScriptedClient([
        R({"present": True, "match": 1}),                          # sections: Locations -> #1
        R({"present": True, "match": 1}, {"present": False, "match": None}),  # presence: Real yes, Ghost no
    ])
    result = grade(wof, "w.md", rubric, votes=1, use_model=True, model_client=client, now=_NOW)
    items = _by_id(result)
    # present but cites nothing -> FAIL, no model call for support
    real_src = items["rubric.sourcing.locations.real"]
    assert real_src.status == Status.FAIL and "cites no footnote" in real_src.evidence.detail
    # absent -> na (don't double-penalize; its presence item already failed)
    ghost_src = items["rubric.sourcing.locations.ghost"]
    assert ghost_src.status == Status.NA
    assert items["rubric.presence.locations.ghost"].status == Status.FAIL
    # only the two model calls happened (sections + presence); support was gated out.
    assert client.calls == 2


def test_category_with_no_matching_section_fails_loud_not_silent(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    # rubric requires a locations entity but declares NO Locations section -> no match -> FAIL.
    rubric = Rubric.model_validate({"sections": ["Characters"], "locations": ["Aldenburg"]})
    wof = ("# W\n\nsub\n\n---\n\n## Characters\n\n### <a id=\"x\"></a>X\n\nbody.\n")
    client = ScriptedClient([R({"present": True, "match": 1})])     # sections: Characters present
    result = grade(wof, "w.md", rubric, votes=1, use_model=True, model_client=client, now=_NOW)
    items = _by_id(result)
    pres = items["rubric.presence.locations.aldenburg"]
    assert pres.status == Status.FAIL and "section absent" in pres.evidence.detail
    assert any("[REVIEW]" in r.getMessage() and "no matching" in r.getMessage()
               for r in caplog.records)                            # loud, not silent
