"""Phase 3.10 -- LIVE / integration tests for the six extractors.

These make REAL Anthropic API calls over hand-picked messages and check that the
right entities come out of the right extractor (and, just as importantly, do NOT
come out of the wrong one). They are the slow, network, costs-credits companion
to the fast mocked unit tests (``test_locations_extractor.py`` etc.), which are
untouched.

How they're gated (see pytest.ini + conftest.py):
  * Marked ``integration`` and DESELECTED by default -- a plain ``pytest`` run
    skips them. Opt in with ``pytest -m integration``.
  * They ``skipif`` themselves when ``ANTHROPIC_API_KEY`` is absent. The repo-root
    ``conftest.py`` calls ``load_dotenv()`` first, so a key sitting in ``.env`` is
    visible by the time this skip condition is evaluated (no false skip).

Assertions are deliberately LOOSE -- LLM output isn't deterministic, so we check
name/alias substring membership, counts, scope, and emptiness, not exact strings.
The fact-vs-quote faithfulness (did Claude hang the right fact on a real quote?)
is a human job: every run writes a per-extractor eyeball report to
``tests/output/`` for that.

The fixtures split by privacy: the committed synthetic subset
(``tests/fixtures/synthetic_messages.py``) always runs; the gitignored
real-campaign subset (``tests/fixtures/real_messages.py``) runs only when present.
"""

import os
import shutil
from pathlib import Path

import pytest

from agents.characters_extractor import CharactersExtractor
from agents.history_extractor import HistoryExtractor
from agents.item_extractor import ItemExtractor
from agents.locations_extractor import LocationsExtractor
from agents.organization_extractor import OrganizationExtractor
from agents.people_and_cultures_extractor import PeopleAndCulturesExtractor
from tests.eyeball import format_entities
from tests.fixtures.synthetic_messages import CASES as SYNTHETIC_CASES

try:
    # Gitignored; a fresh clone won't have it -- fall back to the synthetic subset
    # rather than erroring at collection time.
    from tests.fixtures.real_messages import CASES as REAL_CASES
except ImportError:
    REAL_CASES = []

ALL_CASES = SYNTHETIC_CASES + REAL_CASES


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="live extractor tests need ANTHROPIC_API_KEY",
    ),
]


# --- the test roster for the Characters extractor ---------------------------
# CharactersExtractor REQUIRES the real-player names (its anti-conflation anchor;
# see agents/characters_extractor.py). The five other extractors take no args.
# We use the fixtures' role-label senders as stand-in "real people", plus "Ryan"
# -- the name filled into the #20 name-collision case, which must be on the
# roster for the collision flag to fire.
TEST_ROSTER = ["dm", "exporter", "player_a", "player_b", "player_c", "Ryan"]


def _build_extractor(name):
    if name == "locations":
        return LocationsExtractor()
    if name == "characters":
        return CharactersExtractor(player_names=TEST_ROSTER)
    if name == "history":
        return HistoryExtractor()
    if name == "organizations":
        return OrganizationExtractor()
    if name == "items":
        return ItemExtractor()
    if name == "people":
        return PeopleAndCulturesExtractor()
    raise ValueError(f"unknown extractor short-name: {name!r}")


# Build each extractor at most once, lazily -- only when a non-skipped test
# actually runs (so collection never constructs a client / needs a key).
_EXTRACTOR_CACHE = {}


def _get_extractor(name):
    if name not in _EXTRACTOR_CACHE:
        _EXTRACTOR_CACHE[name] = _build_extractor(name)
    return _EXTRACTOR_CACHE[name]


# --- eyeball report wiring --------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="session", autouse=True)
def _reset_output_dir():
    """Clear tests/output/ once at the start of the run so the eyeball reports are
    fresh. Only runs when a live test actually executes (the module is otherwise
    deselected/skipped), so a plain ``pytest`` never creates the directory."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


def _append_report(extractor_name, case_id, result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{extractor_name}_review.txt"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(format_entities(result, f"{extractor_name} — {case_id}"))


# --- assertion helpers ------------------------------------------------------
def _matches(entity, needle):
    """True if ``needle`` (case-insensitive) is a substring of the entity's name
    or any of its aliases."""
    needle = needle.lower()
    if needle in entity.name.lower():
        return True
    return any(isinstance(a, str) and needle in a.lower() for a in entity.aliases)


def _apply_expectations(extractor_name, case_id, exp, result):
    names = [e.name for e in result]

    # Universal History contract: ordering is the 4.1b timeline pass's job, so the
    # extractor must leave chronological_position unset on EVERY event. Checked
    # whenever the history extractor runs, table or no table.
    if extractor_name == "history":
        unplaced = all(e.chronological_position is None for e in result)
        assert unplaced, (
            f"[{case_id}/{extractor_name}] every event must have "
            f"chronological_position=None; got "
            f"{[(e.name, e.chronological_position) for e in result]}"
        )

    for needle in exp.get("expect", []):
        assert any(_matches(e, needle) for e in result), (
            f"[{case_id}/{extractor_name}] expected an entity matching {needle!r}; "
            f"got {names}"
        )

    expect_any = exp.get("expect_any")
    if expect_any:
        assert any(_matches(e, n) for e in result for n in expect_any), (
            f"[{case_id}/{extractor_name}] expected an entity matching any of "
            f"{expect_any}; got {names}"
        )

    for needle in exp.get("reject", []):
        offenders = [e.name for e in result if _matches(e, needle)]
        assert not offenders, (
            f"[{case_id}/{extractor_name}] expected NO entity matching {needle!r}; "
            f"got offenders {offenders} (full list {names})"
        )

    if "scope" in exp:
        scopes = [e.scope.value for e in result]
        assert any(e.scope == exp["scope"] for e in result), (
            f"[{case_id}/{extractor_name}] expected a history event with scope "
            f"{exp['scope']!r}; got scopes {scopes}"
        )

    if "date_substring" in exp:
        needle = exp["date_substring"]
        dated = [e.date_text for e in result if e.date_text and needle in e.date_text]
        assert dated, (
            f"[{case_id}/{extractor_name}] expected a history event whose date_text "
            f"contains {needle!r}; got date_texts "
            f"{[(e.name, e.date_text) for e in result]}"
        )

    if exp.get("date_is_none"):
        # The negative direction of intent #2: a relative clue / duration is NOT a
        # date, so no returned event may populate date_text.
        offenders = [(e.name, e.date_text) for e in result if e.date_text is not None]
        assert not offenders, (
            f"[{case_id}/{extractor_name}] expected every event's date_text to be None "
            f"(no explicit date stated); got dated events {offenders}"
        )

    if exp.get("empty"):
        assert result == [], (
            f"[{case_id}/{extractor_name}] expected nothing extracted; got {names}"
        )

    if "min_count" in exp:
        assert len(result) >= exp["min_count"], (
            f"[{case_id}/{extractor_name}] expected >= {exp['min_count']} entities; "
            f"got {len(result)}: {names}"
        )


# --- the parametrized live test ---------------------------------------------
# One (case, extractor_name) pair per extractor a case actually targets -- we do
# NOT run all six extractors over every message (wasted credits, mostly
# irrelevant). Empty expectation dicts (e.g. eyeball-only people on #6) still run
# the extractor and write its report; they just assert nothing.
_PAIRS = [(case, name) for case in ALL_CASES for name in case.expect]
_IDS = [f"{case.id}::{name}" for case, name in _PAIRS]


@pytest.mark.parametrize("case,extractor_name", _PAIRS, ids=_IDS)
def test_extractor_over_case(case, extractor_name):
    extractor = _get_extractor(extractor_name)
    result = extractor.extract([case.message])

    # Write the eyeball report BEFORE asserting, so a failing assertion still
    # leaves the human-readable output behind to diagnose from.
    _append_report(extractor_name, case.id, result)

    _apply_expectations(extractor_name, case.id, case.expect[extractor_name], result)
