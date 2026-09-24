"""Tests for evals/extract_metrics.py -- the PURE metric module behind the
extractor A/B eval harness (evals/extract_ab.py).

Fully offline: no network, no API key, no `integration` marker. Only the pure
metric functions are tested -- the harness (evals/extract_ab.py) makes real paid
calls and is never collected by pytest (its filename doesn't match `test_*.py`).

House style mirrors tests/test_orchestrator.py: tiny builder helpers at the top,
REAL models.lore objects for the entity inputs (so the tests exercise the actual
`.name` / `.aliases[i].text` attribute paths a duck-typed fake could drift from),
and a trivial fake normalizer for the quote classifier so that stays pure too.
"""

from evals.extract_metrics import (
    EntityDiff,
    classify_dropped_quote,
    format_report,
    match_entities,
    normalize_name,
)
from models.lore import Alias, Character, HistoryEvent, Location, Scope


# --- tiny builders ---------------------------------------------------------- #

def loc(name, *aliases):
    """A minimal Location -- only `name` + `aliases` matter to match_entities."""
    return Location(name=name, aliases=[Alias(text=a) for a in aliases])


# A fake "production" normalizer for classify_dropped_quote: lowercase ONLY, so it
# does NOT fold punctuation or collapse whitespace. That deliberately-weak fold is
# what lets a "cosmetic" case slip past it while the aggressive skeleton catches
# it. Keeping it a fake keeps the test decoupled from the real _normalize_for_match.
def fake_norm(s):
    return s.lower()


# --- normalize_name --------------------------------------------------------- #

def test_normalize_name_collapses_whitespace_strips_and_casefolds():
    assert normalize_name("  Lake   Mundi ") == "lake mundi"


def test_normalize_name_is_case_insensitive():
    assert normalize_name("REDBRIDGE") == normalize_name("redbridge") == "redbridge"


def test_normalize_name_preserves_punctuation():
    # A real name difference (period present vs absent) must survive -- the alias
    # set, not the name key, is where such variants get reconciled.
    assert normalize_name("St. Merrow") == "st. merrow"
    assert normalize_name("St. Merrow") != normalize_name("St Merrow")


# --- match_entities --------------------------------------------------------- #

def test_match_entities_clean_all_match():
    base = [loc("Riverton"), loc("Lake Mundi")]
    cand = [loc("Lake Mundi"), loc("Riverton")]        # order shouldn't matter
    d = match_entities(base, cand, "locations")
    assert (d.matched, d.baseline_count, d.candidate_count) == (2, 2, 2)
    assert d.missed == [] and d.extra == []


def test_match_entities_missed_keeps_original_casing():
    base = [loc("Riverton"), loc("The Sunken Vault")]
    cand = [loc("riverton")]                            # candidate dropped the vault
    d = match_entities(base, cand, "locations")
    assert d.matched == 1
    assert d.missed == ["The Sunken Vault"]             # ORIGINAL display name, real casing
    assert d.extra == []


def test_match_entities_extra_is_candidate_only():
    base = [loc("Riverton")]
    cand = [loc("Riverton"), loc("Old Mill")]           # candidate invented / renamed
    d = match_entities(base, cand, "locations")
    assert d.matched == 1
    assert d.missed == []
    assert d.extra == ["Old Mill"]


def test_match_entities_alias_match_baseline_name_is_candidate_alias():
    base = [loc("Emperor Krieger")]
    cand = [loc("The Emperor", "Emperor Krieger")]      # baseline's name is candidate's alias
    d = match_entities(base, cand, "characters")
    assert d.matched == 1 and d.missed == [] and d.extra == []


def test_match_entities_alias_match_candidate_name_is_baseline_alias():
    base = [loc("The Emperor", "Emperor Krieger")]      # candidate's name is baseline's alias
    cand = [loc("Emperor Krieger")]
    d = match_entities(base, cand, "characters")
    assert d.matched == 1 and d.missed == [] and d.extra == []


def test_match_entities_case_insensitive_match():
    d = match_entities([loc("Redbridge")], [loc("redbridge")], "locations")
    assert d.matched == 1 and d.missed == [] and d.extra == []


def test_match_entities_empty_baseline_all_extra():
    d = match_entities([], [loc("Old Mill"), loc("Riverton")], "locations")
    assert d.matched == 0
    assert d.missed == []
    assert d.extra == ["Old Mill", "Riverton"]


def test_match_entities_empty_candidate_all_missed():
    d = match_entities([loc("Old Mill"), loc("Riverton")], [], "locations")
    assert d.matched == 0
    assert d.missed == ["Old Mill", "Riverton"]
    assert d.extra == []


def test_match_entities_no_double_match():
    # Two baseline entries both normalize to one candidate name. The first in order
    # claims the single candidate; the second must fall to `missed` -- one candidate
    # can never satisfy two baselines. Locks the consumed-pool behavior.
    base = [loc("Riverton"), loc("riverton")]
    cand = [loc("Riverton")]
    d = match_entities(base, cand, "locations")
    assert d.matched == 1
    assert d.missed == ["riverton"]                     # the second, unclaimed baseline
    assert d.extra == []


def test_match_entities_is_type_agnostic():
    # match_entities reads only .name / .aliases, so it works uniformly across all
    # six lore types. Prove it on Character and HistoryEvent (which need extra
    # required fields to construct) alongside Location.
    base = [Character(name="Kriggy"), HistoryEvent(name="The Founding",
                                                   description="A city was founded.",
                                                   scope=Scope.WORLD)]
    cand = [HistoryEvent(name="The Founding", description="A city was founded.",
                         scope=Scope.WORLD)]
    d = match_entities(base, cand, "history")
    assert d.matched == 1
    assert d.missed == ["Kriggy"]


# --- classify_dropped_quote ------------------------------------------------- #

def test_classify_dropped_quote_cosmetic():
    # The words ARE in the source, but only punctuation/spacing the fake (weak)
    # normalizer doesn't fold differs -> the aggressive skeleton matches while the
    # fake current normalizer does not. That's a recoverable cosmetic drop.
    quote = "hello, world"
    sources = ["well hello  world!"]                    # comma gone, double space, bang
    assert classify_dropped_quote(quote, sources, fake_norm) == "cosmetic"


def test_classify_dropped_quote_reword():
    # The words themselves are absent from every source, even after stripping ALL
    # punctuation -> the model paraphrased or invented. A genuine loss.
    quote = "the bridge burned at dawn"
    sources = ["we crossed the wide river", "and made camp"]
    assert classify_dropped_quote(quote, sources, fake_norm) == "reword"


def test_classify_dropped_quote_unknown_no_sources():
    assert classify_dropped_quote("anything at all", [], fake_norm) == "unknown"


def test_classify_dropped_quote_unknown_blank_quote():
    # A blank/whitespace quote normalizes to nothing -> can't be classified.
    assert classify_dropped_quote("   ", ["some real message"], fake_norm) == "unknown"


def test_classify_dropped_quote_unknown_when_matches_both():
    # Present under BOTH the aggressive skeleton AND the (fake) production
    # normalizer -> inconsistent with a real drop, so we don't force it into a
    # bucket. (Reachable because the harness compares against ALL filtered
    # messages, a superset of the drop's own file-pure batch.)
    quote = "hello world"
    sources = ["hello world here"]
    assert classify_dropped_quote(quote, sources, fake_norm) == "unknown"


# --- format_report ---------------------------------------------------------- #

def test_format_report_has_tiers_numbers_and_specs():
    diffs = [
        EntityDiff("locations", baseline_count=12, candidate_count=10, matched=9,
                   missed=["The Sunken Vault", "Miller's Ford", "Redbridge"], extra=["Old Mill"]),
        EntityDiff("characters", baseline_count=8, candidate_count=8, matched=8,
                   missed=[], extra=[]),
    ]
    quote_stats = {
        "survival": {
            "locations": {"baseline": 20, "candidate": 14},
            "characters": {"baseline": 10, "candidate": 10},
        },
        "drops": {"cosmetic": 4, "reword": 7, "unknown": 1},
        "drop_samples": [("reword", "we burned the bridge"), ("cosmetic", "beneath the mill")],
    }
    report = format_report(
        models={"baseline": "claude-sonnet-4-6",
                "candidate": "openrouter:deepseek/deepseek-v4-flash",
                "filtered_messages": 342},
        per_type_diffs=diffs,
        quote_stats=quote_stats,
        json_retry_stats={"baseline": 0, "candidate": 3},
    )

    # Both tier labels are present and clearly distinguished.
    assert "PRIMARY" in report
    assert "DIAGNOSTIC" in report
    # Both model specs are echoed.
    assert "claude-sonnet-4-6" in report
    assert "openrouter:deepseek/deepseek-v4-flash" in report

    # Pin the coverage TOTAL row's actual cells: base 12+8=20, cand 10+8=18,
    # matched 9+8=17, missed 3+0=3, extra 1+0=1. Match on WHOLE-CELL tokens of a
    # line beginning "TOTAL" (via .split()), so this can't be satisfied by "20"
    # or "TOTAL" leaking in from the survival table below -- and it WOULD fail if
    # the summation regressed (e.g. tot["base"] summed candidate_count -> 18).
    total_lines = [ln for ln in report.splitlines() if ln.strip().startswith("TOTAL")]
    assert any(
        all(str(n) in ln.split() for n in (20, 18, 17, 3, 1)) for ln in total_lines
    ), f"coverage TOTAL row 20/18/17/3/1 not found in: {total_lines}"
    # And the survival TOTAL row: base 20+10=30, cand 14+10=24, delta -6.
    assert any(
        all(str(n) in ln.split() for n in (30, 24, -6)) for ln in total_lines
    ), f"survival TOTAL row 30/24/-6 not found in: {total_lines}"

    # The named missed list shows through.
    assert "The Sunken Vault" in report
    # Diagnostic drop buckets + retry counts are present.
    assert "cosmetic: 4" in report
    assert "reword: 7" in report
    assert "candidate: 3" in report
    # The cache caveat on retries is spelled out so a reader isn't misled.
    assert "cache" in report.lower()
