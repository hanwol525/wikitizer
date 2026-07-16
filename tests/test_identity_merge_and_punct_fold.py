"""Regression tests for two debug-branch fixes.

Both were found from a real end-to-end run whose output kept four identical "Gol"
locations (and four "Pyo") as separate pages, and dropped correctly-attributed
facts:

  1. Reconciler: `_short_name_veto` treated IDENTICAL short names as a spelling
     gamble and vetoed them apart. Identity is the strongest merge evidence, not a
     gamble -- the strict short-name rule is only for DIFFERENT look-alike names
     ("CJ" vs "DJ"). Fix: exempt an all-identical group from the veto.

  2. Extractor: `_normalize_for_match` folded curly quotes but NOT the ellipsis or
     en/em-dashes. Claude retypes "…" as "..." and "—"/"–" as "-"/"--", so a
     verbatim quote that was correctly attributed failed the check -- and a failed
     quote drops its whole DETAIL (the fact), not just the quote. Fix: fold the
     ellipsis and dashes to their ASCII form (and collapse hyphen runs).

These are additive (a new file per the repo convention) and run under plain
`pytest` -- no network, no key.
"""

from datetime import datetime

from models.lore import Location, Character
from models.message import Message
from models.reconcile import ReconcileDecision, MergeGroup
from agents.reconciler import Reconciler, _short_name_veto
from agents.base_extractor import (
    BaseExtractor, _normalize_for_match, _quote_is_verbatim,
)


# --------------------------------------------------------------------------- #
# Helpers (mirror the ones in test_reconciler.py / test_base_extractor.py)
# --------------------------------------------------------------------------- #
def loc(name, aliases=None):
    from models.lore import Alias
    al = [Alias(text=a, source_files=[]) for a in (aliases or [])]
    return Location(name=name, aliases=al)


def _decision(members, canonical):
    return ReconcileDecision(merges=[MergeGroup(members=members, canonical=canonical)])


def make_message(content, sender="Matt", source_file="group.txt"):
    return Message(sender=sender, timestamp=datetime(2024, 1, 1, 12, 0, 0),
                   content=content, source_file=source_file)


# ======================================================================= #
# Bug 1 -- identical short names must merge (the Gol / Pyo bug)
# ======================================================================= #
def test_identical_short_names_are_not_vetoed():
    # "Gol" is exactly SHORT_NAME_LEN (3) chars, so the short-name rule engages --
    # but the names are IDENTICAL, so it must NOT veto.
    assert _short_name_veto([loc("Gol"), loc("Gol")]) is False
    assert _short_name_veto([loc("Pyo"), loc("Pyo")]) is False
    # the real failure had four of them
    assert _short_name_veto([loc("Gol")] * 4) is False


def test_identical_short_names_are_normalized_before_the_identity_check():
    # case/whitespace-only differences are still identity, not a spelling gamble
    assert _short_name_veto([loc("Gol"), loc(" gol "), loc("GOL")]) is False


def test_different_short_names_still_vetoed_without_stated_alias():
    # REGRESSION: the CJ/DJ protection must be untouched by the identity exemption.
    assert _short_name_veto([loc("CJ"), loc("DJ")]) is True


def test_stated_alias_between_different_short_names_still_allowed():
    # REGRESSION: an explicit cross-member alias link still permits the merge.
    assert _short_name_veto([loc("CJ", aliases=["DJ"]), loc("DJ")]) is False


def test_short_name_mixed_with_a_different_long_name_still_needs_a_stated_alias():
    # ["Gol", "Golden City"] is NOT all-identical, so the identity exemption does
    # not apply; the short name genuinely might not be the long one -> still veto.
    assert _short_name_veto([loc("Gol"), loc("Golden City")]) is True


def test_long_identical_names_unaffected():
    # No short name in play -> the rule never engaged in the first place.
    assert _short_name_veto([loc("Maltraav"), loc("Maltraav")]) is False


def test_apply_merges_four_identical_gol_locations_into_one():
    # End-to-end through the deterministic assemble path: the exact production bug.
    entries = [loc("Gol"), loc("Gol"), loc("Gol"), loc("Gol")]
    decision = _decision([0, 1, 2, 3], "Gol")
    rec = Reconciler.__new__(Reconciler)  # skip __init__/client; _apply needs none
    result = rec._apply(decision, entries, "Location")
    assert len(result) == 1
    assert result[0].name == "Gol"


def test_apply_still_keeps_cj_dj_separate():
    # REGRESSION at the _apply layer: a proposed CJ/DJ merge is still vetoed apart.
    entries = [Character(name="CJ"), Character(name="DJ")]
    rec = Reconciler.__new__(Reconciler)
    result = rec._apply(_decision([0, 1], "CJ"), entries, "Character")
    assert sorted(e.name for e in result) == ["CJ", "DJ"]


# ======================================================================= #
# Bug 2 -- ellipsis + dash folding in the verbatim check
# ======================================================================= #
def test_verbatim_tolerates_ellipsis_both_directions():
    assert _quote_is_verbatim("I might multiclass...", "I might multiclass…") is True
    assert _quote_is_verbatim("I might multiclass…", "I might multiclass...") is True


def test_verbatim_tolerates_em_and_en_dash_retypings():
    # em-dash retyped as a single hyphen, a double hyphen, or preserved
    assert _quote_is_verbatim("Gol - the material side", "Gol — the material side") is True
    assert _quote_is_verbatim("Gol -- the material side", "Gol — the material side") is True
    assert _quote_is_verbatim("Gol — the material side", "Gol — the material side") is True
    # en-dash retyped as a hyphen
    assert _quote_is_verbatim("years 1-10", "years 1–10") is True


def test_verbatim_still_rejects_a_genuine_reword():
    # The folds must not open the door to a real paraphrase / hallucination.
    assert _quote_is_verbatim("Lake Mundi is huge", "Lake Mundi is a massive lake") is False
    assert _quote_is_verbatim("Dragons rule Gol", "Gol is a continent…") is False


def test_normalize_folds_ellipsis_and_dashes():
    assert _normalize_for_match("wait…") == "wait..."
    assert _normalize_for_match("a—b") == "a-b"
    assert _normalize_for_match("a–b") == "a-b"
    assert _normalize_for_match("a--b") == "a-b"  # hyphen run collapses


def test_resolve_quote_keeps_a_fact_whose_quote_differs_only_by_an_ellipsis():
    # This is the seam that was dropping the FACT: verify_quotes on, the message has
    # a real ellipsis, Claude's quote uses "...". Before the fold this returned None
    # (fact dropped); now it resolves to a Quote with the message's metadata.
    agent = BaseExtractor(client=_NoClient(), verify_quotes=True)
    batch = [make_message("Gol is the material facing side of the world of spirits…",
                          sender="dm", source_file="Gol_June4.txt")]
    q = agent._resolve_quote(
        "Gol is the material facing side of the world of spirits...", 0, batch)
    assert q is not None
    assert q.speaker == "dm"
    assert q.source_file == "Gol_June4.txt"


class _NoClient:
    """Stand-in so BaseExtractor.__init__ doesn't build a real Anthropic client;
    _resolve_quote never calls it."""
    class messages:
        @staticmethod
        def create(**kwargs):  # pragma: no cover - never invoked here
            raise AssertionError("no API call expected")


# ======================================================================= #
# Bug 3 (found during adversarial verification) -- whole-batch recovery when
# Claude cites the WRONG batch-local source_id. The quote is verbatim in a
# DIFFERENT message of the same file-pure batch, so it must be recovered, not
# dropped. This is now the dominant remaining fact-drop cause.
# ======================================================================= #
def _extractor(verify=True):
    return BaseExtractor(client=_NoClient(), verify_quotes=verify)


def test_wrong_source_id_recovers_fact_from_true_message():
    # file-pure batch (same source_file); the quote lives in message 2 but Claude
    # cited message 0. The fact must survive, with message 2's speaker.
    batch = [
        make_message("what's everyone's AC for tomorrow", sender="Matt", source_file="Gol.txt"),
        make_message("rolling initiative", sender="Ana", source_file="Gol.txt"),
        make_message("Gol is the material facing side of the world of spirits",
                     sender="dm", source_file="Gol.txt"),
    ]
    q = _extractor()._resolve_quote(
        "Gol is the material facing side of the world of spirits", 0, batch)
    assert q is not None
    assert q.speaker == "dm"            # recovered from the TRUE message, not the cited one
    assert q.source_file == "Gol.txt"


def test_valid_but_wrong_in_range_id_recovers_from_true_message():
    # The dominant real case: a well-formed, in-range id that points at the wrong
    # row. Claude cited id 0 but the quote is verbatim in id 1 (same file) -> recover.
    batch = [
        make_message("what's the plan", sender="Ana", source_file="Gol.txt"),
        make_message("Pyo is a frontier country", sender="dm", source_file="Gol.txt"),
    ]
    q = _extractor()._resolve_quote("Pyo is a frontier country", 0, batch)
    assert q is not None and q.speaker == "dm"


def test_malformed_id_still_drops_even_if_quote_is_in_batch(caplog):
    # A malformed id (out of range / wrong type) is a severe Claude malfunction, not
    # the ordinary off-by-one, so it stays an immediate drop -- recovery is scoped to
    # a valid, in-range id whose message merely didn't hold the quote. (Preserves the
    # existing bool/out-of-range drop tests + their anti-wrong-attachment rationale.)
    import logging
    batch = [make_message("Pyo is a frontier country", sender="dm", source_file="Gol.txt")]
    with caplog.at_level(logging.WARNING, logger="agents.base_extractor"):
        q = _extractor()._resolve_quote("Pyo is a frontier country", 99, batch)
    assert q is None
    assert any("out of range" in r.getMessage() for r in caplog.records)


def test_quote_verbatim_nowhere_in_batch_is_still_dropped(caplog):
    # A genuine hallucination/reword is verbatim in NO message -> still dropped.
    batch = [make_message("Gol is a continent", sender="dm", source_file="Gol.txt")]
    import logging
    with caplog.at_level(logging.WARNING, logger="agents.base_extractor"):
        q = _extractor()._resolve_quote("Dragons secretly rule Gol", 0, batch)
    assert q is None
    assert any("verbatim" in r.getMessage() for r in caplog.records)


def test_fast_path_keeps_exact_cited_attribution_even_if_quote_recurs():
    # When Claude cites correctly, the CITED message wins even if the same string
    # appears elsewhere -- recovery must not override a correct citation.
    batch = [
        make_message("we made camp", sender="Ana", source_file="Gol.txt"),
        make_message("we made camp", sender="Matt", source_file="Gol.txt"),
    ]
    q = _extractor()._resolve_quote("we made camp", 1, batch)
    assert q.speaker == "Matt"  # cited id 1, not the first occurrence


def test_verify_off_unusable_id_still_drops():
    # With verification off there's nothing to relocate against -> drop (unchanged).
    batch = [make_message("Gol is a continent", sender="dm", source_file="Gol.txt")]
    assert _extractor(verify=False)._resolve_quote("Gol is a continent", 7, batch) is None
