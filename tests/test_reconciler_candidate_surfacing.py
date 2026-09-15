"""Widened candidate-pair surfacing (run-1.15 §D).

The advisory candidate list (which the LLM must adjudicate -- it never auto-merges) used
to surface only spelling-close or token-SUBSET pairs. So "Krieger Family" vs "Krieger
Royal House" -- which share only "krieger" and are neither a subset nor one letter apart
-- never reached the model as a "look harder at this pair" hint. Now a pair that shares a
DISTINCTIVE word (>=4 chars, held by few entries of this type) is surfaced too, while a
common word stays excluded so it can't flood the list. Offline, pure, no API.
"""

from agents.reconciler import _candidate_pairs
from models.lore import Organization


def org(name):
    return Organization(name=name)


def _index_pairs(pairs):
    return {(i, j) for i, j, _ in pairs}


# --- the new distinctive-token branch --------------------------------------- #
def test_distinctive_shared_token_surfaces_the_pair():
    entries = [org("Krieger Family"), org("Krieger Royal House"), org("Ostrana Council")]
    pairs = _candidate_pairs(entries)
    assert (0, 1) in _index_pairs(pairs)
    reason = next(r for i, j, r in pairs if (i, j) == (0, 1))
    assert "distinctive" in reason and "krieger" in reason


def test_common_token_does_not_flood():
    # "krieger" now spans 5 entries -> not distinctive -> a pair sharing ONLY it is not
    # surfaced (and none of these are subsets or spelling-close), so nothing floods.
    entries = [org("Krieger Family"), org("Krieger Royal House"), org("Krieger Guard"),
               org("Krieger Bank"), org("Krieger Watch")]
    assert _candidate_pairs(entries) == []


def test_short_shared_token_does_not_surface():
    # "cj" is shared but < 4 chars -> not a distinctive word; the pair is left alone.
    entries = [org("CJ Ashford"), org("CJ Beaumont")]
    assert _candidate_pairs(entries) == []


# --- the pre-existing branches still fire (no regression from the new else) -- #
def test_subset_still_surfaces():
    entries = [org("Kraken"), org("Kraken Clan")]
    pairs = _candidate_pairs(entries)
    assert (0, 1) in _index_pairs(pairs)
    reason = next(r for i, j, r in pairs if (i, j) == (0, 1))
    assert "extra words" in reason


def test_one_letter_off_still_surfaces():
    entries = [org("Maltaav"), org("Maltraav")]
    pairs = _candidate_pairs(entries)
    assert (0, 1) in _index_pairs(pairs)
    reason = next(r for i, j, r in pairs if (i, j) == (0, 1))
    assert "one letter off" in reason
