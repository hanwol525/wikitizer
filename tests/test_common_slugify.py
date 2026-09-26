"""Tests for evals/common/slugify.py -- the golden-convention slug rules + the tolerant oracle.

Fully offline. The golden cases are drawn straight from output/gol-lore-full.md so the pinned
rules stay locked to the file they grade.
"""

import pytest

from evals.common.slugify import (
    anchor_matches_name,
    has_non_ascii_letter,
    slug_bases,
    slugify,
    strip_leading_article,
)


@pytest.mark.parametrize("name,expected", [
    ("Aldenburg", "aldenburg"),
    ("Cloud Mountains", "cloud-mountains"),
    ("Crown's Nest", "crowns-nest"),                     # apostrophe dropped, not hyphenated
    ("Camden \"Cam\" Stonefoot", "camden-cam-stonefoot"),  # quotes dropped
    ("Lake Mundi (The Great Well, The Pond)", "lake-mundi-the-great-well-the-pond"),  # parens flattened
    ("Dwarven Stronghold (Taken Lands of Tiber)", "dwarven-stronghold-taken-lands-of-tiber"),
    ("Half-Elf", "half-elf"),                            # internal hyphen kept
    ("Skjöldr Aldvarðr", "skjöldr-aldvarðr"),            # non-ASCII PRESERVED (not folded)
    ("The Maltraav–Kriega War", "the-maltraavkriega-war"),  # en-dash is a separator, punctuation drops
])
def test_slugify_golden_cases(name, expected):
    assert slugify(name) == expected


def test_slugify_lowercases():
    assert slugify("ALL CAPS Name") == "all-caps-name"


def test_slugify_collapses_and_trims_hyphens():
    assert slugify("  Spaced   Out  ") == "spaced-out"
    assert slugify("-Leading & trailing-") == "leading-trailing"


def test_slugify_all_punctuation_is_empty():
    assert slugify("!!!") == ""
    assert slugify("   ") == ""


def test_slugify_decomposed_accent_is_normalized():
    # A decomposed "o + combining diaeresis" must fold to the same slug as precomposed "ö".
    precomposed = "öystein"          # ö
    decomposed = "öystein"          # o + U+0308
    assert slugify(precomposed) == slugify(decomposed)


def test_strip_leading_article():
    assert strip_leading_article("The Citadel") == "Citadel"
    assert strip_leading_article("the pond") == "pond"
    assert strip_leading_article("Theodore") == "Theodore"   # only a whole leading "the "
    assert strip_leading_article("Citadel") == "Citadel"


def test_slug_bases_includes_article_stripped():
    assert slug_bases("The Citadel") == {"the-citadel", "citadel"}
    assert slug_bases("Aldenburg") == {"aldenburg"}


def test_anchor_matches_name_exact_and_article_drop():
    assert anchor_matches_name("aldenburg", "Aldenburg")
    assert anchor_matches_name("the-dvergarim", "The Dvergarim")   # article kept
    assert anchor_matches_name("dvergarim", "The Dvergarim")       # article dropped -- also accepted


def test_anchor_matches_name_disambiguation_suffix():
    assert anchor_matches_name("citadel-2", "The Citadel")             # article drop + -N
    assert anchor_matches_name("krieger-imperium-2", "The Krieger Imperium")
    assert anchor_matches_name("aldenburg-3", "Aldenburg")


def test_anchor_matches_name_rejects_transliteration_and_junk():
    # A transliterated accent must NOT match -- that's the whole point of preserving non-ASCII.
    assert not anchor_matches_name("skjoldr-aldvarr", "Skjöldr Aldvarðr")
    assert anchor_matches_name("skjöldr-aldvarðr", "Skjöldr Aldvarðr")
    assert not anchor_matches_name("something-else", "Aldenburg")
    assert not anchor_matches_name("aldenburg-x", "Aldenburg")     # non-numeric suffix
    assert not anchor_matches_name("", "!!!")                      # empty base can't match


def test_has_non_ascii_letter():
    assert has_non_ascii_letter("Skjöldr")
    assert has_non_ascii_letter("Aldvarðr")
    assert not has_non_ascii_letter("Aldenburg")
    assert not has_non_ascii_letter("citadel-2")                   # digit is not a non-ASCII letter
