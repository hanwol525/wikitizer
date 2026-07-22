"""Case-insensitive matching for proper-noun surfaces (run-1.13, notes #1 and #5).

Prose sometimes carries a chat-caps spelling ("DOBROVIC is the capital of Eglon") that
strict case-sensitive matching neither linked nor re-cased. We now match case-safe
surfaces (multi-word, or a distinctive single word >= 4 chars that isn't a common English
word) case-INSENSITIVELY and emit the canonical casing -- but only when the matched prose
isn't all-lowercase, so an ALL-CAPS / Title mention links while ordinary lowercase prose
(a common word) stays plain. Fabricated fixtures, offline, no LLM.
"""

from models.lore import Alias, Location
from renderer.crosslink import build_crosslink_map, add_crosslinks


def loc(name, aliases=None):
    return Location(name=name,
                    aliases=[Alias(text=a, source_files=["g.txt"]) for a in (aliases or [])])


# --- ALL-CAPS / mixed proper nouns link AND re-case ------------------------- #
def test_all_caps_single_word_links_and_recases():
    cmap = build_crosslink_map([loc("Dobrovic"), loc("Eglon")])
    out = add_crosslinks("DOBROVIC is the capital of Eglon.", cmap, None)
    assert "[Dobrovic](#dobrovic)" in out                # linked AND re-cased
    assert "[Eglon](#eglon)" in out
    assert "DOBROVIC" not in out


def test_all_caps_multiword_links_with_article_outside():
    cmap = build_crosslink_map([loc("Kraken Clan")])
    out = add_crosslinks("The KRAKEN CLAN struck.", cmap, None)
    assert "The [Kraken Clan](#kraken-clan) struck." == out


def test_ci_lookup_has_distinctive_single_word():
    cmap = build_crosslink_map([loc("Maltraav")])
    assert cmap._ci_lookup.get("maltraav") == "Maltraav"


# --- the anti-false-link guards --------------------------------------------- #
def test_lowercase_multiword_stays_plain():
    # Preserves the existing case-sensitive guarantee: an all-lowercase mention never
    # fires a case-insensitive match (the safe direction).
    cmap = build_crosslink_map([loc("Lake Mundi")])
    assert add_crosslinks("near lake mundi", cmap, None) == "near lake mundi"


def test_distinctive_single_word_lowercase_plain_but_caps_links():
    cmap = build_crosslink_map([loc("Maltraav")])
    assert add_crosslinks("the maltraav lands", cmap, None) == "the maltraav lands"
    assert "[Maltraav](#maltraav)" in add_crosslinks("MALTRAAV rose", cmap, None)


def test_common_word_entity_stays_case_sensitive():
    # A single-word common English name ('Will') must NOT get the case-insensitive upgrade,
    # or it would link the lowercase verb everywhere.
    cmap = build_crosslink_map([loc("Will")])
    assert "will" not in cmap._ci_lookup
    assert add_crosslinks("he will go home", cmap, None) == "he will go home"   # lowercase verb
    assert add_crosslinks("WILL it rain", cmap, None) == "WILL it rain"          # all-caps: still no
    # exact-case matching is UNCHANGED (the pre-existing behavior)
    assert "[Will](#will)" in add_crosslinks("Will arrived", cmap, None)


def test_casefold_collision_disables_ci_for_both():
    # Two entities whose names differ only by case can't be re-cased unambiguously, so
    # both fall back to exact-only (no case-insensitive entry).
    cmap = build_crosslink_map([loc("Aten"), loc("ATEN")])
    assert "aten" not in cmap._ci_lookup
    assert add_crosslinks("aten in lowercase", cmap, None) == "aten in lowercase"


# --- CI matches still obey first-occurrence + self-suppression -------------- #
def test_first_occurrence_holds_for_ci_match():
    cmap = build_crosslink_map([loc("Dobrovic")])
    out = add_crosslinks("DOBROVIC and DOBROVIC again", cmap, None)
    assert out.count("](#dobrovic)") == 1               # only the first mention links


def test_self_suppression_holds_for_ci_match():
    cmap = build_crosslink_map([loc("Dobrovic")])
    out = add_crosslinks("DOBROVIC is home", cmap, "dobrovic")   # this page IS Dobrovic
    assert "](#dobrovic)" not in out                    # never links to itself


def test_ci_match_via_alias_recases_to_alias_surface():
    cmap = build_crosslink_map([loc("CJ", aliases=["Clara Jane"])])
    out = add_crosslinks("we met CLARA JANE today", cmap, None)
    assert "[Clara Jane](#cj)" in out                   # alias links to CJ's anchor, re-cased
