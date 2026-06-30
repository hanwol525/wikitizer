"""Unit tests for renderer/crosslink.py (Phase 4.3: the cross-link pass).

Plain pytest -- no network, no API key, no integration marker. Like the cross-
linker itself, every fixture here is FABRICATED synthetic markdown (safe to
commit; never real campaign text) and every assertion is deterministic and exact.

Grouped to match the brief: slugify, map build / collisions, the walk / matching,
first-occurrence + self-suppression, no-go zones, history, and edges. A handful of
additive cases (NFC byte-variant MATCHING, hyphen boundaries, casing preservation)
go beyond the brief's list -- extra coverage, never replacing a listed case.
"""

import logging

import pytest

from models.lore import (
    Character,
    HistoryEvent,
    Item,
    Location,
    Organization,
    PeopleAndCultures,
    Scope,
)
from renderer.crosslink import (
    REVIEW_PREFIX,
    add_crosslinks,
    build_crosslink_map,
    load_crosslink_words,
    slugify,
)


# --- tiny builders ---------------------------------------------------------- #

def loc(name, aliases=None):
    return Location(name=name, aliases=aliases or [])


def char(name, aliases=None):
    return Character(name=name, aliases=aliases or [])


def org(name, aliases=None):
    return Organization(name=name, aliases=aliases or [])


def item(name, aliases=None):
    return Item(name=name, aliases=aliases or [])


def people(name, aliases=None):
    return PeopleAndCultures(name=name, aliases=aliases or [])


def event(name, description, aliases=None):
    return HistoryEvent(name=name, description=description, scope=Scope.WORLD, aliases=aliases or [])


def surfaces(cmap):
    """Just the surface strings in the resolved pool (drop the anchors)."""
    return [s for s, _ in cmap.sources]


# ===========================================================================
# slugify
# ===========================================================================

def test_slugify_basic():
    assert slugify("Lake Mundi") == "lake-mundi"


def test_slugify_ascii_folds_accents():
    assert slugify("Théoden") == "theoden"
    assert slugify("Canción") == "cancion"


def test_slugify_keeps_internal_hyphen():
    assert slugify("Half-Elf") == "half-elf"


def test_slugify_strips_apostrophe_closing_the_word_up():
    # An apostrophe is removed, NOT turned into a hyphen -- so "Mal'taav" slugs the
    # same as its apostrophe-free spelling.
    assert slugify("Mal'taav") == "maltaav"


def test_slugify_collapses_spaces_and_trims_punctuation():
    assert slugify("  Lake   Mundi!!  ") == "lake-mundi"
    assert slugify("--Lake--Mundi--") == "lake-mundi"


def test_slugify_nfc_byte_variants_slug_identically():
    # Precomposed "\u00e9" (one codepoint) and "e" + combining acute (U+0301) are
    # pixel-identical but different BYTES; they must produce the SAME slug.
    precomposed = "Caf\u00e9"        # e-acute as a single codepoint
    decomposed = "Cafe\u0301"        # plain e + combining acute
    assert precomposed != decomposed  # genuinely different byte sequences
    assert slugify(precomposed) == slugify(decomposed) == "cafe"


def test_slugify_all_punctuation_is_empty_string():
    # slugify itself returns "" for a no-ASCII-letter name; the entity-<N> fallback
    # is the MAP's job (it has the index). The map-level test below proves no
    # id="" is ever emitted.
    assert slugify("!!!") == ""
    assert slugify("'''") == ""


# ===========================================================================
# map build / collisions
# ===========================================================================

def test_slug_collision_suffixes_the_later_entity(caplog):
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([loc("Riverton"), loc("RIVERTON")])
    assert cmap.entity_anchors == ["riverton", "riverton-2"]
    assert REVIEW_PREFIX in caplog.text
    assert "slug collision" in caplog.text


def test_empty_slug_falls_back_and_never_emits_empty_id(caplog):
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([loc("!!!"), loc("Lake Mundi")])
    assert cmap.entity_anchors[0] == "entity-0"
    assert cmap.anchors["!!!"] == "entity-0"
    # No anchor anywhere is the empty string.
    assert "" not in cmap.entity_anchors
    assert "" not in cmap.anchors.values()
    assert REVIEW_PREFIX in caplog.text


def test_alias_colliding_with_a_real_name_loses_to_the_name(caplog):
    # Entity A claims "Riverton" as an alias; entity B is actually NAMED "Riverton".
    # The real name wins: the surface "Riverton" points at B's anchor, A's alias
    # claim is dropped, and it's flagged.
    a = loc("Town", aliases=["Riverton"])
    b = loc("Riverton")
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([a, b])
    assert ("Riverton", "riverton") in cmap.sources
    assert ("Riverton", "town") not in cmap.sources
    assert REVIEW_PREFIX in caplog.text


def test_same_alias_on_two_entities_is_held_out_of_pool(caplog):
    # Both entities claim "The Order" -> we can't pick -> not in the pool, flagged.
    a = loc("Guild A", aliases=["The Order"])
    b = org("Guild B", aliases=["The Order"])
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([a, b])
    assert "Order" not in surfaces(cmap)
    assert REVIEW_PREFIX in caplog.text


def test_identical_name_across_types_both_anchored_surface_held_out(caplog):
    # The canonical scenario the pass-4 ambiguity branch is written for: a realm that
    # is BOTH a Location (the place) and an Organization (the governing body), sharing
    # one IDENTICAL name. Unlike test_slug_collision_suffixes_the_later_entity (two
    # DIFFERENT names -- "Riverton"/"RIVERTON" -- that merely slug alike), here the
    # name string itself is the same, so on top of the slug-suffixing we also hit the
    # "two entities share the name" path and Pass 4's ambiguity hold-out.
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([loc("Krieger Imperium"), org("Krieger Imperium")])
    # Each entity still gets its own distinct, suffixed anchor (correct in the
    # per-entity list even though the name->anchor dict can only hold the first).
    assert cmap.entity_anchors == ["krieger-imperium", "krieger-imperium-2"]
    assert cmap.anchors["Krieger Imperium"] == "krieger-imperium"
    # The shared surface is genuinely ambiguous (claimed by both anchors), so it is
    # held out of the source pool entirely -- no inbound link anywhere, by design.
    assert "Krieger Imperium" not in surfaces(cmap)
    out = add_crosslinks("They marched on Krieger Imperium at dawn.", cmap, None)
    assert out == "They marched on Krieger Imperium at dawn."  # nothing linked
    # Both the shared-name and the ambiguity hold-out are flagged for a human.
    assert REVIEW_PREFIX in caplog.text
    assert "share the name" in caplog.text
    assert "left out of the link pool" in caplog.text


def test_require_article_member_is_in_pool_but_article_required():
    cmap = build_crosslink_map([loc("Founding")], common_words={"require_article": ["Founding"]})
    assert ("Founding", "founding") in cmap.sources
    # The walk tests below prove the firing behavior; here we pin the flag on the
    # private lookup so a regression in the gate is caught at build time too.
    assert cmap._lookup["Founding"] == ("founding", True)


def test_never_link_member_is_fully_held_out_but_anchor_still_exists():
    cmap = build_crosslink_map([loc("Pond")], common_words={"never_link": ["Pond"]})
    assert "Pond" not in surfaces(cmap)
    # Held out of the SOURCE pool, but it still has its anchor as a target.
    assert cmap.anchors["Pond"] == "pond"


def test_sentence_ish_name_is_target_only_not_a_source(caplog):
    name = "The council that secretly rules the southern reach and collects its tithes"
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([loc(name), loc("Lake Mundi")])
    # Its anchor exists (it is a valid link TARGET)...
    assert cmap.anchors[name] == slugify(name)
    # ...but it is NOT hunted in prose as a source.
    assert all("council" not in s for s in surfaces(cmap))
    assert REVIEW_PREFIX in caplog.text


def test_sentence_ish_by_internal_punctuation_is_held_out():
    # A short name with sentence punctuation is gated too (not just long ones).
    cmap = build_crosslink_map([loc("It was destroyed.")])
    assert surfaces(cmap) == []
    # ...but the anchor is still minted.
    assert cmap.anchors["It was destroyed."] == "it-was-destroyed"


def test_inert_typod_alias_sits_in_pool_and_matches_nothing():
    # A misspelled alias the reconciler folded in: it sits in the pool but never
    # appears in clean prose, so it simply never matches. No scrubbing needed.
    cmap = build_crosslink_map([loc("Maltaav", aliases=["Maltraav"])])
    assert "Maltraav" in surfaces(cmap)
    out = add_crosslinks("Maltaav arrived at dawn.", cmap, None)
    assert "[Maltaav](#maltaav)" in out
    # The typo never appears in prose, so nothing fires off it -- and no crash.
    assert "Maltraav" not in out


def test_pool_is_longest_first_across_names_and_aliases_together():
    # An ALIAS longer than another entity's whole NAME must win the longest match,
    # so names and aliases share one line-up sorted together.
    phoenix = char("Phoenix")
    brotherhood = org("Brotherhood", aliases=["The Brotherhood of the Phoenix Blood"])
    cmap = build_crosslink_map([phoenix, brotherhood])
    assert cmap.sources[0][0] == "Brotherhood of the Phoenix Blood"
    out = add_crosslinks("The Brotherhood of the Phoenix Blood marched.", cmap, None)
    assert "[Brotherhood of the Phoenix Blood](#brotherhood)" in out
    # "Phoenix" inside the longer match is consumed, never separately linked.
    assert "#phoenix" not in out


# ===========================================================================
# the walk / matching
# ===========================================================================

def test_longest_match_beats_its_own_prefix():
    cmap = build_crosslink_map([loc("Krieger Imperium"), char("Krieger")])
    out = add_crosslinks("They fled the Krieger Imperium at dawn.", cmap, None)
    assert "the [Krieger Imperium](#krieger-imperium)" in out
    # The bare "Krieger" entity must NOT fire inside the longer phrase.
    assert "(#krieger)" not in out


def test_no_relink_of_own_output():
    # The classic trap: one pass, one link, no nesting -- finditer's non-overlapping
    # walk makes re-scanning consumed spans impossible.
    cmap = build_crosslink_map([loc("Krieger Imperium"), char("Krieger")])
    out = add_crosslinks("the Krieger Imperium", cmap, None)
    assert out == "the [Krieger Imperium](#krieger-imperium)"
    assert "[[" not in out and "]]" not in out


def test_word_boundaries_keep_similar_names_distinct():
    cmap = build_crosslink_map([loc("Riverton")])
    out = add_crosslinks("Rivertown and Riverside lie near Riverton today.", cmap, None)
    assert "[Riverton](#riverton)" in out
    # The bigger words are untouched -- "Riverton" never fires inside them.
    assert "[Rivertown" not in out and "Rivertown" in out
    assert "[Riverside" not in out and "Riverside" in out


def test_apostrophe_is_internal_bare_prefix_never_fires_inside():
    cmap = build_crosslink_map([char("Mal'taav"), char("Mal")])
    out = add_crosslinks("Mal'taav drew his blade.", cmap, None)
    assert "[Mal'taav](#maltaav)" in out
    # "Mal" must not fire inside "Mal'taav".
    assert "(#mal)" not in out
    # ...but "Mal" as a whole word still links elsewhere.
    out2 = add_crosslinks("Mal waited.", cmap, None)
    assert "[Mal](#mal)" in out2


def test_matching_is_case_sensitive():
    cmap = build_crosslink_map([loc("Lake Mundi")])
    out = add_crosslinks("Lake Mundi differs from lake mundi.", cmap, None)
    assert "[Lake Mundi](#lake-mundi)" in out
    assert "lake mundi" in out  # the lowercase spelling stays plain
    assert out.count("](#lake-mundi)") == 1


def test_article_rule_proper_noun_fires_with_or_without_the():
    cmap = build_crosslink_map([loc("Krieger Imperium")])
    # With "the": article stays OUTSIDE the brackets, original casing preserved.
    with_the = add_crosslinks("We saw The Krieger Imperium rise.", cmap, None)
    assert "The [Krieger Imperium](#krieger-imperium) rise" in with_the
    # Without "the": still links (proper noun).
    without_the = add_crosslinks("Krieger Imperium rose.", cmap, None)
    assert without_the.startswith("[Krieger Imperium](#krieger-imperium)")


def test_article_rule_common_word_requires_the():
    cmap = build_crosslink_map([loc("Founding")], common_words={"require_article": ["Founding"]})
    linked = add_crosslinks("After the Founding, things changed.", cmap, None)
    assert "the [Founding](#founding)," in linked
    # Bare / sentence-start "Founding" must NOT false-match.
    assert add_crosslinks("Founding members got perks.", cmap, None) == "Founding members got perks."


def test_alias_links_to_canonical_anchor_not_its_own_slug():
    cmap = build_crosslink_map([loc("Lake Mundi", aliases=["The Great Well"])])
    out = add_crosslinks("They reached The Great Well by noon.", cmap, None)
    # Article outside, link points at the CANONICAL anchor, never #the-great-well.
    assert "The [Great Well](#lake-mundi) by noon" in out
    assert "#the-great-well" not in out


# ===========================================================================
# first-occurrence + self-suppression
# ===========================================================================

def test_first_occurrence_linked_later_left_plain():
    cmap = build_crosslink_map([char("Garrval")])
    out = add_crosslinks("Garrval met Garrval again.", cmap, None)
    assert out == "[Garrval](#garrval) met Garrval again."


def test_self_suppression_including_via_alias():
    # On Lake Mundi's own page, neither its name nor an alias resolving to its
    # anchor should link (the page anchor is pre-seeded into `seen`).
    cmap = build_crosslink_map([loc("Lake Mundi", aliases=["The Great Well"])])
    block = "Lake Mundi is also called The Great Well by locals."
    out = add_crosslinks(block, cmap, "lake-mundi")
    assert out == block  # nothing linked


def test_full_lake_mundi_trace():
    # The brief's trace, exactly: seen pre-seeded with #lake-mundi.
    cmap = build_crosslink_map([loc("Lake Mundi", aliases=["The Great Well"]), char("Garrval")])
    block = "Garrval visited. The Great Well is here. Garrval left."
    out = add_crosslinks(block, cmap, "lake-mundi")
    assert out == "[Garrval](#garrval) visited. The Great Well is here. Garrval left."


def test_seen_resets_per_block():
    # The same entity links once on character A's page AND once on character B's.
    cmap = build_crosslink_map([char("Garrval"), char("Alice"), char("Bob")])
    page_a = add_crosslinks("Alice knew Garrval well.", cmap, "alice")
    page_b = add_crosslinks("Bob knew Garrval too.", cmap, "bob")
    assert "[Garrval](#garrval)" in page_a
    assert "[Garrval](#garrval)" in page_b


# ===========================================================================
# no-go zones
# ===========================================================================

def test_headings_are_not_linkified():
    cmap = build_crosslink_map([loc("Lake Mundi")])
    out = add_crosslinks("# Lake Mundi\n\nLake Mundi is wet.", cmap, None)
    # Heading line untouched; body linked.
    assert out.startswith("# Lake Mundi\n")
    assert "# [Lake Mundi]" not in out
    assert "[Lake Mundi](#lake-mundi) is wet." in out


def test_fenced_code_block_is_untouched():
    cmap = build_crosslink_map([loc("Lake Mundi")])
    out = add_crosslinks("```\nLake Mundi\n```\n\nLake Mundi here.", cmap, None)
    assert "```\nLake Mundi\n```" in out          # code intact
    assert "[Lake Mundi](#lake-mundi) here." in out


def test_inline_code_span_is_untouched():
    cmap = build_crosslink_map([loc("Lake Mundi")])
    out = add_crosslinks("`Lake Mundi` then Lake Mundi.", cmap, None)
    assert out == "`Lake Mundi` then [Lake Mundi](#lake-mundi)."


def test_existing_markdown_link_is_not_nested():
    cmap = build_crosslink_map([loc("Lake Mundi")])
    block = "Read [about Lake Mundi](#elsewhere) now."
    out = add_crosslinks(block, cmap, None)
    assert out == block  # the name inside the existing link is left alone


def test_footnote_markers_are_inert():
    cmap = build_crosslink_map([loc("Lake Mundi")])
    out = add_crosslinks("Lake Mundi[^1] is wet.[^2]", cmap, None)
    assert "[Lake Mundi](#lake-mundi)[^1]" in out
    assert "[^2]" in out


# ===========================================================================
# history
# ===========================================================================

def test_history_description_links_noun_mentions():
    # Events are NOT in the map; their description is run with this_anchor=None and
    # the noun mentions inside it still light up.
    cmap = build_crosslink_map([loc("Riverton"), org("Krieger Imperium")])
    ev = event("The Sacking of Riverton", "Riverton was sacked by the Krieger Imperium.")
    out = add_crosslinks(ev.description, cmap, None)
    assert "[Riverton](#riverton)" in out
    assert "the [Krieger Imperium](#krieger-imperium)" in out


def test_history_event_name_is_neither_target_nor_source():
    # Build over nouns only -- the event is excluded entirely.
    cmap = build_crosslink_map([loc("Riverton"), org("Krieger Imperium")])
    ev = event("The Sacking of Riverton", "Riverton was sacked.")
    event_slug = slugify(ev.name)
    assert event_slug not in cmap.anchors.values()      # no anchor minted
    assert ev.name not in cmap.anchors
    assert all("Sacking" not in s for s in surfaces(cmap))  # not a source


# ===========================================================================
# edges
# ===========================================================================

def test_empty_map_returns_block_unchanged():
    cmap = build_crosslink_map([])
    assert cmap.pattern is None
    assert add_crosslinks("Lake Mundi is here.", cmap, None) == "Lake Mundi is here."


def test_empty_block_returns_unchanged():
    cmap = build_crosslink_map([loc("Lake Mundi")])
    assert add_crosslinks("", cmap, None) == ""


def test_block_with_no_matches_unchanged():
    cmap = build_crosslink_map([loc("Lake Mundi")])
    assert add_crosslinks("Nothing here matches anything.", cmap, None) == "Nothing here matches anything."


# ===========================================================================
# additive coverage (beyond the brief's list)
# ===========================================================================

def test_nfc_byte_variant_prose_matches_precomposed_surface():
    # Step B in action: prose written with a COMBINING accent matches a surface
    # stored PRECOMPOSED, because both are NFC'd before comparison.
    surface_name = "Th\u00e9oden"            # precomposed e-acute in the model
    cmap = build_crosslink_map([char(surface_name)])
    decomposed_prose = "We met The\u0301oden today."  # plain e + U+0301
    assert surface_name not in decomposed_prose        # different bytes pre-NFC
    out = add_crosslinks(decomposed_prose, cmap, None)
    assert "[Th\u00e9oden](#theoden)" in out


def test_hyphen_is_internal_to_a_name():
    cmap = build_crosslink_map([people("Half-Elf")])
    out = add_crosslinks("A Half-Elf walked by.", cmap, None)
    assert "[Half-Elf](#half-elf)" in out


def test_lowercase_the_article_casing_is_preserved():
    cmap = build_crosslink_map([loc("Krieger Imperium")])
    out = add_crosslinks("deep in the Krieger Imperium they hid", cmap, None)
    assert "the [Krieger Imperium](#krieger-imperium)" in out


def test_item_and_people_types_participate_as_full_citizens():
    # All five noun types are targets+sources, not just Location/Character.
    cmap = build_crosslink_map([item("Sunblade"), people("The Krieg")])
    out = add_crosslinks("The Krieg forged the Sunblade.", cmap, None)
    assert "[Sunblade](#sunblade)" in out
    assert "[Krieg](#the-krieg)" in out  # alias-less name, "The" stripped to source


# --- hardening tests added after an adversarial review (Phase 4.3) ----------- #
# These are ADDITIVE: the listed tests above are untouched. Each closes a gap the
# review surfaced where a plausible regression would manufacture a WRONG link --
# exactly the false positive the anti-hallucination ethos most wants to prevent --
# yet would still pass the original suite.


def test_lowercase_prose_is_never_linked_isolated_from_seen():
    # Companion to test_matching_is_case_sensitive, which mixes "Lake Mundi" and
    # "lake mundi" in ONE block. There the lowercase spelling stays plain partly
    # because the capitalized one already linked first and seeded #lake-mundi into
    # `seen` -- so that test can't tell "case-sensitive" apart from "case-insensitive
    # + first-occurrence-suppressed". This isolates the CASE rule: lowercase-only
    # prose, with no capitalized occurrence to seed `seen`, must come back fully
    # unchanged. A re.IGNORECASE regression (which would link a common lowercase
    # word to the wrong-context anchor) is caught here, not by the mixed-case test.
    cmap = build_crosslink_map([loc("Lake Mundi")])
    block = "We rowed across lake mundi at dusk."
    assert add_crosslinks(block, cmap, None) == block


def test_left_word_boundary_does_not_fire_after_a_word_char():
    # Companion to test_word_boundaries_keep_similar_names_distinct, which only
    # exercises the RIGHT/trailing boundary ("Rivertown"/"Riverside" are suffix
    # extensions, caught by the lookAHEAD). This pins the lookBEHIND: a name
    # preceded by a word char must NOT fire. A regression swapping the custom
    # lookbehind for a plain \b would pass the suffix-only test yet manufacture
    # "pre[Riverton]" -- a wrong link.
    cmap = build_crosslink_map([loc("Riverton")])
    for block in (
        "The town of preRiverton is fake.",
        "Both aRiverton and Riverton5 are decoys.",
    ):
        assert add_crosslinks(block, cmap, None) == block

def test_two_entities_with_identical_name_drop_shared_surface_as_ambiguous(caplog):
    # A realm that's both a Location AND an Organization, same exact name string.
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([loc("Riverton"), org("Riverton")])
    assert cmap.entity_anchors == ["riverton", "riverton-2"]  # both anchored
    assert "Riverton" not in surfaces(cmap)  # shared surface dropped
    assert REVIEW_PREFIX in caplog.text

# --- loader ----------------------------------------------------------------- #

def test_load_crosslink_words_missing_file_defaults_empty(tmp_path):
    missing = tmp_path / "nope.json"
    assert load_crosslink_words(str(missing)) == {"require_article": [], "never_link": []}


def test_load_crosslink_words_reads_file(tmp_path):
    p = tmp_path / "words.json"
    p.write_text('{"require_article": ["Founding"], "never_link": ["Pond"]}', encoding="utf-8")
    assert load_crosslink_words(str(p)) == {"require_article": ["Founding"], "never_link": ["Pond"]}
