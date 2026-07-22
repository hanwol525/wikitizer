"""Display-name casing normalization (output-quality batch).

Players type a place in ALL CAPS for emphasis ("EGLON") and the extractor keeps the name
verbatim, so it renders as a shouting heading. `_normalize_display_name` title-cases such
screaming names for display while preserving genuine short acronyms/initials. render_wiki
applies it to a rebuilt copy of every entity + event BEFORE building the cross-link map, so
the heading and the (case-sensitive) cross-link surface pool stay in sync. Offline, pure.
"""

from models.lore import Alias, Detail, Location
from renderer.markdown import _normalize_display_name, render_wiki


# --- the pure helper -------------------------------------------------------- #
def test_screaming_single_word():
    assert _normalize_display_name("EGLON") == "Eglon"


def test_whole_name_all_caps_multiword():
    assert _normalize_display_name("THE CITADEL") == "The Citadel"


def test_partial_caps_token_is_normalized():
    assert _normalize_display_name("Lake MUNDI") == "Lake Mundi"


def test_preserves_short_acronyms_and_initials():
    assert _normalize_display_name("CJ") == "CJ"
    assert _normalize_display_name("DM") == "DM"
    assert _normalize_display_name("FBI") == "FBI"        # 3 letters -> below the >=4 gate


def test_leaves_mixed_case_names_untouched():
    assert _normalize_display_name("Lake Mundi") == "Lake Mundi"
    assert _normalize_display_name("Mal'taav") == "Mal'taav"


def test_apostrophe_uses_slice_not_title():
    assert _normalize_display_name("MAL'TAAV") == "Mal'taav"   # NOT "Mal'Taav"


def test_empty_string_is_safe():
    assert _normalize_display_name("") == ""


# --- lowercase + full title-casing (bucket B extension) --------------------- #
def test_all_lowercase_and_all_caps_reach_the_same_canonical():
    assert _normalize_display_name("crown's nest") == "Crown's Nest"
    assert _normalize_display_name("CROWN'S NEST") == "Crown's Nest"


def test_lowercase_leading_article_is_capitalized():
    assert _normalize_display_name("the deathsworn") == "The Deathsworn"


def test_interior_stopwords_stay_lowercase():
    assert _normalize_display_name("houses of maltaav") == "Houses of Maltaav"
    assert _normalize_display_name("the 12 houses of maltraav") == "The 12 Houses of Maltraav"


def test_hyphenated_word_capitalizes_each_part():
    assert _normalize_display_name("half-elf") == "Half-Elf"


def test_already_wellcased_mixed_name_is_left_alone():
    # A name that already has good mixed case (incl. a lowercase stop-word) is untouched.
    assert _normalize_display_name("The 12 Houses of Maltraav") == "The 12 Houses of Maltraav"
    assert _normalize_display_name("Lake Mundi") == "Lake Mundi"


# --- through render_wiki (heading + cross-link stay in sync) ----------------- #
def _loc(name, details):
    return Location(name=name,
                    details=[Detail(text=d, source_files=["g.txt"]) for d in details])


def test_render_wiki_normalizes_heading_and_still_crosslinks():
    eglon = _loc("EGLON", ["A great capital city"])
    riverton = _loc("Riverton", ["A village on the road to Eglon"])
    md = render_wiki([eglon, riverton], [], [], [], [], [])
    # The heading is title-cased, not shouting -- nowhere in the doc is the ALL-CAPS form.
    assert "Eglon" in md
    assert "EGLON" not in md
    # And the normalized surface still links from another entity's prose (the pool was
    # built from the SAME normalized name, so the case-sensitive match succeeds).
    assert "[Eglon](#eglon)" in md


def test_render_wiki_lowercase_heading_is_title_cased():
    md = render_wiki([_loc("crown's nest", ["A cliffside fortress town."])],
                     [], [], [], [], [])
    assert "Crown's Nest" in md
    assert "crown's nest" not in md               # the lowercase form is gone from the heading


def test_render_wiki_normalizes_alias_surface_for_crosslinking():
    # An alias typed lowercase is canonicalized in the surface pool, so a canonically-cased
    # mention in ANOTHER entity's prose links to it (it would not, case-sensitively, otherwise).
    hold = Location(name="Crown's Nest",
                    aliases=[Alias(text="crowns hold", source_files=["g.txt"])],
                    details=[Detail(text="A fortress.", source_files=["g.txt"])])
    road = _loc("Riverton", ["The road runs from Crowns Hold to the coast"])
    md = render_wiki([hold, road], [], [], [], [], [])
    assert "[Crowns Hold](#crowns-nest)" in md
