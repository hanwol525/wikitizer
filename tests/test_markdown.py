"""Unit tests for renderer/markdown.py (Phase 4.4 Brief 2: render_history).

Plain pytest -- no network, no API key, no integration marker. render_history is
pure given its inputs, and build_crosslink_map (used here to get a real, parallel
event_anchors list plus a working add_crosslinks) is pure too, so this whole suite
runs offline. Every fixture is fabricated synthetic lore; every assertion exact.
"""

from models.lore import (
    Character,
    HistoryEvent,
    Item,
    Location,
    Organization,
    PeopleAndCultures,
    Quote,
    Scope,
)
from renderer.crosslink import build_crosslink_map
from renderer.footnotes import FootnoteRegistry
from renderer.markdown import render_history, render_wiki


# --- tiny builders ---------------------------------------------------------- #

def hev(name, description="An event happened.", calendar_system=None,
        chronological_position=None, date_text=None, aliases=None, quotes=None):
    # Build a HistoryEvent with the stamped fields set directly, as if the weave
    # had already run. Matches test_reconciler.py's construction for the required
    # fields (scope especially); only the fields a test asserts on matter.
    return HistoryEvent(
        name=name,
        description=description,
        scope=Scope.WORLD,
        date_text=date_text,
        calendar_system=calendar_system,
        chronological_position=chronological_position,
        aliases=aliases or [],
        supporting_quotes=quotes or [],
    )


def loc(name, aliases=None):
    # A minimal Location, same as test_crosslink.py's builder.
    return Location(name=name, aliases=aliases or [])


def test_no_events_returns_empty_string():
    assert render_history([], build_crosslink_map([]), FootnoteRegistry()) == ""


def test_single_system_no_subhead_no_note():
    events = [hev("The Founding", calendar_system="AR", chronological_position=0, date_text="1 AR"),
              hev("The Sundering", calendar_system="AR", chronological_position=1, date_text="50 AR")]
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    assert out.startswith("## History")
    assert "**Important Note:**" not in out   # single system -> no note
    assert "### AR" not in out                # single system -> no per-system subhead
    # chronological order
    assert out.index("The Founding") < out.index("The Sundering")


def test_multi_system_has_note_and_orders_sections_by_count():
    # System "AR" has 2 events, "Elvish" has 1 -> AR section first.
    events = [
        hev("Founding", calendar_system="AR", chronological_position=0),
        hev("Sundering", calendar_system="AR", chronological_position=1),
        hev("Starfall", calendar_system="Elvish", chronological_position=0),
    ]
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    assert out.startswith("## History")
    assert "**Important Note:**" in out
    assert "### AR" in out and "### Elvish" in out
    assert out.index("### AR") < out.index("### Elvish")   # more events -> first


def test_all_undated_uses_timeline_header_and_no_subhead():
    events = [hev("A Quiet Season", chronological_position=0),
              hev("A Long Winter", chronological_position=1)]
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    assert out.startswith("## Timeline")
    assert "### Undated Events" not in out   # they ARE the body
    assert "**Important Note:**" not in out


def test_could_not_place_only_uses_history_header_with_note_and_no_subhead():
    events = [hev("A Rumor"), hev("A Whisper")]   # no system, no position
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    assert out.startswith("## History")
    assert "**Important Note:**" in out
    assert "### Could Not Place" not in out   # they ARE the body -> note, not subhead


def test_undated_and_could_not_place_subheads_appear_below_dated_in_order():
    events = [
        hev("Founding", calendar_system="AR", chronological_position=0),
        hev("A Quiet Season", chronological_position=0),   # undated-but-placed
        hev("A Rumor"),                                     # could-not-place
    ]
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    assert "### Undated Events" in out
    assert "### Could Not Place" in out
    assert out.index("### Undated Events") < out.index("### Could Not Place")


def test_dated_entry_atom_has_anchor_bold_name_date_tag_and_footnote():
    q = Quote(text="It split the world", speaker="Matt", source_file="g.txt")
    events = [hev("The Sundering", description="It split the world.", date_text="50 AR",
                  calendar_system="AR", chronological_position=0, quotes=[q])]
    reg = FootnoteRegistry()
    out = render_history(events, build_crosslink_map([], events=events), reg)
    assert '- <a id="the-sundering"></a>**The Sundering** — 50 AR. It split the world.[^1]' in out
    assert reg.render_definitions() != ""   # the quote was registered


def test_undated_entry_atom_has_no_date_tag():
    events = [hev("A Quiet Season", description="A long peace.", chronological_position=0)]
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    assert '**A Quiet Season**. A long peace.' in out
    assert "—" not in out   # no date -> no em-dash date tag anywhere


def test_ineligible_event_renders_without_an_anchor():
    # A sentence-shaped name is gated out of anchoring (event_anchors slot is None),
    # so it still renders but with no <a id>.
    long_name = "The party finally reached the distant gates just before nightfall"
    events = [hev(long_name, description="They made camp.", chronological_position=0)]
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    assert "<a id=" not in out                 # no anchor stamped
    assert f"**{long_name}**" in out           # but the event still renders


def test_anchor_stays_paired_with_its_event_after_display_sort():
    # THE footgun. Feed two dated events in REVERSE chronological order. After
    # render_history sorts them chronologically, each bullet must still carry ITS
    # OWN anchor, not the anchor of whatever event now sits at that position.
    later = hev("The Sundering", calendar_system="AR", chronological_position=1)
    earlier = hev("The Founding", calendar_system="AR", chronological_position=0)
    events = [later, earlier]                          # input order: later first
    cmap = build_crosslink_map([], events=events)      # event_anchors parallel to input
    out = render_history(events, cmap, FootnoteRegistry())
    founding_line = next(l for l in out.splitlines() if "The Founding" in l)
    sundering_line = next(l for l in out.splitlines() if "The Sundering" in l)
    assert 'id="the-founding"' in founding_line        # not swapped
    assert 'id="the-sundering"' in sundering_line
    assert out.index("The Founding") < out.index("The Sundering")   # sorted chronologically


def test_could_not_place_sorted_alphabetically():
    events = [hev("Zephyr Omen"), hev("Ashen Sign"), hev("Moonrise")]
    out = render_history(events, build_crosslink_map([], events=events), FootnoteRegistry())
    assert out.index("Ashen Sign") < out.index("Moonrise") < out.index("Zephyr Omen")


def test_description_is_crosslinked_and_self_link_suppressed():
    # A noun mention in a description becomes a link; the event's own name inside
    # its own description does NOT link back to itself (this_anchor suppression).
    events = [hev("The Sundering",
                  description="The Sundering struck Lake Mundi hard.",
                  calendar_system="AR", chronological_position=0)]
    cmap = build_crosslink_map([loc("Lake Mundi")], events=events)
    out = render_history(events, cmap, FootnoteRegistry())
    assert "[Lake Mundi](#lake-mundi)" in out          # the noun mention linked
    assert "[Sundering](#the-sundering)" not in out    # self-mention NOT linked


# --- Phase 4.4 Brief 3: entity sections + render_wiki -----------------------

def test_entity_renders_as_heading_with_anchor_and_prose_body():
    loc = Location(name="Rivendell", details=["A hidden valley.", "Home to Elrond."])
    out = render_wiki([loc], [], [], [], [], [])
    assert "## Locations" in out
    assert '### <a id="rivendell"></a>Rivendell' in out
    assert "A hidden valley. Home to Elrond." in out   # details joined as prose


def test_section_order_is_fixed():
    # The event is DATED (calendar_system set) so the history section renders as
    # "## History"; an undated-only history would render "## Timeline" (Brief 2's
    # locked render_history rule), which is a different string than this test pins.
    out = render_wiki(
        [Location(name="Bree")], [Character(name="Aragorn")],
        [hev("The Founding", calendar_system="AR", chronological_position=0)],
        [Organization(name="Rangers")], [Item(name="Anduril")],
        [PeopleAndCultures(name="Elves")],
    )
    assert (out.index("## Locations") < out.index("## History")
            < out.index("## People & Cultures") < out.index("## Organizations")
            < out.index("## Characters") < out.index("## Items"))


def test_empty_sections_are_dropped():
    out = render_wiki([Location(name="Bree")], [], [], [], [], [])
    assert "## Locations" in out
    assert "## Characters" not in out
    assert "## History" not in out


def test_empty_world_returns_empty_string():
    assert render_wiki([], [], [], [], [], []) == ""


def test_entities_sorted_alphabetically_within_section():
    out = render_wiki([Location(name="Zephyr Peak"), Location(name="Ashford")],
                      [], [], [], [], [])
    assert out.index("Ashford") < out.index("Zephyr Peak")


def test_entity_type_routed_to_correct_section():
    # A People & Cultures entry must land under People & Cultures, not Characters.
    out = render_wiki([], [Character(name="Gimli")], [], [], [],
                      [PeopleAndCultures(name="Dwarves")])
    people_section = out[out.index("## People & Cultures"):out.index("## Characters")]
    assert "Dwarves" in people_section          # routed to the right section
    assert "Gimli" not in people_section


def test_entity_anchor_stays_paired_after_type_split_and_sort():
    # The footgun, entity edition. Concatenation puts the location first, so it keeps
    # "riverton" and the org suffixes to "riverton-2". Each heading must carry ITS
    # OWN anchor after the isinstance-split and the alphabetical sort -- not swapped.
    out = render_wiki([Location(name="Riverton")], [], [],
                      [Organization(name="Riverton")], [], [])
    assert '### <a id="riverton"></a>Riverton' in out
    assert '### <a id="riverton-2"></a>Riverton' in out


def test_entity_footnotes_land_at_end_of_body():
    q = Quote(text="a fact", speaker="M", source_file="g.txt")
    out = render_wiki([Location(name="Bree", details=["A town."], supporting_quotes=[q])],
                      [], [], [], [], [])
    assert "A town.[^1]" in out    # marker at end of body
    assert "[^1]:" in out          # definition block rendered at the end


def test_entity_body_is_crosslinked_and_self_link_suppressed():
    bree = Location(name="Bree", details=["Bree lies near Rivendell."])
    rivendell = Location(name="Rivendell", details=["A valley."])
    out = render_wiki([bree, rivendell], [], [], [], [], [])
    assert "[Rivendell](#rivendell)" in out    # cross-reference linked
    assert "[Bree](#bree)" not in out          # own name in own body not linked


def test_footnotes_numbered_in_render_order_across_sections():
    q1 = Quote(text="loc fact", speaker="M", source_file="g.txt")
    q2 = Quote(text="char fact", speaker="M", source_file="g.txt")
    out = render_wiki([Location(name="Bree", details=["x"], supporting_quotes=[q1])],
                      [Character(name="Aragorn", details=["y"], supporting_quotes=[q2])],
                      [], [], [], [])
    # Locations render before Characters, so the location's quote is [^1].
    assert "x[^1]" in out
    assert "y[^2]" in out


def test_history_section_is_included():
    out = render_wiki([], [], [hev("The Founding", chronological_position=0,
                                   description="It began.")], [], [], [])
    assert "## Timeline" in out          # render_history produced its section
    assert "The Founding" in out


def test_entity_with_empty_details_renders_clean_heading_no_trailing_blanks():
    # A named-but-factless entity (empty details) is a supported extractor output.
    # It must render as a clean heading with NO empty body paragraph trailing it.
    out = render_wiki([Location(name="Bree")], [], [], [], [], [])
    assert out == '## Locations\n\n### <a id="bree"></a>Bree'   # exact: no trailing blank
    assert "[^" not in out                                     # no stray footnote marker


def test_two_empty_details_entities_have_no_triple_blank_gap():
    out = render_wiki([Location(name="Ashford"), Location(name="Bree")], [], [], [], [], [])
    assert "\n\n\n" not in out   # exactly one blank line between the two headings


def test_empty_details_entity_with_quote_hangs_footnote_on_heading():
    # With no body, a supporting quote's marker would otherwise float alone on its
    # own paragraph; instead it hangs on the heading so provenance is kept, cleanly.
    q = Quote(text="Bree is mentioned", speaker="M", source_file="g.txt")
    out = render_wiki([Location(name="Bree", supporting_quotes=[q])], [], [], [], [], [])
    assert '### <a id="bree"></a>Bree[^1]' in out    # marker attached to heading
    assert "></a>Bree\n\n[^" not in out              # NOT floating on its own paragraph
    assert "[^1]:" in out                            # definition still rendered


def test_entity_heading_anchor_matches_its_own_name_across_types():
    # Companion to the homonym footgun test above: DISTINCT names couple each
    # heading's name to its expected anchor, so a pure entity<->anchor swap (which
    # the same-name test can't detect, since both anchors appear either way) would
    # mismatch a name with the wrong slug and fail here.
    out = render_wiki([Location(name="Riverton")], [], [],
                      [Organization(name="Ironhold")], [], [])
    assert '### <a id="riverton"></a>Riverton' in out
    assert '### <a id="ironhold"></a>Ironhold' in out
