"""C6a: render_wiki drops truly-empty entities.

An entity with no prose, no details, and no supporting quotes has nothing to show
and would render as a bare heading. render_wiki now prunes these BEFORE building the
cross-link map, so a dropped entity is neither an anchor target nor a link source
(mentions of its name render as plain text, never a link to a missing page). This is
how a business wrongly grabbed as a name-only Location vanishes while the correct
Organization copy of the same name survives. Offline, pure.
"""

from models.lore import Character, Detail, Location, Organization, Quote
from renderer.markdown import _is_renderable, render_wiki


def det(text, *sources):
    return Detail(text=text, source_files=list(sources))


# --- the predicate --------------------------------------------------------- #
def test_is_renderable_truth_table():
    assert _is_renderable(Location(name="X", details=[det("a fact")])) is True
    assert _is_renderable(Location(name="X", prose="polished")) is True
    q = Quote(text="X exists", speaker="M", source_file="g.txt")
    assert _is_renderable(Location(name="X", supporting_quotes=[q])) is True
    assert _is_renderable(Location(name="X")) is False   # nothing to show


# --- render_wiki drops them ------------------------------------------------ #
def test_factless_quoteless_entity_is_dropped_entirely():
    out = render_wiki([Location(name="Ghosttown")], [], [], [], [], [])
    assert out == ""                          # no heading, no section


def test_name_only_entity_with_a_quote_is_kept():
    q = Quote(text="Realton is a place", speaker="M", source_file="g.txt")
    out = render_wiki([Location(name="Realton", supporting_quotes=[q])], [], [], [], [], [])
    assert '### <a id="realton"></a>Realton' in out


def test_dropped_entity_name_is_not_linked_in_other_prose():
    # Ghosttown has no facts -> dropped -> NOT an anchor, so a mention of "Ghosttown"
    # in another entity's body renders as plain text, not a dangling link.
    ghost = Location(name="Ghosttown")
    town = Location(name="Riverton", details=[det("Riverton lies near Ghosttown.")])
    out = render_wiki([ghost, town], [], [], [], [], [])
    assert "Ghosttown" in out                 # the word still appears in Riverton's body
    assert "(#ghosttown)" not in out          # but never as a link to the dropped page


def test_cross_type_empty_location_dropped_populated_org_survives():
    # The Wizard's-Tower scenario: a business grabbed as an empty Location + the correct
    # populated Organization of the SAME name. Only the org should render, and a mention
    # links to the org's anchor.
    empty_loc = Location(name="Wizard's Tower Brewing Company")
    org = Organization(name="Wizard's Tower Brewing Company",
                       details=[det("Based in Oberveis, known for its Red Squash Lager.")])
    mentioner = Character(name="Barkeep",
                          details=[det("Sells ale from Wizard's Tower Brewing Company.")])
    out = render_wiki([empty_loc], [mentioner], [], [org], [], [])
    # exactly one heading for the name (the org), which keeps the clean slug because the
    # empty Location was pruned before the map was built.
    assert out.count('### <a id="wizards-tower-brewing-company"></a>') == 1
    assert '<a id="wizards-tower-brewing-company-2">' not in out   # no cross-type suffix
    assert "## Locations" not in out                              # the empty location section is gone
    # the mention in the Barkeep body links to the surviving org
    assert "[Wizard's Tower Brewing Company](#wizards-tower-brewing-company)" in out
