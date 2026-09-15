"""Realm Location+Organization dual keeps working cross-links (run-1.15 §C).

A realm is legitimately BOTH a Location (the place) and the Organization that governs
it, so the reconciler keeps two pages. Before, the cross-linker dropped their shared
surface as "ambiguous" -- so every mention of the realm went unlinked. Now that ONE
legit cross-type dual routes its surface to the Location's anchor and stays linkable,
while a genuinely ambiguous collision (two same-type entities, or any kind combo that
isn't exactly Location+Organization) still drops. Offline, fabricated fixtures, no API.
"""

import logging

from models.lore import Item, Location, Organization, PeopleAndCultures
from renderer.crosslink import add_crosslinks, build_crosslink_map


from models.lore import Alias


def loc(name, aliases=None):
    return Location(name=name, aliases=[Alias(text=a, source_files=[]) for a in (aliases or [])])


def org(name, aliases=None):
    return Organization(name=name, aliases=[Alias(text=a, source_files=[]) for a in (aliases or [])])


def item(name):
    return Item(name=name)


def peo(name):
    return PeopleAndCultures(name=name)


def surfaces(cmap):
    return [s for s, _ in cmap.sources]


# --- the dual routes to the Location and stays linkable --------------------- #
def test_identical_name_dual_routes_to_location():
    cmap = build_crosslink_map([loc("Krieger Imperium"), org("Krieger Imperium")])
    assert ("Krieger Imperium", "krieger-imperium") in cmap.sources
    out = add_crosslinks("Envoys of Krieger Imperium arrived.", cmap, None)
    assert out == "Envoys of [Krieger Imperium](#krieger-imperium) arrived."


def test_article_differing_dual_still_routes_to_location():
    # The user's Citadel case: "Citadel" (the place) + "The Citadel" (the governing
    # order). Their anchors differ by a "the-" prefix, but they share ONE article-
    # stripped surface, so the dual is recognized on KINDS alone and routed.
    cmap = build_crosslink_map([loc("Citadel"), org("The Citadel")])
    assert cmap.entity_anchors == ["citadel", "the-citadel"]
    assert ("Citadel", "citadel") in cmap.sources
    # Both a bare and an article-led mention link to the Location page.
    assert add_crosslinks("The Citadel holds.", cmap, None) == "The [Citadel](#citadel) holds."
    assert add_crosslinks("We saw Citadel.", cmap, None) == "We saw [Citadel](#citadel)."


def test_dual_routes_even_when_organization_listed_first():
    # Routing is by KIND (Location wins), not by list order -- so an Org-first concat
    # still points the surface at the Location's anchor.
    cmap = build_crosslink_map([org("Krieger Imperium"), loc("Krieger Imperium")])
    # org is first -> org gets the clean slug, loc gets -2; routing still picks the loc.
    assert ("Krieger Imperium", "krieger-imperium-2") in cmap.sources


# --- non-dual collisions still drop as ambiguous ---------------------------- #
def test_location_item_collision_drops(caplog):
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([loc("Aegis"), item("Aegis")])
    assert "Aegis" not in surfaces(cmap)
    assert "left out of the link pool" in caplog.text


def test_people_org_collision_drops(caplog):
    # People+Organization is NOT the legit realm dual (that's strictly Location+Org).
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([peo("Crowsworn"), org("Crowsworn")])
    assert "Crowsworn" not in surfaces(cmap)
    assert "left out of the link pool" in caplog.text


def test_two_locations_same_name_drop(caplog):
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([loc("Riverton"), loc("Riverton")])
    assert "Riverton" not in surfaces(cmap)
    assert "left out of the link pool" in caplog.text


def test_loc_org_sharing_only_an_alias_still_drops(caplog):
    # Kinds ARE {Location, Organization}, but the shared surface is an ALIAS of two
    # DIFFERENTLY-named entities -- not a realm dual. Routing is gated on the surface
    # being a NAME of both, so this correctly stays ambiguous and drops.
    with caplog.at_level(logging.WARNING):
        cmap = build_crosslink_map([loc("Aldermere", aliases=["The Reach"]),
                                    org("Vosshold", aliases=["The Reach"])])
    assert "Reach" not in surfaces(cmap)
    assert "left out of the link pool" in caplog.text
