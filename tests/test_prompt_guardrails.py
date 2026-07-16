"""Regression tripwires for the prompt-only fixes (clusters C3/C4/C5/C6b/C6c/C7).

These fixes are LLM-behavior — their real effect only shows on a live call (the
integration suite covers that). What a plain `pytest` CAN guard is that the
distinctive new guidance is still PRESENT in each prompt constant, so a later edit
can't silently revert it. Same "assert the prompt contains X" precedent as
test_people_location_boundary.py / test_characters_extractor.py.
"""

from agents.reconciler import RECONCILER_SYSTEM_PROMPT, DATE_EXTRACTION_PROMPT
from agents.characters_extractor import SYSTEM_PROMPT as CHARACTERS_PROMPT
from agents.organization_extractor import SYSTEM_PROMPT as ORG_PROMPT
from agents.locations_extractor import SYSTEM_PROMPT as LOCATIONS_PROMPT
from agents.noise_filter import SYSTEM_PROMPT as NOISE_PROMPT


# C3 — prefer a stated real name over an assumed/fake alias
def test_reconciler_prefers_real_name_over_assumed():
    assert "A REAL name over an ASSUMED one" in RECONCILER_SYSTEM_PROMPT
    assert "EVEN IF the assumed name appears more often" in RECONCILER_SYSTEM_PROMPT


def test_characters_extractor_prefers_real_name():
    assert "REAL/true/actual/birth name" in CHARACTERS_PROMPT


# C4 — timeline defaults to one calendar; continuous reigns are not a new system
def test_timeline_defaults_to_single_calendar():
    assert "STRONGLY DEFAULT TO A SINGLE CALENDAR SYSTEM" in DATE_EXTRACTION_PROMPT
    assert "NO reset to 0 per ruler" in DATE_EXTRACTION_PROMPT


# C5 — org label reads a place embedded in the group's own name
def test_org_label_uses_place_in_name():
    assert "The baseball team from Valdenmoor" in ORG_PROMPT
    assert "when the name itself embeds a place" in ORG_PROMPT.lower()


# C6b — a business is an organization, not a location
def test_locations_business_boundary():
    assert "A business, company, shop, guild, or brewery is NOT a location" in LOCATIONS_PROMPT


# C6c — a single individual by role is a character, not an organization
def test_org_role_singleton_boundary():
    assert '"Kriggy\'s guard"' in ORG_PROMPT
    assert "structured GROUP of people" in ORG_PROMPT


# C7 — out-of-world / meta content is noise
def test_noise_filter_flags_out_of_world_meta():
    assert "out-of-world" in NOISE_PROMPT
    assert "regular person from Earth" in NOISE_PROMPT


# Fix C — a country/realm with only geography (no governance) is Locations-only, not an org
def test_org_excludes_pure_geography_countries():
    assert "pure geography" in ORG_PROMPT
    # the "name alone is enough" rule is explicitly scoped away from bare place names
    assert "does NOT license capturing a bare place, country, or realm name" in ORG_PROMPT
    # a negative worked example demonstrating the drop
    assert "is a Location, not an organization" in ORG_PROMPT


# Fix D — merge same-referent entities even without a shared name/alias
def test_reconciler_merges_same_referent_without_shared_name():
    assert "The SAME referent under two descriptions that share NO words" in RECONCILER_SYSTEM_PROMPT
    assert "shared defining tie" in RECONCILER_SYSTEM_PROMPT
    # the file-pure rationale is captured so a later edit knows WHY this rule exists
    assert "file-pure" in RECONCILER_SYSTEM_PROMPT
