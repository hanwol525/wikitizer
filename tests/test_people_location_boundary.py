"""Regression tripwire for the People-vs-Location boundary in the People & Cultures
prompt.

Background: the People & Cultures extractor was minting bare COUNTRIES as bogus
"peoples" (e.g. a country described only as "a country in the empire with X as its
capital" got its own People & Cultures page, on top of the correct "People of X"
entry). Root cause: the prompt listed "a nation's people" as an inclusion item and
had boundary bullets vs Characters and Organizations but NONE vs Locations. The fix
added a Locations boundary bullet, tightened the inclusion phrasing, and added a
negative example.

Why this test is prompt-string-only: the behavior it fixes is only observable via a
real LLM call. The offline suite drives a FakeClient that returns canned JSON
regardless of the system prompt, so it CANNOT assert prompt semantics. The live
behavioral guard is the `country_vs_people` Case in
tests/fixtures/synthetic_messages.py (run with `pytest -m integration`). What this
file buys is a DEFAULT-RUN tripwire: a plain `pytest` deselects integration tests, so
without this nothing here would catch an accidental deletion of the boundary. Same
"assert the prompt contains X" precedent as test_characters_extractor.py.
"""

from agents.people_and_cultures_extractor import PeopleAndCulturesExtractor


def test_people_prompt_has_a_locations_boundary_bullet():
    """The primary fix: an explicit 'a PLACE is not a people' boundary vs Locations,
    mirroring the Organization prompt's leading place-is-not-us bullet."""
    prompt = PeopleAndCulturesExtractor.system_prompt
    assert "A PLACE is NOT a people or culture" in prompt
    # The bullet must name the locations extractor as the place's home and give the
    # decisive drop rule (place-only description -> not extracted here).
    assert "captured by the locations extractor" in prompt
    assert "leave the place to the locations extractor" in prompt


def test_people_prompt_disambiguates_nation_vs_its_people():
    """The 'a nation's people' inclusion item was the phrase the model over-read as
    'the nation itself'; it must now spell out that the inhabitants, not the nation,
    are what belong here."""
    prompt = PeopleAndCulturesExtractor.system_prompt
    assert "the people of a nation (its inhabitants, NOT the nation itself)" in prompt
    # The bare, misreadable phrasing must be gone.
    assert "a nation's people" not in prompt


def test_people_prompt_has_a_negative_country_example():
    """The example gained a bare-country message that must yield NOTHING here, with a
    note explaining why -- the highest-value example for an over-extraction bug."""
    prompt = PeopleAndCulturesExtractor.system_prompt
    assert "Gol is a country in the empire, with Crown's Nest as its capital" in prompt
    assert 'Only "the people of Gol", described as a group, would belong here.' in prompt
