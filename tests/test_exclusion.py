"""Tests for exclusion.py (Phase 4.6 Part 2: the restricted-view carving).

Pure, no network, no API key -- exclusion.py is LLM-free. Two jobs under test:
`validate_exclusions` (the fail-fast guard) and `_filter_entity`/`filter_entities`
(the confidential-fact/quote carving). The orchestration that wires these into a
second wiki (and the History re-run) is covered in test_orchestrator.py.
"""

import pytest

from exclusion import validate_exclusions, filter_entities, _filter_entity
from models.lore import (
    Detail, Quote, Location, Character, Organization, Item, PeopleAndCultures,
)


def det(text, *source_files):
    return Detail(text=text, source_files=list(source_files))


def q(text, source="group.txt"):
    return Quote(text=text, speaker="M", source_file=source)


# --- validate_exclusions ----------------------------------------------------

def test_validate_exclusions_clean_set_does_not_raise():
    validate_exclusions(["dm.txt"], ["logs/group.txt", "logs/dm.txt"])   # no raise


def test_validate_exclusions_unknown_name_raises():
    with pytest.raises(ValueError):
        validate_exclusions(["ghost.txt"], ["logs/group.txt"])


def test_validate_exclusions_path_form_of_real_file_raises_with_bare_hint():
    # Strict: even a path to a REAL input file is rejected (messages store bare
    # names), and the message points at the bare name to use instead.
    with pytest.raises(ValueError) as exc:
        validate_exclusions(["logs/dm.txt"], ["logs/group.txt", "logs/dm.txt"])
    assert "dm.txt" in str(exc.value)
    assert "path" in str(exc.value)   # the "looks like a path" hint fired


def test_validate_exclusions_empty_list_does_not_raise():
    validate_exclusions([], ["logs/group.txt"])   # no raise


def test_validate_exclusions_message_lists_valid_names():
    with pytest.raises(ValueError) as exc:
        validate_exclusions(["nope.txt"], ["logs/group.txt", "logs/dm.txt"])
    # the helpful message enumerates the real bare filenames
    assert "group.txt" in str(exc.value) and "dm.txt" in str(exc.value)


# --- _filter_entity: the per-fact / per-quote drop rules --------------------

def test_fact_from_public_and_secret_survives_sources_untouched():
    # any-source-public: a fact stated in both a public and a secret file is public.
    e = Location(name="X", details=[det("f", "group.txt", "secret.txt")], supporting_quotes=[])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert [d.text for d in kept.details] == ["f"]
    assert kept.details[0].source_files == ["group.txt", "secret.txt"]   # NOT rewritten


def test_fact_only_secret_source_is_dropped():
    e = Location(name="X", details=[det("secret fact", "secret.txt")],
                 supporting_quotes=[q("public quote", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert [d.text for d in kept.details] == []   # secret-only fact gone


def test_fact_with_no_sources_is_dropped():
    # A source-less Detail can't be proven public -> over-hide (drop it).
    e = Location(name="X", details=[det("unsourced fact")],
                 supporting_quotes=[q("public quote", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert [d.text for d in kept.details] == []


def test_quote_from_excluded_file_dropped_public_kept():
    e = Location(name="X", details=[],
                 supporting_quotes=[q("public", "group.txt"), q("secret", "secret.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert [qq.text for qq in kept.supporting_quotes] == ["public"]


def test_entity_kept_when_only_a_public_quote_survives():
    # all facts secret, but a public quote keeps the entity alive (details -> []).
    e = Location(name="X", details=[det("secret", "secret.txt")],
                 supporting_quotes=[q("public", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert [d.text for d in kept.details] == []
    assert [qq.text for qq in kept.supporting_quotes] == ["public"]


def test_entity_kept_when_only_a_public_fact_survives():
    # all quotes secret, but a public fact keeps the entity alive (quotes -> []).
    e = Location(name="X", details=[det("public", "group.txt")],
                 supporting_quotes=[q("secret", "secret.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert [d.text for d in kept.details] == ["public"]
    assert kept.supporting_quotes == []


def test_entity_dropped_when_nothing_public_survives():
    e = Location(name="X", details=[det("secret", "secret.txt")],
                 supporting_quotes=[q("secret", "secret.txt")])
    assert _filter_entity(e, {"secret.txt"}) is None


def test_filter_entity_does_not_mutate_input():
    e = Location(name="X",
                 details=[det("public", "group.txt"), det("secret", "secret.txt")],
                 supporting_quotes=[q("public", "group.txt"), q("secret", "secret.txt")])
    _filter_entity(e, {"secret.txt"})
    # the reconciled original (which the FULL doc still needs) is untouched
    assert [d.text for d in e.details] == ["public", "secret"]
    assert [qq.text for qq in e.supporting_quotes] == ["public", "secret"]


def test_filter_entity_preserves_character_specific_fields():
    c = Character(name="CJ", is_pc=True, player_name="Hannah",
                  details=[det("public", "group.txt"), det("secret", "secret.txt")],
                  supporting_quotes=[q("public", "group.txt")])
    kept = _filter_entity(c, {"secret.txt"})
    assert kept.is_pc is True and kept.player_name == "Hannah"   # carried by model_copy
    assert [d.text for d in kept.details] == ["public"]


@pytest.mark.parametrize("Model", [Location, Character, Organization, Item, PeopleAndCultures])
def test_filter_entity_mixed_public_secret_over_all_five_types(Model):
    e = Model(name="X",
              details=[det("public fact", "group.txt"), det("secret fact", "secret.txt")],
              supporting_quotes=[q("public quote", "group.txt"), q("secret quote", "secret.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert [d.text for d in kept.details] == ["public fact"]
    assert [qq.text for qq in kept.supporting_quotes] == ["public quote"]


# --- filter_entities: the list wrapper --------------------------------------

def test_filter_entities_drops_wholly_secret_and_maps_the_rest():
    pub = Location(name="Publicton", details=[det("f", "group.txt")], supporting_quotes=[])
    sec = Location(name="Secretville", details=[det("f", "secret.txt")], supporting_quotes=[])
    out = filter_entities([pub, sec], {"secret.txt"})
    assert [e.name for e in out] == ["Publicton"]   # the secret-only entity is gone


def test_filter_entities_empty_in_empty_out():
    assert filter_entities([], {"secret.txt"}) == []
