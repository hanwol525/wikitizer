"""Tests for exclusion.py (Phase 4.6 Part 2 carving + Part 3 alias/name re-head).

Pure, no network, no API key -- exclusion.py is LLM-free. Under test:
`validate_exclusions` (the fail-fast guard, incl. the duplicate-basename check) and
`_filter_entity`/`filter_entities` (the confidential-fact/quote/alias carving plus
the Part 3 secret-name re-head). The orchestration that wires these into a second
wiki is covered in test_orchestrator.py.

Part 3 note: a surviving entity keeps its `name` only when that name is provably
public -- i.e. `name_sources` has a non-excluded entry. So every entity that should
keep its name sets `name_sources` to a public file; a secret-only name is re-headed
to a public alias (or dropped if none survives).
"""

import logging

import pytest

from exclusion import validate_exclusions, filter_entities, _filter_entity
from models.lore import (
    Detail, Alias, Quote, Location, Character, Organization, Item, PeopleAndCultures,
)


def det(text, *source_files):
    return Detail(text=text, source_files=list(source_files))


def al(text, *source_files):
    return Alias(text=text, source_files=list(source_files))


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


def test_validate_exclusions_rejects_duplicate_input_basenames():
    # Two inputs sharing a basename make the bare-name match ambiguous (both would be
    # excluded/kept together), so the guard rejects it up front.
    with pytest.raises(ValueError):
        validate_exclusions(["x.txt"], ["a/x.txt", "b/x.txt"])


# --- _filter_entity: the per-fact / per-quote drop rules --------------------

def test_fact_from_public_and_secret_survives_sources_untouched():
    # any-source-public: a fact stated in both a public and a secret file is public.
    e = Location(name="X", name_sources=["group.txt"],
                 details=[det("f", "group.txt", "secret.txt")], supporting_quotes=[])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert [d.text for d in kept.details] == ["f"]
    assert kept.details[0].source_files == ["group.txt", "secret.txt"]   # NOT rewritten


def test_fact_only_secret_source_is_dropped():
    e = Location(name="X", name_sources=["group.txt"], details=[det("secret fact", "secret.txt")],
                 supporting_quotes=[q("public quote", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert [d.text for d in kept.details] == []   # secret-only fact gone


def test_fact_with_no_sources_is_dropped():
    # A source-less Detail can't be proven public -> over-hide (drop it).
    e = Location(name="X", name_sources=["group.txt"], details=[det("unsourced fact")],
                 supporting_quotes=[q("public quote", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert [d.text for d in kept.details] == []


def test_quote_from_excluded_file_dropped_public_kept():
    e = Location(name="X", name_sources=["group.txt"], details=[],
                 supporting_quotes=[q("public", "group.txt"), q("secret", "secret.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert [qq.text for qq in kept.supporting_quotes] == ["public"]


def test_entity_kept_when_only_a_public_quote_survives():
    # all facts secret, but a public quote keeps the entity alive (details -> []).
    e = Location(name="X", name_sources=["group.txt"], details=[det("secret", "secret.txt")],
                 supporting_quotes=[q("public", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert [d.text for d in kept.details] == []
    assert [qq.text for qq in kept.supporting_quotes] == ["public"]


def test_entity_kept_when_only_a_public_fact_survives():
    # all quotes secret, but a public fact keeps the entity alive (quotes -> []).
    e = Location(name="X", name_sources=["group.txt"], details=[det("public", "group.txt")],
                 supporting_quotes=[q("secret", "secret.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert [d.text for d in kept.details] == ["public"]
    assert kept.supporting_quotes == []


def test_entity_dropped_when_nothing_public_survives():
    e = Location(name="X", name_sources=["group.txt"], details=[det("secret", "secret.txt")],
                 supporting_quotes=[q("secret", "secret.txt")])
    assert _filter_entity(e, {"secret.txt"}) is None


def test_filter_entity_preserves_character_specific_fields():
    c = Character(name="CJ", name_sources=["group.txt"], is_pc=True, player_name="Hannah",
                  details=[det("public", "group.txt"), det("secret", "secret.txt")],
                  supporting_quotes=[q("public", "group.txt")])
    kept = _filter_entity(c, {"secret.txt"})
    assert kept.is_pc is True and kept.player_name == "Hannah"   # carried by model_copy
    assert [d.text for d in kept.details] == ["public"]


@pytest.mark.parametrize("Model", [Location, Character, Organization, Item, PeopleAndCultures])
def test_filter_entity_mixed_public_secret_over_all_five_types(Model):
    e = Model(name="X", name_sources=["group.txt"],
              aliases=[al("public aka", "group.txt"), al("secret aka", "secret.txt")],
              details=[det("public fact", "group.txt"), det("secret fact", "secret.txt")],
              supporting_quotes=[q("public quote", "group.txt"), q("secret quote", "secret.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert kept.name == "X"                                  # public name kept, not re-headed
    assert [a.text for a in kept.aliases] == ["public aka"]  # secret alias carved
    assert [d.text for d in kept.details] == ["public fact"]
    assert [qq.text for qq in kept.supporting_quotes] == ["public quote"]


# --- Part 3: alias carving + secret-name re-head ----------------------------

def test_alias_kept_when_any_source_public_sources_untouched():
    e = Location(name="X", name_sources=["group.txt"],
                 aliases=[al("keep me", "group.txt", "secret.txt")],  # public AND secret
                 details=[det("f", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert [a.text for a in kept.aliases] == ["keep me"]
    assert kept.aliases[0].source_files == ["group.txt", "secret.txt"]   # NOT rewritten


def test_alias_dropped_when_secret_only():
    e = Location(name="X", name_sources=["group.txt"],
                 aliases=[al("secret alias", "secret.txt"), al("public alias", "group.txt")],
                 details=[det("f", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert [a.text for a in kept.aliases] == ["public alias"]


def test_alias_with_no_sources_is_dropped():
    # A source-less Alias can't be proven public -> over-hide, same as a Detail.
    e = Location(name="X", name_sources=["group.txt"], aliases=[al("unsourced")],
                 details=[det("f", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept.aliases == []
    assert kept.name == "X"        # public name -> no re-head


def test_no_rehead_when_name_is_public():
    e = Location(name="Publicton", name_sources=["group.txt"], aliases=[al("aka", "group.txt")],
                 details=[det("f", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept.name == "Publicton"                        # name untouched
    assert kept.name_sources == ["group.txt"]
    assert [a.text for a in kept.aliases] == ["aka"]


def test_rehead_when_name_is_secret_only():
    e = Location(name="Blackspire Keep", name_sources=["secret.txt"],
                 aliases=[al("the old fort", "group.txt")],
                 details=[det("An old fort.", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert kept.name == "the old fort"                     # re-headed to the public alias
    assert kept.name_sources == ["group.txt"]              # the promoted alias's sources
    assert "the old fort" not in [a.text for a in kept.aliases]   # heading not self-aliased


def test_unsourced_name_reheads_too():
    # name_sources=[] behaves like secret-only (can't prove public) -> re-head.
    e = Location(name="Mystery", name_sources=[], aliases=[al("the fort", "group.txt")],
                 details=[det("A fort.", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept.name == "the fort"


def test_rehead_picks_first_surviving_public_alias():
    e = Location(name="Secret", name_sources=["secret.txt"],
                 aliases=[al("secret aka", "secret.txt"),     # dropped (secret)
                          al("first public", "group.txt"),    # first surviving -> the heading
                          al("second public", "group.txt")],
                 details=[det("f", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept.name == "first public"
    assert [a.text for a in kept.aliases] == ["second public"]   # first popped as the heading


def test_impossible_case_no_public_name_or_alias_drops_and_flags(caplog):
    # secret name + no public alias, but a public quote keeps the fact/quote gate open
    # -> the re-head has nothing to promote -> drop + loud [REVIEW].
    e = Location(name="Blackspire Keep", name_sources=["secret.txt"],
                 aliases=[al("secret aka", "secret.txt")], details=[],
                 supporting_quotes=[q("a public mention", "group.txt")])
    with caplog.at_level(logging.WARNING, logger="exclusion"):
        kept = _filter_entity(e, {"secret.txt"})
    assert kept is None
    assert "[REVIEW]" in caplog.text and "file-pure" in caplog.text


def test_survives_on_public_quote_alone_and_reheads():
    # all details secret, name secret, but a public quote + public alias -> re-head.
    e = Location(name="Blackspire Keep", name_sources=["secret.txt"],
                 aliases=[al("the old fort", "group.txt")], details=[det("secret", "secret.txt")],
                 supporting_quotes=[q("the old fort stands", "group.txt")])
    kept = _filter_entity(e, {"secret.txt"})
    assert kept is not None
    assert kept.name == "the old fort"
    assert [d.text for d in kept.details] == []


def test_filter_entity_does_not_mutate_input():
    e = Location(name="Blackspire Keep", name_sources=["secret.txt"],
                 aliases=[al("the old fort", "group.txt"), al("secret aka", "secret.txt")],
                 details=[det("public", "group.txt"), det("secret", "secret.txt")],
                 supporting_quotes=[q("public", "group.txt"), q("secret", "secret.txt")])
    _filter_entity(e, {"secret.txt"})   # this re-heads on the COPY, never the input
    # the reconciled original (which the FULL doc still needs) is untouched -- name,
    # name_sources, both aliases, all details and quotes
    assert e.name == "Blackspire Keep" and e.name_sources == ["secret.txt"]
    assert [a.text for a in e.aliases] == ["the old fort", "secret aka"]
    assert [d.text for d in e.details] == ["public", "secret"]
    assert [qq.text for qq in e.supporting_quotes] == ["public", "secret"]


# --- filter_entities: the list wrapper --------------------------------------

def test_filter_entities_drops_wholly_secret_and_maps_the_rest():
    pub = Location(name="Publicton", name_sources=["group.txt"],
                   details=[det("f", "group.txt")], supporting_quotes=[])
    sec = Location(name="Secretville", name_sources=["secret.txt"],
                   details=[det("f", "secret.txt")], supporting_quotes=[])
    out = filter_entities([pub, sec], {"secret.txt"})
    assert [e.name for e in out] == ["Publicton"]   # the secret-only entity is gone


def test_filter_entities_empty_in_empty_out():
    assert filter_entities([], {"secret.txt"}) == []
