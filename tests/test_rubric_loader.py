"""Tests for evals/rubric/loader.py -- the rubric YAML models + loader. Fully offline."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import evals.rubric
from evals.rubric.loader import Rubric, RubricEntity, load_rubric


def test_bare_string_coerces_to_name():
    r = Rubric.model_validate({"locations": ["Aldenburg", {"name": "Foo", "aliases": ["Bar"]}]})
    assert r.locations[0].name == "Aldenburg" and r.locations[0].aliases == []
    assert r.locations[1].name == "Foo" and r.locations[1].aliases == ["Bar"]


def test_pc_flag_parsed_and_defaults_false():
    r = Rubric.model_validate({"characters": [{"name": "Aerin", "pc": True}, "NPC Bob"]})
    assert r.characters[0].pc is True
    assert r.characters[1].name == "NPC Bob" and r.characters[1].pc is False


def test_extra_forbidden_on_entity_and_rubric():
    with pytest.raises(ValidationError):
        RubricEntity(name="x", bogus=1)
    with pytest.raises(ValidationError):
        Rubric.model_validate({"unknown_category": ["x"]})


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rubric(tmp_path / "does_not_exist.yaml")


def test_malformed_shape_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    # a well-formed YAML that is the wrong SHAPE (an entity with an unknown key) -> ValidationError
    p.write_text("characters:\n  - {name: A, bogus: 1}\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_rubric(p)


def test_empty_file_is_an_empty_rubric(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    r = load_rubric(p)
    assert r.sections == [] and r.locations == [] and r.characters == []


def test_example_rubric_loads():
    example = Path(evals.rubric.__file__).with_name("example_rubric.yaml")
    r = load_rubric(example)
    assert any(s.name == "Locations" for s in r.sections)
    assert any(c.pc for c in r.characters)             # at least one declared PC
    assert any(l.aliases for l in r.locations)         # aliases preserved
