"""The rubric data file: ``output/rubric.yaml`` -> a validated ``Rubric``.

The rubric IS the checklist, so a **missing file is a hard error** (not a skip) and a
parsed-but-wrong-shape file raises pydantic's ``ValidationError``. A fabricated
``example_rubric.yaml`` (committed next to this module) is the schema reference and the
offline-test fixture; the real ``output/rubric.yaml`` stays gitignored.

The YAML shape (a bare string coerces to ``{name: <str>}``; ``aliases`` covers other names
AND acceptable alternative phrasings; ``pc`` is only meaningful under ``characters``)::

    sections:
      - name: Locations
        aliases: [Places]
      - {name: People & Cultures}
    locations:
      - Aldenburg
      - {name: Dwarven Stronghold, aliases: [Taken Lands of Tiber]}
    characters:
      - {name: Aerin Wakestrider, pc: true}
    organizations: [ ... ]
    people_and_cultures: [ ... ]
    events:
      - {name: Imperial Expansion Begins, aliases: [Start of the Imperial Expansion]}

There is intentionally no ``items`` entity list -- the rubric only requires the *Items
section* exists (covered by ``sections``); add an ``items:`` list later if required Item
entries are ever defined.
"""

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RubricEntity(BaseModel):
    """One required section or entity. ``name`` is the canonical form; ``aliases`` are other
    acceptable names/spellings/phrasings the WOF may use; ``pc`` marks a player character
    (only read under ``characters``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    aliases: List[str] = Field(default_factory=list)
    pc: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce_bare_string(cls, data):
        # A bare YAML string (``- Aldenburg``) is shorthand for ``{name: Aldenburg}``. This
        # runs before field validation because a plain str can't be validated as the model
        # otherwise (there's no dict to pull ``name`` from).
        if isinstance(data, str):
            return {"name": data}
        return data


class Rubric(BaseModel):
    """The full campaign rubric. ``sections`` drives the section-presence check; the five
    entity categories drive per-entity presence + sourcing (see ``ENTITY_CATEGORIES``)."""

    model_config = ConfigDict(extra="forbid")

    sections: List[RubricEntity] = Field(default_factory=list)
    locations: List[RubricEntity] = Field(default_factory=list)
    characters: List[RubricEntity] = Field(default_factory=list)
    organizations: List[RubricEntity] = Field(default_factory=list)
    people_and_cultures: List[RubricEntity] = Field(default_factory=list)
    events: List[RubricEntity] = Field(default_factory=list)


def load_rubric(path) -> Rubric:
    """Read + validate the rubric YAML. Raises ``FileNotFoundError`` when the file is absent
    (it is the checklist), ``yaml.YAMLError`` on a syntax error, and pydantic's
    ``ValidationError`` on a wrong shape. The CLI turns all three into a clean error."""
    with Path(path).open(encoding="utf-8") as f:      # missing file -> FileNotFoundError
        data = yaml.safe_load(f)
    if data is None:                                   # an empty file -> an empty (valid) rubric
        data = {}
    return Rubric.model_validate(data)
