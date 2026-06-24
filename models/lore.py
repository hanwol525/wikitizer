from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class Scope(str, Enum):
    """Allowed values for HistoryEvent.scope. The ``(str, Enum)`` mix-in (3.9's
    way to get what 3.11's StrEnum gives) makes each member behave like its text
    -- ``Scope.WORLD == "world"`` is True -- so it serializes to JSON cleanly and
    any renderer doing ``scope == "world"`` keeps working. Being an Enum means
    pydantic REJECTS any off-menu value (a typo like "wrold" no longer stores)."""
    WORLD = "world"
    REGIONAL = "regional"
    PERSONAL = "personal"


class Quote(BaseModel):
    text: str
    speaker: str
    source_file: str

class Location(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

class Character(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    is_pc: bool = False
    player_name: Optional[str] = None
    details: list[str] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

class HistoryEvent(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str
    scope: Scope
    # The extractor never sets this -- a later timeline pass (which sees every
    # event at once) fills it, and the renderer (4.4) reads it to decide the
    # "Could Not Place" pile. The field stays as that mailbox even though
    # extraction always leaves it None.
    chronological_position: Optional[int] = None
    supporting_quotes: list[Quote] = Field(default_factory=list)

# The three typed categories below replace the retired `OtherDetail` catch-all.
# Each is shaped EXACTLY like Location (same four fields) on purpose: the Phase
# 4.1 reconciler dedups all six entity types with the same name+alias matching,
# and the renderer treats them uniformly. The category-specific discipline lives
# in each one's extractor prompt, not in the schema.

class Organization(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

class Item(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

# PascalCase identifier; the pretty "People & Cultures" display label lives in
# the renderer (a Python name can't hold a space or the bare word `and`).
class PeopleAndCultures(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)