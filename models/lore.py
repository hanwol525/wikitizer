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


class Detail(BaseModel):
    """One extracted fact about an entity, plus where it came from.

    The fact-grain twin of ``Quote``. ``text`` is the fact itself -- exactly the
    string that used to live bare in ``details: list[str]`` -- and ``source_files``
    records which chat-log file(s) that fact was stated in.

    ``source_files`` is a LIST on purpose, even though a freshly-extracted fact
    always has exactly one source. When the reconciler merges two mentions of the
    same entity and finds the SAME fact stated in two different files (say a public
    log and a confidential one), it collapses them into one ``Detail`` -- and that
    survivor has to remember it came from BOTH, so a future exclusion pass drops it
    only when EVERY one of its sources is excluded. A single string couldn't say
    "this fact is public AND secret".

    Nothing reads ``source_files`` yet -- it's dormant plumbing for ``--exclude-
    sources``. Rendering ignores it and uses only ``text``, which is why adding it
    changes nothing you can see in the wiki.
    """
    text: str
    source_files: list[str] = Field(default_factory=list)


class Location(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    details: list[Detail] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

class Character(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    is_pc: bool = False
    player_name: Optional[str] = None
    details: list[Detail] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

class HistoryEvent(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str
    scope: Scope
    # The date EXACTLY as the campaign stated it ("342 AR", "the Third Age",
    # "year 1247"), or None when no date is stated. The extractor captures this
    # verbatim -- it does NOT normalize, parse, or convert it to a number. The
    # 4.1b timeline pass is what interprets it into ordering. (Most campaigns
    # state no dates, so this is None on most events -- that's expected.)
    date_text: Optional[str] = None
    # Which calendar system this event's date belongs to ("AR years", "Elder
    # Scrolls eras", "Hebrew calendar", ...), or None. Like chronological_position,
    # the EXTRACTOR never sets this -- the 4.1b timeline pass derives it (it needs
    # the whole-timeline view to assign a CONSISTENT label across events, and to
    # recognize that two different notations -- e.g. "1347" and "Third Era 347" --
    # are the same system), and the renderer (4.4) groups events into one timeline
    # per system. None until 4.1b fills it.
    calendar_system: Optional[str] = None
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
    details: list[Detail] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

class Item(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    details: list[Detail] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)

# PascalCase identifier; the pretty "People & Cultures" display label lives in
# the renderer (a Python name can't hold a space or the bare word `and`).
class PeopleAndCultures(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    details: list[Detail] = Field(default_factory=list)
    supporting_quotes: list[Quote] = Field(default_factory=list)