"""Phase 4.1a: the *decision* objects the reconciler LLM hands back.

The reconciler never builds merged entries itself. It only emits a DECISION --
"entries 3 and 7 are the same thing, call the result 'Lake Mundi'" -- and a
pure-Python combiner (in agents/reconciler.py) does the actual fact-merging from
the ORIGINAL entry objects. These models are the contract between those two
halves: the only things the LLM is allowed to say.

One model set covers all six entity types, because a decision is just indices +
names; it never names a field that's specific to Characters or History.
"""

from typing import Optional

from pydantic import BaseModel, Field


class DetailConflict(BaseModel):
    """One pair of details (from two entries being merged) that the LLM judged to
    genuinely DISAGREE -- e.g. "ruled by X" vs "ruled by Y". Carries the text and
    source of each side so the human reviewer can act on it.

    Purely informational: Python keeps BOTH details in the merged entry and just
    logs this; it never drops either side. We carry the TEXT here (not a "detail
    #2" coordinate) on purpose -- our stored detail/quote lists don't keep a clean
    index-to-index mapping, so a coordinate would be fragile. A reworded note here
    is harmless: the real detail is still carried verbatim by Python in the merged
    entry; this object is only a pointer for the review queue.
    """
    detail_a: str
    source_a: str
    detail_b: str
    source_b: str
    note: str = ""  # the LLM's plain-English "why these disagree", for the reviewer


class MergeGroup(BaseModel):
    """One set of entries the LLM is CONFIDENT are the same entity.

    `members` are INDICES into the input list (so [3, 7] means "the 4th and 8th
    entries are one thing", counting from 0). `canonical` is the winning name for
    the merged entry's heading; Python rejects it unless it's a name/alias that
    already appears among those members (so even the heading can't be invented).
    `conflicts` is usually empty -- it only fills when two merged details disagree.
    """
    members: list[int]
    canonical: str
    conflicts: list[DetailConflict] = Field(default_factory=list)


class PossibleDuplicate(BaseModel):
    """A pair the LLM found SUSPICIOUS but wasn't confident enough to merge, so it
    deliberately left them separate. These never get merged; Python just writes a
    findable log line so you can confirm/reject them by hand later."""
    members: list[int]
    note: str = ""


class ReconcileDecision(BaseModel):
    """The whole thing the LLM returns for one entity-type list.

    Anything NOT named in `merges` is implicitly a singleton -- the LLM never
    echoes unique entries back, which keeps the common case cheap. A list with no
    duplicates returns both lists empty, and Python passes the input straight
    through untouched.
    """
    merges: list[MergeGroup] = Field(default_factory=list)
    possible_duplicates: list[PossibleDuplicate] = Field(default_factory=list)


# --- Phase 4.1b: timeline-ordering decision contracts -----------------------
# Two LLM calls feed the timeline engine. Call 1 (date extraction) returns a
# DateDecision; Python sorts it into a spine + gaps; Call 2 (placement) returns a
# PlacementDecision dropping the undated events into those gaps. The LLM NEVER
# emits a position integer in either -- Python owns every chronological_position.

class DatedEvent(BaseModel):
    """One event the LLM judged to carry a usable date (Call 1) -- either an explicit
    stated date, or a present-relative offset ("200 years ago") resolved against a
    reference year."""
    index: int          # which history event this is, by position in the list we sent
    system: str         # the calendar system the date is in: "AR years", "Elder Scrolls eras", ...
    parts: list[int]    # the date as a sortable tuple, biggest unit FIRST:
                        #   [1347] for "1347", or [4, 200] for "4th Era 200".
                        # Python tuple-sorts these, so era-then-year orders for free.
    anchor_relative: bool = False
                        # True when `parts` was COMPUTED from a present-relative offset
                        # against the reference year (e.g. date_text "200 years ago",
                        # reference 1424 -> parts [1224]). It tells _sanity_guard_parts
                        # to SKIP its digit-match check, because a resolved offset
                        # legitimately shares no digit with its source phrase (1224 vs
                        # 200). Defaults False, so an explicit stated date is unaffected.


class DateDecision(BaseModel):
    """Call 1's whole reply. Any event NOT listed in `dated` has no usable stated
    date -> it's a candidate for relative placement in Call 2 (or Could Not Place).
    Implicit, like 4.1a's unlisted-means-singleton."""
    dated: list[DatedEvent] = Field(default_factory=list)
    # The reference ("present-day") year the LLM used to resolve any relative offsets,
    # and the calendar system it belongs to -- reported back for logging/traceability
    # only (Python does not re-derive positions from them). None when no reference year
    # was available or needed.
    reference_year: Optional[int] = None
    reference_system: Optional[str] = None


class GapPlacement(BaseModel):
    """One gap and the relative events that fall into it, in order (Call 2)."""
    gap: str            # a gap ID Python handed the LLM, e.g. "AR years#1" or "(undated)#0"
    events: list[int]   # the relative events in this gap, EARLIEST FIRST (this list IS
                        # the within-gap ordering; Python lays them down in this order)


class PlacementDecision(BaseModel):
    """Call 2's whole reply. Any relative event not placed in any gap -> Could Not
    Place (position None). Implicit + safe, same as 4.1a's unlisted convention."""
    placements: list[GapPlacement] = Field(default_factory=list)
