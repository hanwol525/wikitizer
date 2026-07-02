"""Phase 4.4: pure-Python markdown rendering (no LLM, no I/O).

This module turns reconciled + woven lore objects into the wiki markdown. It's a
sibling of renderer/footnotes.py and renderer/crosslink.py in the cheap,
deterministic outer layer -- every expensive LLM step is already done upstream, so
rendering is mechanical and repeatable.

This file currently holds only `render_history` (Phase 4.4 Brief 2); the five
entity render functions and the top-level assembly arrive in Brief 3.
"""

from typing import Optional

from renderer.crosslink import CrosslinkMap, add_crosslinks
from renderer.footnotes import FootnoteRegistry


# Both notes are heads-ups TO THE READER -- they never claim the tool reconciles
# anything, matching the "flag for human review, not agent reconciliation" call.
# They share the "**Important Note:**" label on purpose (they never co-occur -- the
# multi-system note needs dated systems, the could-not-place note needs none).
_MULTI_SYSTEM_NOTE = (
    "**Important Note:** This campaign uses more than one timekeeping system. Each "
    "is listed separately below, and events in different systems can't be reliably "
    "ordered against one another — so their relative timing may need a human "
    "eye to reconcile."
)

_COULD_NOT_PLACE_ONLY_NOTE = (
    "**Important Note:** None of these events have a stated date or any clue to "
    "their order, so they couldn't be placed in a timeline. They're listed "
    "alphabetically — not chronologically — and putting them in sequence "
    "would need a human eye."
)


def _event_bullet(event, anchor: Optional[str], cmap: CrosslinkMap,
                  registry: FootnoteRegistry) -> str:
    """Render ONE history event as a single markdown list item (the entry atom).

    Name-forward and uniform: the bold name always leads, the date is a trailing
    tag present only when the event has one, then the description, then any footnote
    markers. The shape is identical for dated and undated events -- only the
    "— {date}" clause appears or disappears -- which is what keeps a mixed
    dated/undated list reading cleanly (a bold year leading some bullets and a name
    leading others is the raggedness name-forward avoids).
    """
    # 1. Anchor, only if this event earned one. `anchor is None` means it was gated
    #    out upstream (a sentence-shaped name), so we stamp nothing -- the event
    #    still renders, it just isn't a jump target. `is not None`, not truthiness.
    #    The <a id> sits INLINE at the very front of the bullet: a raw-HTML line of
    #    its own between list items can break list continuity in some markdown
    #    parsers, so keeping it inline preserves the list.
    anchor_html = f'<a id="{anchor}"></a>' if anchor is not None else ""

    # 2. Date tag, only when the event states a date. `date_text` is None (or "")
    #    for every undated and could-not-place event, so the clause simply vanishes.
    #    Truthiness is the RIGHT check here (unlike position): an empty date string
    #    should hide the tag exactly like None does.
    date_tag = f" — {event.date_text}" if event.date_text else ""

    # 3. The description is the ONLY prose we cross-link -- we link it alone and
    #    build the bullet around it, never feeding the bold name or the date to the
    #    linker. `this_anchor` is this event's own anchor (or None), so a mention of
    #    the event's own name inside its description won't link back to itself.
    linked_desc = add_crosslinks(event.description, cmap, anchor)

    # 4. Footnote markers for the event's supporting quotes, in order, appended at
    #    the END of the description -- a history event is one description block, so
    #    there's no per-fact place to hang them (that per-fact split is an entity
    #    thing). registry.add mints a stable, deduplicated number per distinct quote.
    refs = "".join(f"[^{registry.add(q)}]" for q in event.supporting_quotes)

    return f"- {anchor_html}**{event.name}**{date_tag}. {linked_desc}{refs}"


def _bullets(rows, cmap: CrosslinkMap, registry: FootnoteRegistry) -> str:
    """Render already-sorted (event, anchor, index) rows to a tight bullet list --
    one line per event. Callers sort `rows` first; this just renders them in order.
    """
    return "\n".join(_event_bullet(ev, anchor, cmap, registry) for ev, anchor, _ in rows)


def render_history(events, cmap: CrosslinkMap, registry: FootnoteRegistry) -> str:
    """Render the whole History (or Timeline) section from the woven events.

    `events` MUST be the same list, in the same order, that was passed to
    build_crosslink_map -- `cmap.event_anchors` is parallel to it by index. We pair
    each event with its anchor FIRST (before any sort) so re-ordering events for
    display can never desync them from their anchors.

    Returns "" when there are no events at all, so the caller can drop the whole
    section without peeking inside (same emptiness contract the footnote registry
    uses).
    """
    if not events:
        return ""

    # Pair-then-bucket. Carry the ORIGINAL index in each row so any sort has a
    # stable, unique final tiebreak, and so the anchor can never drift off its event.
    paired = [(ev, cmap.event_anchors[i], i) for i, ev in enumerate(events)]

    dated = {}              # calendar_system -> [rows]
    undated = []            # calendar_system None, position not None
    could_not_place = []    # calendar_system None, position None
    for row in paired:
        ev = row[0]
        if ev.calendar_system is not None:
            dated.setdefault(ev.calendar_system, []).append(row)
        elif ev.chronological_position is not None:
            undated.append(row)
        else:
            could_not_place.append(row)

    n_systems = len(dated)

    # Header: default "## History"; "## Timeline" ONLY when there are no dated
    # systems but there IS an ordered (undated) sequence to justify the word.
    header = "## Timeline" if (n_systems == 0 and undated) else "## History"
    blocks = [header]

    # Notes (both labelled "**Important Note:**"). Multi-system note when 2+ systems;
    # could-not-place-only note when there's nothing but could-not-place events.
    if n_systems >= 2:
        blocks.append(_MULTI_SYSTEM_NOTE)
    elif n_systems == 0 and not undated and could_not_place:
        blocks.append(_COULD_NOT_PLACE_ONLY_NOTE)

    # Dated sections, most-documented first (count desc, then system name for a
    # stable tiebreak). Per-system subhead only when there are 2+ systems.
    for system in sorted(dated, key=lambda s: (-len(dated[s]), s)):
        if n_systems >= 2:
            blocks.append(f"### {system}")
        rows = sorted(dated[system], key=lambda r: r[0].chronological_position)
        blocks.append(_bullets(rows, cmap, registry))

    # Undated-but-ordered events. Subhead only when there's dated content above them
    # (in the "## Timeline" case they ARE the body, so no subhead).
    if undated:
        if n_systems >= 1:
            blocks.append("### Undated Events")
        rows = sorted(undated, key=lambda r: r[0].chronological_position)
        blocks.append(_bullets(rows, cmap, registry))

    # Could Not Place: no positions (that's what put them here), so sort
    # alphabetically by name with the original index as the tiebreak. Subhead
    # appears unless these are the whole body (that case used the note instead).
    if could_not_place:
        only_content = (n_systems == 0 and not undated)
        if not only_content:
            blocks.append("### Could Not Place")
        rows = sorted(could_not_place, key=lambda r: (r[0].name, r[2]))
        blocks.append(_bullets(rows, cmap, registry))

    # Blank line between every block so markdown doesn't fuse a heading into the
    # list below it (or two blocks into one paragraph) when rendered.
    return "\n\n".join(blocks)
