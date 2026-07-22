"""Phase 4.4: pure-Python markdown rendering (no LLM, no I/O).

This module turns reconciled + woven lore objects into the wiki markdown. It's a
sibling of renderer/footnotes.py and renderer/crosslink.py in the cheap,
deterministic outer layer -- every expensive LLM step is already done upstream, so
rendering is mechanical and repeatable.

It holds the six section renderers -- `render_history` (the timeline, Brief 2)
plus the five entity renderers (`render_locations` / `render_characters` /
`render_organizations` / `render_items` / `render_people`, Brief 3) -- and the
top-level `render_wiki` that wires all six sections plus the footnotes block into
one document.
"""

import re
from typing import Optional

from renderer.crosslink import CrosslinkMap, add_crosslinks, build_crosslink_map
from renderer.footnotes import FootnoteRegistry
from models.lore import Location, Character, Organization, Item, PeopleAndCultures


def _flatten(text: str) -> str:
    """Collapse internal whitespace to single spaces so a value renders as ONE
    markdown list item. A merged HistoryEvent's description is ``"\\n\\n".join(...)``
    of the pieces (the reconciler keeps both), and a line-wrapped ``date_text`` can
    carry a stray newline ("151 to\\n200"); dropped straight into a ``- `` bullet,
    that blank line ENDS the list item and spills the rest as an un-bulleted
    paragraph. Flattening first keeps the bullet intact -- the same reason the entity
    renderer joins its details into one paragraph."""
    return re.sub(r"\s+", " ", text).strip()


# Function words kept lowercase inside a title-cased name (never the first token). Matches
# the campaign's real entities ("Houses of Maltaav", "The 12 Houses of Maltraav").
_TITLE_STOPWORDS = frozenset(
    "a an and at by for from in of on or the to vs with".split()
)


def _cap_token(tok: str) -> str:
    """Capitalize the first letter of each HYPHEN-separated part, lowercasing the rest --
    so a hyphen starts a new word ("half-elf" -> "Half-Elf") but an apostrophe does NOT
    ("crown's" -> "Crown's", "MAL'TAAV" -> "Mal'taav", never "Crown'S"/"Mal'Taav")."""
    return "-".join(p[:1].upper() + p[1:].lower() for p in tok.split("-"))


def _is_screaming(tok: str) -> bool:
    letters = [c for c in tok if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _normalize_display_name(name: str) -> str:
    """Render an entity/event name in one canonical Title Case regardless of how it was typed
    in chat, WITHOUT disturbing names that are already well-cased.

    Two paths:
      * A SINGLE-CASE name (every letter upper, or every letter lower) that has an alphabetic
        word >= 4 chars is fully title-cased: each token capitalized (apostrophe/hyphen-safe
        via _cap_token), the first token always capitalized, and interior stop-words lowered.
        This fixes both "CROWN'S NEST" and "crown's nest" -> "Crown's Nest", and
        "the deathsworn" -> "The Deathsworn". The >=4 guard preserves short all-caps
        acronyms/initials ("CJ", "DM", "FBI").
      * Otherwise (MIXED case, or no long word) only a screaming ALL-CAPS token >= 4 is
        title-cased ("Lake MUNDI" -> "Lake Mundi"); an already-good mixed name ("Lake Mundi",
        "McDonald's") and short acronyms are left UNCHANGED.

    Purely cosmetic: the reconciler already merged case-variants (_name_key lowercases) and
    slug anchors lowercase too, so this creates no new pages and shifts no anchors."""
    letters = [c for c in name if c.isalpha()]
    single_case = bool(letters) and (
        all(c.isupper() for c in letters) or all(c.islower() for c in letters))
    has_long_word = any(len(w) >= 4 for w in re.findall(r"[A-Za-z]+", name))

    if single_case and has_long_word:
        out = []
        first = True
        for tok in name.split(" "):
            if not tok:                      # keep any double-space structure intact
                out.append(tok)
                continue
            capped = _cap_token(tok)
            if not first and capped.lower() in _TITLE_STOPWORDS:
                capped = capped.lower()
            out.append(capped)
            first = False
        return " ".join(out)

    # Mixed case (or a short all-caps acronym): only tame a screaming ALL-CAPS word.
    tokens = name.split(" ")
    if not any(_is_screaming(t) and sum(c.isalpha() for c in t) >= 4 for t in tokens):
        return name
    return " ".join(_cap_token(t) if _is_screaming(t) else t for t in tokens)


_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _alpha_key(name: str) -> str:
    """Alphabetical sort key that ignores a leading article, so "The Deathsworn" files
    under D (not T) while the article stays in the displayed name. Case-insensitive."""
    return _LEADING_ARTICLE_RE.sub("", name).strip().lower()


def _is_renderable(entity) -> bool:
    """True if an entity has anything to show: a polished prose body, at least one
    detail, or at least one supporting quote (a footnote). A name-only entity with
    none of these would render as a bare heading, so render_wiki drops it. A quote
    alone still counts -- a named-but-factless entity that carries its name as a
    supporting quote renders as a clean heading + footnote (the existing behaviour)."""
    return bool(getattr(entity, "prose", None)) or bool(entity.details) or bool(entity.supporting_quotes)


def _smart_join_details(details) -> str:
    """Join an entity's `details` into one prose paragraph, giving each fact a
    terminal period so they don't run together.

    The extractors write each `Detail.text` as a short fragment WITHOUT a trailing
    period (e.g. "A massive central lake"), so a bare space join produced run-ons
    ("A massive central lake Home to three rings"). We add a "." to any fact that
    doesn't already end in sentence punctuation, then space-join -- so a fact that
    already ends in ".", "!" or "?" is left alone (no doubled period). A "." is not
    a cross-link word-boundary char, so this is applied BEFORE add_crosslinks with no
    effect on matching. This is the fallback body source; when the prose agent has
    run, `entity.prose` is used instead (see _render_entity)."""
    parts = []
    for d in details:
        t = d.text.strip()
        if not t:
            continue
        if t[-1] not in ".!?":
            t += "."
        parts.append(t)
    return " ".join(parts)


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
    date_tag = f" — {_flatten(event.date_text)}" if event.date_text else ""

    # 3. The description is the ONLY prose we cross-link -- we link it alone and
    #    build the bullet around it, never feeding the bold name or the date to the
    #    linker. `this_anchor` is this event's own anchor (or None), so a mention of
    #    the event's own name inside its description won't link back to itself.
    #    Source: the prose agent's polished description if it ran (`event.prose`), else
    #    the raw `event.description`. Flatten first: a merged event's raw description is
    #    "\n\n"-joined, and that blank line would otherwise break the bullet (spill a
    #    second, un-bulleted paragraph); a polished body is one line but flattening is
    #    harmless there.
    raw_desc = event.prose if event.prose else event.description
    linked_desc = add_crosslinks(_flatten(raw_desc), cmap, anchor)

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
        rows = sorted(could_not_place, key=lambda r: (_alpha_key(r[0].name), r[2]))
        blocks.append(_bullets(rows, cmap, registry))

    # Blank line between every block so markdown doesn't fuse a heading into the
    # list below it (or two blocks into one paragraph) when rendered.
    return "\n\n".join(blocks)


def _render_entity(entity, anchor: str, cmap: CrosslinkMap,
                   registry: FootnoteRegistry) -> str:
    """Render ONE entity: a ### heading (with its <a id> stamped inline) plus a
    prose body and end-of-entity footnote markers.

    Entities always have an anchor (unlike events, they're never gated out), so we
    always stamp the <a id>. The body is the entity's `details` joined into one
    prose paragraph -- we don't try to footnote individual facts, because the
    stored `details` and `supporting_quotes` lists have no position-by-position
    mapping (the quote list is deduplicated at extraction), so the quotes back the
    entity as a whole and their markers land at the end, same as an event.
    """
    # The <a id> sits INLINE at the START of the heading text ("### <a id>Name"),
    # not on a line of its own above it -- a raw-HTML line above the heading would
    # stop the "### " from being read as a heading. (The display renderer must pass
    # raw HTML through; already logged against the renderer-choice decision.)
    heading = f'### <a id="{anchor}"></a>{entity.name}'

    # Body source: the prose agent's polished paragraph if it ran (`entity.prose`),
    # else the details joined into one paragraph with per-fact terminal periods
    # (`_smart_join_details`). Either way the WHOLE paragraph is ONE cross-link
    # block, with the entity's own anchor as `this_anchor`, so a mention of the
    # entity's own name inside its body doesn't link back to itself.
    raw_body = entity.prose if entity.prose else _smart_join_details(entity.details)
    body = add_crosslinks(raw_body, cmap, anchor)

    # Footnote markers for the entity's quote pool, appended at the end of the body.
    refs = "".join(f"[^{registry.add(q)}]" for q in entity.supporting_quotes)

    # A named-but-factless entity (empty `details`) is a supported extractor output
    # -- a location/etc. named with no surviving facts is emitted anyway. Rendering
    # the empty body paragraph regardless would leave a lone floating `[^N]` marker
    # (or, with no quotes, a heading trailed by blank lines). So when there's no
    # body, drop the paragraph and hang any footnote markers on the heading itself:
    # the entity still renders as a clean heading and keeps its quote provenance.
    if not body:
        return f"{heading}{refs}"
    return f"{heading}\n\n{body}{refs}"


def _render_entity_section(label: str, pairs, cmap: CrosslinkMap,
                           registry: FootnoteRegistry) -> str:
    """Render one whole entity SECTION: a "## {label}" header plus every entity
    under it, sorted alphabetically by name. Returns "" when there are no entities
    of this type, so the caller drops the section (never a bare empty header).

    `pairs` is a list of (entity, anchor) already paired by the caller -- pairing
    up front means an anchor can't desync from its entity when we sort.
    """
    if not pairs:
        return ""
    # Alphabetical by name; the anchor (always present, always unique) is the
    # tiebreak, so two same-named entries render in a stable order every run.
    rows = sorted(pairs, key=lambda p: (_alpha_key(p[0].name), p[1]))
    bodies = "\n\n".join(_render_entity(ent, anchor, cmap, registry)
                         for ent, anchor in rows)
    return f"## {label}\n\n{bodies}"


def render_locations(pairs, cmap: CrosslinkMap, registry: FootnoteRegistry) -> str:
    return _render_entity_section("Locations", pairs, cmap, registry)


def render_characters(pairs, cmap: CrosslinkMap, registry: FootnoteRegistry) -> str:
    # NOTE: Characters render uniformly with the other entities -- `is_pc` and
    # `player_name` are intentionally not surfaced yet (stored-but-unrendered, like
    # HistoryEvent.scope). A future phase can label PCs with their player.
    return _render_entity_section("Characters", pairs, cmap, registry)


def render_organizations(pairs, cmap: CrosslinkMap, registry: FootnoteRegistry) -> str:
    return _render_entity_section("Organizations", pairs, cmap, registry)


def render_items(pairs, cmap: CrosslinkMap, registry: FootnoteRegistry) -> str:
    return _render_entity_section("Items", pairs, cmap, registry)


def render_people(pairs, cmap: CrosslinkMap, registry: FootnoteRegistry) -> str:
    # The pretty display label lives here (the class is PeopleAndCultures, which a
    # Python identifier can't spell with a space or the bare word "and").
    return _render_entity_section("People & Cultures", pairs, cmap, registry)


def render_wiki(locations, characters, events, organizations, items, people,
                common_words=None) -> str:
    """Top-level: assemble the whole wiki markdown from the six reconciled lore
    lists. Builds the cross-link map ONCE over every entity + event, then renders
    each section in a FIXED order -- which is also the order footnote numbers get
    assigned, so they climb in order down the page.

    Section / render / footnote-numbering order:
        Locations, History, People & Cultures, Organizations, Characters, Items.

    Returns "" for a completely empty world (every section empty, no footnotes).

    `common_words` is the loaded crosslink_words.json config (or None). This
    function is the seam the eventual whole-pipeline entry point (main.py) plugs
    into -- it takes the six typed lists explicitly.
    """
    # Canonicalize name casing for display BEFORE anything reads `.name`/aliases. We rebuild
    # each entity + event with a Title-Cased name AND Title-Cased alias texts, so the heading,
    # the alias display, and the (case-sensitive) cross-link surface pool are all built from the
    # SAME canonical strings and stay in sync -- normalizing only the heading would desync it
    # from the pool and stop the name linking. Anchors are unaffected (slugify lowercases);
    # model_copy leaves the caller's originals untouched.
    def _cased(seq):
        out = []
        for e in seq:
            aliases = [a.model_copy(update={"text": _normalize_display_name(a.text)})
                       for a in e.aliases]
            out.append(e.model_copy(update={
                "name": _normalize_display_name(e.name), "aliases": aliases}))
        return out
    locations = _cased(locations)
    characters = _cased(characters)
    events = _cased(events)
    organizations = _cased(organizations)
    items = _cased(items)
    people = _cased(people)

    # Drop truly-empty entities BEFORE building the cross-link map. An entity with no
    # renderable body (no prose, no details) AND no supporting quotes has nothing to
    # show -- it would render as a bare heading (or a lone footnote-less stub). This is
    # where a business wrongly grabbed as a name-only Location, or a name-only entity
    # whose only "fact" failed the verbatim-quote check, gets pruned. Filtering HERE
    # (not inside _render_entity) is load-bearing: a dropped entity must be neither an
    # anchor target NOR a cross-link source, so a mention of its name elsewhere renders
    # as plain text instead of a link to a page that doesn't exist. Events are never
    # filtered -- a HistoryEvent always has a (required) description.
    locations = [e for e in locations if _is_renderable(e)]
    characters = [e for e in characters if _is_renderable(e)]
    organizations = [e for e in organizations if _is_renderable(e)]
    items = [e for e in items if _is_renderable(e)]
    people = [e for e in people if _is_renderable(e)]

    # Concatenate the five entity types into ONE list for the map (build_crosslink_map
    # takes a single entity list). The order here only decides which entity keeps
    # the clean slug on a cross-type name clash -- cosmetic, since both still get an
    # anchor and the ambiguous surface drops out either way -- so we just match the
    # display order for least surprise.
    all_entities = (list(locations) + list(people) + list(organizations)
                    + list(characters) + list(items))

    cmap = build_crosslink_map(all_entities, events=events, common_words=common_words)

    # Pair every entity with its anchor, THEN split back apart by type. We split on
    # isinstance (an object's type can't drift) rather than by counting how many of
    # each we concatenated -- no fragile length arithmetic, and the anchor stays
    # glued to its entity through the split.
    paired = list(zip(all_entities, cmap.entity_anchors))
    loc_pairs = [pa for pa in paired if isinstance(pa[0], Location)]
    people_pairs = [pa for pa in paired if isinstance(pa[0], PeopleAndCultures)]
    org_pairs = [pa for pa in paired if isinstance(pa[0], Organization)]
    char_pairs = [pa for pa in paired if isinstance(pa[0], Character)]
    item_pairs = [pa for pa in paired if isinstance(pa[0], Item)]

    # ONE registry shared across every section, so footnote numbers are unique and
    # assigned in render order. History takes (events, cmap, registry), not pairs --
    # it reads event_anchors off the map itself, so it needs the events list.
    registry = FootnoteRegistry()
    sections = [
        render_locations(loc_pairs, cmap, registry),
        render_history(events, cmap, registry),
        render_people(people_pairs, cmap, registry),
        render_organizations(org_pairs, cmap, registry),
        render_characters(char_pairs, cmap, registry),
        render_items(item_pairs, cmap, registry),
    ]

    # The footnote-definitions block is rendered LAST (after every section has
    # registered its quotes) and appended at the very end.
    footnotes = registry.render_definitions()

    # Drop empty pieces so we never emit a bare header or stray blank, then join
    # with one blank line between blocks.
    blocks = [b for b in sections + [footnotes] if b]
    return "\n\n".join(blocks)
