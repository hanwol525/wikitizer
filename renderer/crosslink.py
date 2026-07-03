"""Phase 4.3: the cross-link pass.

Pure Python, no LLM, no network -- the sibling of renderer/footnotes.py. It lives
in the cheap, deterministic OUTER layer of the pipeline: every expensive LLM step
(extraction, reconciliation, timeline ordering) is already done and cached
upstream, so all this module does is mechanical, repeatable text-munging over
already-rendered markdown.

What it does: turn a bare entity name sitting in body prose ("Lake Mundi") into a
clickable internal link ("[Lake Mundi](#lake-mundi)") that jumps to that entity's
page. We OWN our anchors -- the renderer (Phase 4.4) stamps an explicit
``<a id="...">`` on each entity heading -- so `slugify` here is the single source
of truth for anchor ids and nothing depends on any markdown library's auto-anchor
scheme.

Design in one breath: build ONE resolved structure (`CrosslinkMap`) over the full
noun-entity list, exactly once, where every messy case (slug collisions, ambiguous
aliases, sentence-shaped fallback names, common-word over-matching) is decided at
BUILD time and never re-litigated during the walk. Then the walk over each block
is a single left-to-right pass of one compiled, longest-first regex via
``re.finditer`` -- which makes re-linking your own output structurally impossible,
because a span the walk has already consumed can never be re-scanned.

Anti-hallucination ethos (the whole reason this is conservative): a cross-link is a
CLAIM that "this text refers to that entity." A wrong link is a quiet little lie in
a tool whose pitch is "trust the source." So the bias is always to MISS a legit
link before ever manufacturing a wrong one. There is NO fuzzy / edit-distance /
typo-tolerant matching anywhere -- matching is exact, modulo only NFC + case +
self-defined word boundaries + the article rule. The prose we scan is the model's
clean output; typos survive only inside verbatim quotes, which are quarantined in
the footnotes block and never routed here.

Builds and tested in ISOLATION: Phase 4.4's renderer does not exist yet. This
module is fully self-contained; 4.4 will import `build_crosslink_map` /
`add_crosslinks` and feed them real rendered markdown later.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Same convention as agents/reconciler.py: every loud, human-actionable flag
# carries this prefix so a later review pass can funnel them out of the routine
# log noise. Quiet, routine auto-resolutions use logging.debug and stay unflagged.
REVIEW_PREFIX = "[REVIEW]"

# Default location of the (optional) common-word config. Mirrors how
# config/speaker_map.json sits next to its loader -- the file is DATA, not source,
# and both lists default empty so a fresh clone runs clean with no config at all.
DEFAULT_WORDS_PATH = "config/crosslink_words.json"

# A surface form with more than this many words is treated as sentence-shaped and
# held out of the source pool (it keeps its anchor as a target). A real entity name
# is short; a long one is almost always an extractor fallback name (a truncated
# detail sentence). See `_looks_sentence_ish`.
_MAX_SOURCE_WORDS = 6

# Internal sentence punctuation that also marks a surface form as sentence-shaped.
# Comma is deliberately excluded: it shows up in some legitimate compound names and
# the word-count gate already catches the long fallback sentences that matter.
_SENTENCE_PUNCT = ".!?;:"

# The character set that counts as "inside a name" for self-defined word
# boundaries. `\w` is unicode-aware (so accented letters in fantasy names count),
# plus BOTH apostrophes (straight ' and curly U+2019) and the hyphen -- because in
# a name like "Mal'taav" or "Half-Elf" the apostrophe/hyphen is part of the word,
# NOT a boundary. Python's bare ``\b`` treats apostrophe/hyphen as boundaries and
# would happily fire "Mal" inside "Mal'taav", so we define our own boundaries with
# lookarounds against this class instead of trusting ``\b``.
_NAME_CHARS = r"\w'’\-"


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O, no state) -- each unit-tests on its own.
# --------------------------------------------------------------------------- #

def _nfc(text: str) -> str:
    """Canonical NFC form. Two pixel-identical strings can differ as raw BYTES (a
    precomposed "é" vs an "e" + combining accent); NFC folds both to one form so a
    surface form and the prose it should match line up instead of silently
    missing. We NFC at exactly two seams -- the surface forms at build time and the
    prose at match time -- so both sides of every comparison are in the same form."""
    return unicodedata.normalize("NFC", text)


def slugify(name: str) -> str:
    """Turn an entity name into an anchor id. The single source of truth for anchor
    ids -- the renderer stamps ``<a id="{slug}">`` and the cross-linker emits
    ``(#{slug})``, both reading the SAME table, so they physically cannot drift.

    Pipeline, in order:
      1. ASCII-fold via NFKD: decompose, drop combining marks, keep only ASCII.
         ("Théoden" -> "theoden", "Canción" -> "cancion"). NFKD inherently
         normalizes byte-variants (both spellings of "é" decompose to "e" + mark),
         so a separate NFC step is NOT needed here -- NFC's real job is at the
         MATCHING layer, not the anchor layer.
      2. Lowercase.
      3. Spaces (any whitespace run) -> hyphens.
      4. Keep internal hyphens (the slug separator); strip apostrophes and anything
         else outside [a-z0-9-]. ("Mal'taav" -> "maltaav", "Half-Elf" -> "half-elf").
      5. Collapse runs of hyphens to one; trim leading/trailing hyphens.

    Returns "" for an all-punctuation / no-ASCII-letter name. The EMPTY-SLUG
    FALLBACK (a deterministic ``entity-<N>`` id) lives in `build_crosslink_map`, not
    here, because only the map knows the entity's stable index -- this function is a
    pure string -> string transform with no notion of "which entity".
    """
    # 1. ASCII-fold: decompose so accents become separate combining marks we can
    #    drop, then keep only the plain-ASCII skeleton.
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(
        ch for ch in decomposed
        if not unicodedata.combining(ch) and ord(ch) < 128
    )
    # 2 + 3. Lowercase, then every whitespace run becomes a single hyphen (the
    #        multi-space collapse falls out of step 5 anyway, but doing it here
    #        keeps the intent obvious).
    lowered = ascii_only.lower()
    hyphenated = re.sub(r"\s+", "-", lowered)
    # 4. Drop everything that isn't a slug char. Internal hyphens survive; an
    #    apostrophe "Mal'taav" just closes up to "maltaav" (NOT "mal-taav") so a
    #    name and its apostrophe-free spelling slug the same.
    kept = re.sub(r"[^a-z0-9-]", "", hyphenated)
    # 5. Collapse hyphen runs and trim the edges.
    return re.sub(r"-+", "-", kept).strip("-")


def _split_article(surface: str) -> str:
    """Strip a leading "the "/"The " from a surface form for MATCHING purposes.

    The article rule says "the" is NEVER part of a link -- it always renders as
    plain prose outside the brackets. So the surface forms we store and hunt for
    must exclude a leading article; the walk's regex re-attaches an optional "the"
    at match time and keeps it outside the brackets. We require whitespace after
    "the" so a name like "Theoden" (no space) is left whole, not chopped to
    "oden". A bare "The" with nothing after is left as-is (degenerate)."""
    m = re.match(r"(?i:the)\s+(.+)", surface)
    return m.group(1) if m else surface


def _looks_sentence_ish(surface: str) -> bool:
    """Build-time classification of a surface form's SHAPE (not runtime
    sentence-parsing, which is a fragile rabbit hole we explicitly rejected).

    The four Location-shaped noun types can pick up a sentence-ish FALLBACK name
    via the extractors' "missing name -> first detail" path. Those go into the
    source pool, so this gate is the reason they can't false-fire on a verbatim
    sentence repeat. (History-event names face the same gate, but harder: a
    sentence-ish event name earns no anchor at all and never enters the pool --
    see `build_crosslink_map`'s Pass 1b.) It's
    belt-and-suspenders over longest-match + word boundaries (a sentence-ish name
    would only ever match a full verbatim sentence, which ~never recurs) -- it turns
    "safe by luck" into "safe by design." Such a name still gets its ANCHOR as a
    target; it just isn't hunted for in prose as a source."""
    if any(ch in _SENTENCE_PUNCT for ch in surface):
        return True
    return len(surface.split()) > _MAX_SOURCE_WORDS


def _compile_pattern(surfaces: list) -> Optional["re.Pattern"]:
    """Compile ONE regex whose alternatives are the pool's surface forms, ordered
    LONGEST first, with self-defined word boundaries and an optional leading
    article. Returns None for an empty pool.

    Why one regex + ``re.finditer`` (the load-bearing choice): finditer walks
    left-to-right in reading order and returns NON-overlapping matches, so a span
    we've already turned into a link can never be re-scanned -- re-linking our own
    output is impossible by construction, with no manual cursor and no sequential
    ``str.replace`` (which WOULD re-scan and double-link). Longest-first ordering is
    what makes "Krieger Imperium" win over a bare "Krieger": regex alternation takes
    the FIRST alternative that matches at a position, so the longer phrase must come
    first in the alternation or it would be shadowed by its own prefix.

    The two capture groups (only two, regardless of pool size -- the surface forms
    live inside a single alternation, not one group each):
      - ``the``: an OPTIONAL leading "the"/"The" + following whitespace, with its
        own left boundary so the "the" in "breathe" isn't mistaken for an article.
        Present or absent, it is always emitted OUTSIDE the brackets with its
        original casing. Whether it is REQUIRED for a match to fire is decided
        per-surface at build time and enforced in the match handler, not here.
      - ``name``: the matched surface form itself, wrapped in self-defined
        boundaries (lookbehind/lookahead against `_NAME_CHARS`) so apostrophes and
        hyphens count as word-internal.
    """
    if not surfaces:
        return None
    # Re-sort defensively so the alternation is longest-first regardless of how the
    # caller ordered `surfaces`; ties break alphabetically for a byte-stable regex.
    ordered = sorted(surfaces, key=lambda s: (-len(s), s))
    alternation = "|".join(re.escape(s) for s in ordered)
    the_part = rf"(?P<the>(?<![{_NAME_CHARS}])(?:[Tt]he)\s+)?"
    name_part = rf"(?<![{_NAME_CHARS}])(?P<name>{alternation})(?![{_NAME_CHARS}])"
    return re.compile(the_part + name_part)


# --------------------------------------------------------------------------- #
# The resolved, build-once structure.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CrosslinkMap:
    """The single source of truth for both readers, built ONCE by
    `build_crosslink_map` and never mutated.

    - `anchors`: entity name -> final anchor id, for the convenient common case
      where names are unique (the renderer can stamp headings from this). When two
      entities genuinely share a name, this dict can only hold one -- use
      `entity_anchors` for the authoritative per-entity mapping.
    - `entity_anchors`: parallel to the input list -- ``entity_anchors[i]`` is
      entity i's final anchor, correct even when two entities share a name (each
      still gets its own suffixed anchor). This is what the renderer should zip with
      the entity list to stamp headings.
    - `sources`: the resolved match-pool as ``(surface_form, target_anchor)`` pairs,
      ordered LONGEST surface first (the brief-required exposure). Names and aliases
      are mixed into ONE pool and sorted together.
    - `pattern`: the compiled longest-first regex over `sources`, or None if the
      pool is empty. Built once here so every block reuses it.
    - `_lookup`: surface_form -> (anchor, article_required), the match handler's
      private lookup. Underscore-prefixed: it's an implementation detail of the
      walk, not part of the public contract.
    - `event_anchors`: parallel to the `events` passed to `build_crosslink_map` --
      ``event_anchors[i]`` is event i's anchor, or None where the event earned none
      (its sentence-shaped name was gated out). The same parallel-list pattern as
      `entity_anchors`, just with an opt-out slot the renderer skips.
    """
    anchors: dict
    entity_anchors: list
    sources: list
    pattern: Optional["re.Pattern"]
    _lookup: dict = field(default_factory=dict)
    # parallel to the `events` passed to build_crosslink_map: event_anchors[i] is
    # event i's anchor, or None where the event earned none (a sentence-shaped name
    # that was gated out). The renderer zips this with the events to stamp anchors,
    # stamping nothing where it's None -- the same parallel-list pattern as
    # entity_anchors, just with an opt-out slot.
    event_anchors: list = field(default_factory=list)


def load_crosslink_words(path: str = DEFAULT_WORDS_PATH) -> dict:
    """Load the optional common-word config: ``{"require_article": [...],
    "never_link": [...]}``. Like `speaker_map.load_speaker_map` it's a thin reader
    the orchestrator calls and threads into `build_crosslink_map` (which itself does
    NO file I/O -- same separation as the speaker map).

    Unlike the speaker-map loader, a MISSING file is not an error: both lists
    default empty so a fresh clone with no config behaves exactly like an empty
    config. A malformed JSON file still raises -- and so does a file that PARSES
    but is the wrong shape (a non-dict, or a typo'd key like "neverlink" that
    would otherwise fall through .get() to an empty default and silently disable
    the config). A typo'd config is worth surfacing, not silently swallowing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"require_article": [], "never_link": []}

    # A file that PARSED but is the wrong shape is a silent trap. A non-dict, or a
    # typo'd key like "neverlink", would otherwise fall through .get() to an empty
    # default and quietly disable the config with no warning. Catch it loudly.
    if not isinstance(data, dict):
        raise ValueError(
            f"crosslink_words.json must be a JSON object, got {type(data).__name__}."
        )
    known = {"require_article", "never_link"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"crosslink_words.json has unrecognized key(s): {sorted(unknown)}. "
            f"Expected only {sorted(known)} — check for a typo."
        )

    return {
        "require_article": list(data.get("require_article", [])),
        "never_link": list(data.get("never_link", [])),
    }


def build_crosslink_map(noun_entities: list, events: Optional[list] = None, common_words: Optional[dict] = None) -> CrosslinkMap:
    """Build the resolved `CrosslinkMap` over the five NOUN types (`Location`,
    `Character`, `Organization`, `Item`, `PeopleAndCultures`) plus, optionally, the
    `HistoryEvent`s. An ELIGIBLE event -- one whose article-stripped name isn't
    sentence-ish, judged by the SAME `_looks_sentence_ish` gate the entity source
    pool uses -- is both a target (it earns an anchor, assigned AFTER the entities
    so an event-vs-entity slug clash suffixes the event, never the entity) and a
    source (its name + aliases join the same claim pools). A sentence-shaped event
    name (the extractor's "missing name -> first detail" fallback) is NEITHER: no
    anchor (None in `event_anchors`) and nothing in the pool. Entities always get an
    anchor regardless of the gate; events must clear it to earn one. Either way an
    event's `description` is still run through `add_crosslinks` at render time, so
    noun mentions inside it light up.

    All the messy cases are resolved HERE, once, at build time -- never during the
    walk. `common_words` is the loaded `crosslink_words.json` config (or None for
    the empty default); this function does no file I/O of its own.
    """
    common_words = common_words or {}
    # NFC the config words so they line up with NFC'd surface forms. Match is exact
    # and case-sensitive (the words are stored capitalized on purpose -- "common
    # even when capitalized"), against the article-stripped surface form.
    require_article = {_nfc(w) for w in common_words.get("require_article", [])}
    never_link = {_nfc(w) for w in common_words.get("never_link", [])}

    # --- Pass 1: give EVERY entity a unique anchor (slug-collision suffixing). --- #
    # Anchors are assigned to every entity regardless of any later source gate, so a
    # sentence-ish name still has a valid target even though it won't be a source.
    entity_anchors = []
    anchors = {}            # entity name -> anchor (first occurrence wins)
    used_anchors = set()
    for i, ent in enumerate(noun_entities):
        base = slugify(ent.name)
        if not base:
            # Empty-slug fallback: a deterministic id by stable index, never an
            # empty ``id=""``. Flagged because an all-punctuation name is odd enough
            # for a human to want to see.
            base = f"entity-{i}"
            logger.warning("%s crosslink: name %r slugs to empty; using fallback anchor %r.",
                           REVIEW_PREFIX, ent.name, base)
        anchor = base
        if anchor in used_anchors:
            # Slug collision: two DIFFERENT entities whose names slug the same
            # ("Riverton!" and "Riverton" both -> "riverton"). Suffix the later one
            # deterministically (riverton, riverton-2, riverton-3, ...) and flag it
            # -- usually means the reconciler should have merged them, or it's a real
            # coincidence worth a human glance.
            k = 2
            while f"{base}-{k}" in used_anchors:
                k += 1
            anchor = f"{base}-{k}"
            logger.warning("%s crosslink: slug collision on %r; %r -> anchor %r.",
                           REVIEW_PREFIX, base, ent.name, anchor)
        used_anchors.add(anchor)
        entity_anchors.append(anchor)
        if ent.name not in anchors:
            anchors[ent.name] = anchor
        else:
            # Two entities literally share a name (e.g. a realm that is both a
            # Location and the Organization that governs it). Both keep their own
            # anchor in `entity_anchors`; the name->anchor dict can only hold one.
            logger.warning("%s crosslink: two entities share the name %r; "
                           "anchors %r and %r both exist (see entity_anchors).",
                           REVIEW_PREFIX, ent.name, anchors[ent.name], anchor)

    # --- Pass 1b: give each ELIGIBLE event a unique anchor, AFTER the entities. --- #
    # Events differ from entities twice over. (1) They're gated: a sentence-shaped
    # event name (the extractor's "missing name -> first detail" fallback) is not a
    # useful link target, so it earns NO anchor -- we store None in the parallel
    # list and the renderer stamps nothing there. (Entities always get an anchor;
    # an event has to clear the gate.) (2) They suffix AFTER the entities: we keep
    # using the SAME `used_anchors` set from Pass 1, so on an event-vs-entity slug
    # clash the ENTITY keeps the clean slug ("riverton") and the event takes the
    # suffix ("riverton-2"). That ordering IS the "entities before events" tiebreak
    # -- it falls out of running this loop second, with no explicit type-ranking.
    events = events or []
    event_anchors = []      # parallel to `events`; None where an event earns no anchor
    for i, ev in enumerate(events):
        # Gate on the SAME article-stripped, NFC'd form the entity source-gate uses,
        # so "eligible event" means exactly what "sentence-ish source" means for
        # entities -- one rule, no second definition to drift.
        if _looks_sentence_ish(_nfc(_split_article(ev.name))):
            event_anchors.append(None)     # gated out -> no anchor, no pool entry
            continue
        base = slugify(ev.name)
        if not base:
            # Same empty-slug guard as Pass 1, but with an event-scoped fallback id
            # so it can never collide with an entity-<N> fallback. Flagged, because
            # an all-punctuation name is odd enough for a human to want to see.
            base = f"event-{i}"
            logger.warning("%s crosslink: event name %r slugs to empty; using fallback "
                           "anchor %r.", REVIEW_PREFIX, ev.name, base)
        anchor = base
        if anchor in used_anchors:
            # Identical suffixing to Pass 1: bump -2, -3, ... until unique, and flag.
            k = 2
            while f"{base}-{k}" in used_anchors:
                k += 1
            anchor = f"{base}-{k}"
            logger.warning("%s crosslink: slug collision on %r; event %r -> anchor %r.",
                           REVIEW_PREFIX, base, ev.name, anchor)
        used_anchors.add(anchor)
        event_anchors.append(anchor)

    # --- Pass 2: gather surface-form CLAIMS (name claims and alias claims). --- #
    # A claim is "surface form S points at anchor A". We track names and aliases
    # separately so the name-beats-alias rule can be applied before we look for
    # genuine ambiguity. Each surface is article-stripped + NFC'd; the actual prose
    # match is against this (un-folded) string -- folding is ONLY for anchor ids.
    name_claims = {}     # surface -> set(anchor)
    alias_claims = {}    # surface -> set(anchor)
    for i, ent in enumerate(noun_entities):
        s = _nfc(_split_article(ent.name))
        if s.strip():
            name_claims.setdefault(s, set()).add(entity_anchors[i])
        for alias in ent.aliases:
            a = _nfc(_split_article(alias))
            if a.strip():
                alias_claims.setdefault(a, set()).add(entity_anchors[i])

    # An event with an anchor (event_anchors[i] is not None -> it cleared the gate)
    # is a real target, so its name and aliases join the SAME claim pools the
    # entities use. From here it's just another claimant: Passes 3-4 resolve any
    # event-vs-entity or event-vs-event clash uniformly (a surface claimed by >1
    # anchor drops out as ambiguous, and each thing keeps its own anchor). We check
    # `is not None`, never truthiness -- an anchor is never an empty string, but the
    # None opt-out is the real signal and we key on it explicitly.
    for i, ev in enumerate(events):
        if event_anchors[i] is None:
            continue                       # gated out -> not a source either
        s = _nfc(_split_article(ev.name))
        if s.strip():
            name_claims.setdefault(s, set()).add(event_anchors[i])
        for alias in ev.aliases:
            a = _nfc(_split_article(alias))
            if a.strip():
                alias_claims.setdefault(a, set()).add(event_anchors[i])

    # --- Pass 3a: name beats alias. --- #
    # If an alias surface equals some entity's real NAME surface, the name wins and
    # the alias claim is dropped. If the alias only ever pointed where the name
    # already points (same anchor), it's harmless redundancy -- drop it quietly. If
    # it pointed somewhere ELSE, it's a genuine alias-vs-name clash -- drop it loud.
    for s in list(alias_claims):
        if s in name_claims:
            if not alias_claims[s] <= name_claims[s]:
                logger.warning("%s crosslink: alias %r collides with a real entity "
                               "name; name wins, alias dropped.", REVIEW_PREFIX, s)
            del alias_claims[s]

    # --- Pass 3b: merge surviving claims into one surface -> anchors map. --- #
    combined = {}
    for source_claims in (name_claims, alias_claims):
        for s, anchor_set in source_claims.items():
            combined.setdefault(s, set()).update(anchor_set)

    # --- Pass 4: resolve ambiguity + gates into the final pool. --- #
    sources = []     # (surface, anchor), sorted longest-first at the end
    lookup = {}      # surface -> (anchor, article_required)
    for s, anchor_set in combined.items():
        if len(anchor_set) > 1:
            # Same surface form claimed by two DIFFERENT entities (two same-named
            # entities, or the same alias on two entities). We genuinely can't pick
            # -- linking to one arbitrarily is a coin-flip that's wrong half the
            # time -- so it goes nowhere and a human is told.
            logger.warning("%s crosslink: surface %r is claimed by %d entities (%s); "
                           "left out of the link pool.",
                           REVIEW_PREFIX, s, len(anchor_set), sorted(anchor_set))
            continue
        (anchor,) = tuple(anchor_set)
        if s in never_link:
            # The truly-cursed escape hatch: a surface so generically common that
            # even article-required would over-fire. Held fully out of the pool.
            logger.warning("%s crosslink: surface %r is on the never_link list; "
                           "held out of the source pool.", REVIEW_PREFIX, s)
            continue
        if _looks_sentence_ish(s):
            logger.warning("%s crosslink: surface %r looks sentence-shaped; kept as "
                           "a target but not hunted in prose.", REVIEW_PREFIX, s)
            continue
        article_required = s in require_article
        sources.append((s, anchor))
        lookup[s] = (anchor, article_required)

    # Names and aliases compete for the longest match in ONE line-up: an alias can
    # be longer than some other entity's whole name, so they must be sorted
    # together, not "all names then all aliases".
    sources.sort(key=lambda pair: (-len(pair[0]), pair[0]))
    pattern = _compile_pattern([s for s, _ in sources])

    return CrosslinkMap(
        anchors=anchors,
        entity_anchors=entity_anchors,
        sources=sources,
        pattern=pattern,
        _lookup=lookup,
        event_anchors=event_anchors,
    )


# --------------------------------------------------------------------------- #
# The no-go pre-pass: regions matching must never touch.
# --------------------------------------------------------------------------- #

# Tape off the trim before painting the open wall: find the regions that must stay
# untouched, run the link weave only on the plain-prose stretches between them.
# Order in the alternation is precedence -- a fenced code block is tried first so
# its contents (which may contain '#', backticks, or '[x](y)') are protected as one
# unit rather than mis-read by the later branches. finditer is non-overlapping, so
# whatever the fence consumes the other branches never see.
_PROTECTED = re.compile(
    # Fenced code block: ``` ... ``` across lines (non-greedy to the next fence).
    r"(?P<fence>```[\s\S]*?```)"
    # ATX heading: a line starting with optional indent then '#'. (?m:...) scopes
    # MULTILINE to just this branch so ^/$ mean line-start/line-end.
    r"|(?P<heading>(?m:^[ \t]*#.*$))"
    # Existing markdown link [text](url): protect the WHOLE thing so we never nest a
    # link inside it (text has no ']', url has no ')').
    r"|(?P<link>\[[^\]]*\]\([^)]*\))"
    # Inline code span: a run of backticks, then anything up to the same-length
    # closing run (CommonMark pairs spans by run length). A backstop -- footnote
    # quotes render inside backticks, so even one that slipped through is protected.
    r"|(?P<code>(?P<bt>`+)(?:(?!(?P=bt)).)*?(?P=bt))"
)


def _weave(segment: str, crosslink_map: CrosslinkMap, seen: set) -> str:
    """Run the single longest-first regex over ONE plain-prose stretch and weave in
    links. `seen` is threaded across all stretches of a block so first-occurrence is
    per-block, not per-stretch.

    The whole link/skip decision is ONE rule keyed on the target anchor:
      - article-required surface with no leading "the" present -> leave plain
        (consumed but not linked);
      - else if the target anchor is already in `seen` -> leave plain;
      - else -> wrap it and record the anchor in `seen`.
    That single `seen` check delivers BOTH first-occurrence-per-page and self-link
    suppression (the page's own anchor is pre-seeded), because suppression keys on
    the target ANCHOR, not the surface string -- so an alias that resolves to the
    page's own anchor is suppressed too.
    """
    pattern = crosslink_map.pattern
    if pattern is None:
        return segment
    out = []
    last = 0
    for m in pattern.finditer(segment):
        out.append(segment[last:m.start()])
        name = m.group("name")
        the = m.group("the") or ""          # "" when no article precedes
        anchor, article_required = crosslink_map._lookup[name]
        if article_required and not m.group("the"):
            # A common-word surface (e.g. "Founding") fires only with a leading
            # "the". No article here -> this is sentence-prose like "Founding
            # members got perks", not a reference. Consume it but leave it plain.
            out.append(m.group(0))
        elif anchor in seen:
            # First-occurrence-per-page OR self-suppression -- same check, no
            # special-casing. The "the" (if any) stays as it was, outside any link.
            out.append(m.group(0))
        else:
            seen.add(anchor)
            # "the" is ALWAYS rendered outside the brackets with its original casing.
            out.append(f"{the}[{name}](#{anchor})")
        last = m.end()
    out.append(segment[last:])
    return "".join(out)


def add_crosslinks(block: str, crosslink_map: CrosslinkMap, this_anchor: Optional[str]) -> str:
    """Weave internal links over ONE rendered block -- an entity page (heading +
    body) or a single history-event description.

    `this_anchor` is the block's own anchor, pre-seeded into `seen` so the page is
    never linked to itself. For a history-event description, pass the event's own
    anchor from `event_anchors` when it has one -- an anchored event IS a target,
    and its own description must not link back to it -- and None only for a
    gated-out event (no anchor -> not a target -> nothing to self-suppress). The
    `seen` set is fresh per call, so each page is its own little article:
    first-mention-per-article, reset on the next block.
    """
    # Empty block or empty pool: nothing to do, return the input untouched (no NFC
    # rewrite either, so it's byte-identical).
    if not block or crosslink_map.pattern is None:
        return block

    # NFC the whole block ONCE up front so the no-go pre-pass and the match walk
    # operate on the same canonical text (stable indices), and byte-variant prose
    # lines up with the NFC'd surface forms in the pool.
    block = _nfc(block)

    seen = set()
    if this_anchor:
        seen.add(this_anchor)

    # Walk the protected regions; weave links only in the gaps between them.
    out = []
    last = 0
    for m in _PROTECTED.finditer(block):
        out.append(_weave(block[last:m.start()], crosslink_map, seen))
        out.append(m.group(0))   # protected region: emit verbatim
        last = m.end()
    out.append(_weave(block[last:], crosslink_map, seen))
    return "".join(out)
