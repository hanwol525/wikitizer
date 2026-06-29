"""Phase 4.1a -- the reconciler's MERGE step. De-duplicates a list of ONE entity
type. See models/reconcile.py for the decision-object contract.

Design in one breath: the LLM only DECIDES which entries merge (+ the canonical
name, + which details conflict, + which pairs look suspicious). Pure Python does
the actual fact-merging from the ORIGINAL objects, so rewriting a fact is
impossible by construction. Conservative on purpose -- a false merge fabricates a
Frankenstein entity (the sin we avoid); a false split is just a duplicate page
(harmless, caught on review).

Inherits BaseAgent DIRECTLY (like the noise filter), not BaseExtractor: no
batching (de-dup needs the whole list at once) and no new quotes to verify.
"""

import json
import logging
import re

from pydantic import ValidationError

from agents.base import BaseAgent, ClaudeJSONError, strip_code_fences
from models.lore import Character, HistoryEvent, Scope
from models.reconcile import ReconcileDecision, DateDecision, PlacementDecision

logger = logging.getLogger(__name__)

# All loud, human-actionable flags share this prefix so Phase 5.3 can funnel them
# into review.txt without them drowning in routine warnings (dropped details, JSON
# retries). Quiet auto-resolved clashes (is_pc / scope) DON'T use it -- they log at
# debug level and stay out of the review queue.
REVIEW_PREFIX = "[REVIEW]"

# Initials / very short names get the strict "stated-alias-only" rule (see below).
SHORT_NAME_LEN = 3


# --------------------------------------------------------------------------- #
# Pure helpers (no self, no Claude) -- these unit-test on their own, exactly
# like the verbatim-check helpers in base_extractor.py.
# --------------------------------------------------------------------------- #

def _type_label(entry) -> str:
    """Human-friendly type name for prompts/logs. PeopleAndCultures -> the pretty
    'People & Cultures'; everything else uses the class name as-is."""
    name = entry.__class__.__name__
    return "People & Cultures" if name == "PeopleAndCultures" else name


def _extract_json_object(text):
    """Best-effort: pull the first balanced top-level ``{...}`` object out of TEXT.

    `strip_code_fences` peels a markdown wrapper, but Sonnet sometimes prefaces the
    JSON with a conversational preamble ("Looking at these entries...") that the
    fence-stripper leaves in place -- which then fails `model_validate_json` and
    burns all three retries, dropping a real merge. This recovers the embedded
    object. It scans from the first ``{`` tracking brace depth while respecting
    string literals + escapes, so a brace inside a quoted value can't throw off the
    count. Returns the substring, or ``None`` if no balanced object is found."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_json_model(raw, model_cls):
    """Parse raw LLM text into the given Pydantic model, or None if it can't. Same
    tolerant strategy as _parse_decision -- the fence-stripped text first, then the
    first balanced {...} object (so a conversational preamble Sonnet adds doesn't
    burn all the retries) -- generalized to any model. The two timeline calls reuse
    it for DateDecision / PlacementDecision."""
    for candidate in (strip_code_fences(raw), _extract_json_object(raw)):
        if not candidate:
            continue
        try:
            return model_cls.model_validate_json(candidate)
        except (ValidationError, ClaudeJSONError, json.JSONDecodeError):
            continue
    return None


def _dedup_preserve_order(items):
    """Drop EXACT duplicates, keep first-seen order. A light .strip() guards against
    stray surrounding whitespace masquerading as a new value -- nothing cleverer,
    because a redundant detail is harmless but an over-eager dedup that eats a
    subtly-DIFFERENT fact is not. Returns the original (un-stripped) items."""
    seen = set()
    out = []
    for it in items:
        key = it.strip()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _dedup_quotes(quotes):
    """Dedup Quote objects on the FULL (text, speaker, source_file) triple, first-
    seen order. Two DIFFERENT speakers attesting the same fact are two distinct
    quotes -- both survive, because that's MORE evidence, not redundancy."""
    seen = set()
    out = []
    for q in quotes:
        key = (q.text, q.speaker, q.source_file)
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def _resolve_scope(scopes):
    """Broadest scope wins. Matches the History extractor's 'default to world'
    philosophy: a too-broad event is loud (front-of-wiki) and gets corrected; a
    too-narrow one hides in `personal` and gets missed. So never narrow on a merge.
    """
    order = {Scope.PERSONAL: 0, Scope.REGIONAL: 1, Scope.WORLD: 2}
    return max(scopes, key=lambda s: order[s])


def _resolve_date_text(members):
    """date_text for a merged HistoryEvent. A stated date beats None (None just
    means that particular mention didn't give one -- absence of evidence, same
    logic as is_pc/player_name). If two merged events state DIFFERENT dates,
    that's a contradiction in the source: we keep the first stated one and log it
    quietly. The dropped date is removed from the `date_text` field; the merged
    event's concatenated description and unioned quotes may or may not still
    mention it, so the clash is logged for traceability. Returns the date string,
    or None."""
    dates = _dedup_preserve_order(
        [m.date_text.strip() for m in members if m.date_text and m.date_text.strip()]
    )
    if not dates:
        return None
    if len(dates) > 1:
        logger.debug("Reconciler: date_text clash %s -> kept first.", dates)
    return dates[0]


class _VetoMerge(Exception):
    """Raised by the combiner when a proposed merge turns out to be unsafe to
    actually merge (right now: two Characters with two DIFFERENT real player_names
    -- usually a sign the LLM grouped two genuinely different PCs). The apply loop
    catches it, leaves those entries UNMERGED (each keeps its own data), and writes
    a loud flag. We veto rather than hard-block forever because a player handoff
    (one human takes over another's character) is a real, if rare, case."""
    def __init__(self, names, message):
        super().__init__(message)
        self.names = names
        self.message = message


def _resolve_player_name(members):
    """name-vs-None  -> the name  (None just means 'this batch didn't know').
    name-vs-DIFFERENT-name -> VETO: two different humans behind one character is
    strong evidence the merge itself is wrong, so we don't pick a winner (that
    would fabricate a player) -- we bail and let the apply loop flag it."""
    names = [m.player_name for m in members if m.player_name]
    distinct = _dedup_preserve_order(names)
    if len(distinct) <= 1:
        return distinct[0] if distinct else None
    raise _VetoMerge(distinct, f"player_name clash across a merge: {distinct}")


def _short_name_veto(members) -> bool:
    """Return True if this merge should be VETOED under the short-name rule.

    For very short names (<= SHORT_NAME_LEN chars, e.g. initials like 'CJ'), one
    wrong letter is a HUGE fraction of the name, so spelling-closeness is not
    trustworthy: 'CJ' vs 'DJ' are different people; 'CJ' vs 'CL' could be anything.
    So we only allow a short-name merge when the alias was EXPLICITLY stated --
    which we detect structurally: one member's name appears in another member's
    alias list (that's how the extractor records a stated alias). No such link ->
    veto. Over-blocking here just yields a harmless duplicate page; under-blocking
    risks a fabricated merge, so we err toward veto. Only fires when a SHORT name
    is involved -- long-name merges are unaffected."""
    names = [m.name for m in members]
    if not any(len(nm.strip()) <= SHORT_NAME_LEN for nm in names):
        return False  # no short names in play -> rule doesn't apply
    # A "stated alias" must be a CROSS-member link: one member's name appearing in
    # a DIFFERENT member's alias list. A member self-aliasing (its own name among
    # its own aliases -- which the extractors can emit) is NOT evidence that two
    # short names are the same entity, so it must not satisfy the rule -- else
    # CJ(aliases=['CJ']) + DJ would wrongly merge. Hence the `other is not m` guard.
    stated = any(
        m.name.strip().lower() in {
            a.strip().lower()
            for other in members if other is not m
            for a in other.aliases
        }
        for m in members
    )
    return not stated  # veto UNLESS the alias was stated cross-member


def _resolve_canonical(members, canonical):
    """Snap the LLM's canonical to the EXACT member name/alias string it matched.

    `_validate_decision` accepts a canonical case-/whitespace-insensitively, so the
    LLM may hand back a recased/padded form ("lake mundi" for "Lake Mundi"). But the
    merged heading must be a string that LITERALLY appeared among the members
    (intent: even the heading can't be invented) -- otherwise the recased form
    becomes the heading AND the real name, no longer ``==`` to it, gets demoted into
    the alias list. So we look the normalized form back up and return the verbatim
    stored value: a member NAME if one matches (the heading should be a real name),
    else a matching alias. Falls back to the given canonical unchanged if nothing
    matches (validation should already have rejected that case)."""
    key = canonical.strip().lower()
    for m in members:
        if m.name.strip().lower() == key:
            return m.name
    for m in members:
        for a in m.aliases:
            if a.strip().lower() == key:
                return a
    return canonical


def _combine_group(members, canonical):
    """Merge a list of same-type entries into ONE, given the canonical name the
    LLM picked. Pure + deterministic -- only unions/concatenates fields from the
    ORIGINAL objects, never rephrasing anything. Returns a new entry of the same
    class. May raise _VetoMerge (Characters with clashing player_names)."""
    cls = members[0].__class__
    # Snap the canonical back to the verbatim member name/alias it matched: the
    # validator is case-/whitespace-insensitive, but the heading must be a string
    # that literally appeared in the data, and the loser-to-alias test below relies
    # on an exact `!=` (see _resolve_canonical).
    canonical = _resolve_canonical(members, canonical)

    # aliases: every member's aliases + every member's NAME that isn't the canonical
    # pick (so no name is ever lost -- losers become aliases).
    aliases = []
    for m in members:
        aliases.extend(m.aliases)
        if m.name != canonical:
            aliases.append(m.name)
    aliases = _dedup_preserve_order(aliases)
    aliases = [a for a in aliases if a != canonical]  # canonical shouldn't alias itself

    # quotes: concat all, dedup on the triple.
    quotes = []
    for m in members:
        quotes.extend(m.supporting_quotes)
    quotes = _dedup_quotes(quotes)

    # HistoryEvent is the odd one out: no `details`, a single prose `description`.
    if isinstance(members[0], HistoryEvent):
        # keep both -> concatenate descriptions. Faithful if a little clunky, which
        # is fine because History merges extra-conservatively, so this is rare.
        descriptions = _dedup_preserve_order([m.description for m in members])
        description = "\n\n".join(descriptions)
        scopes = [m.scope for m in members]
        if len(set(scopes)) > 1:
            logger.debug("Reconciler: scope clash %s -> kept broadest.", set(scopes))
        return cls(
            name=canonical,
            aliases=aliases,
            description=description,
            scope=_resolve_scope(scopes),
            date_text=_resolve_date_text(members),
            calendar_system=None,         # 4.1b fills this; None at merge time (like chronological_position)
            chronological_position=None,  # always None until the 4.1b timeline pass
            supporting_quotes=quotes,
        )

    # everyone else has a `details` list: concat + exact-dupe-only removal.
    details = []
    for m in members:
        details.extend(m.details)
    details = _dedup_preserve_order(details)

    # Characters carry two scalars to resolve; the other three types don't.
    if isinstance(members[0], Character):
        if len({m.is_pc for m in members}) > 1:
            logger.debug("Reconciler: is_pc clash in '%s' -> True wins.", canonical)
        return cls(
            name=canonical,
            aliases=aliases,
            is_pc=any(m.is_pc for m in members),       # True wins (False = absence of evidence)
            player_name=_resolve_player_name(members),  # may raise _VetoMerge
            details=details,
            supporting_quotes=quotes,
        )

    # Location / Organization / Item / PeopleAndCultures: identical 4-field shape.
    return cls(name=canonical, aliases=aliases, details=details, supporting_quotes=quotes)


def _validate_decision(decision, entries):
    """Sanity-check the LLM's decision BEFORE acting on it. Returns a list of
    human-readable problems; empty list == clean. We RETRY on any problem (a
    structurally broken decision is treated like bad JSON).

    Checks (all on `merges`; possible_duplicates is logging-only):
      1. every index is in range 0..len-1
      2. no index appears in two different merge groups (one entry can't be two)
      3. each `canonical` is actually one of the names/aliases among that group's
         members -- so even the HEADING can't be invented out of thin air.
    """
    problems = []
    n = len(entries)
    seen = set()
    for gi, group in enumerate(decision.merges):
        # A "merge" of fewer than two entries isn't a duplicate group, and an empty
        # members list would crash the combiner (members[0]). Flag it so a malformed
        # decision is retried like any other bad output -- never crashed on.
        if len(group.members) < 2:
            problems.append(
                f"merge {gi}: a merge needs >= 2 members, got {len(group.members)}"
            )
        valid = []
        for idx in group.members:
            if idx < 0 or idx >= n:
                problems.append(f"merge {gi}: index {idx} out of range 0..{n - 1}")
            elif idx in seen:
                problems.append(f"merge {gi}: index {idx} already used in another group")
            else:
                seen.add(idx)
                valid.append(idx)
        # canonical must appear among the (valid) members' names/aliases
        allowed = set()
        for idx in valid:
            allowed.add(entries[idx].name.strip().lower())
            allowed.update(a.strip().lower() for a in entries[idx].aliases)
        if valid and group.canonical.strip().lower() not in allowed:
            problems.append(
                f"merge {gi}: canonical {group.canonical!r} is not a name/alias of any member"
            )
    return problems


# --------------------------------------------------------------------------- #
# Phase 4.1b -- the pure-Python timeline engine. Runs AFTER the 4.1a merge, over
# the deduplicated HistoryEvents, to fill `calendar_system` + `chronological_position`.
# The model: a sorted SPINE of dated events with GAPS between them; relative
# (undated) events get woven into the gaps. Dates sort as tuples of parts, biggest
# unit first, so Python owns every position and the LLM never emits one. These are
# the deterministic pieces; Brief B wires the two Claude calls on top of them.
# --------------------------------------------------------------------------- #

# The pseudo-system that holds relatively-ordered, date-LESS events. A campaign
# with no dates at all runs entirely on this one "system" (its single gap); a
# mixed campaign uses it for events that order among themselves but anchor to no
# date. Events woven here get calendar_system=None (the label is internal only).
UNDATED = "(undated)"


def _gap_id(system: str, k: int) -> str:
    """The ID for gap k of a system. Gap k sits immediately BEFORE rank-group k
    (so gap 0 = before everything, gap m = after the last group). Callers match
    these by EXACT STRING against the gap-lookup dict -- they never parse the id
    back apart, so a system label containing a '#' or space can't break anything."""
    return f"{system}#{k}"


def _sanity_guard_parts(parts, date_text) -> bool:
    """Best-effort check that a parts tuple is grounded in the event's stated date.
    True = usable; False = looks unmoored, so the caller drops this event from the
    dated spine (it falls back to relative placement / Could Not Place).

    Pure Python can't fully verify a WORD-based date ("the Third Age" -> era 3)
    without parsing calendar notation (the era arithmetic we deliberately skip), so:
      - date_text has NO numbers -> can't verify -> trust the LLM's local read (True).
      - date_text HAS numbers -> at least one parts value must match one of the
        extracted numeric tokens (the tuple is anchored to a real stated number).
        Zero overlap means the parts are unmoored from the date_text -> reject (False).
    This catches the clear numeric mismatch (e.g. [343] for "342 AR", or a tuple
    sharing none of the date_text's numbers). A near-miss that still shares a number slips
    through, but the verbatim date renders anyway, so the worst case is a minor
    mis-sort, never lost or fabricated lore."""
    digits = {int(n) for n in re.findall(r"\d+", date_text or "")}
    if not digits:
        return True
    return bool(set(parts) & digits)


def _validate_date_decision(decision, events) -> list:
    """Structural sanity-check on Call 1's DateDecision BEFORE the spine is built.
    Returns a list of human-readable problems; empty list == clean. (Brief B retries
    on any problem, like 4.1a.) This is SEPARATE from _sanity_guard_parts: validation
    failures are structural and retry the whole call; a guard failure drops a single
    event to relative placement without failing the call.

    Checks:
      1. every index is in range 0..len-1
      2. the event at that index actually HAS a date_text (you can't date a date-less
         event) -- a non-None, non-blank string
      3. parts is a non-empty list of positive integers
      4. no index appears twice in `dated`
    """
    problems = []
    n = len(events)
    seen = set()
    for d in decision.dated:
        if d.index < 0 or d.index >= n:
            problems.append(f"dated: index {d.index} out of range 0..{n - 1}")
            continue
        if d.index in seen:
            problems.append(f"dated: index {d.index} listed more than once")
        seen.add(d.index)
        dt = events[d.index].date_text
        if not (isinstance(dt, str) and dt.strip()):
            problems.append(f"dated: index {d.index} has no date_text to extract from")
        if not d.parts or not all(isinstance(p, int) and p > 0 for p in d.parts):
            problems.append(f"dated: index {d.index} has invalid parts {d.parts!r}")
        if not d.system or not d.system.strip():
            problems.append(f"dated: index {d.index} has an empty system label")
    return problems


def _build_spine_and_gaps(dated):
    """Group dated events by system, tuple-sort within each, collapse ties into
    rank-groups, and enumerate the gaps. `dated` is a list of (index, system, parts)
    tuples that already passed _validate_date_decision AND _sanity_guard_parts.

    Returns (spines, gap_lookup):
      spines: dict[system_label, list[list[int]]] -- per system, the ordered
              rank-groups (events with identical parts share a group, in original
              order). ALWAYS includes the UNDATED pseudo-system (zero rank-groups ->
              one gap) so relatively-ordered date-less events have somewhere to land.
      gap_lookup: dict[gap_id, (system_label, k)] -- every gap ID the LLM may
              reference, mapped to its system and insert-slot k. m rank-groups -> m+1
              gaps. Opaque keys: match by exact string, never parse.
    """
    by_system = {}
    for index, system, parts in dated:
        by_system.setdefault(system, []).append((tuple(parts), index))

    spines = {}
    for system, items in by_system.items():
        items.sort(key=lambda pair: pair[0])          # tuple-sort by parts
        groups = []                                    # collapse ties into rank-groups
        for parts_key, index in items:
            if groups and groups[-1][0] == parts_key:
                groups[-1][1].append(index)
            else:
                groups.append((parts_key, [index]))
        spines[system] = [members for _, members in groups]

    spines.setdefault(UNDATED, [])                     # always available; zero markers -> one gap

    gap_lookup = {}
    for system, rank_groups in spines.items():
        for k in range(len(rank_groups) + 1):          # m groups -> m+1 gaps
            gap_lookup[_gap_id(system, k)] = (system, k)
    return spines, gap_lookup


def _validate_placement_decision(decision, events, dated_indices, gap_lookup) -> list:
    """Structural sanity-check on Call 2's PlacementDecision BEFORE weaving. Returns
    a list of problems; empty == clean. (Brief B retries on any problem.)

    Checks:
      1. every `gap` is a real gap ID Python handed out (in gap_lookup)
      2. no gap appears in more than one placement (so the gap->events dict can't
         silently drop a collision)
      3. every event index is in range
      4. every placed event is a RELATIVE event -- NOT one of the dated_indices (a
         dated event's position is Python's, not the LLM's to move)
      5. no event index appears in more than one gap (no double-placement)
    """
    problems = []
    n = len(events)
    seen_gaps = set()
    seen_events = set()
    for p in decision.placements:
        if p.gap not in gap_lookup:
            problems.append(f"placement: unknown gap {p.gap!r}")
        if p.gap in seen_gaps:
            problems.append(f"placement: gap {p.gap!r} listed more than once")
        seen_gaps.add(p.gap)
        for idx in p.events:
            if idx < 0 or idx >= n:
                problems.append(f"placement: event index {idx} out of range 0..{n - 1}")
                continue
            if idx in dated_indices:
                problems.append(f"placement: event {idx} is a dated event; can't be placed by gap")
            if idx in seen_events:
                problems.append(f"placement: event {idx} placed in more than one gap")
            seen_events.add(idx)
    return problems


def _weave_and_stamp(events, spines, placements, gap_lookup):
    """Produce the final events with calendar_system + chronological_position set.
    Pure + deterministic; assumes `placements` already passed
    _validate_placement_decision. `placements` is dict[gap_id, list[int]] (ordered
    relative-event indices per gap). Walks each system's spine, drops placed
    relatives into their gaps in order, and stamps sequential positions PER SYSTEM
    (each timeline numbers from 0). Events placed nowhere -> Could Not Place
    (calendar_system=None, position=None). Returns NEW event objects (model_copy);
    never mutates the inputs."""
    # bucket placements by system + insert-slot; skip any gap id we didn't define
    # (a stray/invalid gap ref -> its events simply aren't placed -> Could Not Place)
    placed = {}  # system -> {k: [event indices in order]}
    for gap_id, idxs in placements.items():
        if gap_id not in gap_lookup:
            continue
        system, k = gap_lookup[gap_id]
        placed.setdefault(system, {})[k] = list(idxs)

    updates = {}  # event index -> (calendar_system, position)
    for system, rank_groups in spines.items():
        cal = None if system == UNDATED else system
        gaps = placed.get(system, {})
        sequence = []
        for k in range(len(rank_groups) + 1):
            sequence.extend(gaps.get(k, []))          # relatives in gap k (before group k)
            if k < len(rank_groups):
                sequence.extend(rank_groups[k])       # the (possibly tied) markers at rank k
        for position, index in enumerate(sequence):
            updates[index] = (cal, position)

    out = []
    for i, e in enumerate(events):
        cal, position = updates.get(i, (None, None))  # not woven anywhere -> Could Not Place
        out.append(e.model_copy(update={
            "calendar_system": cal,
            "chronological_position": position,
        }))
    return out


RECONCILER_SYSTEM_PROMPT = """\
You are the reconciler for a D&D lore wiki. You are given a list of already-extracted lore entries that are ALL THE SAME KIND of thing (all locations, or all characters, etc.). Your ONLY job is to find which entries are duplicates of each other -- the same real entity written down more than once under different names or phrasings -- and report your findings as JSON.

You do NOT write merged entries, you do NOT rewrite any text, and you do NOT invent any facts. You only identify duplicates and pick which EXISTING name should be the heading. Other code does the actual merging from the original entries.

## What counts as a duplicate
Two entries are duplicates only when they are clearly the SAME entity. For example:
- The same place under two names: "Lake Mundi" and "The Great Well".
- A simple misspelling of one name: "Maltaav" and "Maltraav".

## Be conservative -- when unsure, DO NOT merge
Wrongly merging two DIFFERENT entities is a serious error: it fuses unrelated facts into one fake entity. Wrongly leaving a true duplicate separate is only a minor annoyance (a stray extra entry). So: merge ONLY on strong evidence, and when it is genuinely unclear, leave them separate and report them under "possible_duplicates" instead.

Strong evidence to merge:
- The text actually states the alias (e.g. "Lake Mundi, also called the Great Well").
- One entry's name already appears in another entry's "aliases" list.

Weak evidence (do NOT merge on this alone -- report as a possible duplicate):
- Two entries just SOUND like they might be the same, but nothing ever says so.

## Similar spelling means LOOK HARDER, never an automatic merge
Names close in spelling are a prompt to investigate, not a decision. Use the surrounding facts:
- Do both names appear at the same time as DIFFERENT things? (e.g. "CJ and her sister DJ" -> two different people. Do NOT merge.)
- Does the text state a relationship or distinction between them? (e.g. "DJ is CJ's sister" -> two entities. Do NOT merge.)
- Do the facts read as ONE entity described twice, or TWO coherent separate entities?
A real misspelling shows NONE of these distinctness signals -- that is what makes it a typo. If you see ANY of them, keep the entries separate.

## Very short names (initials) -- strict rule
For very short names (about 3 characters or fewer, like "CJ", "DJ", "AJ"), do NOT merge on spelling similarity AT ALL. One different letter in a 2-3 letter name is a totally different name, not a typo. Only merge two short names if the text EXPLICITLY states they are the same (or one already appears in the other's aliases).

## Watch the "almost the same name but different thing" trap
Some names are nearly identical but refer to DIFFERENT entities -- a person vs. the empire named after them. "Krieger" (a person/family) and "Krieger Imperium" (an empire) are NOT the same entity. Do not merge a name with a longer name that adds words.

## Worked examples -- follow these closely
- "CJ" and "DJ": the text says DJ is CJ's sister. -> DO NOT MERGE. Two different people whose names happen to differ by one letter.
- "Maltaav" and "Maltraav": the same place, one just missing a letter, and nothing in the text treats them as two things. -> MERGE. Canonical: "Maltraav" (the correctly-spelled / more frequent form).

## Picking the canonical name (for entries you DO merge)
The canonical name MUST be one of the names or aliases that already appears among the entries you are merging -- NEVER a new name you invent. Prefer:
1. A real proper name over a vague descriptive phrase ("Lake Mundi" over "the big lake up north").
2. Among proper names, the one used most often across the supporting quotes.
Every other name automatically becomes an alias (handled by other code) -- you only name the winner.

## Flagging contradictions
While identifying a merge, if two of its entries state details that genuinely DISAGREE about the same fact (e.g. "ruled by the Kriega" vs "controlled by the Maltraav Imperium"), report that pair under the merge's "conflicts". Do NOT pick a side and do NOT drop either detail -- both are kept by other code. Only flag REAL disagreements, not two compatible facts (e.g. "sacred to the Kriega" and "fed by a spring" are both true at once -- NOT a conflict).

## Output -- return ONLY this JSON, nothing else
{
  "merges": [
    {
      "members": [<integer indices of the entries that are the same entity>],
      "canonical": "<the winning name, copied from one of those entries>",
      "conflicts": [
        {
          "detail_a": "<a detail from one entry>",
          "source_a": "<that detail's source file>",
          "detail_b": "<the disagreeing detail from another entry>",
          "source_b": "<that detail's source file>",
          "note": "<one short sentence: why these disagree>"
        }
      ]
    }
  ],
  "possible_duplicates": [
    {
      "members": [<integer indices of a suspicious-but-not-confirmed pair>],
      "note": "<one short sentence: why you suspect it but did not merge>"
    }
  ]
}

Rules for the JSON:
- "members" are the integer indices shown in [brackets] before each entry, counting from 0.
- Only include entries that have duplicates. Any entry NOT listed in "merges" is assumed unique -- do not list unique entries anywhere.
- "conflicts" is usually an empty list; only include real disagreements.
- If there are no duplicates at all, return exactly {"merges": [], "possible_duplicates": []}.
- Every "members" list must contain at least two indices -- a one-entry "merge" is meaningless; just leave a unique entry out entirely.
- Return ONLY the JSON object: your entire reply must be that single JSON object, with no sentence before it and nothing after it, and no markdown code fences.
"""


DATE_EXTRACTION_PROMPT = """\
You are building a timeline for a D&D lore wiki. You are given history events that each state an explicit in-world DATE. For each one, identify (a) which calendar SYSTEM the date is in, and (b) the date as a sortable list of numeric PARTS.

You do NOT order the events and you do NOT assign any position number -- you only read each date into a system label + parts. Sorting is done by other code.

## Parts: a list of numbers, biggest unit FIRST
Turn the stated date into a list of integers, largest unit first, so they sort correctly:
- "1347" -> [1347]
- "342 AR" -> [342]
- "4th Era 200" -> [4, 200]        (era 4, year 200)
- "Third Age, year 12" -> [3, 12]  ("Third" is the 3rd age)
Read a written-out ordinal as its number ("Third" -> 3, "Fourth Era" -> 4). Use the SAME unit structure for every event in one system (e.g. if one Elder Scrolls date is [era, year], they all are), so the lists compare correctly. If a date gives only the larger unit (e.g. just "the Third Age", no year), give just that ([3]).

## System: a consistent label for the calendar
Give each date a short label for its calendar -- "AR years", "Elder Scrolls eras", "Hebrew calendar", etc. Use the SAME label for every event in the same calendar so they group together. If two events use DIFFERENT notations but the text states they are the SAME calendar (e.g. "1347, that is, Third Era 347"), give them the same label and put their parts on the same scale.

## CRITICAL: only in-world, stated information
- Use ONLY what the event's date text states. Do NOT use real-world calendar knowledge -- you must NOT convert between calendars using facts you know about the real world (e.g. that a Hebrew year equals some Gregorian year). Only an equivalence the campaign text itself states counts.
- If a date is too vague to become any number ("long ago", "in the early days"), leave that event OUT entirely -- it's handled as an undated event.

## Output -- return ONLY this JSON, nothing else
{"dated": [{"index": <event index>, "system": "<calendar label>", "parts": [<numbers, biggest unit first>]}, ...]}

- "index" is the integer shown in [brackets] before the event.
- Include only events whose date you could turn into parts; omit any you couldn't.
- If none, return {"dated": []}.
- Return ONLY the JSON object -- no preamble, no markdown fences.
"""


PLACEMENT_PROMPT = """\
You are building a timeline for a D&D lore wiki. The dated events are already sorted into one or more TIMELINES, each a list of dated markers with numbered GAPS between them. Your job: take each UNDATED event and choose which gap it belongs in, from the relative-time clues in its description ("before the Empire fell", "after the dynasty collapsed").

You do NOT assign position numbers -- you only choose a gap (by its ID) for each event. The numbering is done by other code.

## How gaps work
Each timeline is shown as gaps and markers in order: [gap S#0], marker, [gap S#1], marker, [gap S#2], ... So gap S#0 is BEFORE the first marker, gap S#1 is BETWEEN the first and second, and the last gap is AFTER the final marker. "Before the Sundering" -> the gap immediately before the Sundering marker; "after the Sundering" -> the gap immediately after it.

## Markers that share a spot (ties)
Two markers can sit at the SAME spot (same date) -- they're shown together on one marker line, e.g. "marker: Founding (1300), Charter (1300)". Because they're at the same spot, "after Founding" and "after Charter" mean the SAME gap (the one right after that shared marker line). Don't be thrown by two names at one spot; treat them as one boundary.

## Several events in one gap
If several undated events fall in the same gap, list them in that gap IN ORDER, earliest first -- so "the feud, which came after the rebellion" puts the rebellion before the feud within their shared gap.

## The undated timeline
There is also an "Undated timeline" for events that have a relative order among THEMSELVES but are NOT tied to any dated marker ("the Aldward rise, then the Borren feud", neither linked to a date). Place those in that timeline's single gap, in order.

## CRITICAL: don't guess
- Place an event ONLY where its clue actually supports. If an event has no usable relative clue and no link to any marker or other event, LEAVE IT OUT entirely -- it goes under "Could Not Place" rather than being guessed into a spot.
- Use only what the descriptions state. No real-world knowledge.

## Output -- return ONLY this JSON, nothing else
{"placements": [{"gap": "<a gap ID from above>", "events": [<event indices in that gap, earliest first>]}, ...]}

- Use the EXACT gap IDs shown (e.g. "AR years#1", "(undated)#0").
- Each undated event goes in AT MOST one gap. Omit any you can't place.
- If you can place none, return {"placements": []}.
- Return ONLY the JSON object -- no preamble, no markdown fences.
"""


class Reconciler(BaseAgent):
    """Phase 4.1a -- de-duplicates a list of ONE entity type. See module docstring."""

    system_prompt = RECONCILER_SYSTEM_PROMPT

    def reconcile(self, entries: list) -> list:
        # 0 or 1 entries can't contain a duplicate -> nothing to do, and don't spend
        # an API call. (Also guards the type inference below against an empty list.)
        if len(entries) < 2:
            return list(entries)

        label = _type_label(entries[0])

        # Ask the LLM for its decision, retrying up to 3x on bad JSON / invalid
        # decision (the 5.3 "retry malformed JSON up to 3 times" rule). The retry
        # has to re-run _validate_decision, not just re-parse, so it's a custom loop
        # even if BaseAgent offers a parse helper.
        decision = None
        for attempt in range(3):
            raw = self.call_claude(self.system_prompt, self._build_user_message(entries))
            decision = self._parse_decision(raw)
            if decision is None:
                logger.warning("Reconciler[%s]: bad decision JSON (attempt %d/3); raw was: %r",
                               label, attempt + 1, raw)
                continue
            problems = _validate_decision(decision, entries)
            if problems:
                logger.warning("Reconciler[%s]: invalid decision (attempt %d/3): %s",
                               label, attempt + 1, "; ".join(problems))
                decision = None
                continue
            break  # clean decision

        if decision is None:
            # Failed 3x -> contain the failure: merge nothing, pass entries through,
            # log loudly. Under-merge is the harmless direction.
            logger.error("Reconciler[%s]: no usable decision after 3 attempts; "
                         "returning %d entries unmerged.", label, len(entries))
            return list(entries)

        return self._apply(decision, entries, label)

    def _build_user_message(self, entries) -> str:
        label = _type_label(entries[0])
        lines = [
            f"These are {label} entries to de-duplicate. Each is prefixed by its "
            f"index in [brackets], counting from 0.\n"
        ]
        for i, e in enumerate(entries):
            lines.append(f"[{i}] {json.dumps(e.model_dump(mode='json'), ensure_ascii=False)}")
        return "\n".join(lines)

    def _parse_decision(self, raw):
        """Parse raw LLM text into a `ReconcileDecision`, or `None` if it can't.

        Tries the fence-stripped text first (the common case), then falls back to
        pulling the first balanced ``{...}`` object out of the raw reply -- Sonnet
        sometimes prefixes a conversational preamble ("Looking at these entries...")
        that `strip_code_fences` (markdown-fence-only) can't peel, which would
        otherwise fail validation and burn all three retries, dropping a real merge.
        Structural sanity (indices/canonical) is still checked separately by
        `_validate_decision`; this only handles getting a parseable object out."""
        for candidate in (strip_code_fences(raw), _extract_json_object(raw)):
            if not candidate:
                continue
            try:
                return ReconcileDecision.model_validate_json(candidate)
            except (ValidationError, ClaudeJSONError, json.JSONDecodeError):
                continue
        return None

    def _apply(self, decision, entries, label):
        """Turn a validated decision into the final list, in pure Python."""
        merged_out = []
        consumed = set()  # indices folded into a merge -> not also emitted as singletons

        for group in decision.merges:
            try:
                members = [entries[i] for i in group.members]
            except IndexError:
                logger.warning(
                    "Reconciler[%s]: skipping merge group with out-of-range member index: %s",
                    label,
                    group.members,
                )
                continue
            # Defense-in-depth: a merge needs >= 2 members. _validate_decision
            # already rejects smaller groups (so this never fires in the real flow),
            # but guarding here means _apply can't be crashed by a malformed decision
            # handed to it directly -- _combine_group's members[0] would IndexError on
            # an empty group. Members stay unconsumed -> fall through as singletons.
            if len(members) < 2:
                logger.warning("Reconciler[%s]: skipping merge group with %d member(s).",
                               label, len(members))
                continue

            # short-name guard: veto a merge joining very short names with no stated-
            # alias link (protects CJ/DJ etc.). Veto -> leave them separate + flag.
            if _short_name_veto(members):
                logger.warning("%s Reconciler[%s]: vetoed short-name merge %s (no stated "
                               "alias); kept separate.", REVIEW_PREFIX, label,
                               [m.name for m in members])
                continue  # members stay unconsumed -> fall through as singletons

            try:
                merged_entry = _combine_group(members, group.canonical)
            except _VetoMerge as veto:
                logger.warning("%s Reconciler[%s]: vetoed merge of %s -- %s; kept separate.",
                               REVIEW_PREFIX, label, [m.name for m in members], veto.message)
                continue

            # merge stuck -> log any detail conflicts (rich + findable); keep BOTH sides.
            for c in group.conflicts:
                logger.warning("%s Reconciler[%s] contradiction in '%s': %r (%s) vs %r (%s) -- %s",
                               REVIEW_PREFIX, label, merged_entry.name,
                               c.detail_a, c.source_a, c.detail_b, c.source_b, c.note)

            merged_out.append(merged_entry)
            consumed.update(group.members)

        # log suspicious-but-unmerged pairs for hand review
        for pd in decision.possible_duplicates:
            pair = [entries[i].name for i in pd.members if 0 <= i < len(entries)]
            logger.warning("%s Reconciler[%s]: possible duplicate, NOT merged: %s -- %s",
                           REVIEW_PREFIX, label, pair, pd.note)

        # everything never folded into a merge passes through unchanged (singletons)
        singletons = [e for i, e in enumerate(entries) if i not in consumed]
        result = merged_out + singletons

        logger.info("Reconciler[%s]: %d entries -> %d (%d merge groups, %d possible-dupes "
                    "flagged).", label, len(entries), len(result),
                    len(decision.merges), len(decision.possible_duplicates))
        return result

    # ----------------------------------------------------------------------- #
    # Phase 4.1b -- the timeline pass (two Claude calls on top of the Brief A
    # engine above). The LLM emits sort keys (Call 1) and gap choices (Call 2);
    # Python owns every chronological_position.
    # ----------------------------------------------------------------------- #

    def order_history(self, events: list) -> list:
        """Phase 4.1b -- the timeline pass. Fills calendar_system + chronological_position
        on a list of (already-deduplicated) HistoryEvents, via two Claude calls on top of
        the pure-Python engine. Call 1 reads each event's stated date into a sortable
        tuple; Python sorts those into a spine + gaps; Call 2 drops the undated events into
        the gaps; Python weaves + stamps. The LLM never emits a position.

        Degrades gracefully: a failed Call 1 -> empty spine (events ordered relatively, or
        Could Not Place); a failed Call 2 -> the dated spine still stamps, only the relative
        events fall to Could Not Place. _weave_and_stamp always runs, so a total LLM failure
        yields a less-ordered -- never corrupted -- timeline."""
        if not events:
            return list(events)

        # Call 1 only runs if something actually states a date.
        dated = []  # (index, system, parts), post-validation + post-guard
        if any(e.date_text and e.date_text.strip() for e in events):
            dated = self._extract_dates(events)

        spines, gap_lookup = _build_spine_and_gaps(dated)
        dated_indices = {idx for idx, _, _ in dated}

        # Call 2 only runs if there's something undated to place.
        placements = {}
        if any(i not in dated_indices for i in range(len(events))):
            placements = self._place_relatives(events, dated_indices, spines, gap_lookup)

        return _weave_and_stamp(events, spines, placements, gap_lookup)

    def _extract_dates(self, events) -> list:
        """Call 1. Returns a list of (index, system, parts) that passed validation AND the
        grounding guard. Retries up to 3x on malformed/invalid JSON; total failure -> []
        (empty spine). A per-event guard failure drops just that event to relative placement."""
        decision = None
        for attempt in range(3):
            raw = self.call_claude(DATE_EXTRACTION_PROMPT, self._build_date_message(events))
            decision = _parse_json_model(raw, DateDecision)
            if decision is None:
                logger.warning("Timeline dates: bad JSON (attempt %d/3); raw: %r", attempt + 1, raw)
                continue
            problems = _validate_date_decision(decision, events)
            if problems:
                logger.warning("Timeline dates: invalid (attempt %d/3): %s",
                               attempt + 1, "; ".join(problems))
                decision = None
                continue
            break
        if decision is None:
            logger.error("Timeline dates: no usable decision after 3 attempts; "
                         "treating all events as undated.")
            return []

        dated = []
        for d in decision.dated:
            if _sanity_guard_parts(d.parts, events[d.index].date_text):
                dated.append((d.index, d.system, d.parts))
            else:
                logger.warning("%s Timeline: parts %r for '%s' (date_text %r) look unmoored "
                               "from the stated date; dropping it to relative placement.",
                               REVIEW_PREFIX, d.parts, events[d.index].name,
                               events[d.index].date_text)
        return dated

    def _place_relatives(self, events, dated_indices, spines, gap_lookup) -> dict:
        """Call 2. Returns {gap_id: [event indices in order]} for _weave_and_stamp. Retries
        up to 3x; total failure -> {} (the relatives fall to Could Not Place; the dated
        spine still stamps)."""
        relative = [i for i in range(len(events)) if i not in dated_indices]
        decision = None
        for attempt in range(3):
            raw = self.call_claude(
                PLACEMENT_PROMPT, self._build_placement_message(events, dated_indices, spines))
            decision = _parse_json_model(raw, PlacementDecision)
            if decision is None:
                logger.warning("Timeline placement: bad JSON (attempt %d/3); raw: %r", attempt + 1, raw)
                continue
            problems = _validate_placement_decision(decision, events, dated_indices, gap_lookup)
            if problems:
                logger.warning("Timeline placement: invalid (attempt %d/3): %s",
                               attempt + 1, "; ".join(problems))
                decision = None
                continue
            break
        if decision is None:
            logger.error("Timeline placement: no usable decision after 3 attempts; "
                         "%d relative event(s) -> Could Not Place.", len(relative))
            return {}
        return {p.gap: p.events for p in decision.placements}

    def _build_date_message(self, events) -> str:
        """Call 1's user message: only the events that actually state a date, each with its
        ORIGINAL index (so the engine maps results back correctly)."""
        lines = ["These history events state a date. For each, give its calendar system and "
                 "sortable parts. Each is prefixed by its index.\n"]
        for i, e in enumerate(events):
            if e.date_text and e.date_text.strip():
                lines.append(f"[{i}] date={e.date_text!r}  (event: {e.name})")
        return "\n".join(lines)

    def _build_placement_message(self, events, dated_indices, spines) -> str:
        """Call 2's user message: each timeline shown as interleaved gaps + markers (markers
        named with their dates so relative clues can match them), then the undated events
        with their descriptions (which carry the relative clues)."""
        lines = ["Place each undated event into a gap on one of the timelines below, using "
                 "the gap's ID.\n"]
        for system, rank_groups in spines.items():
            if system == UNDATED:
                lines.append("\nUndated timeline (events with no date, ordered only relative "
                             "to each other):")
            else:
                lines.append(f"\nTimeline: {system}")
            for k in range(len(rank_groups) + 1):
                lines.append(f"  [gap {_gap_id(system, k)}]")
                if k < len(rank_groups):
                    markers = ", ".join(f"{events[i].name} ({events[i].date_text})"
                                        for i in rank_groups[k])
                    lines.append(f"  -- marker: {markers}")
        lines.append("\nUndated events to place (read each description for its relative order):")
        for i in range(len(events)):
            if i not in dated_indices:
                lines.append(f"  [{i}] {events[i].name}: {events[i].description!r}")
        return "\n".join(lines)
