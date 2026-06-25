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

from pydantic import ValidationError

from agents.base import BaseAgent, ClaudeJSONError, strip_code_fences
from models.lore import Character, HistoryEvent, Scope
from models.reconcile import ReconcileDecision

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
            members = [entries[i] for i in group.members]

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
