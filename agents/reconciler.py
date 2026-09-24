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
from collections import Counter
from typing import Optional

from pydantic import ValidationError

from agents.base import BaseAgent, ClaudeJSONError, loads_tolerant, strip_code_fences
from models.lore import Alias, Character, HistoryEvent, Scope
from text_norm import name_key as _fold_name_key
from models.reconcile import (
    ReconcileDecision, MergeGroup, DateDecision, PlacementDecision, CrossTypeDecision,
)
from player_map import declared_characters, declared_groups, declared_groups_with_players

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
    # Last resort: the same tolerant parse the extractors use (peels a preamble AND
    # re-escapes stray double-quotes), then validate. A ReconcileDecision whose
    # `conflicts` quote verbatim detail text is vulnerable to the very unescaped-quote
    # breakage the extractors hit, so it earns the json_repair fallback too. Pure
    # prose repairs to "" -> loads_tolerant raises -> None (retry preserved).
    try:
        return model_cls.model_validate(loads_tolerant(raw))
    except (ValidationError, json.JSONDecodeError, ValueError):
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


def _dedup_details(details):
    """Collapse Details that state the same fact into one, unioning their sources.

    Replaces the old ``_dedup_preserve_order`` call for details now that details are
    ``Detail`` objects, not strings. On the VISIBLE axis it behaves identically to
    the old string de-dup -- key on the stripped ``text`` (so a subtly-different fact
    is never eaten), keep the first-seen ``Detail`` verbatim, preserve order -- so
    the rendered wiki is byte-for-byte unchanged. The new, invisible part: when the
    same fact arrives from two files, the survivor keeps the ORDERED UNION of every
    file it came from, so the future exclusion pass strips it only when ALL its
    sources are excluded (the same rule the whole-entity case already uses).

    Never mutates an input Detail: a union produces a fresh Detail via model_copy.
    """
    by_key = {}   # stripped text -> surviving Detail (source_files accumulated)
    order = []    # stripped keys, in first-seen order
    for d in details:
        key = d.text.strip()
        if key in by_key:
            survivor = by_key[key]
            merged = _dedup_preserve_order(survivor.source_files + d.source_files)
            by_key[key] = survivor.model_copy(update={"source_files": merged})
        else:
            by_key[key] = d
            order.append(key)
    return [by_key[k] for k in order]


def _dedup_aliases(aliases):
    """Collapse Aliases with the same text into one, unioning their sources.

    The alias twin of ``_dedup_details``, and it replaces the old
    ``_dedup_preserve_order`` call on aliases now that they're objects. Same visible
    behaviour as the old string de-dup -- key on the stripped ``text``, keep the
    first-seen ``Alias`` verbatim, preserve order -- plus the invisible part: an
    alias stated in two files keeps the ORDERED UNION of both, so the carve strips
    it only when ALL its sources are excluded.

    Never mutates an input Alias: a union produces a fresh one via model_copy.
    """
    by_key = {}   # stripped text -> surviving Alias (source_files accumulated)
    order = []    # stripped keys, in first-seen order
    for a in aliases:
        key = a.text.strip()
        if key in by_key:
            survivor = by_key[key]
            merged = _dedup_preserve_order(survivor.source_files + a.source_files)
            by_key[key] = survivor.model_copy(update={"source_files": merged})
        else:
            by_key[key] = a
            order.append(key)
    return [by_key[k] for k in order]


def _entity_for_prompt(entry):
    """Serialise an entry the way the reconciler LLM has ALWAYS seen it: details and
    aliases as bare text strings, with no source tags and no name_sources. The
    provenance fields are plumbing the LLM never saw, so we strip them back out to
    keep the prompt's SHAPE identical -- same keys, same order, same values -- which
    keeps Claude's merge decisions driven by exactly the information they always
    were, and keeps the prompt cache hitting. HistoryEvent has no ``details``, so
    that guard makes it a no-op there.

    Key order survives because `name_sources` is declared right after `name`, so
    popping it leaves {name, aliases, details, supporting_quotes, ...} in the
    original order, and `{**data, k: v}` replaces a key in place.
    """
    data = entry.model_dump(mode="json")
    if "details" in data:
        data = {**data, "details": [d["text"] for d in data["details"]]}
    if "aliases" in data:
        data = {**data, "aliases": [a["text"] for a in data["aliases"]]}
    data.pop("name_sources", None)
    # `prose` is the prose agent's later output (always None at reconcile time,
    # since the prose pass runs AFTER reconcile). Pop it so a "prose": null key never
    # appears in the merge prompt -- keeps the prompt shape byte-identical and the
    # prompt cache hitting, exactly like the name_sources pop above.
    data.pop("prose", None)
    # `pronouns` is declared-party ground truth (display metadata), never a merge signal --
    # pop it so a new key never shifts the merge prompt (keeps the prompt cache hitting,
    # same reason as name_sources/prose). Absent on the other five types, so a no-op there.
    data.pop("pronouns", None)
    return data


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


def _resolve_player_name(members, allow_majority=False):
    """Resolve one player_name across a merge group.

    name-vs-None -> the name (None just means "this batch didn't know"). A clash of
    two or more DIFFERENT real player_names is handled by ``allow_majority``:

    - ``allow_majority=False`` (the identical-name FLOOR path): any clash VETOes. Two
      different humans behind one same-NAMED character is strong evidence the "merge"
      is really two distinct PCs, and the floor merges on name identity alone (no
      semantic evidence), so we refuse rather than fabricate a player.
    - ``allow_majority=True`` (the LLM EVIDENCE-based path): the LLM grouped these on
      real duplicate evidence (shared aliases, stated identity), so one odd
      player_name is far more likely a SINGLE mislabeled record than a genuine two-PC
      collision. A STRICT plurality wins (loudly flagged); only a true tie -- no
      single most-common name -- still VETOes (a real player handoff is the case we
      protect). This backstops the extractor's speaker-grounded null (Fix A), which
      usually removes the bad value before it ever reaches here."""
    names = [m.player_name for m in members if m.player_name]
    distinct = _dedup_preserve_order(names)
    if len(distinct) <= 1:
        return distinct[0] if distinct else None
    if not allow_majority:
        raise _VetoMerge(distinct, f"player_name clash across a merge: {distinct}")
    # Evidence-based merge: let a strict plurality win. Count case-insensitively but
    # keep each name's first verbatim spelling so the winner is returned as written.
    counts = {}
    first_form = {}
    for nm in names:
        key = nm.strip().lower()
        counts[key] = counts.get(key, 0) + 1
        first_form.setdefault(key, nm)
    top = max(counts.values())
    leaders = [k for k, c in counts.items() if c == top]
    if len(leaders) > 1:
        # No single most-common name -> genuinely ambiguous (possible player handoff).
        raise _VetoMerge(distinct, f"player_name clash across a merge with no majority: {distinct}")
    winner = first_form[leaders[0]]
    overridden = [first_form[k] for k in counts if k != leaders[0]]
    logger.warning("%s Reconciler: player_name clash %s in an evidence-based merge -> kept the "
                   "majority %r, overrode %s (likely a mislabeled record).",
                   REVIEW_PREFIX, distinct, winner, overridden)
    return winner


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
    # IDENTITY is not a spelling gamble. The short-name rule guards against merging
    # DIFFERENT short names that merely look alike ("CJ" vs "DJ"). Two entries whose
    # names are literally the SAME string ("Gol"/"Gol") are the strongest possible
    # merge evidence -- there's no similarity to misjudge -- so they must never be
    # vetoed. Without this, four extracted "Gol" locations (identical name, no cross-
    # alias) were vetoed apart into four duplicate pages. Compared normalized
    # (strip+lower) so "Gol"/" gol " count as identical. Only exempts the fully-
    # identical group: a mix like ["Gol", "Golden City"] still faces the stated-
    # alias rule below, because there the short name really might not be the long one.
    if len({nm.strip().lower() for nm in names}) == 1:
        return False
    # A "stated alias" must be a CROSS-member link: one member's name appearing in
    # a DIFFERENT member's alias list. A member self-aliasing (its own name among
    # its own aliases -- which the extractors can emit) is NOT evidence that two
    # short names are the same entity, so it must not satisfy the rule -- else
    # CJ(aliases=['CJ']) + DJ would wrongly merge. Hence the `other is not m` guard.
    stated = any(
        m.name.strip().lower() in {
            a.text.strip().lower()
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
    # Fold via _name_key (the same fold the deterministic floors use) so a straight-vs-
    # curly quote -- or accent/apostrophe/case -- variant still matches its own member.
    # The fold is for LOOKUP only; we always return the verbatim stored string.
    key = _name_key(canonical)
    for m in members:
        if _name_key(m.name) == key:
            return m.name
    for m in members:
        for a in m.aliases:
            if _name_key(a.text) == key:
                return a.text
    return canonical


def _canonical_sources(members, canonical):
    """The source file(s) of the winning canonical name.

    ``_resolve_canonical`` has already snapped `canonical` to a string that
    LITERALLY appears among the members (as a name or an alias) -- an invented
    canonical is rejected by ``_validate_decision`` before we ever get here -- so its
    provenance is always derivable. We just look it up the same way and union across
    every member that asserted it: the same name stated in two files is attested by
    both, so a later exclusion drops it only when ALL of them are excluded (the
    Detail rule again).

    Match is stripped but CASE-SENSITIVE, matching how `_dedup_details` /
    `_dedup_aliases` key. That's deliberate: `_resolve_canonical` returns a verbatim
    stored string, so an exact match is right, and if the LLM picked the
    differently-cased twin ("RIVERTON" over "Riverton") we WANT only that twin's
    sources -- the other spelling becomes an alias carrying its own.

    Returns [] if nothing matches (a defensive path `_resolve_canonical` allows but
    validation should have prevented). [] means "not provably public", so the carve
    re-heads -- the same over-hiding default as a source-less Detail.
    """
    key = canonical.strip()
    sources = []
    for m in members:
        if m.name.strip() == key:
            sources.extend(m.name_sources)
        for a in m.aliases:
            if a.text.strip() == key:
                sources.extend(a.source_files)
    return _dedup_preserve_order(sources)


def _combine_group(members, canonical, allow_majority=False):
    """Merge a list of same-type entries into ONE, given the canonical name the
    LLM picked. Pure + deterministic -- only unions/concatenates fields from the
    ORIGINAL objects, never rephrasing anything. Returns a new entry of the same
    class. May raise _VetoMerge (Characters with clashing player_names).

    ``allow_majority`` is forwarded to `_resolve_player_name`: True for the LLM's
    evidence-based merges (a minority player_name loses to the majority), False for
    the deterministic identical-name floor (any player_name clash still vetoes)."""
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
            # A losing name becomes an alias -- and carries ITS OWN provenance
            # across, which is what lets the carve re-head to it later.
            aliases.append(Alias(text=m.name, source_files=list(m.name_sources)))
    aliases = _dedup_aliases(aliases)
    aliases = [a for a in aliases if a.text != canonical]  # canonical shouldn't alias itself

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
            name_sources=_canonical_sources(members, canonical),
            aliases=aliases,
            description=description,
            scope=_resolve_scope(scopes),
            date_text=_resolve_date_text(members),
            calendar_system=None,         # 4.1b fills this; None at merge time (like chronological_position)
            chronological_position=None,  # always None until the 4.1b timeline pass
            supporting_quotes=quotes,
        )

    # everyone else has a `details` list: concat + exact-dupe-only removal.
    # `_dedup_details` collapses same-text Details (visible behaviour identical to
    # the old string de-dup) while unioning their source_files (the invisible
    # provenance axis for --exclude-sources).
    details = []
    for m in members:
        details.extend(m.details)
    details = _dedup_details(details)

    # Characters carry two scalars to resolve; the other three types don't.
    if isinstance(members[0], Character):
        if len({m.is_pc for m in members}) > 1:
            logger.debug("Reconciler: is_pc clash in '%s' -> True wins.", canonical)
        return cls(
            name=canonical,
            name_sources=_canonical_sources(members, canonical),
            aliases=aliases,
            is_pc=any(m.is_pc for m in members),       # True wins (False = absence of evidence)
            player_name=_resolve_player_name(members, allow_majority),  # may raise _VetoMerge
            details=details,
            supporting_quotes=quotes,
        )

    # Location / Organization / Item / PeopleAndCultures: identical shape.
    return cls(name=canonical, name_sources=_canonical_sources(members, canonical),
               aliases=aliases, details=details, supporting_quotes=quotes)


# --- Cluster 2: deterministic under-merge safety net ----------------------- #
# The reconciler makes ONE giant call per type; at scale (50+ entries) the model
# silently omits valid merges from its decision -- even byte-identical names it is
# explicitly told to merge -- with no log. These two pure helpers backstop that: a
# deterministic floor force-merges identical names AFTER the LLM decision, and an
# advisory candidate list surfaces spelling-close pairs so the model can't skip them.

CANDIDATE_EDIT_THRESHOLD = 2   # names within this edit distance are surfaced (not merged)
# Declared-name typo floor: a name one edit away from a DECLARED character name/alias is
# folded in as a misspelling -- but only when BOTH names are at least this long, so
# initials / very short names (CJ vs DJ) are never fuzzy-matched.
DECLARED_TYPO_MIN_LEN = 4
# A pair of names sharing a DISTINCTIVE word -- at least this long, and held by no more
# than _DISTINCTIVE_TOKEN_MAX_ENTITIES entries of this type -- is surfaced to the LLM to
# adjudicate even when neither name is a subset of the other ("Krieger Family" vs
# "Krieger Royal House" share the distinctive "krieger"). A common word spans many
# entries and is excluded by the frequency cap, so it can't flood the candidate list.
_DISTINCTIVE_TOKEN_MIN_LEN = 4
_DISTINCTIVE_TOKEN_MAX_ENTITIES = 3
_ARTICLES = ("the ", "a ", "an ")


def _name_key(name: str) -> str:
    """Normalized grouping key for a name -- delegates to text_norm.name_key, the SAME
    fold the cross-linker's slugify uses (case / whitespace / apostrophes / accents /
    stray punctuation all folded away). This is load-bearing: two names that will
    become the same crosslink anchor now always share a merge bucket, so an
    apostrophe/accent variant like "Scoia'tael" can't split into several reconciler
    pages that then collide on one dead anchor. Keeps articles -- article-stripping is
    the separate _strip_article layer. Returns "" for an all-punctuation name."""
    return _fold_name_key(name)


def _merge_identical_names(entries, label):
    """Force-merge entries whose NAMES are identical after normalization -- the merge
    the prompt mandates but the giant call sometimes silently omits. Runs on the
    POST-decision result (so no index conflict with the LLM's own merges), preserving
    first-seen order. Skipped for HistoryEvent (event names are model-generated LABELS,
    so two identical labels are not necessarily the same event). A Character
    player_name clash still vetoes via _combine_group -> two different same-named PCs
    stay separate."""
    if not entries or isinstance(entries[0], HistoryEvent):
        return list(entries)
    groups = {}
    order = []
    for e in entries:
        # An un-normalizable name (all punctuation / emoji -> "") gets a per-entity
        # sentinel key so two such junk names never force-merge into one page.
        k = _name_key(e.name) or "\x00{}".format(id(e))
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(e)
    out = []
    for k in order:
        members = groups[k]
        if len(members) < 2:
            out.append(members[0])
            continue
        try:
            merged = _combine_group(members, members[0].name)
        except _VetoMerge as veto:
            logger.warning("%s Reconciler[%s]: identical-name merge of %s vetoed -- %s; "
                           "kept separate.", REVIEW_PREFIX, label,
                           [m.name for m in members], veto.message)
            out.extend(members)
            continue
        logger.info("Reconciler[%s]: deterministic identical-name merge of %d '%s' entries.",
                    label, len(members), members[0].name)
        out.append(merged)
    return out


def _merge_article_variants(entries, label):
    """Force-merge entries whose names are IDENTICAL after stripping a leading article
    ("The Citadel"/"Citadel", "The Kraken Clan"/"Kraken Clan") -- a pure-article difference
    the identical-name floor misses (it keys the full name) and the LLM often leaves split
    across a long list. An un-merged pair survives as two pages AND makes the cross-link
    pass drop their shared surface as ambiguous ("claimed by 2 entities"), so neither
    links. Runs AFTER _merge_identical_names on the post-decision result (first-seen order
    preserved). Merges only on a leading the/a/an, so it CANNOT touch the descriptor
    KIND-trap ("Krieger" vs "Krieger Imperium", which differ by a content word). Skipped
    for HistoryEvent (event names are model-generated LABELS). A Character player_name
    clash still vetoes via _combine_group -> two genuinely-different same-named PCs stay
    separate."""
    if not entries or isinstance(entries[0], HistoryEvent):
        return list(entries)
    groups = {}
    order = []
    for e in entries:
        k = _strip_article(_name_key(e.name)) or "\x00{}".format(id(e))
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(e)
    out = []
    for k in order:
        members = groups[k]
        if len(members) < 2:
            out.append(members[0])
            continue
        # >=2 members here means DIFFERENT full names that collapse to one key only after
        # stripping the article (identical names were already merged upstream) -- a genuine
        # article variant. Merge, keeping the first-seen name as the heading.
        try:
            merged = _combine_group(members, members[0].name)
        except _VetoMerge as veto:
            logger.warning("%s Reconciler[%s]: article-variant merge of %s vetoed -- %s; "
                           "kept separate.", REVIEW_PREFIX, label,
                           [m.name for m in members], veto.message)
            out.extend(members)
            continue
        logger.info("%s Reconciler[%s]: article-variant merge of %d entries -> '%s' (%s).",
                    REVIEW_PREFIX, label, len(members), merged.name,
                    ", ".join(m.name for m in members))
        out.append(merged)
    return out


def _surface_keys(entity):
    """Article-stripped name keys of an entity's name AND every alias -- every surface
    form it answers to. Blank keys (all-punctuation) are dropped."""
    keys = {_strip_article(_name_key(entity.name))}
    keys.update(_strip_article(_name_key(a.text)) for a in entity.aliases)
    keys.discard("")
    return keys


def _share_quote(a, b):
    """True when two entities cite at least one identical supporting quote (the same
    (text, speaker, source_file) triple `_dedup_quotes` keys on) -- the same chat line
    backing both is hard evidence they describe one thing."""
    qa = {(q.text, q.speaker, q.source_file) for q in a.supporting_quotes}
    return any((q.text, q.speaker, q.source_file) in qa for q in b.supporting_quotes)


def _declined_name_pairs(entries, decision):
    """Name pairs the LLM was SHOWN as possible duplicates and did NOT merge -- the
    advisory `_candidate_pairs` list (recomputed deterministically from the same entries,
    so it matches what the user message carried) minus pairs that landed in one merge
    group, plus every pair the LLM parked in `possible_duplicates`. Returned as
    frozensets of article-stripped name keys: a name survives the merges as either a
    name or an alias, so post-merge entities are matched by surface key, not index."""
    group_of = {}
    for g, mg in enumerate(decision.merges):
        for i in mg.members:
            group_of[i] = g
    pairs = set()

    def _add(i, j):
        if not (0 <= i < len(entries) and 0 <= j < len(entries)):
            return
        ki = _strip_article(_name_key(entries[i].name))
        kj = _strip_article(_name_key(entries[j].name))
        if ki and kj and ki != kj:
            pairs.add(frozenset((ki, kj)))

    for i, j, _reason in _candidate_pairs(entries):
        if group_of.get(i, -1 - i) != group_of.get(j, -1 - j):  # not merged together
            _add(i, j)
    for pd in decision.possible_duplicates:
        for a in range(len(pd.members)):
            for b in range(a + 1, len(pd.members)):
                _add(pd.members[a], pd.members[b])
    return pairs


def _merge_unique_subset_names(entries, label, declined_pairs=None):
    """Force-merge a fragment entry into its UNIQUE containing entity ("Mundi" ->
    "Lake Mundi") -- but ONLY on corroborating evidence, never on token shape alone.
    Fold A into B iff ALL of:

      * A's name tokens are a PROPER subset of the tokens of B's name OR one of B's
        aliases, AND B is the ONLY entity of this type whose name/aliases contain A's
        tokens. Zero or >=2 candidate supersets -> skip A. Aliases count, so a superset
        the LLM already merged into another page ("White Tower" now an alias of
        "Citadel") still makes a bare "Tower" ambiguous.
      * EVIDENCE links them: A's name is already one of B's aliases (or B's name one of
        A's), or A and B cite an identical supporting quote. A unique token subset alone
        is NOT identity -- "Free Islands"/"Dwarven Free Islands", "Elves"/"High Elves",
        "Krieger"/"Krieger Imperium" (the KIND trap) all pass the token test and are
        different things.
      * The LLM did NOT decline the pair: a pair it was shown in the candidate advisory
        and left unmerged, or parked in possible_duplicates (`declined_pairs`, from
        `_declined_name_pairs`), is its considered keep-separate call and is respected.

    The fuller/superset name wins the heading; each fragment name becomes an alias.
    Runs AFTER _merge_article_variants on the post-decision result, first-seen order
    preserved. Skipped for HistoryEvent (event names are model-generated LABELS). A
    short fragment (<= SHORT_NAME_LEN, e.g. initials like "CJ") is never folded, and a
    Character player_name clash still vetoes via _combine_group. Every fold is
    [REVIEW]-logged. If a bad class appears the fix is to tighten this rule, never to
    widen it."""
    if not entries or isinstance(entries[0], HistoryEvent):
        return list(entries)
    declined_pairs = declined_pairs or set()
    n = len(entries)
    tokens = [set(_strip_article(_name_key(e.name)).split()) for e in entries]
    # Every surface (name + aliases) of each entity, as token sets, for the superset scan.
    surfaces = [_surface_keys(e) for e in entries]
    surface_toks = [[set(k.split()) for k in s] for s in surfaces]
    # fold_target[i] = j: fragment i folds into its unique containing entity j.
    fold_target = {}
    for i, e in enumerate(entries):
        if not tokens[i]:                               # all-punctuation name -> never a fragment
            continue
        if len(e.name.strip()) <= SHORT_NAME_LEN:       # never fuzzy-fold initials ("CJ")
            continue
        supers = [j for j in range(n)
                  if j != i and any(tokens[i] < ts for ts in surface_toks[j])]  # PROPER subset
        if len(supers) != 1:                            # none, or ambiguous -> skip
            continue
        j = supers[0]
        key_i = _strip_article(_name_key(e.name))
        key_j = _strip_article(_name_key(entries[j].name))
        aliased = key_i in surfaces[j] or key_j in surfaces[i]
        if not (aliased or _share_quote(e, entries[j])):
            continue                                    # token shape alone is not identity
        if any(frozenset((a, b)) in declined_pairs
               for a in surfaces[i] for b in surfaces[j] if a != b):
            logger.info("Reconciler[%s]: unique-subset fold %r -> %r skipped -- the LLM "
                        "declined this pair.", label, e.name, entries[j].name)
            continue
        fold_target[i] = j
    if not fold_target:
        return list(entries)
    # The uniqueness gate makes fold-chains impossible (A<B<C would give A two supersets
    # -> A skipped), so targets and fragments are disjoint. Defensively drop any edge from
    # an index that is itself a target, so a stray chain can never build a bad group.
    targets = set(fold_target.values())
    fold_target = {i: t for i, t in fold_target.items() if i not in targets}
    targets = set(fold_target.values())
    frags = {t: [] for t in targets}
    for i in sorted(fold_target):                       # ascending -> stable fragment order
        frags[fold_target[i]].append(i)
    consumed = set(fold_target)
    out = []
    for i, e in enumerate(entries):
        if i in consumed:
            continue                                    # folded into its target below
        if i in targets:
            members = [entries[i]] + [entries[f] for f in frags[i]]
            try:
                merged = _combine_group(members, entries[i].name)
            except _VetoMerge as veto:
                logger.warning("%s Reconciler[%s]: unique-subset fold of %s vetoed -- %s; "
                               "kept separate.", REVIEW_PREFIX, label,
                               [m.name for m in members], veto.message)
                out.extend(members)
                continue
            for f in frags[i]:
                logger.warning("%s Reconciler[%s]: unique-subset fold %r -> %r.",
                               REVIEW_PREFIX, label, entries[f].name, entries[i].name)
            out.append(merged)
        else:
            out.append(e)
    return out


def _declared_canonical(members, main_name, ordered_names):
    """Heading for a declared-party merge: the user's declared MAIN NAME (original casing),
    even when it only ever surfaced as an alias -- or never surfaced -- in the chat. This is
    user-declared ground truth, so it wins the heading. `_combine_group`'s `_resolve_canonical`
    returns an unmatched canonical UNCHANGED, so a main_name that isn't a verbatim member
    string still becomes the heading (its name_sources are just empty, fine for a heading).
    Falls back to the earliest declared name that DID surface as a member's own `.name`, then
    `members[0].name`, when no main_name is configured (e.g. an old flat-list back-compat
    group still has main_name = its first entry, so the fallback is rarely reached)."""
    if main_name and main_name.strip():
        return main_name
    by_name = {m.name.strip().lower(): m.name for m in members}
    for dn in ordered_names:
        if dn in by_name:
            return by_name[dn]
    return members[0].name


def _strip_aliases(entity, names_lower):
    """Return a copy of `entity` with any alias whose text matches (case-insensitively) a
    name in `names_lower` removed. Used after a player-named PC is folded into its declared
    character, to keep the player's REAL name from surviving as an in-world alias
    ('CJ (aka Hannah)'). Returns the entity unchanged when nothing matches."""
    kept = [a for a in entity.aliases if a.text.strip().lower() not in names_lower]
    if len(kept) == len(entity.aliases):
        return entity
    return entity.model_copy(update={"aliases": kept})


def _merge_declared_characters(entries, label, groups, player_keys=None, possible_dup_names=None,
                               main_names=None, pronouns=None):
    """Deterministically merge the Character entries the user DECLARED as one character
    (grouped under one player in the player_map). Each group's names are aliases of ONE
    character, so any entries whose name OR alias falls in the same group are force-merged.

    Runs on the POST-_apply/POST-identical-floor result, Characters ONLY (`groups` is a
    per-player list of normalized name-lists; empty for other types or no declared party
    -> no-op). Every member carries the SAME authoritative player_name (stamped at
    extraction from the same map), so _combine_group won't veto -- but we still catch it.
    This is the ground-truth backstop for the reconciler's own (LLM + identical-name)
    merging: the user's declaration that Kriggy/Krigius/Ambrose are one character is
    honored even if the model split them.

    `player_keys` (aligned with `groups`; None -> off) enables the PLAYER-NAME FOLD: a
    character named exactly like a declared player (a player_map KEY that owns a declared
    character group) is that player's PC mis-titled with the player's own name (e.g. a
    'Sam' page that should be Krigius, or a 'Conrad' NPC page that should be CJ). It folds
    into that player's declared character and the player's real name is stripped from the
    result's aliases. We do NOT gate on is_pc/player_name: the extractor nulls player_name
    for undeclared characters (config is authoritative), so gating on it would make this
    fold impossible to fire in the real pipeline (the phantom arrives is_pc=True/False,
    player_name=None). Trade-off (chosen by the user): a coincidental NPC sharing a
    player's first name is swept in too -- so every fold is logged loudly [REVIEW]. This
    removes the phantom page AND unblocks prose de-conflation (which otherwise bails
    because the player's name is also an in-world character name).

    `possible_dup_names` (lowercased) are names the LLM parked in `possible_duplicates`
    (looked-at-but-NOT-merged, e.g. 'Gaerin' flagged as Aerin's sibling). The FUZZY folds
    below (typo + token) SKIP any entry the LLM flagged this way, so a deterministic floor
    can't override the model's stated distinctness. The exact + player folds are
    ground-truth and always apply.

    The TOKEN fold: a compound name that CONTAINS a declared bare alias as a whole word
    ('Kriggy' inside 'Kriggy Krieger') folds into that declared character -- the surname
    variant the typo floor (edit-distance-1) and exact match both miss. Bare aliases are
    the single-word declared names >= DECLARED_TYPO_MIN_LEN, so a shared SURNAME ('Krieger'
    is not a bare declared alias) or a short token can't trigger it."""
    if not groups or label != "Character":
        return list(entries)
    group_sets = [set(g) for g in groups]
    # Per group: the bare SINGLE-WORD declared aliases >= DECLARED_TYPO_MIN_LEN (e.g.
    # "kriggy"), used by the token fold. A multi-word declared name ("krigius krieger")
    # contributes no bare token, so its surname can't leak in as a match key.
    group_bare_tokens = [
        {g for g in gs if " " not in g and len(g) >= DECLARED_TYPO_MIN_LEN}
        for gs in group_sets
    ]
    # Each declared player's own name -> the group index(es) it owns. A LIST because a
    # player may own several declared characters (multi-PC) -- a page named after such a
    # player can't be disambiguated, so the player-fold only fires when the list is length 1.
    player_to_gis = {}
    for gi, pk in enumerate(player_keys or []):
        player_to_gis.setdefault(pk, []).append(gi)
    flagged = possible_dup_names or frozenset()

    def _keys(entry):
        return {entry.name.strip().lower()} | {a.text.strip().lower() for a in entry.aliases}

    def _group_index(entry):
        """Return (group_index, reason) or (None, None); reason in {exact, typo, token, player}."""
        keys = _keys(entry)
        for gi, gs in enumerate(group_sets):
            if keys & gs:
                return gi, "exact"
        # Distinctness gate: if the LLM parked any of this entry's names as a
        # possible-but-NOT-merged duplicate, do NOT let the fuzzy folds override it.
        if not (keys & flagged):
            # Declared-name TYPO floor: a name one edit away from a declared name/alias
            # (both >= DECLARED_TYPO_MIN_LEN chars, so initials/short names are never
            # fuzzy-matched) is a misspelling of that declared character -> fold it in
            # ("Kriggius" -> the declared "Krigius"). Anchored ONLY to ground-truth
            # declared names. The typo name is kept as an alias; the merge loop logs it.
            for gi, gs in enumerate(group_sets):
                for k in keys:
                    if len(k) >= DECLARED_TYPO_MIN_LEN and any(
                            len(g) >= DECLARED_TYPO_MIN_LEN and _edit_distance(k, g) == 1
                            for g in gs):
                        return gi, "typo"
            # TOKEN fold: a compound name containing a declared bare alias as a whole word
            # ("Kriggy Krieger" -> the group whose bare alias "kriggy" is one of its tokens).
            name_tokens = {t for t in re.findall(r"[^\W_]+", entry.name.lower())}
            for gi, bare in enumerate(group_bare_tokens):
                if name_tokens & bare:
                    return gi, "token"
        # Ground-truth fold: a character named exactly like a declared player (a
        # player_map KEY owning a declared group) is that player's PC mis-titled with the
        # player's own name -> fold it in. NOT gated on is_pc and NOT gated on a NULL
        # player_name: the extractor nulls player_name for undeclared characters, so gating
        # on it would make this dead code in the real pipeline. The one exception -- an
        # EXPLICIT, contrary player_name (the LLM said a DIFFERENT player plays this one) --
        # vetoes the name fold; it's inert in the real flow (player_name is null there) and
        # just respects a stated contradiction. A coincidental NPC sharing a player's name
        # IS swept in (accepted by the user); the merge loop logs every fold [REVIEW].
        name_l = entry.name.strip().lower()
        gis = player_to_gis.get(name_l)
        if gis:
            pn = (getattr(entry, "player_name", None) or "").strip().lower()
            if pn and pn != name_l:
                return None, None
            if len(gis) == 1:
                return gis[0], "player"
            # The player owns several declared PCs -> can't tell which this page is; leave
            # it separate + [REVIEW] rather than fold it into an arbitrary one.
            logger.warning("%s Reconciler[%s]: character %r is named after a player who has "
                           "%d declared characters; can't disambiguate -- leaving it separate.",
                           REVIEW_PREFIX, label, entry.name, len(gis))
            return None, None
        return None, None

    buckets = {}       # group index -> [members]
    plan = []          # output order: ("solo", entry) | ("group", gi) first-seen
    folded = {}        # group index -> {player-name-str folded in} (to strip from aliases)
    typo_folded = {}   # group index -> {typo name folded in} (logged [REVIEW], kept as alias)
    token_folded = {}  # group index -> {compound name folded in} (logged [REVIEW], kept as alias)
    for e in entries:
        gi, reason = _group_index(e)
        if gi is None:
            plan.append(("solo", e))
        else:
            if gi not in buckets:
                buckets[gi] = []
                plan.append(("group", gi))
            buckets[gi].append(e)
            nm = e.name.strip().lower()
            if reason == "player":
                folded.setdefault(gi, set()).add(nm)   # phantom -> strip player name from aliases
            elif reason == "typo":
                typo_folded.setdefault(gi, set()).add(e.name)
            elif reason == "token":
                token_folded.setdefault(gi, set()).add(e.name)

    def _stamp_pronouns(c, gi):
        """Stamp the declared character's ground-truth pronouns onto a merged/folded entry
        (a fresh list per entity). A no-op when no pronouns are configured for that group."""
        p = pronouns[gi] if pronouns and gi < len(pronouns) else None
        return c.model_copy(update={"pronouns": list(p)}) if p else c

    out = []
    for kind, val in plan:
        if kind == "solo":
            out.append(val)
            continue
        members = buckets[val]
        strip = folded.get(val)
        if len(members) < 2:
            # A lone player-named phantom (no declared sibling character was extracted to
            # fold into). We can't rename it to the declared PC without that PC's original
            # casing, so keep it but surface it loudly for review.
            if strip:
                logger.warning("%s Reconciler[%s]: character %r is named after declared "
                               "player(s) %s but no declared sibling character exists to fold "
                               "it into; leaving it as-is for review.", REVIEW_PREFIX, label,
                               members[0].name, sorted(strip))
            out.append(_stamp_pronouns(members[0], val))
            continue
        main_name = main_names[val] if main_names and val < len(main_names) else None
        canonical = _declared_canonical(members, main_name, groups[val])
        try:
            merged = _combine_group(members, canonical, allow_majority=True)
        except _VetoMerge as veto:
            logger.warning("%s Reconciler[%s]: declared-party merge of %s vetoed -- %s; "
                           "kept separate.", REVIEW_PREFIX, label,
                           [m.name for m in members], veto.message)
            out.extend(members)
            continue
        if strip:
            merged = _strip_aliases(merged, strip)   # keep the player's real name out of aliases
            logger.warning("%s Reconciler[%s]: folded character(s) named after player(s) %s "
                           "into declared character %r (ground-truth player_map fold -- verify "
                           "none is a coincidental NPC sharing a player's name).",
                           REVIEW_PREFIX, label, sorted(strip), merged.name)
        tf = typo_folded.get(val)
        if tf:
            logger.warning("%s Reconciler[%s]: folded likely-misspelling(s) %s into declared "
                           "character %r (one edit from a declared name).",
                           REVIEW_PREFIX, label, sorted(tf), merged.name)
        tk = token_folded.get(val)
        if tk:
            logger.warning("%s Reconciler[%s]: folded compound name(s) %s into declared "
                           "character %r (contains a declared alias as a whole word).",
                           REVIEW_PREFIX, label, sorted(tk), merged.name)
        logger.info("Reconciler[%s]: declared-party merge of %d entries -> %r.",
                    label, len(members), merged.name)
        out.append(_stamp_pronouns(merged, val))
    return out


def flag_subject_bleed(characters) -> None:
    """Log a [REVIEW] line for any Character detail whose LEADING SUBJECT is a DIFFERENT
    extracted character -- the "CJ's fact landed on Aerin's page" backstory bleed the
    Characters extractor can produce (a detail's subject is the LLM's free choice, which no
    code can verify). LOG-ONLY, never drops: the subject of free-form prose can't be proven,
    so a fact naming another character MID-sentence ("cousin of Skjoldr") is legitimate and
    left alone -- only a detail that STARTS with another character's name/alias as a whole
    word ("CJ doesn't think she's lame") is surfaced. Call it AFTER character reconcile +
    cross-type, when the full roster is known."""
    owners = {}   # surface_lower -> {owning character name, ...}
    for c in characters:
        for s in [c.name] + [a.text for a in c.aliases]:
            key = s.strip().lower()
            if key:
                owners.setdefault(key, set()).add(c.name)
    surfaces_longest_first = sorted(owners.items(), key=lambda kv: -len(kv[0]))
    for c in characters:
        own = {c.name.strip().lower()} | {a.text.strip().lower() for a in c.aliases}
        for d in c.details:
            t = d.text.strip()
            tl = t.lower()
            for surf, surf_owners in surfaces_longest_first:
                if len(surf) < 2 or not tl.startswith(surf):
                    continue
                after = t[len(surf):len(surf) + 1]
                if after and after.isalnum():
                    continue                       # not a whole-word prefix ("CJ" in "CJohn")
                if surf in own:
                    break                          # leads with THIS character's own name -> fine
                logger.warning("%s Reconciler[Character]: possible subject-bleed on %r -- a "
                               "detail begins with a different character (%s): %r",
                               REVIEW_PREFIX, c.name, sorted(surf_owners), d.text)
                break                              # at most one flag per detail
        # (no return -- pure logging side effect)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, iterative two-row (no dependency)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _strip_article(key: str) -> str:
    for art in _ARTICLES:
        if key.startswith(art):
            return key[len(art):]
    return key


def _candidate_pairs(entries, limit: int = 50):
    """Advisory list of spelling-close / one-plus-extra-words / shared-distinctive-word
    name pairs for the LLM to explicitly adjudicate -- it NEVER auto-merges (so CJ/DJ-style
    one-letter siblings stay the LLM's call, decided from context). Identical names are
    handled by the deterministic floor, not here. Returns [(i, j, reason), ...], capped at
    `limit`."""
    keys = [_name_key(e.name) for e in entries]
    toks = [set(_strip_article(k).split()) for k in keys]
    # token -> how many entries (of this type) contain it. A token in few entries is
    # "distinctive" enough that two names sharing it deserve a second look even when
    # neither is a subset of the other; a common word spans many entries -> excluded.
    tok_freq = Counter()
    for ts in toks:
        for t in ts:
            tok_freq[t] += 1
    pairs = []
    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = keys[i], keys[j]
            if not ki or not kj or ki == kj:
                continue  # blanks skipped; identical handled by the floor
            reason = None
            # a 1-edit gap in a very short name (initials) is a DIFFERENT name, not a
            # typo, so only the length>=4 names go through the edit-distance branch.
            if min(len(ki), len(kj)) >= 4 and 1 <= _edit_distance(ki, kj) <= CANDIDATE_EDIT_THRESHOLD:
                reason = "one letter off"
            elif toks[i] and toks[j] and (toks[i] < toks[j] or toks[j] < toks[i]):
                reason = "one name is the other plus extra words (article/title/descriptor)"
            else:
                # Neither spelling-close nor subset, but they share a distinctive word --
                # e.g. "Krieger Family" / "Krieger Royal House" share "krieger". Surface
                # for adjudication; the frequency cap keeps a common word from flooding.
                shared = sorted(
                    t for t in (toks[i] & toks[j])
                    if len(t) >= _DISTINCTIVE_TOKEN_MIN_LEN
                    and tok_freq[t] <= _DISTINCTIVE_TOKEN_MAX_ENTITIES
                )
                if shared:
                    reason = "share the distinctive word(s) {}".format(", ".join(shared))
            if reason:
                pairs.append((i, j, reason))
                if len(pairs) >= limit:
                    return pairs
    return pairs


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
        # canonical must appear among the (valid) members' names/aliases. Fold via
        # _name_key (the deterministic floors' fold) so a straight-vs-curly quote variant
        # of a real member name isn't rejected as "invented" -- the CJ bug, where Opus
        # emitted a straight-quote canonical for a curly-quote member and burned 3 retries.
        allowed = set()
        for idx in valid:
            allowed.add(_name_key(entries[idx].name))
            allowed.update(_name_key(a.text) for a in entries[idx].aliases)
        if valid and _name_key(group.canonical) not in allowed:
            problems.append(
                f"merge {gi}: canonical {group.canonical!r} is not a name/alias of any member"
            )
    return problems


def _valid_merge_subset(decision, entries, label):
    """Salvage only the VALID merge groups from a decision that failed whole-decision
    validation, dropping (with a log) the broken ones. Same per-group checks
    `_validate_decision` applies globally -- >= 2 members, no repeat within a group,
    every index in range, no index reused by an already-kept group, canonical is a
    real member name/alias -- but PER GROUP, so one bad group no longer sinks every
    good merge. Returns a fresh `ReconcileDecision` (possible_duplicates preserved).

    This is the degrade path behind the all-or-nothing failure the run exposed: a
    single reused index in a 115-location decision was discarding ~40 correct merges
    and burning a retry. First-kept group wins an index (matches `_validate_decision`
    order); a dropped group frees its indices for a later valid group."""
    n = len(entries)
    seen = set()
    kept = []
    for gi, group in enumerate(decision.merges):
        members = group.members
        problems = []
        if len(members) < 2:
            problems.append(f">= 2 members required, got {len(members)}")
        if len(set(members)) != len(members):
            problems.append("an index is repeated within the group")
        if any(idx < 0 or idx >= n for idx in members):
            problems.append("an index is out of range")
        elif any(idx in seen for idx in members):
            problems.append("an index is already used by a kept group")
        else:
            allowed = set()
            for idx in members:
                allowed.add(_name_key(entries[idx].name))
                allowed.update(_name_key(a.text) for a in entries[idx].aliases)
            if _name_key(group.canonical) not in allowed:
                problems.append(f"canonical {group.canonical!r} is not a member name/alias")
        if problems:
            logger.warning(
                "%s Reconciler[%s]: dropping invalid merge group %d %s: %s",
                REVIEW_PREFIX, label, gi,
                [entries[i].name for i in members if 0 <= i < n], "; ".join(problems),
            )
            continue
        seen.update(members)
        kept.append(group)
    return ReconcileDecision(merges=kept, possible_duplicates=decision.possible_duplicates)


# A merge component larger than this is treated as a suspected runaway CHAIN (one bad
# overlap link stitching unrelated entities together): its merges are DROPPED (members
# fall back to singletons) + a loud [REVIEW]. Under-merge is the safe direction; a
# generous cap so a genuinely-well-aliased entity still merges. Tunable.
MERGE_COMPONENT_CAP = 8


def _coalesce_overlapping_groups(merges, entries, label):
    """Union merge groups that SHARE an index into one connected component BEFORE
    validation, so the LLM listing one entry in two groups -- {A,B} + {B,C}, the "index
    already used in another group" failure -- becomes ONE merge {A,B,C} instead of the
    validator rejecting the overlap and the salvage dropping a legit merge (the Lake
    Mundi+Mundi / Skjoldr / Ambrose losses this class caused). Non-overlapping groups pass
    through untouched. A component whose union exceeds MERGE_COMPONENT_CAP is a suspected
    runaway -> its merges are DROPPED + [REVIEW].

    ONLY groups that would pass validation on their own (>= 2 distinct, in-range members)
    are eligible for union; a BROKEN group (out-of-range / duplicate / < 2) is passed
    THROUGH unchanged, so it can't drag a valid merge into a doomed component -- it's left
    for the existing validate -> salvage machinery to drop on its own. Returns a NEW list of
    MergeGroups; the coalesced components are disjoint, so downstream `_validate_decision` /
    `_apply` work unchanged and the "index reused" rule simply stops firing on real overlaps."""
    if not merges:
        return []
    n = len(entries)

    def _clean(g):
        return (len(g.members) >= 2
                and len(set(g.members)) == len(g.members)
                and all(0 <= m < n for m in g.members))

    clean = [g for g in merges if _clean(g)]
    passthrough = [g for g in merges if not _clean(g)]
    if not clean:
        return list(merges)                # nothing unionable; hand originals to validation

    # Union-find over the clean groups' member indices.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:           # path-compress
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for g in clean:
        for m in g.members[1:]:
            union(g.members[0], m)

    # Group the clean merge-groups by component root (first-seen order = deterministic).
    comp_members = {}   # root -> set(indices)
    comp_groups = {}    # root -> [MergeGroup, ...]
    for g in clean:
        root = find(g.members[0])
        comp_members.setdefault(root, set()).update(g.members)
        comp_groups.setdefault(root, []).append(g)

    out = []
    for root, members in comp_members.items():
        groups = comp_groups[root]
        if len(groups) == 1:
            out.append(groups[0])          # no overlap -> unchanged
            continue
        member_list = sorted(members)
        if len(member_list) > MERGE_COMPONENT_CAP:
            names = [entries[i].name for i in member_list if 0 <= i < n]
            logger.warning("%s Reconciler[%s]: overlapping merge groups union to %d members "
                           "(> cap %d) -- suspected runaway chain; dropping the merge, keeping "
                           "them separate: %s", REVIEW_PREFIX, label, len(member_list),
                           MERGE_COMPONENT_CAP, names)
            continue
        # Canonical = the canonical of the LARGEST constituent group (ties -> first). It is
        # a name/alias of one of ITS members, all of which are in the union, so
        # _validate_decision's "canonical is a member" rule still passes.
        biggest = max(groups, key=lambda g: len(g.members))
        conflicts = [c for g in groups for c in g.conflicts]
        out.append(MergeGroup(members=member_list, canonical=biggest.canonical,
                              conflicts=conflicts))
    out.extend(passthrough)                # broken groups: untouched, validation drops them
    return out


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

# A calendar-system LABEL that betrays an UNRESOLVED present-relative offset: the LLM was
# told to resolve "200 years ago" to an absolute year, but sometimes it instead invents a
# "years ago" system whose parts are the raw offset magnitudes. Those count BACKWARDS
# (bigger = older), so ascending-sorting them renders the timeline in reverse. We detect
# and drop such a system to relative placement rather than show a reversed dated timeline.
_UNRESOLVED_OFFSET_SYSTEM_RE = re.compile(r"years?\s+ago|before\s+present|\bago\b", re.IGNORECASE)


def _gap_id(system: str, k: int) -> str:
    """The ID for gap k of a system. Gap k sits immediately BEFORE rank-group k
    (so gap 0 = before everything, gap m = after the last group). Callers match
    these by EXACT STRING against the gap-lookup dict -- they never parse the id
    back apart, so a system label containing a '#' or space can't break anything."""
    return f"{system}#{k}"


def _sanity_guard_parts(parts, date_text, anchor_relative=False) -> bool:
    """Best-effort check that a parts tuple is grounded in the event's stated date.
    True = usable; False = looks unmoored, so the caller drops this event from the
    dated spine (it falls back to relative placement / Could Not Place).

    Pure Python can't fully verify a WORD-based date ("the Third Age" -> era 3)
    without parsing calendar notation (the era arithmetic we deliberately skip), so:
      - anchor_relative -> the parts were COMPUTED from a present-relative offset
        against the reference year ("200 years ago", ref 1424 -> [1224]), so they
        legitimately share no digit with the source phrase (1224 vs 200). The digit
        check would wrongly reject every such event, so we SKIP it and trust the
        LLM's arithmetic (True). The verbatim date_text still renders, so a bad
        subtraction is at worst a minor mis-sort, never lost/fabricated lore -- the
        same safety posture as the word-date case.
      - date_text has NO numbers -> can't verify -> trust the LLM's local read (True).
      - date_text HAS numbers -> at least one parts value must match one of the
        extracted numeric tokens (the tuple is anchored to a real stated number).
        Zero overlap means the parts are unmoored from the date_text -> reject (False).
    This catches the clear numeric mismatch (e.g. [343] for "342 AR", or a tuple
    sharing none of the date_text's numbers). A near-miss that still shares a number slips
    through, but the verbatim date renders anyway, so the worst case is a minor
    mis-sort, never lost or fabricated lore."""
    if anchor_relative:
        return True
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
      3. parts is a non-empty list of non-negative integers (year 0 is a real date)
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
        # >= 0, not > 0: **year 0 is a real in-world date** (this campaign literally
        # has "Ferridus Krieger [0 to 50]"). Rejecting parts:[0] used to sink the
        # WHOLE date decision through all 3 retries -> every event treated as undated
        # -> the timeline lost its chronological order. Still reject negatives/empty.
        if not d.parts or not all(isinstance(p, int) and p >= 0 for p in d.parts):
            problems.append(f"dated: index {d.index} has invalid parts {d.parts!r}")
        if not d.system or not d.system.strip():
            problems.append(f"dated: index {d.index} has an empty system label")
    return problems


def _valid_dated_subset(decision, events):
    """Salvage only the VALID `DatedEvent`s from a DateDecision that failed
    whole-decision validation -- the per-entry version of `_validate_date_decision`
    (index in range, not repeated, the event HAS a date_text, parts a non-empty list
    of non-negative ints, non-empty system). Drops (with a log) only the bad entries,
    so one malformed date can't undate the whole campaign. Returns a `list[DatedEvent]`
    (the same objects), first-kept wins on a repeated index."""
    n = len(events)
    seen = set()
    kept = []
    for d in decision.dated:
        problems = []
        if d.index < 0 or d.index >= n:
            problems.append("index out of range")
        elif d.index in seen:
            problems.append("index repeated")
        else:
            dt = events[d.index].date_text
            if not (isinstance(dt, str) and dt.strip()):
                problems.append("event has no date_text")
            if not d.parts or not all(isinstance(p, int) and p >= 0 for p in d.parts):
                problems.append(f"invalid parts {d.parts!r}")
            if not d.system or not d.system.strip():
                problems.append("empty system label")
        if problems:
            name = events[d.index].name if 0 <= d.index < n else "?"
            logger.warning("%s Timeline dates: dropping invalid dated entry (index %r, '%s'): %s",
                           REVIEW_PREFIX, d.index, name, "; ".join(problems))
            continue
        seen.add(d.index)
        kept.append(d)
    return kept


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


def _coalesce_placements(placements):
    """Merge repeated gap IDs into one entry per gap, concatenating their event lists
    (first-seen order, de-duplicated within the gap). The LLM sometimes emits the SAME gap
    in two GapPlacement objects when several events share a gap; that is benign -- combine
    them rather than treating it as a fatal 'gap listed more than once' (the error that
    burned all three retries in the run and stranded every reign event). Returns a list of
    (gap_id, [event indices]) in first-seen gap order."""
    order = []
    merged = {}
    for p in placements:
        if p.gap not in merged:
            merged[p.gap] = []
            order.append(p.gap)
        for idx in p.events:
            if idx not in merged[p.gap]:
                merged[p.gap].append(idx)
    return [(g, merged[g]) for g in order]


def _validate_placement_decision(decision, events, dated_indices, gap_lookup) -> list:
    """Structural sanity-check on Call 2's PlacementDecision BEFORE weaving. Returns
    a list of problems; empty == clean. (Brief B retries on any problem.) Runs on the
    COALESCED placements, so a repeated gap ID is no longer a failure -- it is combined
    (see _coalesce_placements). What remains fatal:
      1. every `gap` is a real gap ID Python handed out (in gap_lookup)
      2. every event index is in range
      3. every placed event is a RELATIVE event -- NOT one of the dated_indices (a
         dated event's position is Python's, not the LLM's to move)
      4. no event index appears in more than one DIFFERENT gap (no double-placement;
         the same event repeated within one gap is de-duped by coalescing, not flagged)
    """
    problems = []
    n = len(events)
    seen_events = set()
    for gap, evs in _coalesce_placements(decision.placements):
        if gap not in gap_lookup:
            problems.append(f"placement: unknown gap {gap!r}")
        for idx in evs:
            if idx < 0 or idx >= n:
                problems.append(f"placement: event index {idx} out of range 0..{n - 1}")
                continue
            if idx in dated_indices:
                problems.append(f"placement: event {idx} is a dated event; can't be placed by gap")
            if idx in seen_events:
                problems.append(f"placement: event {idx} placed in more than one gap")
            seen_events.add(idx)
    return problems


def _valid_placement_subset(decision, events, dated_indices, gap_lookup) -> dict:
    """Salvage only the VALID placements from a decision that failed whole-decision
    validation -- the per-gap version of `_validate_placement_decision`. A gap with an
    unknown ID is dropped whole; within a kept gap, an event index that is out of range,
    is a dated event, or was already placed in another gap is dropped; the rest are kept.
    So one bad gap/event no longer strands every relative event in Could Not Place (the
    all-or-nothing failure the run exposed). Returns {gap_id: [event indices]} ready for
    _weave_and_stamp."""
    n = len(events)
    seen_events = set()
    kept = {}
    for gap, evs in _coalesce_placements(decision.placements):
        if gap not in gap_lookup:
            logger.warning("%s Timeline placement: dropping unknown gap %r (%d event(s)).",
                           REVIEW_PREFIX, gap, len(evs))
            continue
        good = []
        for idx in evs:
            if idx < 0 or idx >= n:
                logger.warning("%s Timeline placement: dropping out-of-range event index %r.",
                               REVIEW_PREFIX, idx)
                continue
            if idx in dated_indices:
                logger.warning("%s Timeline placement: dropping event %r ('%s') -- it is dated, "
                               "not a relative event.", REVIEW_PREFIX, idx, events[idx].name)
                continue
            if idx in seen_events:
                logger.warning("%s Timeline placement: dropping event %r ('%s') -- already placed "
                               "in another gap.", REVIEW_PREFIX, idx, events[idx].name)
                continue
            seen_events.add(idx)
            good.append(idx)
        if good:
            kept[gap] = good
    return kept


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

Strong evidence to merge (these ARE strong -- put them in "merges", NOT "possible_duplicates"):
- The text actually states the alias (e.g. "Lake Mundi, also called the Great Well").
- One entry's name already appears in another entry's "aliases" list.
- A clear MISSPELLING: two names differ by only a letter or two and NOTHING marks them as distinct things (see the spelling section). This is a merge, not a maybe -- e.g. "Maltaav"/"Maltraav" (the same place, one letter off).
- A longer name that is plainly the SAME entity plus an article, a title, or a descriptor (see the trap section) -- e.g. "The Imperium"/"Krieger Imperium", "Krieger family"/"Krieger royal family".
- The SAME referent under two descriptions that share NO words, when the entries clearly describe one and the same thing. This happens two ways: (a) a shared defining tie -- the same owner/subject AND the same kind of thing (e.g. "CJ's family adventuring agency" and "the adventuring agency run by CJ's parents" are ONE agency; "the guard Kriggy travels with" and "Skjoldr, Kriggy's bodyguard" are ONE person); or (b) two names the text uses for one thing (e.g. "the old fort" and "Blackspire Keep" for one keep). Merge only when the descriptions unambiguously point to a SINGLE referent of the same KIND; if you are not sure they are the same one, leave them separate and report a possible_duplicate. (Extractor batches are kept file-pure, so two files describing the same thing under different names arrive as separate entries with no shared alias -- recognizing them as one referent is your job here.)
- CAUTION on the shared-tie rule above -- a GUARD and the person they GUARD are TWO DIFFERENT people, never one. "the young prince who is guarded by a bodyguard" (the protected) and "the bodyguard who protects the prince" (the protector) are separate entities; the tie "bodyguard of the prince" identifies the GUARD, and the prince is someone else. Do NOT merge a protector with the protected -- likewise a servant with their master, or a mount with its rider. Their backstories are different people's backstories.

Weak evidence (do NOT merge on this alone -- report as a possible duplicate):
- Two entries just SOUND like they might be the same, but nothing ever says so AND there is no clear misspelling or shared-entity descriptor linking them.

"possible_duplicates" is for the genuinely UNCERTAIN pair -- not a place to park a clear typo or an obvious article/descriptor variant. If the evidence above is present, MERGE.

## Similar spelling means LOOK HARDER, never an automatic merge
Names close in spelling are a prompt to investigate, not a decision. Use the surrounding facts:
- Do both names appear at the same time as DIFFERENT things? (e.g. "CJ and her sister DJ" -> two different people. Do NOT merge.)
- Does the text state a relationship or distinction between them? (e.g. "DJ is CJ's sister" -> two entities. Do NOT merge.)
- Do the facts read as ONE entity described twice, or TWO coherent separate entities?
A real misspelling shows NONE of these distinctness signals -- that is what makes it a typo. If you see ANY of them, keep the entries separate.

## Very short names (initials) -- strict rule
For very short names (about 3 characters or fewer, like "CJ", "DJ", "AJ"), do NOT merge on spelling SIMILARITY at all. One different letter in a 2-3 letter name is a totally different name, not a typo. Only merge two DIFFERENT short names if the text EXPLICITLY states they are the same (or one already appears in the other's aliases).

This strict rule is about names that merely LOOK alike. It does NOT apply to names that are IDENTICAL: two or more entries with the exact same name ("Gol" and "Gol", "Pyo" and "Pyo") are the same entity and you MUST merge them, no matter how short the name is -- unless the surrounding facts clearly describe two genuinely different things (the same distinctness signals as above). An identical name is the strongest evidence of a duplicate, not a spelling gamble.

## Watch the "almost the same name but different thing" trap
Some names are nearly identical but refer to DIFFERENT KINDS of entity -- a person vs. the empire named after them. "Krieger" (a person/family) and "Krieger Imperium" (an empire) are NOT the same entity: one is a person, the other a realm. Keep those apart.

But do NOT let this rule block a merge when a longer and a shorter name clearly name the SAME entity and the extra words are just an article, a title, or a descriptor:
- "The Imperium" and "Krieger Imperium" -> the same empire (article/name variant). MERGE.
- "Krieger family" and "Krieger royal family" -> the same family (descriptor). MERGE.
- "Free Islands" and "Dwarven Free Islands" -> the same territory. MERGE.
- Two names for the SAME realm or group that differ only by an interchangeable SYNONYM or translation of one word -- "Krieger Empire" and "Krieger Imperium" (Empire = Imperium), "the Free City" and "the Free Town" -- are one entity. MERGE. (Unlike the person-vs-realm trap above, the KIND is the same here; only a synonym word differs.)
- A bare group name and that same group named by its homeland or origin, when the lore itself ties them together -- "Orcs" and "Orcs of the frozen wild" where the text says the orcs come from the frozen wild -- is ONE people described broadly and by where they live, NOT a new subgroup. MERGE.
- A DEMONYM and "the people of <that place>" name ONE people -- "the Ohwadians" and "the people of Owhad", "the Krieg" and "the people of Kriega" are the same group under a demonym vs. a descriptive phrase. MERGE.
The test is KIND, not length: two names for the ONE same thing merge; a person and the empire named after them stay apart even though one name contains the other.

## Worked examples -- follow these closely
- "CJ" and "DJ": the text says DJ is CJ's sister. -> DO NOT MERGE. Two different people whose names happen to differ by one letter.
- "Maltaav" and "Maltraav": the same place, one just missing a letter, and nothing in the text treats them as two things. -> MERGE. Canonical: "Maltraav" (the correctly-spelled / more frequent form).
- "CJ's family adventuring agency" and "the adventuring agency of CJ's parents": same owner (CJ's parents/family) and same kind of thing (an adventuring agency), with nothing marking them as two different agencies. -> MERGE. The shared defining tie is strong evidence even though the two phrasings share no proper name. Pick whichever existing name reads best as the heading.

## Picking the canonical name (for entries you DO merge)
The canonical name MUST be one of the names or aliases that already appears among the entries you are merging -- NEVER a new name you invent. Prefer, in order:
1. A REAL name over an ASSUMED one. If the text indicates that one name is the entity's real, true, actual, or birth name and another is an assumed name -- an alias, cover, fake name, or one they merely "go by" or "give to others" -- pick the REAL name as canonical, EVEN IF the assumed name appears more often in the quotes. (A character travelling under a false name is still canonically their real self; the false name becomes an alias.)
2. A real proper name over a vague descriptive phrase ("Lake Mundi" over "the big lake up north").
3. Among names of equal standing (no real-vs-assumed signal), the one used most often across the supporting quotes.
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
You are building a timeline for a D&D lore wiki. You are given history events that each state an in-world DATE -- either an explicit date ("342 AR") or a present-relative offset ("200 years ago"). For each one, identify (a) which calendar SYSTEM the date is in, and (b) the date as a sortable list of numeric PARTS.

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

STRONGLY DEFAULT TO A SINGLE CALENDAR SYSTEM. Put every dated event on ONE shared scale and label unless the campaign clearly uses genuinely different, incompatible dating schemes that cannot be placed on one axis. In particular, a run of rulers' reigns or regnal years that counts CONTINUOUSLY -- each reign's years follow on from the previous one with NO reset to 0 per ruler (e.g. one reign is "0 to 50", the next "51 to 100", the next "151 to 200") -- is ONE calendar, not several: give them all the same label and place their parts on the same scale. A founding or epoch anchor stated as "Year 0" (e.g. "the Citadel was founded in Year 0") -- the zero point the same-scale reign years or era counts are measured FROM -- is part of that SAME system: put it on the shared scale with parts [0]. Do NOT spin a Year-0 or founding date off into its own one-event system; it belongs at the START of the main timeline, and giving it a separate system would render it as its own tiny timeline and bury the earliest event mid-page. Only introduce a SECOND system when two dates truly cannot be ordered on one axis (two unrelated calendars the text never equates).

## Present-relative offsets and the reference year
Some events give their date as an offset from the present ("200 years ago", "40 years ago", "two centuries ago") instead of an absolute year. Resolve these against a REFERENCE YEAR -- the campaign's present day:
- Find the reference year. If one is provided in the input as "Reference year (present-day), from config: N", USE THAT -- it overrides everything. Otherwise, if an event states the campaign's current/present/starting year (e.g. "the party is in 1424", "the starting year of 1424"), use that. If you can determine NO reference year, leave every present-relative-offset event OUT (it is handled as an undated event).
- Resolve by SUBTRACTING the offset from the reference year: reference 1424, "200 years ago" -> parts [1224]; "40 years ago" -> [1384]. Read written-out numbers ("two centuries ago" -> 200 years). Put the result in `parts`, on the SAME scale/label as the absolute dates in that calendar (e.g. "AR years"), and set "anchor_relative": true for that event.
- Report the reference year you used in the top-level "reference_year" (and its calendar in "reference_system"), or null if you used none.
- NEVER create a calendar system named "years ago" / "N years ago" / "before present" with the raw offset as its parts. A raw offset counts BACKWARDS (a bigger number is an OLDER event), so it would sort the timeline in REVERSE. Either resolve it to an absolute year on the shared scale (subtract from the reference year, as above), or -- if you have no reference year -- leave the event out entirely.
- A DURATION ("a 200-year war", "30 years of fighting") is NOT a present-relative offset -- it is a span, not a point in time. A clue relative to another EVENT ("before the Empire fell") is NOT one either. Leave both out.

## CRITICAL: only in-world, stated information
- Use ONLY what the event's date text states, plus a reference year that is either provided as config or stated in the events. Do NOT use real-world calendar knowledge -- you must NOT convert between calendars using facts you know about the real world (e.g. that a Hebrew year equals some Gregorian year). Only an equivalence the campaign text itself states counts.
- If a date is too vague to become any number ("long ago", "in the early days"), leave that event OUT entirely -- it's handled as an undated event.

## Output -- return ONLY this JSON, nothing else
{"dated": [{"index": <event index>, "system": "<calendar label>", "parts": [<numbers, biggest unit first>], "anchor_relative": <true if resolved from a present-relative offset, else false>}, ...], "reference_year": <the reference year you used, or null>, "reference_system": "<its calendar label, or null>"}

- "index" is the integer shown in [brackets] before the event.
- "anchor_relative" defaults to false; set it true ONLY for an event you resolved by subtracting an offset from the reference year.
- Include only events whose date you could turn into parts; omit any you couldn't.
- If none, return {"dated": [], "reference_year": null, "reference_system": null}.
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
- The dated MARKERS shown in each timeline are already placed, FIXED reference points -- they are context only. NEVER choose a gap for a marker (a dated event); only the UNDATED events explicitly offered for placement get a gap.
- Use only what the descriptions state. No real-world knowledge.

## Output -- return ONLY this JSON, nothing else
{"placements": [{"gap": "<a gap ID from above>", "events": [<event indices in that gap, earliest first>]}, ...]}

- Use the EXACT gap IDs shown (e.g. "AR years#1", "(undated)#0").
- Each undated event goes in AT MOST one gap. Omit any you can't place.
- If you can place none, return {"placements": []}.
- Return ONLY the JSON object -- no preamble, no markdown fences.
"""


# --------------------------------------------------------------------------- #
# Cross-type resolution. The six extractors run independently, so ONE real entity
# is sometimes captured under several types (a people also extracted as a location).
# An LLM arbiter picks the correct type(s); Python folds the losers' facts into the
# winner and drops the wrong-type pages. Same LLM-decides / Python-assembles split as
# 4.1a: the model only names winning TYPES, never touches a fact.
# --------------------------------------------------------------------------- #

# Iteration + winner-tiebreak order (a legit realm keeps locations over organizations,
# and a place beats a people on a tie); also the noun types the arbiter ever considers.
_CROSS_TYPE_ORDER = ("locations", "organizations", "people", "characters", "items")
# The ONE intended cross-type dual: a realm is both a place (locations) and a governing
# body (organizations). A cluster confined to these is left alone, never arbitrated.
_CROSS_TYPE_LEGIT_DUAL = frozenset({"locations", "organizations"})


CROSS_TYPE_PROMPT = """\
You are the type arbiter for a D&D lore wiki. Six extractors ran independently, so the SAME real thing was sometimes captured under MORE THAN ONE type. For each group below, decide which type(s) correctly classify that one real entity, based ONLY on the facts shown.

The types and what each means:
- locations: a PLACE -- a realm, country, region, city, landmark, or territory (its geography, where it sits).
- organizations: a structured BODY -- a government, guild, order, council, company, army, or a noble house spoken of as a power (who runs it, how it is structured).
- people: a PEOPLE or CULTURE -- a race, species, ethnic group, tribe, clan, or the people of a nation, defined by shared ancestry or heritage.
- characters: a single INDIVIDUAL -- one named person or one creature.
- items: a physical OBJECT a person could pick up or carry.

## A realm is BOTH a place and a power
A realm/nation/kingdom legitimately exists as BOTH a "locations" entry (its territory) AND an "organizations" entry (its government). If a realm is captured under both, KEEP BOTH.

## How to choose
Pick the type(s) that match what the entity actually IS, from its facts:
- A resistance movement, clan, tribe, or race captured as a "locations" is misfiled -> it is "people" (or "organizations" if it is a formal body), not a place.
- A territory or realm captured as a "people" is misfiled -> it is "locations" (and maybe "organizations"), not a people.
- An object, place, or group captured as an "items" is almost always misfiled.
- "organizations" vs "people": a formal, structured BODY (a government, guild, order, army, resistance cell) is an "organizations"; the PEOPLE themselves (a race, tribe, clan, the folk of a nation) are a "people". A named group is usually ONE of these -- pick the better fit; keep BOTH only when the facts clearly describe a formal governing body AND a distinct people/culture.
Keep only the type(s) that genuinely fit; the rest are dropped and their facts folded into what you keep. PREFER a SINGLE type -- the only routine two-type case is the realm (locations + organizations, above).

## Output -- return ONLY this JSON, nothing else
{"choices": [{"cluster": <the [N] id>, "keep_types": ["<type>", ...]}, ...]}
- "keep_types" must be a NON-EMPTY subset of the types actually shown for that cluster. Prefer ONE type; use several only for a realm dual or when the facts genuinely support more than one.
- Give one choice per cluster. If you truly cannot tell, keep ALL of that cluster's types (the safe answer).
- Return ONLY the JSON object -- no preamble, no markdown fences.
"""


def _cross_type_fact_count(entity) -> int:
    """How much evidence an entity carries -- details + supporting quotes. Used to pick
    the most-authoritative winner among the kept types (tie broken by _CROSS_TYPE_ORDER)."""
    return len(entity.details) + len(entity.supporting_quotes)


def _fold_cross_type_facts(winner, losers):
    """Return a copy of `winner` with each loser's details/aliases/quotes unioned in
    (deduped with the same helpers 4.1a uses). The losers are the SAME real entity
    misfiled under another type, so their facts are the winner's facts -- we keep the
    lore and just drop the duplicate page. Never mutates the inputs."""
    details = list(winner.details)
    aliases = list(winner.aliases)
    quotes = list(winner.supporting_quotes)
    for l in losers:
        details.extend(l.details)
        aliases.extend(l.aliases)
        quotes.extend(l.supporting_quotes)
    return winner.model_copy(update={
        "details": _dedup_details(details),
        "aliases": _dedup_aliases(aliases),
        "supporting_quotes": _dedup_quotes(quotes),
    })


def _validate_cross_type_decision(decision, clusters) -> list:
    """Light sanity-check on the arbiter reply (retry on any problem, like 4.1a). Lenient
    on purpose -- the apply step already filters unknown types out of keep_types; we only
    reject a choice that points at an unknown cluster or keeps NONE of that cluster's
    actual types (a sign the model misread the task)."""
    problems = []
    n = len(clusters)
    for c in decision.choices:
        if c.cluster < 0 or c.cluster >= n:
            problems.append(f"choice references unknown cluster {c.cluster}")
            continue
        tmap = clusters[c.cluster][1]
        if not c.keep_types:
            problems.append(f"cluster {c.cluster}: empty keep_types")
        elif not (set(c.keep_types) & set(tmap)):
            problems.append(
                f"cluster {c.cluster}: keep_types {c.keep_types} include none of {sorted(tmap)}")
    return problems


class Reconciler(BaseAgent):
    """Phase 4.1a -- de-duplicates a list of ONE entity type. See module docstring."""

    system_prompt = RECONCILER_SYSTEM_PROMPT

    def __init__(self, player_map=None, **kwargs):
        # The reconciler makes ONE call over an ENTIRE one-type list (no batching),
        # so its decision reply scales with the number of merge groups + any
        # conflicts (which carry verbatim detail sentences). BaseAgent's default
        # max_tokens=4096 can truncate a large decision mid-JSON; a truncated reply
        # has no balanced top-level object, so all three retries re-truncate and
        # reconcile() falls through to "return everything UNMERGED" -- i.e. every
        # duplicate in that type silently becomes its own page. Match the extractors'
        # 8192 headroom. setdefault, so an explicit caller value still wins.
        kwargs.setdefault("max_tokens", 8192)
        super().__init__(**kwargs)
        # The declared party -> one DeclaredCharacter per PC (multi-PC: a player appears in
        # several), for the deterministic declared-merge floor (Characters only). Empty when
        # no party is configured. Aligned lists: recognition names (incl. the last-name
        # cross-product), the owning player key (so a page mis-named after a player folds
        # into that player's PC), the canonical DISPLAY name (the heading), and the declared
        # pronouns (stamped onto the merged Character as ground truth).
        self._declared = declared_characters(player_map or [])
        self._declared_groups = [dc.recognition for dc in self._declared]
        self._declared_player_keys = [dc.player.strip().lower() for dc in self._declared]
        self._declared_main_names = [dc.main_name for dc in self._declared]
        self._declared_pronouns = [dc.pronouns for dc in self._declared]

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
        last_parsed = None  # last decision that PARSED (may be structurally invalid)
        for attempt in range(3):
            raw = self.call_claude(self.system_prompt, self._build_user_message(entries))
            parsed = self._parse_decision(raw)
            if parsed is None:
                logger.warning("Reconciler[%s]: bad decision JSON (attempt %d/3); raw was: %r",
                               label, attempt + 1, raw)
                continue
            # UNION groups the LLM overlapped (one entry listed in two groups) into single
            # components BEFORE validating -- so the "index already used" failure stops
            # dropping legit merges (Lake Mundi+Mundi, Skjoldr, Ambrose). Disjoint by
            # construction, so validation + apply below are unchanged.
            parsed = parsed.model_copy(update={
                "merges": _coalesce_overlapping_groups(parsed.merges, entries, label)})
            last_parsed = parsed
            problems = _validate_decision(parsed, entries)
            if problems:
                logger.warning("Reconciler[%s]: invalid decision (attempt %d/3): %s",
                               label, attempt + 1, "; ".join(problems))
                continue
            decision = parsed
            break  # clean decision

        if decision is None:
            if last_parsed is None:
                # Never even parsed a decision -> merge nothing, pass entries through,
                # log loudly. Under-merge is the harmless direction -- but still apply
                # the deterministic identical-name floor, which is exactly the net for
                # when the LLM gave us nothing usable.
                logger.error("Reconciler[%s]: no usable decision after 3 attempts; "
                             "returning %d entries unmerged.", label, len(entries))
                return _merge_declared_characters(
                    _merge_identical_names(list(entries), label), label,
                    self._declared_groups, self._declared_player_keys,
                    main_names=self._declared_main_names, pronouns=self._declared_pronouns)
            # Parsed but never fully clean. Instead of discarding EVERY merge over one
            # bad group (the all-or-nothing failure that stranded ~40 good merges in a
            # 115-entry run), salvage the valid groups and drop only the broken ones.
            decision = _valid_merge_subset(last_parsed, entries, label)
            logger.warning("%s Reconciler[%s]: no fully-clean decision after 3 attempts; "
                           "applying %d salvageable merge group(s).",
                           REVIEW_PREFIX, label, len(decision.merges))

        # Deterministic floors over the LLM's result: (1) force-merge any same-name
        # entries the model silently left separate (a no-op when it merged everything),
        # (2) force-merge pure-article variants ("The Citadel"/"Citadel") the model left
        # split, (3) collapse a name-fragment into its UNIQUE containing entity
        # ("Mundi" -> "Lake Mundi") when an alias or shared quote corroborates it and the
        # model didn't decline the pair, then (4)
        # force-merge the Character entries the user DECLARED as one character in the
        # player_map. All run on the merged output, so no index conflict.
        merged = _merge_identical_names(self._apply(decision, entries, label), label)
        merged = _merge_article_variants(merged, label)
        merged = _merge_unique_subset_names(
            merged, label, declined_pairs=_declined_name_pairs(entries, decision))
        # Names the LLM parked as possible-but-NOT-merged duplicates (indices into the
        # ORIGINAL entries; a name survives the merges as either a name or an alias). The
        # declared floor's fuzzy folds (typo/token) skip these so a deterministic merge
        # can't override the model's stated distinctness (e.g. 'Gaerin' vs 'Aerin').
        flagged = {entries[i].name.strip().lower()
                   for pd in decision.possible_duplicates
                   for i in pd.members if 0 <= i < len(entries)}
        return _merge_declared_characters(merged, label, self._declared_groups,
                                          self._declared_player_keys, possible_dup_names=flagged,
                                          main_names=self._declared_main_names,
                                          pronouns=self._declared_pronouns)

    def _build_user_message(self, entries) -> str:
        label = _type_label(entries[0])
        lines = [
            f"These are {label} entries to de-duplicate. Each is prefixed by its "
            f"index in [brackets], counting from 0.\n"
        ]
        for i, e in enumerate(entries):
            lines.append(f"[{i}] {json.dumps(_entity_for_prompt(e), ensure_ascii=False)}")

        # Advisory: surface spelling-close / one-plus-extra-words pairs so the model
        # can't silently skip a merge it should make. This rides in the USER message
        # only -- RECONCILER_SYSTEM_PROMPT stays byte-identical, so the prompt cache is
        # untouched. It never auto-merges: the model still decides each pair from the
        # facts (so one-letter-apart siblings like CJ/DJ stay separate).
        candidates = _candidate_pairs(entries)
        if candidates:
            lines.append(
                "\nPOSSIBLE DUPLICATES TO ADJUDICATE -- these entries have similar names "
                "and MAY be duplicates. For EACH pair below, explicitly decide merge or "
                "keep-separate using the entries' facts (do not skip any; a real "
                "misspelling of the same thing merges, two genuinely different things "
                "that just sound alike do not):")
            for i, j, reason in candidates:
                lines.append(f'- [{i}] "{entries[i].name}" vs [{j}] "{entries[j].name}" ({reason})')
        return "\n".join(lines)

    def _parse_decision(self, raw):
        """Parse raw LLM text into a `ReconcileDecision`, or `None` if it can't.

        Tries the fence-stripped text first (the common case), then falls back to
        pulling the first balanced ``{...}`` object out of the raw reply -- Sonnet
        sometimes prefixes a conversational preamble ("Looking at these entries...")
        that `strip_code_fences` (markdown-fence-only) can't peel, which would
        otherwise fail validation and burn all three retries, dropping a real merge.
        Structural sanity (indices/canonical) is still checked separately by
        `_validate_decision`; this only handles getting a parseable object out.
        Delegates to the shared `_parse_json_model` so it also inherits the
        json_repair fallback (a conflicts-heavy decision can carry unescaped quotes)."""
        return _parse_json_model(raw, ReconcileDecision)

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
                # LLM evidence-based merge -> a minority player_name loses to the
                # majority (allow_majority=True); only a genuine tie still vetoes.
                merged_entry = _combine_group(members, group.canonical, allow_majority=True)
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

    def order_history(self, events: list, current_year: Optional[int] = None) -> list:
        """Phase 4.1b -- the timeline pass. Fills calendar_system + chronological_position
        on a list of (already-deduplicated) HistoryEvents, via two Claude calls on top of
        the pure-Python engine. Call 1 reads each event's stated date into a sortable
        tuple; Python sorts those into a spine + gaps; Call 2 drops the undated events into
        the gaps; Python weaves + stamps. The LLM never emits a position.

        `current_year` is the campaign's present-day reference year (config/CLI override).
        When set it is handed to Call 1, which uses it to resolve present-relative offsets
        ("200 years ago" -> current_year - 200) onto the dated spine. When None, Call 1 may
        still auto-detect a reference year stated in the lore; if it finds none, offset
        events stay undated (backward-compatible).

        Degrades gracefully: a failed Call 1 -> empty spine (events ordered relatively, or
        Could Not Place); a failed Call 2 -> the dated spine still stamps, only the relative
        events fall to Could Not Place. _weave_and_stamp always runs, so a total LLM failure
        yields a less-ordered -- never corrupted -- timeline."""
        if not events:
            return list(events)

        # Call 1 only runs if something actually states a date.
        dated = []  # (index, system, parts), post-validation + post-guard
        if any(e.date_text and e.date_text.strip() for e in events):
            dated = self._extract_dates(events, current_year)

        spines, gap_lookup = _build_spine_and_gaps(dated)
        dated_indices = {idx for idx, _, _ in dated}

        # Call 2 only runs if there's something undated to place.
        placements = {}
        if any(i not in dated_indices for i in range(len(events))):
            placements = self._place_relatives(events, dated_indices, spines, gap_lookup)

        return _weave_and_stamp(events, spines, placements, gap_lookup)

    def _extract_dates(self, events, current_year: Optional[int] = None) -> list:
        """Call 1. Returns a list of (index, system, parts) that passed validation AND the
        grounding guard. Retries up to 3x on malformed/invalid JSON; total failure -> []
        (empty spine). A per-event guard failure drops just that event to relative placement."""
        decision = None
        last_parsed = None  # last decision that PARSED (may be structurally invalid)
        for attempt in range(3):
            raw = self.call_claude(DATE_EXTRACTION_PROMPT,
                                   self._build_date_message(events, current_year))
            parsed = _parse_json_model(raw, DateDecision)
            if parsed is None:
                logger.warning("Timeline dates: bad JSON (attempt %d/3); raw: %r", attempt + 1, raw)
                continue
            last_parsed = parsed
            problems = _validate_date_decision(parsed, events)
            if problems:
                logger.warning("Timeline dates: invalid (attempt %d/3): %s",
                               attempt + 1, "; ".join(problems))
                continue
            decision = parsed
            break
        if decision is None:
            if last_parsed is None:
                logger.error("Timeline dates: no usable decision after 3 attempts; "
                             "treating all events as undated.")
                return []
            # Parsed but never fully clean. Salvage the valid dated entries rather than
            # undating the WHOLE timeline over one bad entry (the all-or-nothing failure
            # that scattered the emperor reign years out of chronological order).
            valid = _valid_dated_subset(last_parsed, events)
            logger.warning("%s Timeline dates: no fully-clean decision after 3 attempts; "
                           "using %d salvageable dated event(s).", REVIEW_PREFIX, len(valid))
            decision = DateDecision(dated=valid)

        dated = []
        for d in decision.dated:
            if _UNRESOLVED_OFFSET_SYSTEM_RE.search(d.system or ""):
                # A "years ago"-style system means the LLM left a present-offset unresolved
                # (raw magnitudes sort backwards -> reversed timeline). Drop it to relative
                # placement rather than render a reversed dated timeline.
                logger.warning("%s Timeline: system label %r for '%s' is an unresolved "
                               "present-offset (raw 'years ago' magnitudes sort backwards); "
                               "dropping it to relative placement.",
                               REVIEW_PREFIX, d.system, events[d.index].name)
                continue
            if _sanity_guard_parts(d.parts, events[d.index].date_text, d.anchor_relative):
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
        last_parsed = None  # last decision that PARSED (may be structurally invalid)
        for attempt in range(3):
            raw = self.call_claude(
                PLACEMENT_PROMPT, self._build_placement_message(events, dated_indices, spines))
            parsed = _parse_json_model(raw, PlacementDecision)
            if parsed is None:
                logger.warning("Timeline placement: bad JSON (attempt %d/3); raw: %r", attempt + 1, raw)
                continue
            last_parsed = parsed
            problems = _validate_placement_decision(parsed, events, dated_indices, gap_lookup)
            if problems:
                logger.warning("Timeline placement: invalid (attempt %d/3): %s",
                               attempt + 1, "; ".join(problems))
                continue
            decision = parsed
            break
        if decision is None:
            if last_parsed is None:
                logger.error("Timeline placement: no usable decision after 3 attempts; "
                             "%d relative event(s) -> Could Not Place.", len(relative))
                return {}
            # Parsed but never fully clean -> salvage the valid placements instead of
            # stranding every relative event in Could Not Place over one bad gap/event.
            salvaged = _valid_placement_subset(last_parsed, events, dated_indices, gap_lookup)
            logger.warning("%s Timeline placement: no fully-clean decision after 3 attempts; "
                           "using %d salvageable gap placement(s).", REVIEW_PREFIX, len(salvaged))
            return salvaged
        # Coalesce here too: validation now allows a repeated gap (it's benign), so a clean
        # decision can still carry one -- combine rather than let a dict-comp drop it.
        return {g: evs for g, evs in _coalesce_placements(decision.placements)}

    def _build_date_message(self, events, current_year: Optional[int] = None) -> str:
        """Call 1's user message: only the events that actually state a date, each with its
        ORIGINAL index (so the engine maps results back correctly). The description rides
        along too, so the model can spot a present-relative offset's magnitude and, when no
        config reference year is given, auto-detect one stated in the lore (e.g. "the
        starting year of 1424"). A config current_year, when set, is stated up front and
        OVERRIDES any the model might infer."""
        header = ["These history events state a date. For each, give its calendar system and "
                  "sortable parts. Each is prefixed by its index."]
        if current_year is not None:
            header.append(f"Reference year (present-day), from config: {current_year}")
        lines = ["\n".join(header) + "\n"]
        for i, e in enumerate(events):
            if e.date_text and e.date_text.strip():
                lines.append(f"[{i}] date={e.date_text!r}  (event: {e.name})  desc={e.description!r}")
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

    # ----------------------------------------------------------------------- #
    # Cross-type resolution -- one LLM call after per-type reconcile. The model
    # names the correct type(s) per conflicted name; Python folds + drops.
    # ----------------------------------------------------------------------- #

    def resolve_cross_type(self, by_type: dict) -> dict:
        """De-duplicate a real entity captured under MULTIPLE types. `by_type` is keyed by
        the orchestrator's noun-type strings (locations/characters/organizations/items/
        people); other keys (history) pass through untouched. An LLM arbiter picks the
        correct type(s) for each name held by 2+ types; Python folds the losers' facts into
        the winner and drops the wrong-type pages. A Location+Organization dual (a realm is
        both) is left alone. Degrades to a NO-OP (everything kept) on any LLM/parse failure.
        Returns a NEW dict; never mutates the inputs."""
        noun = [t for t in _CROSS_TYPE_ORDER if t in by_type]

        # 1. cluster entities by normalized name across types (within a type the name is
        #    unique post-reconcile; first index wins if not). Strip a leading article so a
        #    place-vs-org dual split as "Citadel" (Location) / "The Citadel" (Organization)
        #    still clusters -- the per-type article floor can't bridge two type lists.
        clusters_map = {}   # article-stripped name_key -> {type: index}
        for t in noun:
            for idx, e in enumerate(by_type[t]):
                k = _strip_article(_name_key(e.name))
                if k:
                    clusters_map.setdefault(k, {}).setdefault(t, idx)

        # 2. keep only true cross-type conflicts (>= 2 types), minus the legit realm dual.
        clusters = [(k, tmap) for k, tmap in clusters_map.items()
                    if len(tmap) >= 2 and not (set(tmap) <= _CROSS_TYPE_LEGIT_DUAL)]
        if not clusters:
            return dict(by_type)

        # 3. one arbiter call; total failure -> keep everything (safe).
        decision = self._call_cross_type(clusters, by_type)
        if decision is None:
            logger.error("%s cross-type: no usable arbiter decision after 3 attempts; keeping "
                         "all %d conflicted name(s) unmerged.", REVIEW_PREFIX, len(clusters))
            return dict(by_type)
        by_cluster = {c.cluster: c.keep_types for c in decision.choices}

        # 4. apply -- pure Python.
        drops = {t: set() for t in noun}   # type -> {idx to drop}
        absorb = {}                        # (winner_type, winner_idx) -> [loser entities]
        for cid, (k, tmap) in enumerate(clusters):
            keep = {t for t in (by_cluster.get(cid) or []) if t in tmap}
            if not keep:                   # omitted or unusable -> keep all this cluster's types
                continue
            lose = set(tmap) - keep
            if not lose:
                continue
            # primary winner absorbs the losers' facts: most evidence, tie by _CROSS_TYPE_ORDER.
            winner_type = max(keep, key=lambda t: (_cross_type_fact_count(by_type[t][tmap[t]]),
                                                   -_CROSS_TYPE_ORDER.index(t)))
            w_idx = tmap[winner_type]
            absorb.setdefault((winner_type, w_idx), []).extend(by_type[t][tmap[t]] for t in lose)
            for t in lose:
                drops[t].add(tmap[t])
            logger.warning("%s cross-type: %r kept as %s, dropped from %s (facts folded in).",
                           REVIEW_PREFIX, by_type[winner_type][w_idx].name, sorted(keep), sorted(lose))

        # 5. rebuild each type list (drop losers, augment winners); carry non-noun keys.
        out = {}
        for t in noun:
            new_list = []
            for idx, e in enumerate(by_type[t]):
                if idx in drops[t]:
                    continue
                extra = absorb.get((t, idx))
                new_list.append(_fold_cross_type_facts(e, extra) if extra else e)
            out[t] = new_list
        for t, v in by_type.items():
            out.setdefault(t, v)
        return out

    def _call_cross_type(self, clusters, by_type):
        """The arbiter call: parse -> validate -> retry 3x, else None (caller degrades)."""
        for attempt in range(3):
            raw = self.call_claude(CROSS_TYPE_PROMPT,
                                   self._build_cross_type_message(clusters, by_type))
            parsed = _parse_json_model(raw, CrossTypeDecision)
            if parsed is None:
                logger.warning("Cross-type: bad JSON (attempt %d/3); raw: %r", attempt + 1, raw)
                continue
            problems = _validate_cross_type_decision(parsed, clusters)
            if problems:
                logger.warning("Cross-type: invalid (attempt %d/3): %s",
                               attempt + 1, "; ".join(problems))
                continue
            return parsed
        return None

    def _build_cross_type_message(self, clusters, by_type) -> str:
        """One block per conflicted name: its candidate types, each with the entity's
        first few detail texts so the arbiter can judge what it actually IS."""
        lines = ["These names were each extracted under MORE THAN ONE type. For each, choose "
                 "the correct type(s).\n"]
        for cid, (k, tmap) in enumerate(clusters):
            any_t = next(t for t in _CROSS_TYPE_ORDER if t in tmap)
            lines.append(f"[{cid}] the name {by_type[any_t][tmap[any_t]].name!r} appears under "
                         f"{len(tmap)} types:")
            for t in _CROSS_TYPE_ORDER:
                if t in tmap:
                    e = by_type[t][tmap[t]]
                    facts = "; ".join(d.text for d in e.details[:4]) or "(no details)"
                    lines.append(f"    - {t}: {e.name!r} -- {facts}")
        return "\n".join(lines)
