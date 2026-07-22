"""Phase C: the post-reconcile PROSE agent (constrained copyedit + deterministic de-conflation).

Extraction stays ultra-faithful (verbatim quotes, per-fact provenance). But the
extracted `details` are terse fragments and merged HistoryEvent descriptions are
redundant concatenations. This agent runs AFTER reconcile + timeline (and, for the
restricted doc, AFTER the exclusion carve -- so it is leak-safe by construction, like
the History re-run) and cleans each entity body / event description up.

Two layers, deliberately split:

  * **De-conflation is DETERMINISTIC** (pure Python, no LLM): player-name tokens are
    replaced with the character's canonical name via word-boundary matching, with a
    collision guard. It runs BEFORE the LLM, so the LLM never sees a player name and
    never has to reason about who-is-who -- which is what used to make it swap two
    characters. See `deconflate_entities` / `deconflate_events`.
  * **The LLM is a CONSTRAINED COPYEDITOR**: it may only merge duplicate restatements,
    smooth flow, and fix punctuation. It must keep every fact's original subject and
    stated relationships -- it may NOT infer, add, reassign, or invent a relationship,
    and may not change who-did-what. This narrowness is the anti-hallucination fix: the
    old free rewrite invented relationships and inverted subjects from terse first-person
    facts.

It is still "pretty-faithful", not "verbatim-faithful" -- the copyedit is a reword, so
there is no cheap verbatim check; the verbatim quotes ride UNCHANGED into the footnotes
as the audit trail. A per-item failure (empty body, missing id, bad JSON) leaves that
item's `prose` None, so the renderer falls back to the (already de-conflated) raw
details -- degrade toward less-polished, never corrupt.

Inherits BaseAgent DIRECTLY (like the reconciler / noise filter), not BaseExtractor:
it introduces no new quotes to verbatim-verify and does not batch by source_file.
"""

import json
import logging
import re

from agents.base import BaseAgent, ClaudeJSONError

logger = logging.getLogger(__name__)

# Loud, human-actionable flags share this prefix (same convention as the orchestrator);
# defined locally to avoid a circular import (orchestrator imports this module).
REVIEW_PREFIX = "[REVIEW]"


# --------------------------------------------------------------------------- #
# System prompts -- byte-stable module constants so the ephemeral prompt cache
# (BaseAgent.call_claude) hits. The varying data (the entities) rides in the USER
# message. De-conflation is deterministic (below), so these prompts never mention it.
# --------------------------------------------------------------------------- #
ENTITY_PROSE_PROMPT = """You are a copyeditor for a fictional-world wiki (a D&D campaign). You turn terse extracted facts about ONE entity into a clean, readable prose body. You do NOT classify, invent, add, or reinterpret anything.

INPUT: a JSON object with "items": a JSON array; each item is one entity: {"id": <int>, "name": "<entity name>", "facts": ["<fact fragment>", ...]}.

For EACH item, write a clean prose body (one short paragraph) that:
- Uses ONLY the facts given. Invent NOTHING — no new names, dates, numbers, places, or relationships.
- You may ONLY do these three things: merge duplicate or near-duplicate restatements of the SAME fact into one clean statement, smooth the flow between facts, and fix punctuation and capitalization. That is the whole job.
- KEEP each fact's original subject and its stated relationships EXACTLY as given. NEVER change who the subject of a statement is, NEVER swap who did what to whom, and NEVER infer, add, reassign, or invent a relationship (who is whose family, ally, ruler, servant, guard, etc.). If a fact is stated in the first person ("I", "my", "we"), keep its stated subject — do not guess a different person for it.
- Include EVERY distinct fact; drop nothing. If two facts differ at all, keep both.
- Read as encyclopedic prose. Do NOT restate the entity's own name as a title, do NOT add a heading, and do NOT quote — the verbatim source quotes are attached elsewhere as footnotes.
- If an item has no facts, return an empty string for its body.

OUTPUT: ONLY a JSON array, one object per input item, of the form:
[{"id": <the same id>, "body": "<the prose body>"}]
Return a body for every id. Do not output any text outside the JSON array, and do not wrap it in markdown code fences."""


EVENT_PROSE_PROMPT = """You are a copyeditor for a fictional-world wiki (a D&D campaign). You turn a raw history-event description into a clean, readable one. You do NOT classify, invent, add, or reinterpret anything.

INPUT: a JSON object with "items": a JSON array; each item is one event: {"id": <int>, "name": "<event name>", "description": "<the raw description>"}.

The raw description is often a clumsy CONCATENATION of several restatements of the SAME event (because duplicate events were merged). For EACH item, write a clean description (one to a few sentences) that:
- Merges the restatements into ONE coherent account. Collapse the redundancy — do not repeat the same fact two or three times.
- Uses ONLY what the raw description states. Invent NOTHING. Keep EVERY distinct fact; only the repetition goes.
- KEEP each fact's original subject and its stated relationships EXACTLY as given. NEVER change who did what to whom, and NEVER infer, add, or reassign a relationship. If something is stated in the first person, keep its stated subject — do not guess a different person for it.
- Read as encyclopedic prose. Do NOT restate the event's own name as a title and do NOT quote.

OUTPUT: ONLY a JSON array, one object per input item, of the form:
[{"id": <the same id>, "body": "<the cleaned description>"}]
Return a body for every id. Do not output any text outside the JSON array, and do not wrap it in markdown code fences."""


# --------------------------------------------------------------------------- #
# Deterministic de-conflation (pure, no LLM) -- unit-testable on its own.
# --------------------------------------------------------------------------- #
def build_deconflation_map(characters) -> dict:
    """{real player name (lowercased) -> character canonical name} for de-conflation,
    derived from the reconciled Character objects (the only place the player<->character
    association lives). Two COLLISION GUARDS keep it from swapping characters:

      * if one player name maps to 2+ DISTINCT character names (an un-merged duplicate,
        or two PCs mis-tagged with the same player), the key is DROPPED -- de-conflating
        it would rewrite that player to an arbitrary one of the characters everywhere; and
      * if a player-name key equals (case-insensitively) an in-world character name/alias
        OTHER than its own target, it is DROPPED -- de-conflating it would rewrite that
        other character's name into the player's character.

    Each drop logs a [REVIEW] line. NPCs and characters with no player_name are skipped.
    """
    candidates = {}   # player_lower -> [character name, ...]
    for c in characters:
        pn = getattr(c, "player_name", None)
        if getattr(c, "is_pc", False) and pn and pn.strip():
            candidates.setdefault(pn.strip().lower(), []).append(c.name)

    # Every in-world character name + alias, lowercased -- to catch a player key that
    # collides with an in-world name.
    all_names = set()
    for c in characters:
        all_names.add(c.name.strip().lower())
        for a in getattr(c, "aliases", []):
            all_names.add(a.text.strip().lower())

    out = {}
    for key, names in candidates.items():
        distinct = list(dict.fromkeys(names))
        if len(distinct) > 1:
            logger.warning("%s Prose de-conflation: player name %r maps to %d characters "
                           "%s; skipping it (ambiguous, would swap them).",
                           REVIEW_PREFIX, key, len(distinct), distinct)
            continue
        target = distinct[0]
        if key in (all_names - {target.strip().lower()}):
            logger.warning("%s Prose de-conflation: player name %r is also an in-world "
                           "character name/alias; skipping it (would rewrite that name).",
                           REVIEW_PREFIX, key)
            continue
        out[key] = target
    return out


def _compile_deconflation(deconflation_map):
    """Compile one case-insensitive alternation over the player-name keys, longest key
    first (a longer player name wins over a shorter prefix). The word boundary treats
    APOSTROPHE as a boundary on purpose -- so a possessive "Sam's" de-conflates to
    "Kriggy's", while "Sammy" never fires ('m' is a word char). Returns None for an
    empty map (the fast passthrough path)."""
    if not deconflation_map:
        return None
    keys = sorted(deconflation_map, key=len, reverse=True)
    alt = "|".join(re.escape(k) for k in keys)
    return re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)", re.IGNORECASE)


def _deconflate_text(text, pattern, deconflation_map):
    if not text or pattern is None:
        return text
    # A match's lowercase form is always a key (the pattern is built from the lowercased
    # keys with IGNORECASE), so the lookup can't KeyError.
    return pattern.sub(lambda m: deconflation_map[m.group(0).lower()], text)


def deconflate_entities(entities, deconflation_map) -> list:
    """Return new entities with player-name tokens in each `details[].text` replaced by
    the character name. Quotes, aliases, name_sources, and the entity's own name are
    left verbatim (footnotes/provenance stay exact). Empty map -> passthrough."""
    pattern = _compile_deconflation(deconflation_map)
    if pattern is None:
        return list(entities)
    out = []
    for e in entities:
        new_details = [d.model_copy(update={"text": _deconflate_text(d.text, pattern, deconflation_map)})
                       for d in e.details]
        out.append(e.model_copy(update={"details": new_details}))
    return out


def deconflate_events(events, deconflation_map) -> list:
    """Return new events with player-name tokens in `description` AND `name` (title)
    replaced by the character name (e.g. "Sam's Departure" -> "Kriggy's Departure").
    render_wiki rebuilds anchors + crosslinks from the final names, so renaming stays
    internally consistent. Quotes stay verbatim. Empty map -> passthrough."""
    pattern = _compile_deconflation(deconflation_map)
    if pattern is None:
        return list(events)
    return [e.model_copy(update={
        "description": _deconflate_text(e.description, pattern, deconflation_map),
        "name": _deconflate_text(e.name, pattern, deconflation_map),
    }) for e in events]


class ProseAgent(BaseAgent):
    """Constrained-copyedit pass over reconciled (and already de-conflated) entity
    bodies + event descriptions.

    `polish_entities` / `polish_events` return NEW objects (model_copy) with `prose`
    set; the inputs are never mutated. Sonnet @ 0.2 (careful); larger max_tokens because
    a batched reply of full paragraphs is fat."""

    def __init__(self, batch_size: int = 12, **kwargs):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        # Extraction-grade output size; a cut-off reply fails to parse.
        kwargs.setdefault("max_tokens", 8192)
        super().__init__(**kwargs)
        # Named (not in **kwargs) so it isn't forwarded to BaseAgent.
        self.batch_size = batch_size

    # -- entities ----------------------------------------------------------- #
    def polish_entities(self, entities) -> list:
        """Return `entities` with each `prose` set to a copyedited body. An entity whose
        batch fails, or that gets no/empty body, keeps prose=None (the renderer falls
        back to joining its details, which the caller has already de-conflated)."""
        if not entities:
            return list(entities)

        out = []
        for start in range(0, len(entities), self.batch_size):
            chunk = entities[start:start + self.batch_size]
            payload = {"items": [{"id": i, "name": e.name,
                                  "facts": [d.text for d in e.details]}
                                 for i, e in enumerate(chunk)]}
            bodies = self._polish_call(ENTITY_PROSE_PROMPT, payload, len(chunk))
            for i, e in enumerate(chunk):
                body = bodies.get(i)
                if body and body.strip():
                    out.append(e.model_copy(update={"prose": body.strip()}))
                else:
                    if e.details:   # had material to polish -> a miss worth a line
                        logger.warning(
                            "Prose: no polished body for entity %r; leaving un-polished.", e.name)
                    out.append(e)
        return out

    # -- events ------------------------------------------------------------- #
    def polish_events(self, events) -> list:
        """Return `events` with each `prose` set to a de-duplicated description. An event
        whose batch fails / gets no body keeps prose=None (the renderer falls back to
        `event.description`, already de-conflated). Titles are de-conflated deterministically
        upstream, so the LLM does not touch the event name."""
        if not events:
            return list(events)

        out = []
        for start in range(0, len(events), self.batch_size):
            chunk = events[start:start + self.batch_size]
            payload = {"items": [{"id": i, "name": e.name, "description": e.description}
                                 for i, e in enumerate(chunk)]}
            bodies = self._polish_call(EVENT_PROSE_PROMPT, payload, len(chunk))
            for i, e in enumerate(chunk):
                body = bodies.get(i)
                if body and body.strip():
                    out.append(e.model_copy(update={"prose": body.strip()}))
                else:
                    if e.description and e.description.strip():
                        logger.warning(
                            "Prose: no polished body for event %r; leaving un-polished.", e.name)
                    out.append(e)
        return out

    # -- shared call + validation ------------------------------------------ #
    def _polish_call(self, system_prompt, payload, n_items) -> dict:
        """One batch: send the payload, return {batch-local id: body string} for the
        entries that came back valid (int id in range + a string "body"). Never raises --
        a failed/garbled batch returns {} so every item in it falls back to un-polished.
        Mirrors the extractors' wire discipline: batch-local int ids, one-level flatten
        for json_repair wrapping, strict type guards (type(id) is int rejects bool)."""
        user = json.dumps(payload, ensure_ascii=False)
        try:
            resp = self.call_claude_json(system_prompt, user)
        except ClaudeJSONError as exc:
            logger.error("Prose batch failed to return valid JSON; leaving %d item(s) "
                         "un-polished. %s", n_items, exc)
            return {}
        except Exception as exc:
            # ANY other call/parse failure (a raw json.JSONDecodeError -- a sibling of
            # ClaudeJSONError -- a RecursionError from json_repair, or a provider/adapter
            # error) still just leaves this batch un-polished, honoring "never raises":
            # a prose failure must never crash the render, only fall back to raw bodies.
            logger.error("Prose batch failed unexpectedly (%s); leaving %d item(s) "
                         "un-polished. %s", type(exc).__name__, n_items, exc)
            return {}

        if not isinstance(resp, list):
            logger.warning("Prose expected a JSON array but got %s; leaving %d item(s) "
                           "un-polished.", type(resp).__name__, n_items)
            return {}

        # json_repair can wrap the rows one level deep (['', [{...}, ...]]); splice it.
        flattened = []
        for item in resp:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)

        out = {}
        valid_ids = range(n_items)
        for item in flattened:
            if not isinstance(item, dict):
                continue
            iid = item.get("id")
            body = item.get("body")
            if type(iid) is not int or iid not in valid_ids:
                continue
            if not isinstance(body, str):
                continue
            out[iid] = body
        return out
