"""Phase 3.5: the history extractor -- the third and final *extractor* agent.

Runs after the noise filter, over the messages ``select_for_extraction(...)``
keeps (``lore`` + ``ambiguous``), and turns them into validated
:class:`~models.lore.HistoryEvent` objects. It inherits the whole spine from
:class:`~agents.base_extractor.BaseExtractor` -- batching, the Claude call,
``_resolve_quote``, the verbatim-quote check.

Two things make this one different from Locations/Characters:

  * **A different shape.** A :class:`HistoryEvent` has NO ``details`` list -- it's
    one prose ``description`` backed by a flat pile of ``supporting_quotes``. So
    :meth:`_build_entry` does NOT copy the Location/Character detail-loop; it
    loops a flat ``quotes`` list (each entry is just ``quote`` + ``source_id``,
    no per-detail text, because the description IS the single summary).
  * **Scope is an Enum.** ``scope`` must be one of ``world``/``regional``/
    ``personal``. The model rejects anything off-menu, so we *coerce* the raw
    value (case/space-forgiving) and default unknowns to ``"world"`` BEFORE the
    model sees it -- Enum = strict storage, coercion = never trip it. The default
    is ``"world"`` (not the narrowest bucket) on purpose: a misfiled event only
    matters if a human notices, and they only notice if it's visible -- unknowns
    parked in the loud front-of-wiki ``world`` bucket get re-filed; unknowns
    hidden in ``personal`` sit in the basement forever.

What it deliberately does NOT do: set ``chronological_position`` (assigning real
ordering needs a pass that sees every event at once -- the field stays as a
mailbox for that later timeline pass, defaulting to ``None``); merge duplicate
events; order events into a timeline (ordering info is preserved inside the
``description`` sentence instead). All of that is the reconciler's job (4.1+).
"""

import logging
from typing import Optional

from pydantic import ValidationError

from agents.base_extractor import BaseExtractor
from models.lore import Alias, HistoryEvent, Scope
from models.message import Message

logger = logging.getLogger(__name__)


# Built FROM the Enum so the valid-scope list never lives in two places that can
# drift -> {"world", "regional", "personal"}.
VALID_SCOPES = {s.value for s in Scope}


SYSTEM_PROMPT = """You are a worldbuilding extractor for an exported D&D group chat. The campaign is a homebrew tabletop game set in a fictional world. Your job is to find every distinct EVENT in the world's history and lore and extract structured facts about it. You do not classify the chat, summarize it, or invent anything.

A "historical event" is something that happened in the world's story: a war, the founding or fall of a kingdom or empire, a notable death, the rise or fall of a family or power, a cataclysm, a treaty, etc. It is a thing that happened, not a place (that's a different extractor) or a person (also a different extractor).

For each event, extract:
- name: a short canonical label for the event that you choose, e.g. "The Maltraav-Kriega War" or "Founding of the Krieger Imperium". Keep it brief — a title, not a sentence.
- aliases: a list of any OTHER names the same event is called (empty list if none). For example, if a war is also called "the Border War", that is an alias.
- description: ONE prose sentence stating the event factually. IMPORTANT: if the messages say when this event happened relative to other events (before X, after Y, during Z), include that ordering inside this sentence — that is how ordering information is preserved.
- scope: exactly one of "world", "regional", or "personal".
  - world: shaped the whole world or its major powers (the founding or fall of an empire, a world-spanning cataclysm).
  - regional: affects one region, nation, or a few groups (a war between two countries, the founding of a city).
  - personal: about a single person, family, or small group (a noble's death, one family's downfall).
- date_text: an optional in-world date for the event, copied EXACTLY as the text states it (see "Capturing date_text" below); null or omitted if no date is stated.
- quotes: a list of the exact verbatim quotes from the messages that support this event. Each quote is an object with the quote text and the id of the message it came from.

For each quote object, provide:
- quote: the EXACT, VERBATIM text from the message. Copy it character-for-character. Do NOT paraphrase, shorten, fix typos, or change punctuation. It must appear word-for-word in the message.
- source_id: the integer id of the message you took the quote from.

Capturing date_text:
Each event may optionally include a "date_text" field. If the event's text states an explicit in-world date or year -- for example "342 AR", "in the Third Age", "the year 1247", or "Second Age, year 88" -- copy that date string EXACTLY as it appears in the text into "date_text". Copy the words verbatim: do NOT convert it to a number, do NOT normalize the format, and do NOT guess a date that isn't stated. If the event has no explicitly stated date, omit "date_text" or set it to null. Most events have no stated date, and that is expected and fine.

Important: "date_text" is ONLY for a date the text literally states. A relative phrase like "after the old kingdoms collapsed" or "before the Empire fell" is NOT a date -- leave those inside the description as usual and set "date_text" to null. And "date_text" is NOT an ordering: continue to NEVER output any "chronological_position" or position number. Placing events on a timeline is a separate later step.

Hard rules:
- Do NOT invent events, descriptions, or quotes. Every event must be supported by at least one real quote from a real message.
- Do NOT paraphrase quotes. The "description" is yours to phrase; each "quote" must be copied exactly. Each quote is automatically checked against the message you cite in source_id; if it cannot be found there word-for-word, it is thrown away — so copy carefully and cite the right id.
- Only use facts actually stated in the messages. If something is implied but not stated, leave it out.
- An event can be discussed across several messages; pull quotes from wherever it appears. Do not try to merge duplicate events or order events into a timeline beyond stating any ordering inside the description sentence — the rest is handled later.

INPUT: a JSON array of messages, each an object with an integer "id" and a string "content".

OUTPUT: ONLY a JSON array, one object per event, of the form:
{"name": "<string>", "aliases": ["<string>", ...], "description": "<string>", "scope": "<world|regional|personal>", "date_text": "<string or null>", "quotes": [{"quote": "<string>", "source_id": <integer>}, ...]}
If you find no events, return an empty array []. Do not output any text outside the JSON array, and do not wrap it in markdown code fences.

EXAMPLE
Input:
[{"id": 0, "content": "The Krieger Imperium was founded after the old kingdoms collapsed"}, {"id": 1, "content": "Maltraav and Kriega fought a brutal war"}, {"id": 2, "content": "that war happened before the Empire fell"}, {"id": 3, "content": "what's everyone rolling for initiative"}, {"id": 4, "content": "The Aldward family rose to prominence around then too"}, {"id": 5, "content": "everyone just calls it the Border War"}, {"id": 6, "content": "the Sundering of 342 AR shattered the continent"}]
Output:
[{"name": "Founding of the Krieger Imperium", "aliases": [], "description": "The Krieger Imperium was founded after the old kingdoms collapsed.", "scope": "world", "date_text": null, "quotes": [{"quote": "The Krieger Imperium was founded after the old kingdoms collapsed", "source_id": 0}]}, {"name": "The Maltraav-Kriega War", "aliases": ["the Border War"], "description": "A brutal war fought between Maltraav and Kriega, which took place before the Empire fell.", "scope": "regional", "date_text": null, "quotes": [{"quote": "Maltraav and Kriega fought a brutal war", "source_id": 1}, {"quote": "that war happened before the Empire fell", "source_id": 2}]}, {"name": "Rise of the Aldward Family", "aliases": [], "description": "The Aldward family rose to prominence.", "scope": "personal", "date_text": null, "quotes": [{"quote": "The Aldward family rose to prominence", "source_id": 4}]}, {"name": "The Sundering", "aliases": [], "description": "A cataclysm that shattered the continent, occurring in 342 AR.", "scope": "world", "date_text": "342 AR", "quotes": [{"quote": "the Sundering of 342 AR", "source_id": 6}]}]
(Notes on the example: there is NO position/ordering number anywhere — ordering like "after the old kingdoms collapsed" and "before the Empire fell" lives inside the description sentences. Message 3 produced nothing — it's game mechanics, not a world event. The war pulled two quotes, from messages 1 and 2. The first three events have date_text: null because "after the old kingdoms collapsed" / "before the Empire fell" are relative references, not stated dates; only the Sundering states an actual date (342 AR), so only it fills date_text.)"""


class HistoryExtractor(BaseExtractor):
    """Extracts :class:`~models.lore.HistoryEvent` objects from messages.

    ``extract(messages) -> list[HistoryEvent]`` is inherited from
    :class:`~agents.base_extractor.BaseExtractor` unchanged; this subclass adds
    only the prompt and the HistoryEvent-shaped :meth:`_build_entry`. Unlike the
    characters extractor it injects nothing, so it needs no constructor.
    """

    system_prompt = SYSTEM_PROMPT

    def _build_entry(self, raw: dict, batch: list[Message]) -> Optional[HistoryEvent]:
        """Build one :class:`HistoryEvent` from a single response object, or
        ``None`` (logged) if it has no usable description or fails validation.

        Note the order: ``description`` is validated first because the ``name``
        fallback derives from it. Does NOT copy the Location/Character details
        loop -- history has a flat ``quotes`` list instead.
        """
        # 1. description -- the real content. The one thing worth dropping over:
        # an event with no description is a label pointing at nothing.
        description = raw.get("description")
        if not isinstance(description, str) or not description.strip():
            logger.warning(
                "History event has no usable description; skipping it. Raw entry: %r", raw
            )
            return None

        # 2. name -- fall back to a short form of the description rather than drop
        # the event (the description is real lore; losing it over a missing label
        # would be silly).
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            first_sentence = description.strip().split(".")[0].strip()
            name = (first_sentence or description.strip())[:80]
            logger.warning(
                "History event has no usable name; using a short form of the description as the "
                "name. description=%r", description,
            )

        # 3. aliases -- same defensive line as the other two extractors. Batches are
        # file-pure (BaseExtractor.extract), so ONE file owns every name/alias/quote.
        source = batch[0].source_file
        raw_aliases = raw.get("aliases")
        aliases = ([Alias(text=a, source_files=[source]) for a in raw_aliases if isinstance(a, str)]
                   if isinstance(raw_aliases, list) else [])

        # 4. scope -- coerce to an allowed value (case/space-forgiving), defaulting
        # unknowns to "world" so a misfile surfaces for review instead of hiding.
        # This cleans the value BEFORE the strict Enum model sees it, so a bad
        # scope keeps the event instead of tripping validation and dropping it.
        raw_scope = raw.get("scope")
        scope = raw_scope.strip().lower() if isinstance(raw_scope, str) else None
        if scope not in VALID_SCOPES:
            logger.warning(
                "History event scope %r is missing or off-list; defaulting to 'world' (the visible "
                "bucket, so a misfile surfaces for review instead of hiding). description=%r",
                raw_scope, description,
            )
            scope = "world"

        # 4b. date_text -- capture the stated date VERBATIM if the event has one,
        # else None. Purely captured here; the 4.1b timeline pass is what will
        # interpret it into an ordering. Defensive: a non-string or blank value is
        # treated as "no date stated". We .strip() surrounding whitespace (same as
        # name/scope) but never touch the date's actual content -- no normalizing.
        raw_date = raw.get("date_text")
        date_text = raw_date.strip() if isinstance(raw_date, str) and raw_date.strip() else None

        # 5. supporting_quotes -- loop the FLAT quotes list (the structural
        # difference from Locations/Characters: no per-detail text).
        raw_quotes = raw.get("quotes", [])
        if not isinstance(raw_quotes, list):
            logger.warning(
                "History event %r has a non-list 'quotes' (%s); treating as none.",
                name, type(raw_quotes).__name__,
            )
            raw_quotes = []
        quotes_out: list = []
        for qd in raw_quotes:
            if not isinstance(qd, dict):
                logger.warning("History quote entry is not an object, ignoring: %r", qd)
                continue
            # _resolve_quote handles a missing/non-string quote and a bad source_id
            # itself (returns None + logs) and runs the verbatim check.
            q = self._resolve_quote(qd.get("quote"), qd.get("source_id"), batch)
            if q is None:
                continue
            if q not in quotes_out:        # dedup identical quotes within this event
                quotes_out.append(q)

        # 6. Empty-quotes -> keep, but flag loudly. A history event's name is
        # model-generated (unlike a Location's name, which is a real chat mention),
        # so an event with zero surviving quotes has nothing anchoring it to the
        # log -- still kept (missing lore is the worse sin), but surfaced for review.
        if not quotes_out:
            logger.warning(
                "History event %r has no surviving verbatim quotes; keeping it but flagging (its "
                "name was model-generated, so nothing anchors it to the log). description=%r",
                name, description,
            )

        # 7. Build. chronological_position is NOT passed -- it defaults to None.
        try:
            return HistoryEvent(
                name=name,
                name_sources=[source],
                aliases=aliases,
                description=description,
                scope=scope,
                date_text=date_text,
                supporting_quotes=quotes_out,
            )
        except ValidationError as exc:
            logger.warning(
                "History event failed model validation, skipping. name=%r error=%s", name, exc
            )
            return None
