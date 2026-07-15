"""Phase 3.3: the locations extractor -- the first *extractor* agent.

The noise filter (Phase 3.2) is a *classifier*: it only labels messages. This is
the first agent that actually pulls structured lore *out* of the chat. It runs
after the noise filter, over the messages ``select_for_extraction(...)`` keeps
(``lore`` + ``ambiguous``), and turns them into validated :class:`Location`
objects.

After the Step 2 lift, this module is intentionally tiny: all of the batching,
the Claude call, the batch-local-id wire format, and the verbatim-quote
resolution live in :class:`~agents.base_extractor.BaseExtractor`. What's left
here is the *Location-shaped* part -- a prompt, the model
(:class:`~models.lore.Location`), and :meth:`_build_entry`, which turns one
response object into one validated ``Location`` (or ``None``). That asymmetry --
base owns *flow + quote resolution + verification*, this subclass owns *entry
shape* -- is the template-method seam.

Design notes carried by the base (here for the reader, enforced there):

  * **Sonnet @ 0.2, not Haiku @ 0.** Careful extraction wants reasoning, so we
    keep ``BaseAgent``'s defaults and only raise ``max_tokens`` (to 8192) -- a
    cut-off reply fails to parse and dies. Both handled in ``BaseExtractor``.
  * **Claude never supplies metadata.** Each detail cites a message by
    ``source_id``; Python looks the message up and attaches its speaker/source.
  * **Verbatim-quote verification (on by default).** A quote that isn't really in
    its cited message is the "Claude paraphrased or hallucinated" signal; the
    base drops it and logs loudly rather than let it reach the wiki.
  * **No merging.** "Lake Mundi" returned twice is emitted twice; reconciling
    duplicates/aliases is the reconciler's job (Phase 4.1).
"""

import logging
from typing import Optional

from pydantic import ValidationError

from agents.base_extractor import BaseExtractor
from models.lore import Alias, Detail, Location
from models.message import Message

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a worldbuilding extractor for an exported D&D group chat. The campaign is a homebrew tabletop game set in a fictional world. Your job is to find every NAMED location in the messages and extract structured facts about it. You do not classify the chat, summarize it, or invent anything.

A "named location" is any place with a proper name: a lake, mountain range, city, country, region, landmark, building, etc. Examples: "Lake Mundi", "the Cloud Mountains", "Gol", "Eglon". Do NOT extract generic, unnamed places like "the forest", "a tavern", or "the dungeon" unless they are given a proper name. A realm or nation is both a place and a political power. Capture it here as a PLACE — its geography, territory, and where it sits — and leave its government and how it is ruled to the organizations extractor; you do not need to describe its politics.

For each named location, extract:
- name: the location's primary/canonical name.
- aliases: a list of any OTHER names the same location is called (empty list if none). For example, if a lake is called both "The Great Well" and "The Pond", those are aliases of it.
- details: a list of factual statements about the location, each paired with the exact quote that supports it.

For each detail, provide three things:
- detail: a short factual statement about the location, in your own words (e.g. "A massive central lake divided into three rings").
- quote: the EXACT, VERBATIM text from the message that supports this detail. Copy it character-for-character. Do NOT paraphrase, shorten, fix typos, or change punctuation. The quote must appear word-for-word in the message.
- source_id: the integer id of the message you took the quote from.

Hard rules:
- Do NOT invent locations, details, or quotes. Every detail must be supported by a real quote from a real message.
- Do NOT paraphrase quotes. The "detail" is yours to phrase; the "quote" must be copied exactly. Each quote is automatically checked against the message you cite in source_id; if it cannot be found there word-for-word, that detail is thrown away — so copy carefully and cite the right id.
- Only use facts actually stated in the messages. If something is implied but not stated, leave it out.
- A location can be mentioned across several messages; pull details from wherever they appear. Do not try to merge duplicates or decide which mention is "primary" beyond picking a reasonable canonical name — that is handled later.
- If a message names a location but states no facts about it, you may still include it with an empty details list.

INPUT: a JSON array of messages, each an object with an integer "id" and a string "content".

OUTPUT: ONLY a JSON array, one object per location, of the form:
{"name": "<string>", "aliases": ["<string>", ...], "details": [{"detail": "<string>", "quote": "<string>", "source_id": <integer>}, ...]}
If you find no named locations, return an empty array []. Do not output any text outside the JSON array, and do not wrap it in markdown code fences.

EXAMPLE
Input:
[{"id": 0, "content": "The Great Well. The Pond. Lake Mundi."}, {"id": 1, "content": "Lake Mundi is a massive central lake divided into three rings"}, {"id": 2, "content": "what's everyone's AC for tomorrow"}, {"id": 3, "content": "Almost every country southeast of the Cloud Mountains is under the control of the Krieger Imperium"}]
Output:
[{"name": "Lake Mundi", "aliases": ["The Great Well", "The Pond"], "details": [{"detail": "A massive central lake divided into three rings", "quote": "Lake Mundi is a massive central lake divided into three rings", "source_id": 1}]}, {"name": "Cloud Mountains", "aliases": [], "details": [{"detail": "Countries to their southeast are controlled by the Krieger Imperium", "quote": "Almost every country southeast of the Cloud Mountains is under the control of the Krieger Imperium", "source_id": 3}]}]"""


class LocationsExtractor(BaseExtractor):
    """Extracts named :class:`~models.lore.Location` objects from messages.

    ``extract(messages) -> list[Location]``. Inherits the batching, the Claude
    call, and the verbatim-quote resolution from
    :class:`~agents.base_extractor.BaseExtractor`; supplies only the prompt and
    the Location-shaped :meth:`_build_entry`.
    """

    # Read by BaseExtractor._extract_batch via self.system_prompt.
    system_prompt = SYSTEM_PROMPT

    def _build_entry(self, raw: dict, batch: list[Message]) -> Optional[Location]:
        """Build one :class:`Location` from a single response object, or ``None``
        (logged) if it has no usable name or fails model validation."""
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            logger.warning("Location entry has no usable name; skipping it. Raw entry: %r", raw)
            return None

        # Defensive: only keep string aliases, and only if it's actually a list
        # (a stray string would otherwise splat into one-char "aliases").
        raw_aliases = raw.get("aliases")
        # Batches are file-pure (BaseExtractor.extract), so ONE file owns every
        # name, alias, detail and quote in this batch.
        source = batch[0].source_file
        aliases = ([Alias(text=a, source_files=[source]) for a in raw_aliases if isinstance(a, str)]
                   if isinstance(raw_aliases, list) else [])

        # A non-list ``details`` (e.g. an int) would crash ``for d in ...``; the
        # never-crash rule means we coerce it to empty rather than blow up the batch.
        raw_details = raw.get("details", [])
        if not isinstance(raw_details, list):
            logger.warning(
                "Location %r has a non-list 'details' (%s); treating as no details.",
                name, type(raw_details).__name__,
            )
            raw_details = []

        details_out: list = []
        quotes_out: list = []
        for d in raw_details:
            if not isinstance(d, dict):
                logger.warning("Location detail is not an object, ignoring: %r", d)
                continue
            detail_text = d.get("detail")
            quote_text = d.get("quote")
            if not isinstance(detail_text, str) or not isinstance(quote_text, str):
                logger.warning(
                    "Location detail missing a string 'detail'/'quote', ignoring: %r", d
                )
                continue
            q = self._resolve_quote(quote_text, d.get("source_id"), batch)
            if q is None:
                # _resolve_quote already logged the specific reason (bad id /
                # not verbatim); dropping the whole detail without re-logging.
                continue
            details_out.append(Detail(text=detail_text, source_files=[q.source_file]))
            # Dedup identical quotes within this one entry. pydantic v2 compares
            # models by field value, so ``in`` works on Quote instances.
            if q not in quotes_out:
                quotes_out.append(q)

        # A location named but with zero surviving details is fine -- emit it.
        try:
            return Location(
                name=name,
                name_sources=[source],
                aliases=aliases,
                details=details_out,
                supporting_quotes=quotes_out,
            )
        except ValidationError as exc:
            logger.warning("Location failed model validation, skipping. name=%r error=%s", name, exc)
            return None
