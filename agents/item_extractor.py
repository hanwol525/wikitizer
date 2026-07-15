"""Items extractor (suggested Phase 3.8) -- a typed lore extractor.

Runs after the noise filter, over the messages ``select_for_extraction(...)``
keeps (``lore`` + ``ambiguous``), and turns them into validated
:class:`~models.lore.Item` objects. It inherits the whole spine from
:class:`~agents.base_extractor.BaseExtractor` -- batching, the Claude call,
``_resolve_quote``, the verbatim-quote check, the per-detail loop -- so this is
just a prompt + a ``_build_entry``; it injects nothing, so it needs no
``__init__``.

It is shaped exactly like :class:`~agents.organization_extractor.OrganizationExtractor`:

  * **The inclusion gate is wider than "has a proper name".** An item qualifies
    if it is named OR if it is unnamed but clearly tied to a specific
    character/location/event ("Kriggy's family sword"); for a tied-but-unnamed
    object Claude writes a short descriptive label as the ``name``. Ordinary loot
    being bought/sold/counted ("a regular dagger and some rope") is skipped --
    there's nothing notable to build a page around.
  * **Boundary with the other extractors:** an item is a physical object you
    could in principle pick up or carry. A place, person, organization, or
    people/culture is some other extractor's job, not here.

``_build_entry`` mirrors ``LocationsExtractor`` but processes details BEFORE the
name, because a missing name falls back to a short form of the first surviving
detail (the History-style keep-don't-drop move) rather than dropping real facts
over a missing label.
"""

import logging
from typing import Optional

from pydantic import ValidationError

from agents.base_extractor import BaseExtractor
from models.lore import Alias, Detail, Item
from models.message import Message

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a worldbuilding extractor for an exported D&D group chat. The campaign is a homebrew tabletop game set in a fictional world. Your job is to find every notable ITEM in the messages and extract structured facts about it. You do not classify the chat, summarize it, or invent anything.

An "item" is a physical object in the fictional world: a weapon, a piece of armor, an artifact, a relic, a magical or otherwise notable object, or a treasure. Examples: "the Amulet of Destiny", "the Crown of Gol", "Frostbite".

Include an item if EITHER of these is true:
- it has a proper name (e.g. "the Amulet of Destiny", "Frostbite"), OR
- it has no proper name but is clearly tied to a specific character, location, or historical event in the messages (e.g. "Kriggy's family sword", "the relic kept in Gol's temple", "the amulet everyone fought the Great War to obtain"). For a tied-but-unnamed item, write a SHORT descriptive label as its name (e.g. "Kriggy's family sword", "The relic of Gol's temple") so it can be referred to later.

Do NOT extract a generic, unnamed object that has no name and no such tie ("a sword", "some potions", "a dagger"). Skip ordinary equipment and loot that is merely being bought, sold, or counted unless a name or a tie makes it notable.

Boundary with the other extractors:
- A place, a person, an organization, and a people or culture are NOT items — those are other extractors. An item is a physical object you could in principle pick up or carry.

For each item, extract:
- name: the item's primary/canonical name (or your short descriptive label if it has no proper name).
- aliases: a list of any OTHER names the same item is called (empty list if none).
- details: a list of factual statements about the item, each paired with the exact quote that supports it.

For each detail, provide three things:
- detail: a short factual statement about the item, in your own words (e.g. "The only thing that can seal the rift").
- quote: the EXACT, VERBATIM text from the message that supports this detail. Copy it character-for-character. Do NOT paraphrase, shorten, fix typos, or change punctuation. It must appear word-for-word in the message.
- source_id: the integer id of the message you took the quote from.

Hard rules:
- Do NOT invent items, details, or quotes. Every detail must be supported by a real quote from a real message.
- Do NOT paraphrase quotes. The "detail" is yours to phrase; the "quote" must be copied exactly. Each quote is automatically checked against the message you cite in source_id; if it cannot be found there word-for-word, that detail is thrown away — so copy carefully and cite the right id.
- Only use facts actually stated in the messages. If something is implied but not stated, leave it out.
- An item can be mentioned across several messages; pull details from wherever they appear. Do not try to merge duplicates or decide which mention is "primary" beyond picking a reasonable canonical name — that is handled later.
- If a message names an item but states no facts about it, you may still include it with an empty details list.

INPUT: a JSON array of messages, each an object with an integer "id" and a string "content".

OUTPUT: ONLY a JSON array, one object per item, of the form:
{"name": "<string>", "aliases": ["<string>", ...], "details": [{"detail": "<string>", "quote": "<string>", "source_id": <integer>}, ...]}
If you find no items, return an empty array []. Do not output any text outside the JSON array, and do not wrap it in markdown code fences.

EXAMPLE
Input:
[{"id": 0, "content": "The Amulet of Destiny is the only thing that can seal the rift"}, {"id": 1, "content": "they say the Amulet was forged before the Maltraav-Kriega War"}, {"id": 2, "content": "Kriggy's family sword has been passed down for seven generations"}, {"id": 3, "content": "I bought a regular dagger and some rope at the shop"}, {"id": 4, "content": "what's everyone's AC for tomorrow"}]
Output:
[{"name": "Amulet of Destiny", "aliases": ["the Amulet"], "details": [{"detail": "The only thing that can seal the rift", "quote": "The Amulet of Destiny is the only thing that can seal the rift", "source_id": 0}, {"detail": "Said to have been forged before the Maltraav-Kriega War", "quote": "they say the Amulet was forged before the Maltraav-Kriega War", "source_id": 1}]}, {"name": "Kriggy's family sword", "aliases": [], "details": [{"detail": "Has been passed down for seven generations", "quote": "Kriggy's family sword has been passed down for seven generations", "source_id": 2}]}]
(Notes on the example: the Amulet is named, with an alias and two details — message 1 also ties it to a historical event, which is fine; it stays one item. "Kriggy's family sword" has no proper name but is tied to a character, Kriggy, so it is kept with a short descriptive label. Message 3 produced nothing — a regular dagger and rope are ordinary loot with no name or tie. Message 4 is game mechanics.)"""


class ItemExtractor(BaseExtractor):
    """Extracts Item objects. extract(messages) -> list[Item] is inherited from BaseExtractor
    unchanged; this subclass injects nothing, so it needs no __init__."""

    system_prompt = SYSTEM_PROMPT

    def _build_entry(self, raw: dict, batch: list[Message]) -> Optional[Item]:
        """Build one Item, or None (logged) if it has neither a usable name nor any surviving detail,
        or fails validation.

        Order matters: details are processed BEFORE name, because a missing name falls back to a short
        form of the first surviving detail (History-style) so we keep the entity rather than drop real
        facts over a missing label.
        """
        # aliases -- string-only, and a non-list value becomes [] instead of crashing.
        raw_aliases = raw.get("aliases")
        # Batches are file-pure (BaseExtractor.extract), so ONE file owns every
        # name, alias, detail and quote in this batch.
        source = batch[0].source_file
        aliases = ([Alias(text=a, source_files=[source]) for a in raw_aliases if isinstance(a, str)]
                   if isinstance(raw_aliases, list) else [])

        # details FIRST (the name fallback derives from them). Non-list -> empty, never crash.
        raw_details = raw.get("details", [])
        if not isinstance(raw_details, list):
            logger.warning(
                "Item %r has a non-list 'details' (%s); treating as no details.",
                raw.get("name"), type(raw_details).__name__,
            )
            raw_details = []

        details_out: list = []
        quotes_out: list = []
        for d in raw_details:
            if not isinstance(d, dict):
                logger.warning("Item detail is not an object, ignoring: %r", d)
                continue
            detail_text = d.get("detail")
            quote_text = d.get("quote")
            if not isinstance(detail_text, str) or not isinstance(quote_text, str):
                logger.warning("Item detail missing a string 'detail'/'quote', ignoring: %r", d)
                continue
            q = self._resolve_quote(quote_text, d.get("source_id"), batch)
            if q is None:                       # _resolve_quote already logged why
                continue
            details_out.append(Detail(text=detail_text, source_files=[q.source_file]))
            if q not in quotes_out:             # dedup identical quotes within this entry
                quotes_out.append(q)

        # name: real name if present; else a short form of the first surviving detail; else drop.
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            if details_out:
                name = details_out[0].text[:80]
                logger.warning(
                    "Item has no usable name; using a short form of its first detail as the "
                    "name. first detail=%r", details_out[0].text,
                )
            else:
                logger.warning(
                    "Item has no usable name and no surviving details; skipping it. Raw entry: %r", raw,
                )
                return None

        try:
            return Item(
                name=name, name_sources=[source], aliases=aliases, details=details_out, supporting_quotes=quotes_out,
            )
        except ValidationError as exc:
            logger.warning("Item failed model validation, skipping. name=%r error=%s", name, exc)
            return None
