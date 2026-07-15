"""People & Cultures extractor (suggested Phase 3.9) -- a typed lore extractor.

Runs after the noise filter, over the messages ``select_for_extraction(...)``
keeps (``lore`` + ``ambiguous``), and turns them into validated
:class:`~models.lore.PeopleAndCultures` objects. It inherits the whole spine from
:class:`~agents.base_extractor.BaseExtractor` -- batching, the Claude call,
``_resolve_quote``, the verbatim-quote check, the per-detail loop -- so this is
just a prompt + a ``_build_entry``; it injects nothing, so it needs no
``__init__``.

It captures a COLLECTIVE group defined by shared identity -- a race, species,
ethnic group, tribe/clan, a nation's people, or a culture ("the Krieg",
"direwolves" as a kind). Two things shape it:

  * **No inclusion gate** -- unlike Organizations/Items, a people does NOT need a
    proper name or a tie. Any named or clearly-described distinct group counts
    ("the northern tribes"); if it has no proper name, Claude writes a short
    descriptive label as the ``name``.
  * **Two easy-to-confuse boundaries.** A single INDIVIDUAL is a Character ("Kriggy",
    one of the Krieg; one named direwolf in a fight) -- the group as a whole
    ("the Krieg", "direwolves" as a kind) belongs here. A formal, structured BODY
    is an Organization ("the Krieger Imperium", their state) -- the people
    themselves ("the Krieg") belong here. (This pairs with a one-line clarification
    in the characters prompt so a creature-kind isn't grabbed as a character too.)

``_build_entry`` mirrors ``LocationsExtractor`` but processes details BEFORE the
name, because a missing name falls back to a short form of the first surviving
detail (the History-style keep-don't-drop move) rather than dropping real facts
over a missing label.
"""

import logging
from typing import Optional

from pydantic import ValidationError

from agents.base_extractor import BaseExtractor
from models.lore import Alias, Detail, PeopleAndCultures
from models.message import Message

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a worldbuilding extractor for an exported D&D group chat. The campaign is a homebrew tabletop game set in a fictional world. Your job is to find every distinct PEOPLE or CULTURE in the messages and extract structured facts about it. You do not classify the chat, summarize it, or invent anything.

A "people or culture" is a COLLECTIVE group defined by shared identity: a race, a species, an ethnic group, a tribe or clan, a nation's people, or a culture. It is the group as a whole — who these people are and what they are like — not any single member of it. Examples: "the Krieg" as a seafaring people, "direwolves" as a kind of creature, "the hill clans".

Include any people or culture that the messages name or clearly describe as a distinct group. Unlike items, you do NOT need a proper name or a tie — a clearly-described people counts even if it is only ever called "the northern tribes". If the group has no proper name of its own, write a SHORT descriptive label as its name.

Boundaries with the other extractors — this one is easy to confuse, so read carefully:
- A single INDIVIDUAL is NOT a people or culture — that is a character. "Kriggy", one member of the Krieg, is a character; "the Krieg" as a whole people belongs here. One direwolf named in a fight is a character; "direwolves" as a kind of creature belongs here.
- A formal, structured BODY is NOT a people or culture — that is an organization. A culture is "who these people are"; an organization is "a structured group they formed" (a guild, a government, a church). "the Krieg" as a people belongs here; "the Krieger Imperium", their state, is an organization.

For each people or culture, extract:
- name: the group's primary/canonical name (or your short descriptive label if it has no proper name).
- aliases: a list of any OTHER names the same group is called (empty list if none).
- details: a list of factual statements about the group, each paired with the exact quote that supports it.

For each detail, provide three things:
- detail: a short factual statement about the group, in your own words (e.g. "A seafaring people from the northern coasts").
- quote: the EXACT, VERBATIM text from the message that supports this detail. Copy it character-for-character. Do NOT paraphrase, shorten, fix typos, or change punctuation. It must appear word-for-word in the message.
- source_id: the integer id of the message you took the quote from.

Hard rules:
- Do NOT invent peoples, details, or quotes. Every detail must be supported by a real quote from a real message.
- Do NOT paraphrase quotes. The "detail" is yours to phrase; the "quote" must be copied exactly. Each quote is automatically checked against the message you cite in source_id; if it cannot be found there word-for-word, that detail is thrown away — so copy carefully and cite the right id.
- Only use facts actually stated in the messages. If something is implied but not stated, leave it out.
- A people can be mentioned across several messages; pull details from wherever they appear. Do not try to merge duplicates or decide which mention is "primary" beyond picking a reasonable canonical name — that is handled later.
- If a message names a people but states no facts about it, you may still include it with an empty details list.

INPUT: a JSON array of messages, each an object with an integer "id" and a string "content".

OUTPUT: ONLY a JSON array, one object per people or culture, of the form:
{"name": "<string>", "aliases": ["<string>", ...], "details": [{"detail": "<string>", "quote": "<string>", "source_id": <integer>}, ...]}
If you find no peoples or cultures, return an empty array []. Do not output any text outside the JSON array, and do not wrap it in markdown code fences.

EXAMPLE
Input:
[{"id": 0, "content": "The Krieg are a seafaring people from the northern coasts"}, {"id": 1, "content": "Krieg raiders are feared for their longships"}, {"id": 2, "content": "Kriggy is the disgraced son of a noble house"}, {"id": 3, "content": "direwolves hunt in packs across the northern tundra"}, {"id": 4, "content": "what's everyone's AC for tomorrow"}]
Output:
[{"name": "The Krieg", "aliases": [], "details": [{"detail": "A seafaring people from the northern coasts", "quote": "The Krieg are a seafaring people from the northern coasts", "source_id": 0}, {"detail": "Their raiders are feared for their longships", "quote": "Krieg raiders are feared for their longships", "source_id": 1}]}, {"name": "Direwolves", "aliases": [], "details": [{"detail": "Hunt in packs across the northern tundra", "quote": "direwolves hunt in packs across the northern tundra", "source_id": 3}]}]
(Notes on the example: "the Krieg" is a people — kept. Message 2 produced nothing here — "Kriggy" is a single individual, who belongs to the characters extractor, not this one. "Direwolves" as a kind of creature is a people/culture; a single named direwolf would instead be a character. Message 4 is game mechanics.)"""


class PeopleAndCulturesExtractor(BaseExtractor):
    """Extracts PeopleAndCultures objects. extract(messages) -> list[PeopleAndCultures] is inherited
    from BaseExtractor unchanged; this subclass injects nothing, so it needs no __init__."""

    system_prompt = SYSTEM_PROMPT

    def _build_entry(self, raw: dict, batch: list[Message]) -> Optional[PeopleAndCultures]:
        """Build one PeopleAndCultures, or None (logged) if it has neither a usable name nor any
        surviving detail, or fails validation.

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
                "People/culture %r has a non-list 'details' (%s); treating as no details.",
                raw.get("name"), type(raw_details).__name__,
            )
            raw_details = []

        details_out: list = []
        quotes_out: list = []
        for d in raw_details:
            if not isinstance(d, dict):
                logger.warning("People/culture detail is not an object, ignoring: %r", d)
                continue
            detail_text = d.get("detail")
            quote_text = d.get("quote")
            if not isinstance(detail_text, str) or not isinstance(quote_text, str):
                logger.warning("People/culture detail missing a string 'detail'/'quote', ignoring: %r", d)
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
                    "People/culture has no usable name; using a short form of its first detail as the "
                    "name. first detail=%r", details_out[0].text,
                )
            else:
                logger.warning(
                    "People/culture has no usable name and no surviving details; skipping it. "
                    "Raw entry: %r", raw,
                )
                return None

        try:
            return PeopleAndCultures(
                name=name, name_sources=[source], aliases=aliases, details=details_out, supporting_quotes=quotes_out,
            )
        except ValidationError as exc:
            logger.warning(
                "People/culture failed model validation, skipping. name=%r error=%s", name, exc
            )
            return None
