"""Organizations extractor (suggested Phase 3.7) -- a typed lore extractor.

Runs after the noise filter, over the messages ``select_for_extraction(...)``
keeps (``lore`` + ``ambiguous``), and turns them into validated
:class:`~models.lore.Organization` objects. It inherits the whole spine from
:class:`~agents.base_extractor.BaseExtractor` -- batching, the Claude call,
``_resolve_quote``, the verbatim-quote check, the per-detail loop -- so this is
just a prompt + a ``_build_entry``; it injects nothing, so it needs no
``__init__``.

Two things shape this one:

  * **The inclusion gate is wider than "has a proper name".** An organization
    qualifies if it is named OR if it is unnamed but clearly tied to a specific
    location/character/event ("the council that rules Gol"); for a tied-but-
    unnamed group Claude writes a short descriptive label as the ``name``. A
    vague group with neither a name nor a tie ("some merchants") is dropped --
    there's nothing to build a page around.
  * **The Location<->Organization split.** A realm like the Krieger Imperium is a
    PLACE in the locations extractor (its territory/geography) and the governing
    BODY here (who rules it, how it's structured). The same realm legitimately
    appears in both; this extractor captures only the group/governance side. A
    lone individual is a Character, and a whole people/culture is the
    PeopleAndCultures extractor -- not here.

``_build_entry`` mirrors ``LocationsExtractor`` but processes details BEFORE the
name, because a missing name falls back to a short form of the first surviving
detail (the History-style keep-don't-drop move) rather than dropping real facts
over a missing label.
"""

import logging
from typing import Optional

from pydantic import ValidationError

from agents.base_extractor import BaseExtractor
from models.lore import Alias, Detail, Organization
from models.message import Message

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a worldbuilding extractor for an exported D&D group chat. The campaign is a homebrew tabletop game set in a fictional world. Your job is to find every ORGANIZATION in the messages and extract structured facts about it. You do not classify the chat, summarize it, or invent anything.

An "organization" is a structured group or body of people in the fictional world: a government or ruling power, a guild, an order, a military or mercenary company, a church or religion, a council, a faction, an order or corps defined by a profession or function (e.g. "the War Mages", "the royal cartographers"), a company or business, or a noble house spoken of as a power (not merely a surname). Examples: "the Krieger Imperium" as a governing power, "the Adventurers' Guild", "the Order of the Dawn".

Include an organization if EITHER of these is true:
- it has a proper name (e.g. "the Krieger Imperium", "Tansy's Adventuring Agency"), OR
- it has no proper name but is clearly tied to a specific location, character, or historical event in the messages (e.g. "the council that rules Gol", "the guild Kriggy founded", "the army that won the Maltraav-Kriega War"). For a tied-but-unnamed organization, write a SHORT descriptive label as its name (e.g. "The council of Gol", "Kriggy's guild") so it can be referred to later.

A proper name is by itself sufficient reason to capture an organization — it does NOT have to be significant, substantial, or described any further. Capture a named organization EVEN WHEN the message gives no other information about it: the name alone is enough. (This "name alone is enough" rule is about an actual GROUP or BODY of people — a guild, order, team, company, church, council, or noble house spoken of as a power. It does NOT license capturing a bare place, country, or realm name whose only stated content is geographic; a named country with no governing body/rulers/structure stated is a Location, not an organization — see the boundaries below.) When you have only the name and no other facts, do not invent a factual statement — return a single entry whose "quote" is the organization's name exactly as it appears in the message (so the organization still carries that name as its one supporting quote). For its "detail", write the best SHORT description you can from the surrounding context — a nearby header or the framing of the list the name sits in — AND from the group's own NAME. When the name itself embeds a place (a location word at the start of the name), work that place into the label: for "Valdenmoor Dire Wolves" framed as a baseball team, write "The baseball team from Valdenmoor", not just "A baseball team"; for a list introduced as a place's guilds, "A guild of <that place>". If the name embeds no place, use the plain framing ("A baseball team"). Use ONLY context/place actually present in the messages or the name, and never invent specifics; if there is genuinely no context or place-in-name to draw on, fall back to the generic "A named group". If a message is just a comma-separated list of named groups (for example, a list of sports teams), capture EACH listed name as its own name-only organization — one entry per name, each sharing that list's context in its detail.

Do NOT extract a vague, unnamed group that has no name and no such tie ("some merchants", "a few guards", "bandits") — there is nothing to build a page around.

Boundaries with the other extractors:
- A place is NOT an organization. A realm, country, or nation like the Krieger Imperium is captured by the locations extractor as a PLACE (its territory, geography, capital, size, borders, and where it sits); here you capture only the governing BODY (who rules it, how it is structured, its power). The same realm legitimately appears in both — but ONLY capture it here when the messages actually state a governing body, rulers, a political structure, or how power is held. If all you have about a named country/realm/nation is pure geography — that it is a country, its capital, its size, where it borders — with NO governance stated, do NOT create an organization for it even though it is named: that is Locations-only. For example, "Eglon is a major country, with Dobrovic as its capital" is a Location, not an organization.
- A single individual is NOT an organization — that is a character, even when named by a role or relationship rather than a proper name. "Emperor Tiberius" is a character; "the Krieger Imperium" he rules is an organization. Likewise "Kriggy's guard", "the Duke's champion", "so-and-so's servant" are single PEOPLE (the characters extractor's job), NOT organizations — capture an organization only for a structured GROUP of people, never for one person described by their role.
- A whole people or culture is NOT an organization — that is a different extractor. "the Krieg" as a people is not an organization; "the Krieger Imperium", their state, is.

For each organization, extract:
- name: the organization's primary/canonical name (or your short descriptive label if it has no proper name).
- aliases: a list of any OTHER names the same organization is called (empty list if none).
- details: a list of factual statements about the organization, each paired with the exact quote that supports it.

For each detail, provide three things:
- detail: a short factual statement about the organization, in your own words (e.g. "Ruled by Emperor Tiberius and his war council").
- quote: the EXACT, VERBATIM text from the message that supports this detail. Copy it character-for-character. Do NOT paraphrase, shorten, fix typos, or change punctuation. It must appear word-for-word in the message. If the copied text contains quotation marks, keep them EXACTLY as they appear — leave curly “ ” marks curly, do not straighten them — because an unescaped straight quote inside a JSON value breaks the whole batch.
- source_id: the integer id of the message you took the quote from.

Hard rules:
- Do NOT invent organizations, details, or quotes. Every detail must be supported by a real quote from a real message.
- Do NOT paraphrase quotes. The "detail" is yours to phrase; the "quote" must be copied exactly. Each quote is automatically checked against the message you cite in source_id; if it cannot be found there word-for-word, that detail is thrown away — so copy carefully and cite the right id.
- Only use facts actually stated in the messages. If something is implied but not stated, leave it out.
- An organization can be mentioned across several messages; pull details from wherever they appear. Do not try to merge duplicates or decide which mention is "primary" beyond picking a reasonable canonical name — that is handled later.
- If a message names an organization but states no facts about it, you may still include it with an empty details list.

INPUT: a JSON array of messages, each an object with an integer "id" and a string "content".

OUTPUT: ONLY a JSON array, one object per organization, of the form:
{"name": "<string>", "aliases": ["<string>", ...], "details": [{"detail": "<string>", "quote": "<string>", "source_id": <integer>}, ...]}
If you find no organizations, return an empty array []. Do not output any text outside the JSON array, and do not wrap it in markdown code fences.

EXAMPLE
Input:
[{"id": 0, "content": "Almost every country southeast of the Cloud Mountains is under the control of the Krieger Imperium"}, {"id": 1, "content": "The Imperium is ruled by Emperor Tiberius and his war council"}, {"id": 2, "content": "Tansy's Adventuring Agency takes contracts out of Crown's Nest"}, {"id": 3, "content": "the council that rules Gol just raised taxes again"}, {"id": 4, "content": "some merchants were selling potions"}, {"id": 5, "content": "what's everyone's AC for tomorrow"}, {"id": 6, "content": "the local baseball teams are the Valdenmoor Dire Wolves, Crown's Nest Crows, Cloverton Honeybees"}, {"id": 7, "content": "Eglon is one of the major countries, with Dobrovic as its capital"}]
Output:
[{"name": "Krieger Imperium", "aliases": ["the Imperium"], "details": [{"detail": "Controls almost every country southeast of the Cloud Mountains", "quote": "Almost every country southeast of the Cloud Mountains is under the control of the Krieger Imperium", "source_id": 0}, {"detail": "Ruled by Emperor Tiberius and his war council", "quote": "The Imperium is ruled by Emperor Tiberius and his war council", "source_id": 1}]}, {"name": "Tansy's Adventuring Agency", "aliases": [], "details": [{"detail": "Takes contracts out of Crown's Nest", "quote": "Tansy's Adventuring Agency takes contracts out of Crown's Nest", "source_id": 2}]}, {"name": "The council of Gol", "aliases": [], "details": [{"detail": "Rules Gol and recently raised taxes", "quote": "the council that rules Gol just raised taxes again", "source_id": 3}]}, {"name": "Valdenmoor Dire Wolves", "aliases": [], "details": [{"detail": "The baseball team from Valdenmoor", "quote": "Valdenmoor Dire Wolves", "source_id": 6}]}, {"name": "Crown's Nest Crows", "aliases": [], "details": [{"detail": "The baseball team from Crown's Nest", "quote": "Crown's Nest Crows", "source_id": 6}]}, {"name": "Cloverton Honeybees", "aliases": [], "details": [{"detail": "The baseball team from Cloverton", "quote": "Cloverton Honeybees", "source_id": 6}]}]
(Notes on the example: message 3 has no proper name but is tied to a location, Gol, so it is kept with a short descriptive label "The council of Gol". Message 4 produced nothing — "some merchants" is a vague group with no name and no tie. Message 5 is game mechanics. The Krieger Imperium appears here only as the governing BODY; its territory is captured separately by the locations extractor. Message 6 is a comma-separated list of named groups introduced as "baseball teams": each name becomes its own name-only organization, and its detail combines that stated framing with the PLACE embedded in the team's own name — "The baseball team from Valdenmoor" — rather than a bare "A baseball team" or a generic "A named group", carrying the team name itself as its single supporting quote. Message 7 produced nothing — "Eglon" is a named country, but the message states only geography (that it is a country and names its capital) with no governing body, rulers, or political structure, so it is a Location, not an organization; a named place with purely geographic content is captured by the locations extractor, not here.)"""


class OrganizationExtractor(BaseExtractor):
    """Extracts Organization objects. extract(messages) -> list[Organization] is inherited from
    BaseExtractor unchanged; this subclass injects nothing, so it needs no __init__."""

    system_prompt = SYSTEM_PROMPT

    def _build_entry(self, raw: dict, batch: list[Message]) -> Optional[Organization]:
        """Build one Organization, or None (logged) if it has neither a usable name nor any surviving
        detail, or fails validation.

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
                "Organization %r has a non-list 'details' (%s); treating as no details.",
                raw.get("name"), type(raw_details).__name__,
            )
            raw_details = []

        details_out: list = []
        quotes_out: list = []
        for d in raw_details:
            if not isinstance(d, dict):
                logger.warning("Organization detail is not an object, ignoring: %r", d)
                continue
            detail_text = d.get("detail")
            quote_text = d.get("quote")
            if not isinstance(detail_text, str) or not isinstance(quote_text, str):
                logger.warning("Organization detail missing a string 'detail'/'quote', ignoring: %r", d)
                continue
            q = self._resolve_quote(quote_text, d.get("source_id"), batch,
                                    detail_text=detail_text)
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
                    "Organization has no usable name; using a short form of its first detail as the "
                    "name. first detail=%r", details_out[0].text,
                )
            else:
                logger.warning(
                    "Organization has no usable name and no surviving details; skipping it. "
                    "Raw entry: %r", raw,
                )
                return None

        try:
            return Organization(
                name=name, name_sources=[source], aliases=aliases, details=details_out, supporting_quotes=quotes_out,
            )
        except ValidationError as exc:
            logger.warning("Organization failed model validation, skipping. name=%r error=%s", name, exc)
            return None
