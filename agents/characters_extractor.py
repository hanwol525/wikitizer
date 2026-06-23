"""Phase 3.4: the characters extractor -- the second *extractor* agent.

Runs after the noise filter, over the messages ``select_for_extraction(...)``
keeps (``lore`` + ``ambiguous``), and turns them into validated
:class:`~models.lore.Character` objects. It inherits the whole spine from
:class:`~agents.base_extractor.BaseExtractor` -- batching, the Claude call,
``_resolve_quote``, the verbatim-quote check, the per-entry loop -- so this is
mostly ``LocationsExtractor`` plus a few extra fields and one big trap.

The trap is **player/character conflation**: the real humans at the table have
names, and the chat freely mixes "Sam, you free Thursday?" (the person) with
"Sam plays Kriggy" (the person controlling a fictional character). We do NOT
want to mint a character named "Sam" out of table talk.

Two layers guard against that, and neither bakes real names into committed
source (the repo is public):

  * **The roster threads in at runtime (Decision A).** The caller passes
    ``player_names`` -- the real names, sourced from the speaker map's values --
    and the constructor fills a ``__PLAYER_ROSTER__`` placeholder in the prompt
    with them. The prompt tells Claude to treat roster names as people by
    default, with an explicit EXCEPTION for a character that genuinely shares a
    real person's name (a coincidence or an in-joke).
  * **A Python-side roster check (Decision B).** ``player_name`` is validated
    against the roster (case-insensitively) and nulled if it isn't a known real
    person -- and any character whose *name* matches a real person is kept but
    **flagged in the log** for a later human review step. That flag is the safety
    net that makes the softened prompt safe: we don't silently drop the legit
    joke/coincidence cases, and we don't silently keep a genuine mix-up either.

What this agent deliberately does NOT do: capture an alias unless a single
message states it outright; merge duplicate characters across batches; ask the
user anything (no ``input()``); cross-check ``is_pc`` against ``player_name``.
Unifying "Kriggy"/"Kriggy Krieger" fragments and the player->character
confirmation review are the reconciler's / orchestrator's jobs (Phase 4.1+).
"""

import logging
from typing import Optional

from pydantic import ValidationError

from agents.base_extractor import BaseExtractor
from models.lore import Character
from models.message import Message

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a worldbuilding extractor for an exported D&D group chat. The campaign is a homebrew tabletop game set in a fictional world. Your job is to find every NAMED character (a person or creature in the fictional world) in the messages and extract structured facts about them. You do not classify the chat, summarize it, or invent anything.

CRITICAL — players vs characters. The following names are the REAL PEOPLE playing this game: __PLAYER_ROSTER__
- DEFAULT: treat any of those names as the real person (the player), not a character. A message that mentions a real person by name is usually about the CHARACTER that person plays, not a character named after them. Example: "Sam plays Kriggy" means the real person Sam controls the character named Kriggy — so the character's name is "Kriggy", and Sam is recorded as the player, never as a character. And plain table talk about a real person is not a character at all: "Sam, you free Thursday?" is the human Sam sorting out scheduling — extract nothing from it.
- EXCEPTION: a character really can share a name with a real person — by coincidence, or because a player named their character after someone at the table as a joke. So if the context clearly shows a roster name is being used as a CHARACTER — someone acting in-character, an NPC by that name introduced in the world ("we hired a sellsword named Sam at the docks"), or the players explicitly flagging "the character, not the person" — then DO extract it as a character. Set name to that shared name, and set player_name to whoever actually plays them (which is usually NOT the person of the same name, since you don't play yourself; for an NPC, use null).

For each named character, extract:
- name: the character's in-world (fictional) name.
- aliases: a list of any OTHER names this same character is called — nicknames, shortened forms, titles, or epithets (empty list if none). For example, if a character usually called Kriggy is also referred to as Kriggy Krieger, that fuller form is an alias.
- is_pc: true if this character is played by one of the real people listed above (a "player character"); false if it is a non-player character run by the game master (an NPC).
- player_name: if is_pc is true, the real person who plays them — and it MUST be one of the real people listed above. If the character is an NPC, or you are not sure who plays them, use null.
- details: a list of factual statements about the character, each paired with the exact quote that supports it.

For each detail, provide three things:
- detail: a short factual statement about the character, in your own words (e.g. "The disgraced son of a noble house").
- quote: the EXACT, VERBATIM text from the message that supports this detail. Copy it character-for-character. Do NOT paraphrase, shorten, fix typos, or change punctuation. It must appear word-for-word in the message.
- source_id: the integer id of the message you took the quote from.

Keep flavor, drop mechanics. KEEP backstory, personality, relationships, titles, and role in the world. IGNORE game mechanics entirely: character class, subclass, feats, level, ability scores, armor class (AC), hit points, dice, and build choices are NOT character facts. A single message can mix both — keep only the flavor part. Example: from "Kriggy's the disgraced son of a noble house, Battlemaster with 18 AC", extract "the disgraced son of a noble house" and ignore the Battlemaster/AC part. When you quote, quote only the flavor span, not the mechanics.

Hard rules:
- Do NOT invent characters, details, or quotes. Every detail must be supported by a real quote from a real message.
- Do NOT paraphrase quotes. The "detail" is yours to phrase; the "quote" must be copied exactly. Each quote is automatically checked against the message you cite in source_id; if it cannot be found there word-for-word, that detail is thrown away — so copy carefully and cite the right id.
- Only use facts actually stated in the messages. If something is implied but not stated, leave it out.
- A character can be mentioned across several messages; pull details (and any aliases) from wherever they appear. Do not try to merge duplicate characters or unify their names across separate mentions beyond choosing a reasonable canonical name and the aliases a message clearly gives — the rest of the merging is handled later.
- If a character is named but no facts are stated, you may still include them with an empty details list.

INPUT: a JSON array of messages, each an object with an integer "id" and a string "content".

OUTPUT: ONLY a JSON array, one object per character, of the form:
{"name": "<string>", "aliases": ["<string>", ...], "is_pc": <true|false>, "player_name": "<string or null>", "details": [{"detail": "<string>", "quote": "<string>", "source_id": <integer>}, ...]}
If you find no named characters, return an empty array []. Do not output any text outside the JSON array, and do not wrap it in markdown code fences.

EXAMPLE
Input:
[{"id": 0, "content": "Sam plays Kriggy right?"}, {"id": 1, "content": "yeah Kriggy's the disgraced son of a noble house, Battlemaster with 18 AC"}, {"id": 2, "content": "Emperor Tiberius rules the Krieger Imperium with an iron fist"}, {"id": 3, "content": "Kriggy is Tiberius's younger brother"}, {"id": 4, "content": "Sam you free to play Thursday?"}, {"id": 5, "content": "lol Ryan named his character Hannah as a joke"}, {"id": 6, "content": "Kriggy is short for Kriggy Krieger"}]
Output:
[{"name": "Kriggy", "aliases": ["Kriggy Krieger"], "is_pc": true, "player_name": "Sam", "details": [{"detail": "The disgraced son of a noble house", "quote": "Kriggy's the disgraced son of a noble house", "source_id": 1}, {"detail": "Younger brother of Emperor Tiberius", "quote": "Kriggy is Tiberius's younger brother", "source_id": 3}]}, {"name": "Emperor Tiberius", "aliases": [], "is_pc": false, "player_name": null, "details": [{"detail": "Rules the Krieger Imperium with an iron fist", "quote": "Emperor Tiberius rules the Krieger Imperium with an iron fist", "source_id": 2}]}, {"name": "Hannah", "aliases": [], "is_pc": true, "player_name": "Ryan", "details": []}]
(Notes on the example: message 4 produced nothing — "Sam" there is the real person scheduling, not a character. Message 5 is the EXCEPTION case — "Hannah" is a real person's name, but here it's a character Ryan made as a joke, so it IS extracted, with player_name "Ryan", not "Hannah". Message 6 shows aliases — "Kriggy" is the canonical name actually used in the chat, with the fuller "Kriggy Krieger" captured as an alias.)"""


class CharactersExtractor(BaseExtractor):
    """Extracts :class:`~models.lore.Character` objects from messages.

    ``extract(messages) -> list[Character]`` is inherited from
    :class:`~agents.base_extractor.BaseExtractor` unchanged; this subclass adds
    the roster-aware system prompt and the Character-shaped :meth:`_build_entry`.

    ``player_names`` (the real human names from the speaker map) is REQUIRED -- it
    is the anti-conflation anchor, so the caller must supply it rather than let it
    default to empty and silently weaken the trap protection. The extractor itself
    does no file I/O; the orchestrator sources the names.
    """

    def __init__(self, player_names, **kwargs):
        super().__init__(**kwargs)
        # sorted() so the roster string is byte-identical every run -- a stable
        # system prompt is what lets prompt caching kick in later (the cost lever).
        # Use .replace(), NOT .format()/f-string: the prompt is full of literal
        # { } in its JSON example, and .format() treats every brace as a fill-in
        # slot and would choke on them. .replace() doesn't treat braces specially.
        # Drop blank/whitespace-only names first: a roster whose only member is
        # "" or "   " would otherwise build a truthy _roster_lookup of {""} that
        # ENABLES the check and nulls every real player_name -- the inverse of the
        # "empty roster disables the check" intent. Filtering makes an effectively
        # empty roster behave like a truly empty one (and keeps a blank out of the
        # prompt). The orchestrator already drops empties; this is belt-and-suspenders
        # for a whitespace-only name, which its truthiness filter wouldn't catch.
        clean_names = sorted(n for n in player_names if n and n.strip())
        roster_str = ", ".join(clean_names)
        # Instance attribute on purpose: BaseExtractor._extract_batch reads
        # self.system_prompt, and an instance attr quietly takes precedence over
        # the class-level one LocationsExtractor uses -- zero base changes needed.
        self.system_prompt = SYSTEM_PROMPT.replace("__PLAYER_ROSTER__", roster_str)
        # Lowercased set for the case-insensitive player_name check + name-collision flag.
        self._roster_lookup = {n.strip().lower() for n in clean_names}

    def _build_entry(self, raw: dict, batch: list[Message]) -> Optional[Character]:
        """Build one :class:`Character` from a single response object, or ``None``
        (logged) if it has no usable name or fails model validation."""
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            logger.warning("Character entry has no usable name; skipping it. Raw entry: %r", raw)
            return None

        # aliases -- mirror LocationsExtractor: string-only, and a non-list value
        # (e.g. Claude sends a bare string) becomes [] instead of crashing.
        raw_aliases = raw.get("aliases")
        aliases = [a for a in raw_aliases if isinstance(a, str)] if isinstance(raw_aliases, list) else []

        # is_pc -- the model defaults to False (NPC), so "if unsure, just some NPC"
        # is the safe assumption; honor it whenever Claude omits or fumbles the field.
        is_pc = raw.get("is_pc")
        if not isinstance(is_pc, bool):
            is_pc = False

        # player_name -- the roster check (Decision B). Validate, don't rewrite.
        player_name = raw.get("player_name")
        if not isinstance(player_name, str) or not player_name.strip():
            player_name = None
        elif self._roster_lookup and player_name.strip().lower() not in self._roster_lookup:
            # An empty roster disables this check (no ground truth to reject
            # against); a real run never has an empty roster.
            logger.warning(
                "Character %r claims player_name %r, which is not a known real person; nulling it "
                "(likely player/character conflation).", name, player_name,
            )
            player_name = None

        # Name-collision flag: keep the character (it's usually a coincidence or a
        # deliberate joke), but surface it for the later human review step -- this
        # is what catches a genuine player/character mix-up that the player_name
        # check above can't (a real "Sam" character with player_name "Sam" passes
        # that check). Empty roster auto-handles: ``name ... in set()`` is False.
        if name.strip().lower() in self._roster_lookup:
            logger.warning(
                "Character name %r matches a real person in the roster; keeping it but flagging "
                "(coincidence/joke, or a player/character mix-up?). is_pc=%r player_name=%r",
                name, is_pc, player_name,
            )

        # detail/quote loop -- same shape as LocationsExtractor._build_entry.
        raw_details = raw.get("details", [])
        if not isinstance(raw_details, list):
            logger.warning(
                "Character %r has a non-list 'details' (%s); treating as no details.",
                name, type(raw_details).__name__,
            )
            raw_details = []

        details_out: list = []
        quotes_out: list = []
        for d in raw_details:
            if not isinstance(d, dict):
                logger.warning("Character detail is not an object, ignoring: %r", d)
                continue
            detail_text = d.get("detail")
            quote_text = d.get("quote")
            if not isinstance(detail_text, str) or not isinstance(quote_text, str):
                logger.warning(
                    "Character detail missing a string 'detail'/'quote', ignoring: %r", d
                )
                continue
            q = self._resolve_quote(quote_text, d.get("source_id"), batch)
            if q is None:
                # _resolve_quote already logged the specific reason; drop the detail.
                continue
            details_out.append(detail_text)
            if q not in quotes_out:
                quotes_out.append(q)

        try:
            return Character(
                name=name,
                aliases=aliases,
                is_pc=is_pc,
                player_name=player_name,
                details=details_out,
                supporting_quotes=quotes_out,
            )
        except ValidationError as exc:
            logger.warning("Character failed model validation, skipping. name=%r error=%s", name, exc)
            return None
