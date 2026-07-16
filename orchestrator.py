"""Phase 4.5: the pipeline orchestrator -- the traffic cop that runs every stage
in order, hands each stage's output to the next, and keeps the run alive when a
non-critical agent fails. It does NO LLM work of its own; all the clever stuff
lives in the agents. Its job is sequencing, parallelism, and error policy.

The error policy in one line: every stage DEGRADES (log loudly, substitute a
safe fallback, keep going) EXCEPT the noise filter, which is the sole hard-stop.
That mirrors the "degrade toward less-complete, never corrupt" spirit the
reconciler's timeline weave already follows -- an incomplete wiki beats a fake one.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from models.message import Message
from parsers.chat_parser import parse_chat_log
from parsers.reaction_filter import filter_reactions
from agents.noise_filter import NoiseFilterAgent, select_for_extraction
from agents.locations_extractor import LocationsExtractor
from agents.characters_extractor import CharactersExtractor
from agents.history_extractor import HistoryExtractor
from agents.organization_extractor import OrganizationExtractor
from agents.item_extractor import ItemExtractor
from agents.people_and_cultures_extractor import PeopleAndCulturesExtractor
from agents.reconciler import Reconciler
from agents.prose_agent import (
    ProseAgent, build_deconflation_map, deconflate_entities, deconflate_events)
from renderer.markdown import render_wiki
from exclusion import validate_exclusions, filter_entities

# The five noun entity types (History is handled separately: extracted in the parallel
# fan-out, ordered by the timeline pass). Fixed order so the concatenated inbound-mention
# index slicing lines up per type.
NOUN_TYPES = ("locations", "characters", "organizations", "items", "people")

logger = logging.getLogger(__name__)

# Same convention as the agents/renderer: every loud, human-actionable flag
# carries this prefix so the Phase 5 review pass can funnel them out of routine
# log noise. Only the degrade-and-continue logs use it -- an abort (a crash) is
# unmissable on its own and doesn't need flagging.
REVIEW_PREFIX = "[REVIEW]"


def _history_pipeline(messages, history_extractor, reconciler, current_year=None):
    """Run History's three steps (extract -> reconcile -> order_history) over
    `messages`, degrading exactly like the main run: a failed step logs a loud
    [REVIEW] line and falls back to a safe partial result instead of crashing.
    Returns the ordered events (timeline fields filled), or a less-ordered/empty
    list if a step failed.

    Used only for the restricted doc's second History pass. The MAIN run doesn't
    call this -- there, History extraction is entangled with the 6-way parallel
    fan-out and its reconcile with the per-type loop -- so we leave the main path
    exactly as it is and only reuse this for the one extra pass.

    `current_year` is the campaign reference year threaded into order_history so the
    restricted timeline resolves present-relative dates the same way the full one does.
    """
    try:
        raw = history_extractor.extract(messages)
    except Exception as exc:
        logger.error("%s restricted history extract failed; empty history. %s", REVIEW_PREFIX, exc)
        return []
    try:
        deduped = reconciler.reconcile(raw)
    except Exception as exc:
        logger.error("%s restricted history reconcile failed; using un-reconciled. %s", REVIEW_PREFIX, exc)
        deduped = raw
    try:
        return reconciler.order_history(deduped, current_year=current_year)
    except Exception as exc:
        logger.error("%s restricted history order_history failed; events -> 'Could Not "
                     "Place'. %s", REVIEW_PREFIX, exc)
        return deduped   # unordered -> renders under Could Not Place


def _apply_prose(prose_agent, noun_dict, events):
    """Two stages, both returning NEW objects: (1) DETERMINISTIC de-conflation replaces
    player-name tokens with the character name in every detail/description/title; (2) the
    LLM copyeditor cleans the (de-conflated) bodies into readable prose. Returns
    (new_noun_dict, new_events).

    The stages are separate so de-conflation SURVIVES a polish failure: if stage 2 (the
    LLM) dies, we still return the de-conflated set, and the renderer's raw-details
    fallback is already player-name-free. Each stage DEGRADES independently -- a failure
    logs a loud [REVIEW] line and returns the pre-stage objects -- the pipeline's
    "less-complete, never corrupt" rule.

    Called on POST-carve entities in the restricted doc, so it only ever sees facts that
    survived the exclusion, and the de-conflation map is built from the carved (public)
    characters -- leak-safe by construction, like the History re-run. `noun_dict` is
    keyed by NOUN_TYPES; `events` is the ordered HistoryEvent list.
    """
    # --- Stage 1: deterministic de-conflation (pure; never an LLM call). ---
    try:
        deconf = build_deconflation_map(noun_dict["characters"])
        base = dict(noun_dict)   # preserve any non-noun keys (e.g. "history")
        for t in NOUN_TYPES:
            base[t] = deconflate_entities(noun_dict[t], deconf)
        base_events = deconflate_events(events, deconf)
    except Exception as exc:
        logger.error("%s de-conflation failed; using raw objects. %s", REVIEW_PREFIX, exc)
        base, base_events = dict(noun_dict), events

    # --- Stage 2: LLM copyedit on the de-conflated objects; degrade to that set. ---
    try:
        polished = dict(base)
        for t in NOUN_TYPES:
            polished[t] = prose_agent.polish_entities(base[t])
        return polished, prose_agent.polish_events(base_events)
    except Exception as exc:
        logger.error("%s prose copyedit failed; rendering de-conflated (un-polished) "
                     "bodies. %s", REVIEW_PREFIX, exc)
        return base, base_events


@dataclass
class PipelineConfig:
    """Everything the pipeline needs to run, loaded ONCE by main.py before any
    agent is built -- so a broken config fails fast, before any paid LLM call.

    speaker_map / crosslink_words arrive already-loaded (plain dicts), not paths,
    so the orchestrator does no config-file I/O and tests can build a config inline
    with nothing on disk. max_workers caps the extractor fan-out (also the natural
    rate-limit throttle; dial down if you ever hit 429s).
    """
    speaker_map: dict[str, str]
    crosslink_words: dict
    max_workers: int = 6
    # The campaign's "present-day" reference year (e.g. 1424), used by the timeline
    # pass to resolve present-relative dates ("200 years ago" -> 1224). None (the
    # default) means: let order_history auto-detect it from the lore, or leave
    # relative-offset events undated if none is found. A set value OVERRIDES the
    # auto-detected one. Optional[int] (not int | None) for the 3.9 pin.
    current_year: Optional[int] = None
    # The declared party: {player: [character name, alias, ...]} (loaded by main.py
    # from config/player_map.json, gitignored). Empty {} (default) means no party
    # declared -> the characters extractor falls back to the LLM's roster-validated
    # guess. When set, it authoritatively assigns player_name and merges the names
    # grouped under one player as a single character.
    player_map: dict = field(default_factory=dict)


@dataclass
class WikiOutput:
    """What run() hands back. `full` is always the complete wiki. `restricted` is
    the secrets-hidden wiki when exclude_sources was given, or None when it wasn't --
    which is DIFFERENT from an empty string (that would mean 'you asked, but every
    scrap of content was in an excluded file, so nothing public was left'). main.py
    uses that difference to decide whether to write the second file at all.
    """
    full: str
    restricted: Optional[str] = None
    # The reconciled player characters (is_pc) with their assigned players, surfaced
    # so `--confirm-players` can show the user what was discovered and build the
    # config from the real (post-merge) names/aliases. Empty for a normal run.
    characters: list = field(default_factory=list)


class Orchestrator:
    """Runs the whole ingest-to-wiki pipeline. Build once, call run().

    `client` is the single shared Anthropic client, injected so tests can pass a
    fake and never hit the network. In production it's None and we build one real
    client (with the SDK's own 3x transient-HTTP retry) that every agent shares.
    """

    def __init__(self, client=None, cache=False):
        # Injected client wins (tests pass a fake); else build ONE real shared
        # client. max_retries=3 is the SDK's transient-HTTP retry, set once for the
        # whole run. The API key is read by the SDK from the environment (main.py's
        # load_dotenv put it there) -- we never read or store the key here.
        self.client = client if client is not None else anthropic.Anthropic(max_retries=3)
        # Turn on the agents' dev disk cache (main.py's --cache flag). It lives in
        # call_claude_json, so it skips identical extractor + noise-filter calls on a
        # re-run of the same logs -- the big saver while debugging. The reconciler /
        # timeline use call_claude (deliberately cache-free), so those re-run.
        self.cache = cache

    def _build_agents(self, config: PipelineConfig):
        """Construct every agent, all sharing self.client. Its own method so a test
        can subclass and swap in stub agents to exercise sequencing/error-handling
        without any real Claude calls.

        Only the model varies between agents, and that's set INSIDE each agent (the
        noise filter self-selects Haiku; the extractors/reconciler keep BaseAgent's
        Sonnet default) -- so here we only pass the shared client.
        """
        # `cache=self.cache` flows to BaseAgent via **kwargs and turns on the disk
        # cache when --cache is set (default off -> unchanged behaviour).
        noise_filter = NoiseFilterAgent(client=self.client, cache=self.cache)
        # The characters extractor needs the real player names so table-talk about a
        # real person ("Sam, you free Thursday?") isn't minted as a character. The
        # roster is the speaker map's VALUES (every real person, incl. the exporter).
        extractors = {
            "locations": LocationsExtractor(client=self.client, cache=self.cache),
            "characters": CharactersExtractor(
                client=self.client,
                player_names=list(config.speaker_map.values()),
                player_map=config.player_map,
                cache=self.cache,
            ),
            "history": HistoryExtractor(client=self.client, cache=self.cache),
            "organizations": OrganizationExtractor(client=self.client, cache=self.cache),
            "items": ItemExtractor(client=self.client, cache=self.cache),
            "people": PeopleAndCulturesExtractor(client=self.client, cache=self.cache),
        }
        # The reconciler needs the declared party too, for the deterministic
        # declared-merge floor (names grouped under one player -> one character).
        reconciler = Reconciler(client=self.client, cache=self.cache, player_map=config.player_map)
        # The prose agent runs AFTER reconcile/carve to polish bodies + descriptions.
        prose_agent = ProseAgent(client=self.client, cache=self.cache)
        return noise_filter, extractors, reconciler, prose_agent

    def run(self, files, config: PipelineConfig, exclude_sources=None) -> "WikiOutput":
        """Walk the full pipeline and return a WikiOutput (full + optional restricted).

        parse+filter each file -> concat -> noise-filter -> select ->
        extract (6 in parallel) -> reconcile per type + order_history -> render.

        When exclude_sources is given, ALSO builds a restricted doc: entities carved
        by filter_entities (confidential-only facts, quotes, and aliases stripped,
        and a secret-only canonical name re-headed to a surviving public alias --
        Part 3), and History RE-RUN over the messages minus the excluded files --
        leak-proof by construction, since the History extractor never sees the
        secret messages.
        """
        # Fail fast on a bad exclude name BEFORE any work. A name that matches
        # nothing would silently leave those messages in the "restricted" doc.
        if exclude_sources:
            validate_exclusions(exclude_sources, files)

        # --- 1. Parse + reaction-filter every file, concatenate in pass-order. ---
        # Pure Python, no LLM, so it's cheap and sequential. A parse failure is left
        # to RAISE (abort): it's before any paid call, and silently skipping a file
        # the caller asked for would mean silently-missing lore. source_file rides on
        # every Message, so concatenating loses no provenance (4.6 exclusion needs it).
        all_messages: list[Message] = []
        for filepath in files:
            parsed = parse_chat_log(filepath, config.speaker_map)
            all_messages.extend(filter_reactions(parsed))

        if not all_messages:
            logger.warning("No messages parsed from %d file(s); wiki will be empty.", len(files))
            # render_wiki returns "" for an empty world, so we let it fall through.

        noise_filter, extractors, reconciler, prose_agent = self._build_agents(config)

        # --- 2. Noise filter -> select. This is the ONE hard-stop. ---
        # The noise filter is the funnel the whole rest of the pipeline eats. Content
        # failures are already contained inside it (labeled 'ambiguous'); only a
        # sustained HTTP failure reaches us. The only way to "continue" past a dead
        # noise filter is to send the whole UNFILTERED chat to the six Sonnet
        # extractors, which quietly multiplies the bill ~6x on a run that's probably
        # already broken. So we log with context and abort.
        try:
            classified = noise_filter.classify(all_messages)
        except Exception as exc:
            logger.error("Noise filter failed; aborting run. %s", exc)
            raise
        filtered = select_for_extraction(classified)

        # --- 3. Extract: six agents in parallel over the same filtered list. ---
        # Threads (not processes) because each extractor is mostly WAITING on the
        # network -- the case threads speed up -- and they share nothing but the
        # thread-safe client. future.result() re-raises whatever blew up inside the
        # thread, so a dead extractor is caught HERE and DEGRADES: log it loudly and
        # drop just that one section, keeping the rest of the wiki.
        extracted: dict[str, list] = {}
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {
                lore_type: pool.submit(ext.extract, filtered)
                for lore_type, ext in extractors.items()
            }
            for lore_type, future in futures.items():
                try:
                    extracted[lore_type] = future.result()
                except Exception as exc:
                    logger.error(
                        "%s %s extractor failed; omitting that section. %s",
                        REVIEW_PREFIX, lore_type, exc,
                    )
                    extracted[lore_type] = []

        # --- 4. Reconcile each type, then order the history timeline. ---
        # Both DEGRADE on failure (not hard-stops): a failed reconcile falls back to
        # the un-reconciled list (duplicate pages -- the harmless under-merge direction
        # the reconciler itself already takes on a JSON failure), and a failed
        # order_history leaves the events unordered so they render under "Could Not
        # Place" (exactly what _weave_and_stamp already produces internally).
        reconciled: dict[str, list] = {}
        for lore_type, entries in extracted.items():
            try:
                reconciled[lore_type] = reconciler.reconcile(entries)
            except Exception as exc:
                logger.error(
                    "%s %s reconcile failed; using un-reconciled entries. %s",
                    REVIEW_PREFIX, lore_type, exc,
                )
                reconciled[lore_type] = entries

        events = reconciled["history"]
        try:
            events = reconciler.order_history(events, current_year=config.current_year)
        except Exception as exc:
            logger.error(
                "%s order_history failed; history will render under "
                "'Could Not Place'. %s", REVIEW_PREFIX, exc,
            )
            # leave `events` as the reconciled-but-unordered list (timeline fields None)

        # --- 5. Prose polish (full doc): rewrite terse/redundant bodies into clean,
        #        de-duplicated, de-conflated prose. Runs AFTER reconcile+order_history.
        #        Degrades to un-polished on failure (renderer falls back to the join). ---
        reconciled, events = _apply_prose(prose_agent, reconciled, events)

        # --- 6. Render the FULL doc. ---
        full = render_wiki(
            locations=reconciled["locations"],
            characters=reconciled["characters"],
            events=events,
            organizations=reconciled["organizations"],
            items=reconciled["items"],
            people=reconciled["people"],
            common_words=config.crosslink_words,
        )

        # --- 7. If exclusions were requested, build the RESTRICTED doc too. ---
        # Entities are CARVED from the finished lore (no re-extraction). History is
        # RE-RUN over the messages minus the excluded files -- so a secret can't be
        # woven into a description, because the extractor never sees it. The prose pass
        # runs on the CARVED entities (post-exclusion), so it can only ever polish facts
        # that survived -- leak-safe by construction, like the History re-run.
        restricted = None
        if exclude_sources:
            excluded = set(exclude_sources)   # bare filenames; validated above
            restricted_messages = [m for m in filtered if m.source_file not in excluded]
            carved = {t: filter_entities(reconciled[t], excluded) for t in NOUN_TYPES}
            r_events = _history_pipeline(restricted_messages, extractors["history"],
                                         reconciler, current_year=config.current_year)
            carved, r_events = _apply_prose(prose_agent, carved, r_events)
            restricted = render_wiki(
                locations=carved["locations"],
                characters=carved["characters"],
                events=r_events,
                organizations=carved["organizations"],
                items=carved["items"],
                people=carved["people"],
                common_words=config.crosslink_words,
            )

        # Surface the discovered player characters (post-merge names/aliases + assigned
        # players) so `--confirm-players` can build the config from the real names.
        player_characters = [c for c in reconciled["characters"] if getattr(c, "is_pc", False)]
        return WikiOutput(full=full, restricted=restricted, characters=player_characters)
