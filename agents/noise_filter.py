"""Phase 3.2: the noise-filter agent -- the cheap, high-volume funnel.

This is the first LLM agent in the pipeline. It runs after the Phase 2 parser +
reaction filter and before the (future) extractor agents, and its whole reason
to exist is economic: the extractors are expensive (Sonnet, careful prompts,
verbatim-quote discipline) and most of a real chat log is *not* worldbuilding.
So we put a fast, cheap classifier in front of them whose only job is to label
each message ``lore`` / ``mechanic`` / ``noise`` / ``ambiguous`` and let the
orchestrator forward only the messages worth the extractors' attention.

It classifies; it does NOT extract, summarize, rewrite, or quote. Keeping it to
a pure labeling step is exactly what lets it run on Haiku at temperature 0.

Design notes / the "why" behind the shapes here:

  * **Haiku, temperature 0.** This agent sees every surviving message, so it is
    the highest-volume call in the pipeline; sorting into four buckets wants
    speed and low cost, not deep reasoning, and we want the labels boring and
    repeatable rather than creative. Both defaults are overridden in the
    constructor but still pass through ``**kwargs`` so ``client=`` injection
    (and therefore the no-network tests) keeps working.
  * **Batch-local integer IDs.** We hand Claude a JSON array of
    ``{"id", "content"}`` and ask for ``{"id", "label"}`` back. Claude refers to
    messages by number and never echoes their text, and *we* do the rejoin in
    Python -- so a long message can't get truncated or paraphrased on the way
    back, and the response payload stays tiny. IDs reset to 0 each batch so a
    stray id can't drift across batch boundaries.
  * **``ambiguous`` is the universal fallback.** A bad batch, a missing id, an
    unknown id, a junk label -- every failure path lands on ``ambiguous``, never
    ``noise``. The project must not lose lore (Phase 5 tracks every miss): an
    ``ambiguous`` message still flows to the extractors and surfaces in the
    review log later, whereas a wrongly-dropped ``noise`` label loses data
    forever. Erring toward ``ambiguous`` costs a trickle of extra tokens; erring
    toward ``noise`` costs lore. So validation here is deliberately strict and
    paranoid, and every coercion is logged.
  * **Sequential by default, parallel by opt-in.** Batches are fully independent
    (batch-local ids, no cross-batch state), so classifying them concurrently only
    speeds up the wall clock -- the labels and their order are identical. But since
    this is the highest-volume stage, running batches one at a time on a slow backend
    is the pipeline's worst "looks frozen" wait. So :meth:`classify` takes a
    ``max_workers`` knob: the **default 1 keeps it strictly sequential** (simple,
    deterministic, byte-identical for tests), and the orchestrator passes a higher
    value to fan the batches out over a thread pool. Results are always collected in
    submission order, so a parallel run returns the exact same input-ordered labels.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from agents.base import BaseAgent, ClaudeJSONError
from agents.llm_client import DEFAULT_CHEAP_MODEL
from models.message import Message

logger = logging.getLogger(__name__)


VALID_LABELS = {"lore", "mechanic", "noise", "ambiguous"}

SYSTEM_PROMPT = """You are a message classifier for an exported D&D group chat. The campaign is a homebrew tabletop game, and the chat mixes in-world worldbuilding, game-mechanics talk, and ordinary chatter. Your ONLY job is to label each message. You do not extract, summarize, rewrite, or quote anything.

Label every message with exactly one of these four labels:

- lore — in-world worldbuilding worth putting in a wiki: places, characters, factions, history, geography, and in-world facts or events. Examples: the name of a location, who rules a region, what happened in the world's past.
- mechanic — real game/table talk that is NOT worldbuilding: dice, rules, character-sheet numbers (AC, stats, feats, subclasses), spell legality, build planning, scheduling sessions, links to rules pages.
- noise — anything with no worldbuilding or game content: reactions, "lol"/"ok"/"nice", off-topic chatter, logistics, single-word filler. This ALSO covers out-of-world, real-world, or meta / out-of-character content that is not part of THIS fictional setting — a real-world brand or product, a character explicitly framed as "a regular person from Earth" or imported from another game/show, isekai / "hit by a truck" jokes, a reference to "my character in my OTHER campaign", and similar out-of-character riffs. Even when it is phrased like lore (a proper name, a backstory), content explicitly framed as NOT part of this fictional world is noise, not lore.
- ambiguous — you genuinely cannot tell whether it carries worldbuilding. Use this ONLY for real uncertainty, not to avoid deciding. If a message MIGHT carry worldbuilding but you're unsure, prefer ambiguous over noise so it isn't lost.

The tricky boundary is lore vs mechanic. Judge by whether the message tells you something about the WORLD or something about the GAME at the table. A message can name game terms and still be lore if it states a fact about the world — "the royal family is human" is lore even though "human" is a game race. Conversely "what's your AC" or "is that spell banned" is mechanic even if it mentions in-world things. One important case: a statement that ESTABLISHES a specific character — giving a NAMED character a race, class, origin, or defining trait ("Ryan is playing a warforged named Ned", "my character Ned is a warforged", "Sam's character is the exiled prince Kriggy") — is LORE, because it defines someone who exists in the world, even though it names a real player and a game race/class. Only vague, tentative build talk with NO named character ("maybe I'll play a dwarf tank", "thinking of a warlock") stays mechanic.

INPUT: you will receive a JSON array of messages, each an object with an "id" (integer) and a "content" (string).

OUTPUT: respond with ONLY a JSON array, one object per input message, each of the form {"id": <the same id>, "label": "<lore|mechanic|noise|ambiguous>"}. Return a label for every id you were given. Do not output any text outside the JSON array, and do not wrap it in markdown code fences.

EXAMPLE
Input:
[{"id": 0, "content": "The Great Well. The Pond. Lake Mundi."}, {"id": 1, "content": "lol"}, {"id": 2, "content": "Is Silvery Barbs banned in this campaign or am I allowed to be a problem"}, {"id": 3, "content": "Almost every country southeast of the Cloud Mountains is under the control of the Krieger Imperium"}, {"id": 4, "content": "I have a dwarf tank build i wanted to try maybe ill play that"}, {"id": 5, "content": "2014"}]
Output:
[{"id": 0, "label": "lore"}, {"id": 1, "label": "noise"}, {"id": 2, "label": "mechanic"}, {"id": 3, "label": "lore"}, {"id": 4, "label": "mechanic"}, {"id": 5, "label": "ambiguous"}]"""


class NoiseFilterAgent(BaseAgent):
    """Labels each parsed message ``lore`` / ``mechanic`` / ``noise`` / ``ambiguous``.

    Inherits all the Claude plumbing (one low-temperature call, fence-stripping,
    JSON-parse retry) from :class:`~agents.base.BaseAgent`; this subclass adds
    only the prompt, the batching, and the strict response validation.
    """

    def __init__(self, batch_size: int = 50, max_workers: int = 1, **kwargs):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        # Haiku by default: this is the highest-volume agent (sees every surviving
        # message), and labeling into 4 buckets needs speed + low cost, not deep
        # reasoning. Standalone fallback only -- the orchestrator passes an explicit
        # per-role model (which may route this role to a cheaper GLM via OpenRouter).
        kwargs.setdefault("model", DEFAULT_CHEAP_MODEL)
        # Classification wants boring + repeatable, not creative.
        kwargs.setdefault("temperature", 0.0)
        super().__init__(**kwargs)
        # Named (not in **kwargs) so it isn't forwarded to BaseAgent. Tradeoff:
        # bigger batches = fewer calls (cheaper, fewer system-prompt resends) but
        # worse reliability and more output tokens per call. 50 is a safe start;
        # it's a tuning knob, not a constant.
        self.batch_size = batch_size
        # How many batches to classify concurrently. 1 (default) = strictly
        # sequential; the orchestrator raises it to fan batches over a thread pool
        # (the shared client is thread-safe). Also named so it isn't forwarded.
        self.max_workers = max_workers

    def classify(self, messages: list[Message]) -> list[tuple[Message, str]]:
        """Pair every input message with its label, in the original input order.

        Chunks ``messages`` into batches of ``self.batch_size`` and classifies
        each batch independently (a bad batch is contained -- it can't take the
        rest of the run down with it). Empty input short-circuits to ``[]`` with
        no API call.

        With ``self.max_workers > 1`` the batches are classified concurrently over
        a thread pool; results are stitched back together in **batch (input)
        order** regardless, so the output is identical to the sequential path --
        only faster.
        """
        if not messages:
            return []

        batches = [messages[start:start + self.batch_size]
                   for start in range(0, len(messages), self.batch_size)]

        if self.max_workers <= 1 or len(batches) <= 1:
            per_batch = [self._classify_batch(b) for b in batches]
        else:
            # map() yields results in submission order, so labels stay input-ordered
            # even though batches finish out of order. Cap workers at the batch count.
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as pool:
                per_batch = list(pool.map(self._classify_batch, batches))

        results: list[tuple[Message, str]] = []
        for labeled in per_batch:
            results.extend(labeled)
        return results

    def _classify_batch(self, batch: list[Message]) -> list[tuple[Message, str]]:
        """Classify a single batch; never raises and never drops a message.

        Builds a JSON payload of batch-local ``{"id", "content"}`` objects (ids
        0..n-1), asks Claude for ``{"id", "label"}`` objects back, and rejoins by
        id in Python. Every failure mode -- a failed call, a non-list response,
        an unknown id, a junk label, a missing id -- resolves to ``ambiguous`` so
        no message is ever silently lost (see the module docstring for why).
        """
        payload = [{"id": i, "content": msg.content} for i, msg in enumerate(batch)]
        user_message = json.dumps(payload, ensure_ascii=False)

        # A 200-response-with-bad-JSON has already been re-asked max_json_retries
        # times by call_claude_json; if it still failed, don't kill the run --
        # label the whole batch ambiguous and move on.
        try:
            response = self.call_claude_json(SYSTEM_PROMPT, user_message)
        except ClaudeJSONError as exc:
            logger.error(
                "Noise filter batch failed to return valid JSON; labeling all %d "
                "message(s) 'ambiguous'. %s",
                len(batch), exc,
            )
            return [(msg, "ambiguous") for msg in batch]
        except Exception as exc:
            # ANY other call/parse failure (a raw json.JSONDecodeError -- a sibling of
            # ClaudeJSONError, not caught above -- a RecursionError from json_repair, or a
            # provider/adapter error) still resolves to the universal 'ambiguous' fallback:
            # never lose a message, and never let one bad batch raise out of a parallel
            # classify() and take the whole run down.
            logger.error(
                "Noise filter batch failed unexpectedly (%s); labeling all %d "
                "message(s) 'ambiguous'. %s",
                type(exc).__name__, len(batch), exc,
            )
            return [(msg, "ambiguous") for msg in batch]

        if not isinstance(response, list):
            logger.warning(
                "Noise filter expected a JSON array but got %s; labeling all %d "
                "message(s) 'ambiguous'.",
                type(response).__name__, len(batch),
            )
            return [(msg, "ambiguous") for msg in batch]

        # json_repair can recover a response into a WRAPPED shape like ['', [{...},
        # {...}]] (a stray scalar plus the real rows nested one level deep). Splice a
        # single level of nested lists back out so those rows aren't skipped as "not
        # an object" -- otherwise the whole batch falls through to the ambiguous
        # fallback and floods the extractors. Genuine junk (a bare string/int) isn't a
        # list, so it still hits the skip-and-log below. (Same fix as the extractors.)
        flattened = []
        for item in response:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)

        # Fold the response into {local_id: label}, validating as we go. valid_ids
        # is the set of ids we actually sent this batch; anything else is noise
        # from Claude and gets dropped (the message still gets a label below).
        labels_by_id: dict[int, str] = {}
        valid_ids = range(len(batch))
        for item in flattened:
            if not isinstance(item, dict):
                logger.warning(
                    "Noise filter response entry is not an object, ignoring: %r", item
                )
                continue
            item_id = item.get("id")
            label = item.get("label")
            # Guard the id's TYPE before trusting it. Both ``x in range(n)`` and
            # dict keys compare by numeric equality, so a JSON float (``1.0``) or
            # bool (``true``/``false``) would slip past the range check below AND
            # collide with a real integer local id -- e.g. ``labels_by_id[1.0]``
            # is read back by ``labels_by_id.get(1)`` -- silently overwriting a
            # message's label, possibly to ``noise``, which drops it from
            # extraction forever. ``type(...) is int`` (not ``isinstance``)
            # rejects bool too, since ``bool`` subclasses ``int``. A rejected id
            # just falls through to the missing-id -> ``ambiguous`` branch, so
            # the safe-fallback invariant holds.
            if type(item_id) is not int:
                logger.warning(
                    "Noise filter returned a non-integer id %r; ignoring it.", item_id
                )
                continue
            if item_id not in valid_ids:
                logger.warning(
                    "Noise filter returned id %r not in this batch of %d; ignoring it.",
                    item_id, len(batch),
                )
                continue
            if label not in VALID_LABELS:
                logger.warning(
                    "Noise filter returned invalid label %r for id %d; coercing to 'ambiguous'.",
                    label, item_id,
                )
                label = "ambiguous"
            labels_by_id[item_id] = label

        labeled: list[tuple[Message, str]] = []
        for local_id, msg in enumerate(batch):
            label = labels_by_id.get(local_id)
            if label is None:
                logger.warning(
                    "Noise filter response is missing id %d; labeling it 'ambiguous'.",
                    local_id,
                )
                label = "ambiguous"
            labeled.append((msg, label))
        return labeled


def select_for_extraction(classified: list[tuple[Message, str]]) -> list[Message]:
    """Messages worth sending downstream to the extractors: lore + ambiguous.

    Encodes the "default-include ambiguous, let the extractors decide" policy in
    one pure, testable place. ``mechanic`` and ``noise`` are dropped here.
    """
    return [msg for msg, label in classified if label in ("lore", "ambiguous")]
