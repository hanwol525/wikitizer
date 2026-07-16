"""Phase 3.3 (Step 2): the shared base class every *extractor* agent inherits.

The noise filter is a classifier and inherits :class:`~agents.base.BaseAgent`
directly. The extractors (locations, characters, history; organizations, items,
people & cultures) share a second layer on top of that: the same
batching loop, the same batch-local-id
wire format, the same verbatim-quote resolution and verification. That shared
spine lives here so each concrete extractor is just *a prompt + a model +
``_build_entry``*.

This is the **template-method** seam (a fancy name for: the parent runs the whole
loop and calls one method the child fills in at the single spot that differs per
entity). ``BaseExtractor`` owns *flow + quote resolution + verification*; a
subclass owns only *entry shape* -- it supplies ``self.system_prompt`` and
implements :meth:`_build_entry`, which turns one response object into one
validated model (or ``None``).

Extracted from one example (locations). When characters (3.4) and history (3.5)
land -- they add fields but share this "facts-with-quotes" spine and this same
verbatim check -- expect to refine the base slightly. That's expected and fine;
we're not trying to predict their shapes now.

The error-handling philosophy mirrors the noise filter's, minus the universal
``ambiguous`` bucket (extraction has none): *never crash, contain failures, log
loudly* at three levels -- a bad **batch** yields zero entries (loud ``error``),
a bad **entry** is skipped (``warning``), a bad **detail** is dropped
(``warning``, including a failed verbatim check). Good siblings always survive a
bad neighbour.
"""

import json
import logging
import re
from typing import Optional

from agents.base import BaseAgent, ClaudeJSONError
from models.lore import Quote
from models.message import Message

logger = logging.getLogger(__name__)


# --- verbatim-quote check (pure, dependency-free helpers) -------------------
# No ``self`` and no Claude, so they unit-test on their own exactly like
# ``strip_code_fences`` in base.py. Shared by every extractor via this module.

# Typographic punctuation -> ASCII equivalent. These are all characters Claude
# reliably RETYPES when it copies a quote (the export uses the fancy form; Claude
# emits the plain one), so folding both sides to the ASCII form absorbs a cosmetic
# difference without admitting a genuine reword. U+2019 doubles as the apostrophe
# in this dataset (don't, I've), so folding it is especially high-value.
#
# The ellipsis and en/em-dashes were added from a REAL miss (not speculatively):
# the Gol log alone carries 27 ellipses plus en/em-dashes, all inside lore-bearing
# messages, and Claude retypes "…" as "..." and "—"/"–" as "-" -- so verbatim quotes
# that were correctly attributed were being dropped, taking their whole fact with
# them (a failed quote drops the detail, not just the quote). Fold each side to the
# SAME ASCII form so "…"/"..." and "—"/"–"/"-" all compare equal.
_SMART_PUNCT = {
    "‘": "'", "’": "'",   # left/right single curly -> straight apostrophe
    "“": '"', "”": '"',   # left/right double curly  -> straight double quote
    "…": "...",           # ellipsis -> three dots (Claude routinely retypes it)
    "—": "-", "–": "-",   # em / en dash -> hyphen
}


def _normalize_for_match(text: str) -> str:
    """Make text comparable: fold typographic punctuation to its ASCII form, then
    squeeze all whitespace to single spaces and strip the ends. NOT lowercased on
    purpose -- case differences would be a real change, and we still want to catch
    those.

    Deliberately minimal -- only whitespace and the ``_SMART_PUNCT`` characters that
    Claude reliably retypes (curly quotes/apostrophes, the ellipsis, en/em-dashes),
    the near-guaranteed cosmetic mismatches. Folding EVERYTHING would weaken what
    "verbatim" means, so a character earns a spot here only from a real miss in the
    review log, never speculatively.
    """
    for curly, straight in _SMART_PUNCT.items():
        text = text.replace(curly, straight)
    # An em-dash has three plausible retypings ("-", "--", "—"); the map already
    # folds "—"/"–" to a single "-", so collapse runs of hyphens too and all three
    # compare equal. Only strings with 2+ consecutive hyphens are touched (dash
    # substitutes in this dataset), and both sides fold the same way, so this can
    # never turn a genuine reword into a match.
    text = re.sub(r"-{2,}", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _quote_is_verbatim(quote_text: str, message_content: str) -> bool:
    """True if the quote appears inside the message once both are normalized.

    The naive ``quote in message`` is too strict for this dataset: a long message
    has real newlines in ``.content`` but Claude's quote comes back as one flat
    line, and the export uses curly punctuation Claude may retype straight.
    Normalizing both sides absorbs exactly those two cosmetic differences without
    letting a genuine reword (different words) or hallucination slip through.

    An empty or whitespace-only quote normalizes to ``""``, which is a substring
    of *every* string -- so we treat it as NOT verbatim. Otherwise a fabricated
    detail carrying a blank quote would sail past this guard and reach the wiki,
    defeating the very anti-hallucination check this function exists to be.
    """
    normalized_quote = _normalize_for_match(quote_text)
    if not normalized_quote:
        return False
    return normalized_quote in _normalize_for_match(message_content)


class BaseExtractor(BaseAgent):
    """Shared batching + quote-resolution spine for the extractor agents.

    Subclasses supply ``system_prompt`` (a class attribute) and implement
    :meth:`_build_entry`. Everything about chunking messages, talking to Claude,
    and turning a cited ``(quote, source_id)`` into a metadata-correct
    :class:`~models.lore.Quote` lives here.
    """

    def __init__(self, batch_size: int = 20, verify_quotes: bool = True, **kwargs):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        # Extractor output (full entries + verbatim quotes) is far fatter than the
        # noise filter's tiny label rows; a cut-off reply fails to parse and dies.
        # setdefault, not a hard set, so an explicit max_tokens=... still wins.
        kwargs.setdefault("max_tokens", 8192)
        # NB: extractors deliberately do NOT override model/temperature --
        # BaseAgent's Sonnet @ 0.2 is exactly what careful extraction wants.
        super().__init__(**kwargs)
        # Named (kept out of **kwargs) so they aren't forwarded to BaseAgent.
        # The default of 20 is PROVISIONAL -- the real batch size is a later
        # design decision; keep it a trivial one-line default to change.
        self.batch_size = batch_size
        # Verbatim-quote check on/off. On by default: a quote that isn't in its
        # source is the exact "Claude paraphrased or hallucinated" signal.
        self.verify_quotes = verify_quotes

    def extract(self, messages: list[Message]) -> list:
        """Extract lore objects from ``messages``. Element type is set by the
        concrete subclass (e.g. ``list[Location]``); see its docstring.

        Groups messages by ``source_file`` FIRST, then chunks each file's messages
        into batches of ``self.batch_size`` and extracts each independently -- so a
        bad batch is contained AND no batch ever mixes two files. Empty input
        short-circuits with no API call.

        The file-purity is load-bearing, not tidiness. ``_quote_is_verbatim`` proves
        a QUOTE really came from its cited message, but a detail's TEXT is free-form
        prose we never check against anything -- so in a batch mixing two files
        Claude can write a fact citing a quote from file A while weaving in
        something it read from file B. Everything downstream (Detail.source_files,
        Alias.source_files, name_sources, and therefore the whole --exclude-sources
        feature) assumes an entry's text can only contain knowledge from the file
        its batch came from. Keep batches file-pure or that assumption silently
        becomes false. It also lets ``_build_entry`` trust ``batch[0].source_file``
        as THE file for the whole batch.
        """
        if not messages:
            return []
        # dict preserves insertion order (3.7+), so files come out in first-seen
        # order and each file's messages keep their original order -- same output
        # ordering as the old flat chunking for a single-file input.
        by_file: dict = {}
        for m in messages:
            by_file.setdefault(m.source_file, []).append(m)

        out = []
        for file_messages in by_file.values():
            for start in range(0, len(file_messages), self.batch_size):
                batch = file_messages[start:start + self.batch_size]
                out.extend(self._extract_batch(batch))
        return out

    def _extract_batch(self, batch: list[Message]) -> list:
        """Extract one batch; never raises. A failure yields zero entries here,
        not a dead run.

        Reads ``self.system_prompt`` (subclass class attribute) and dispatches to
        ``self._build_entry`` (subclass method) -- the two seams.
        """
        payload = [{"id": i, "content": m.content} for i, m in enumerate(batch)]
        user_message = json.dumps(payload, ensure_ascii=False)
        try:
            response = self.call_claude_json(self.system_prompt, user_message)
        except ClaudeJSONError as exc:
            logger.error(
                "Extractor batch failed to return valid JSON; skipping %d message(s). %s",
                len(batch), exc,
            )
            return []                                  # contained failure: lose one batch, not the run
        if not isinstance(response, list):
            logger.warning(
                "Extractor expected a JSON array but got %s; skipping this batch.",
                type(response).__name__,
            )
            return []
        # json_repair sometimes recovers a preamble + array into a WRAPPED shape like
        # ['', [{...}, {...}]] -- a stray scalar plus the real entries nested one level
        # deep. Splice a single level of nested lists back out so those entries aren't
        # dropped as "not an object" (that wrapping was still losing whole character
        # batches after the tolerant-parse fix). A genuine junk element (a bare string
        # or int) is NOT a list, so it still falls through to the skip-and-log below.
        flattened = []
        for item in response:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        built = []
        for raw in flattened:
            if not isinstance(raw, dict):
                logger.warning("Extractor response entry is not an object, ignoring: %r", raw)
                continue
            entry = self._build_entry(raw, batch)      # subclass seam
            if entry is not None:
                built.append(entry)
        return built

    def _resolve_quote(self, quote_text, source_id, batch: list[Message]) -> Optional[Quote]:
        """Turn Claude's ``(quote_text, source_id)`` into a :class:`Quote` whose
        speaker/source come from the CITED message -- or, when the cited id is off,
        from wherever the quote actually is in this (file-pure) batch. Returns
        ``None`` (logged) only when the quote is verbatim NOWHERE in the batch.

        **Why a whole-batch fallback.** Once the punctuation folding is right, a
        quote copied faithfully from its cited message ALWAYS passes -- the payload
        shows Claude ``m.content`` and we verify against the same ``m.content``, no
        transform. So the one remaining way a *correctly-copied* quote fails is that
        Claude attached the wrong batch-local index (LLMs are unreliable at exact
        index bookkeeping across ~20 near-identical JSON rows): the exact string is
        verbatim in a DIFFERENT message of the batch. Checking only the cited message
        then drops a fact that IS a real utterance in the file -- exactly the
        correctly-attributed fact we want to keep. Because ``extract`` makes batches
        **file-pure**, every message shares one ``source_file``, so relocating the
        quote leaves provenance / ``--exclude-sources`` untouched, and a genuinely
        hallucinated quote is verbatim in NO message so this can't smuggle a
        fabrication through. Recovery needs verification to be ON (matching is the
        very thing that's disabled when it's off).

        Logs the specific reason so a subclass's ``_build_entry`` doesn't re-log.
        """
        if not isinstance(quote_text, str):
            logger.warning(
                "Detail cites a non-string quote %r; dropping this detail.", quote_text
            )
            return None
        # A MALFORMED id (wrong type, or out of range) stays an immediate drop: it's
        # a severe Claude malfunction (not the ordinary off-by-one we recover below),
        # and ``type(...) is int`` rejects bool -- ``isinstance(True, int)`` is True, so
        # a ``true``/``false`` id would otherwise index batch[1]/batch[0] and attach the
        # wrong message (the same trap the noise filter guards). Recovery is scoped to
        # a well-formed, in-range id whose message merely doesn't hold the quote.
        if type(source_id) is not int:
            logger.warning(
                "Detail cites a non-integer source_id %r; dropping this detail.", source_id
            )
            return None
        if not (0 <= source_id < len(batch)):
            logger.warning(
                "Detail cites source_id %d out of range for this batch of %d; "
                "dropping this detail.",
                source_id, len(batch),
            )
            return None

        # Fast path -- keep the EXACT cited attribution when the cited message is
        # usable and either we're not verifying or the quote is verbatim right where
        # Claude said. (Using the cited message, not a search, means a quote that
        # legitimately recurs in several messages still attributes to the cited one.)
        # Speaker + source come from the message, never from Claude.
        cited = batch[source_id]
        if not self.verify_quotes or _quote_is_verbatim(quote_text, cited.content):
            return Quote(text=quote_text, speaker=cited.sender, source_file=cited.source_file)

        # Verification ON and the (valid, in-range) cited message did NOT contain the
        # quote. Claude almost certainly copied it faithfully but attached the wrong
        # batch-local index -- the exact string is verbatim in a DIFFERENT message of
        # this file-pure batch. Relocate it there rather than drop a real fact.
        matches = [m for m in batch if _quote_is_verbatim(quote_text, m.content)]
        if not matches:
            logger.warning(
                "Quote not found verbatim in its cited message id %r NOR anywhere in "
                "its batch; dropping this detail. Claimed quote: %r",
                source_id, quote_text,
            )
            return None
        if len(matches) > 1:
            # The same quote string is verbatim in 2+ messages; source_file is safe
            # (file-pure) but the speaker is ambiguous. Keep the fact (missing lore is
            # the worse sin) and take the first, logging the ambiguity loudly.
            logger.warning(
                "Quote cited to id %r was found verbatim in %d batch messages (not the "
                "cited one); recovering the fact with the first match -- speaker may be "
                "ambiguous. Quote: %r",
                source_id, len(matches), quote_text,
            )
        else:
            logger.info(
                "Quote cited to the wrong id %r recovered from its true message in-batch "
                "(same file, so provenance is safe). Quote: %r",
                source_id, quote_text,
            )
        found = matches[0]
        return Quote(text=quote_text, speaker=found.sender, source_file=found.source_file)

    def _build_entry(self, raw: dict, batch: list[Message]):
        """Build one validated lore model from a single response object, or
        ``None`` if it's unusable. The single per-entity seam -- subclasses fill
        this in. The base is not meant to run on its own."""
        raise NotImplementedError("subclasses build their own model from one response entry")
