"""Phase 3.3 (Step 2): the shared base class every *extractor* agent inherits.

The noise filter is a classifier and inherits :class:`~agents.base.BaseAgent`
directly. The extractors (locations now; characters/history/other to come) share
a second layer on top of that: the same batching loop, the same batch-local-id
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

# Curly punctuation -> straight equivalent. U+2019 doubles as the apostrophe in
# this dataset (don't, I've), so folding it is high-value.
_SMART_PUNCT = {
    "‘": "'", "’": "'",   # left/right single curly -> straight apostrophe
    "“": '"', "”": '"',   # left/right double curly  -> straight double quote
}


def _normalize_for_match(text: str) -> str:
    """Make text comparable: fold curly punctuation to straight, then squeeze all
    whitespace to single spaces and strip the ends. NOT lowercased on purpose --
    case differences would be a real change, and we still want to catch those.

    Deliberately minimal -- only whitespace and curly quotes/apostrophes, the
    high-frequency near-guaranteed mismatches. Em-dashes and the ellipsis could
    also get retyped (``--``, ``...``) but they're rarer; folding everything would
    weaken what "verbatim" means. Add to ``_SMART_PUNCT`` from a real miss in the
    review log, don't pre-fold speculatively.
    """
    for curly, straight in _SMART_PUNCT.items():
        text = text.replace(curly, straight)
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

        Chunks into batches of ``self.batch_size`` and extracts each independently
        so a bad batch is contained. Empty input short-circuits with no API call.
        """
        if not messages:
            return []
        out = []
        for start in range(0, len(messages), self.batch_size):
            batch = messages[start:start + self.batch_size]
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
        built = []
        for raw in response:
            if not isinstance(raw, dict):
                logger.warning("Extractor response entry is not an object, ignoring: %r", raw)
                continue
            entry = self._build_entry(raw, batch)      # subclass seam
            if entry is not None:
                built.append(entry)
        return built

def _resolve_quote(self, quote_text, source_id, batch: list[Message]) -> Optional[Quote]:
    """Turn Claude's ``(quote_text, source_id)`` into a :class:`Quote` whose
    speaker/source come from the cited message -- or ``None`` (logged) if the
    id is unusable or the quote isn't verbatim in that message.

    Logs the specific reason so a subclass's ``_build_entry`` doesn't need to
    re-log when this returns ``None``.
    """
    if not isinstance(quote_text, str):
        logger.warning(
            "Detail cites a non-string quote %r; dropping this detail.", quote_text
        )
        return None
    # Guard the id's TYPE first. ``isinstance(True, int)`` is True in Python,
    # so a ``true``/``false`` source_id would sneak through and index the
    # batch as 0/1 (``batch[True] == batch[1]``), silently attaching the wrong
    # message's quote+speaker. ``type(...) is int`` rejects bool too -- same
    # trap the noise filter guards.
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
        msg = batch[source_id]
        if self.verify_quotes and not _quote_is_verbatim(quote_text, msg.content):
            logger.warning(
                "Quote not found verbatim in its cited message id %d; dropping this detail. "
                "Claimed quote: %r | Message content: %r",
                source_id, quote_text, msg.content,
            )
            return None
        # Speaker + source come from the message, never from Claude.
        return Quote(text=quote_text, speaker=msg.sender, source_file=msg.source_file)

    def _build_entry(self, raw: dict, batch: list[Message]):
        """Build one validated lore model from a single response object, or
        ``None`` if it's unusable. The single per-entity seam -- subclasses fill
        this in. The base is not meant to run on its own."""
        raise NotImplementedError("subclasses build their own model from one response entry")
