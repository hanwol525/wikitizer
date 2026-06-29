"""Phase 4.2: the footnote registry.

Pure Python, no LLM. Turns the verbatim ``Quote`` objects that back each lore
fact into stable, deduplicated footnote numbers, and renders the ``## Footnotes``
definitions block that sits at the bottom of the wiki doc.

By the time a ``Quote`` reaches this registry it is already trustworthy: the
extractor's ``_resolve_quote`` (Phase 3) verified the text is verbatim in its
cited message and pulled the speaker + source FROM that message, never from
Claude. So ``add`` can trust what it's handed -- it does NOT re-police types the
way ``_resolve_quote`` does. That paranoia belongs at the raw-LLM-JSON seam, not
at this internal one where every input is an already-validated pydantic object.

Numbering is assigned in the order ``add`` first sees each distinct quote, which
means RENDER order decides the numbers (the renderer walks Locations, then
Characters, then History, ... in a fixed order). That's fully deterministic as
long as rendering stays single-threaded -- the registry is shared mutable state,
so concurrent ``add`` calls would scramble the count. Do not parallelize
rendering.
"""

import re

from models.lore import Quote

# The section heading. Pulled out as a constant because we may rename it to
# "## Sources" later -- this keeps that a one-line change.
FOOTNOTES_HEADING = "## Footnotes"


def _normalize_quote_text(text: str) -> str:
    """Collapse every run of whitespace -- newlines, tabs, repeated spaces, plus
    stray leading/trailing spaces -- down to single spaces.

    This is THE one normalization for quote text. It is applied exactly once, at
    the ``add`` boundary, so the dedup key and the rendered text are derived from
    the same single-line form and can never drift apart.

    History (so nobody re-splits it): an earlier version applied two DIFFERENT
    normalizations -- ``add`` used ``text.strip()`` (edges only) while rendering
    flattened with ``" ".join(text.split())`` (everything). Two quotes differing
    only by INTERNAL whitespace (a newline vs a space) therefore got two different
    footnote numbers yet rendered to the identical line: duplicate-looking
    footnotes, the exact thing "deduplicated" is supposed to prevent. One shared
    normalizer at one boundary makes that whole class of bug unrepresentable.
    """
    return " ".join(text.split())


class FootnoteRegistry:
    """Mints stable, deduplicated ``[^N]`` footnote numbers for ``Quote`` objects
    and renders their definitions block.

    Single-threaded use only: ``add`` mutates shared state with no locking.
    """

    def __init__(self) -> None:
        # Key: the (text, speaker, source_file) triple the reconciler dedups on
        # (agents/reconciler.py `_dedup_quotes`), EXCEPT the text is first run
        # through `_normalize_quote_text` (whitespace-collapse). The reconciler
        # keys on RAW text, so this is the same field SHAPE with a stricter text
        # match, NOT a byte-identical key. We key on the VALUES, not the Quote
        # object, so two DISTINCT Quote instances with equal fields collapse to a
        # single footnote. Value: just the assigned int.
        #
        # `add` is the ONLY writer here and it always normalizes before storing,
        # so every key's text is guaranteed already single-line -- which is why
        # render_definitions can trust the stored text and never re-flattens it.
        #
        # dict insertion order is a language guarantee on Python 3.7+ (we're on
        # 3.9), and we assign numbers in insert order, so iterating this dict to
        # render definitions yields [^1], [^2], [^3]... already in order. No sort.
        self._numbers: dict[tuple[str, str, str], int] = {}
        # An EXPLICIT counter, never ``len(self._numbers) + 1``: on a dedup hit we
        # return early without inserting, so len() would be the wrong source of
        # truth, and an explicit next-number can never hand out a value that's
        # already in use even if the dict ever lost an entry. Boring and
        # bulletproof, which is what a provenance guarantee wants.
        self._next = 1

    def add(self, quote: Quote) -> int:
        """Register ``quote`` and return its footnote number.

        Idempotent per distinct ``(normalized_text, speaker, source_file)``: the
        first add of a given quote assigns it the next number; every later add of
        an equal quote returns that SAME number. Callers can't tell a fresh mint
        from a repeat, and shouldn't need to.

        The text is run through ``_normalize_quote_text`` (whitespace-collapse) for
        the key so whitespace-only differences -- internal newlines/tabs/repeated
        spaces, plus stray edge spaces -- can't split one quote into multiple
        footnotes. ``add`` does NOT reject empty text -- it can't reach here (the
        upstream verbatim check requires a non-empty match), and since this returns
        a plain ``int`` it has no graceful "no footnote" value to return anyway.
        The tripwire test pins that assumption.
        """
        key_text = _normalize_quote_text(quote.text)
        key = (key_text, quote.speaker, quote.source_file)
        existing = self._numbers.get(key)
        if existing is not None:
            return existing
        number = self._next
        self._numbers[key] = number
        self._next += 1
        return number

    def render_definitions(self) -> str:
        """Render the whole footnotes block as one string, or ``""`` when empty.

        This method owns the section as a UNIT -- the heading plus the ``[^N]:``
        lines. The renderer (4.4) decides WHERE the block lands in the document
        and what blank lines / horizontal rules sit around it; this method never
        reaches outside its own returned string. Returning ``""`` when empty lets
        this single emptiness check gate the heading and the body together, so
        4.4 never has to peek inside the registry to decide whether a footnotes
        section should exist at all.
        """
        if not self._numbers:
            return ""
        lines = [FOOTNOTES_HEADING, ""]
        for (text, speaker, source_file), number in self._numbers.items():
            # `text` is already single-line: `add` ran it through
            # _normalize_quote_text before it ever became a key, and `add` is the
            # only writer, so there's nothing to flatten here -- we render the
            # stored text directly.
            #
            # Wrap the verbatim text in a code span (backticks) so markdown renders
            # it LITERALLY. This is the load-bearing fidelity choice: D&D roleplay
            # is full of asterisks (``*draws his sword*``), and outside a code span
            # markdown eats those as emphasis -- so the reader would see different
            # characters in the footnote than are in the actual chat log, in a tool
            # whose entire pitch is "go check the source yourself." A code span
            # neutralizes asterisks, underscores, brackets, etc. all at once, with
            # no escape table to maintain.
            #
            # The one char a code span CAN'T ignore is the backtick itself: a span
            # is only escape-free when its FENCE outruns the longest backtick run
            # INSIDE the text (CommonMark pairs spans by run length), so a fixed
            # single backtick breaks the instant a quote contains a literal ` --
            # the open fence fuses with the inner backtick and the rest of the
            # quote spills OUTSIDE the span, re-exposing its asterisks (exactly the
            # fidelity loss the code span was meant to prevent). So size the fence
            # to one past the longest inner run; and when the text starts or ends
            # with a backtick, pad with a space (CommonMark strips one leading +
            # one trailing space inside a span, so the pad is invisible) to stop
            # that edge backtick fusing with the fence. Zero inner backticks => a
            # single-backtick fence and a no-op pad, byte-identical to before.
            runs = re.findall("`+", text)
            fence = "`" * (max((len(r) for r in runs), default=0) + 1)
            inner = f" {text} " if (text.startswith("`") or text.endswith("`")) else text
            lines.append(f"[^{number}]: {fence}{inner}{fence} — {speaker}, {source_file}")
        return "\n".join(lines)
