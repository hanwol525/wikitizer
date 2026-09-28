"""``ParsedWOF`` -> narrowed, numbered candidate payloads for the model, plus the entity->
``Entry`` mapping. All matching stays in the model; Python only extracts + numbers candidates
and resolves the model's returned indices back to objects (so a fuzzy name the model returns
never has to be re-parsed).

Pure, no LLM, no I/O.
"""

import hashlib
import re
from typing import List, Optional, Tuple

from evals.common.parse import Entry, ParsedWOF
from evals.common.slugify import slugify

_WS = re.compile(r"\s+")


def section_candidates(parsed: ParsedWOF) -> List[str]:
    """The WOF's section titles, in document order (numbered 1-based in the prompt)."""
    return [s.title for s in parsed.sections]


def entry_candidates(parsed: ParsedWOF, wof_section_index: Optional[int]) -> List[Entry]:
    """The entries under the (1-based) WOF section the model matched to a category, or ``[]``
    when the index is ``None``/out of range (the category's section was judged absent)."""
    if wof_section_index is None:
        return []
    i = wof_section_index - 1
    if 0 <= i < len(parsed.sections):
        return list(parsed.sections[i].entries)
    return []


def _entry_span(parsed: ParsedWOF, entry: Entry) -> Tuple[int, int]:
    """The 1-based ``[first, last]`` line span of an entry's body. ``Entry`` carries only a
    start ``lineno``, so the end is the next entry's ``lineno - 1`` (last entry -> the
    section's last body line). This absorbs the entry's prose + its suppressed detail bullets."""
    for sec in parsed.sections:
        for idx, e in enumerate(sec.entries):
            if e is entry:
                start = e.lineno
                if idx + 1 < len(sec.entries):
                    end = sec.entries[idx + 1].lineno - 1
                elif sec.body:
                    end = sec.body[-1][0]
                else:
                    end = start
                return start, max(start, end)
    return entry.lineno, entry.lineno


def cited_quotes(parsed: ParsedWOF, entry: Entry) -> List[Tuple[int, str]]:
    """The footnote quotes an entry cites: ``[^n]`` refs within the entry's line span, mapped
    to their ``[^n]:`` definition text, deduped by ``n`` (first-seen order). This is the
    sourcing narrowing -- the model sees ONLY these, never all of the WOF's footnotes."""
    start, end = _entry_span(parsed, entry)
    defs_by_n = {}
    for d in parsed.fn_defs:
        defs_by_n.setdefault(d.n, d)          # first definition wins on a duplicate n
    out: List[Tuple[int, str]] = []
    seen = set()
    for ref in parsed.fn_refs:
        if start <= ref.lineno <= end and ref.n not in seen:
            d = defs_by_n.get(ref.n)
            if d is not None:                 # a dangling ref (no matching def) is skipped
                out.append((ref.n, d.raw))
                seen.add(ref.n)
    return out


def entity_slug(name: str) -> str:
    """The slug used in a criterion id, via the shared golden-convention slugifier. A
    letterless/all-punctuation name slugs to ``""`` -- which would make a trailing-dot,
    schema-invalid id (``rubric.presence.locations.``) since ``Item.id`` carries no pattern
    guard -- so it falls back to a stable, collision-resistant ``unnamed-<hash>`` token
    (deterministic, so the no-model and model paths still agree on the id)."""
    s = slugify(name)
    if s:
        return s
    return "unnamed-" + hashlib.sha1((name or "").encode("utf-8")).hexdigest()[:8]


def section_key(name: str) -> str:
    """The normalized join key for matching a category's canonical section name against a
    rubric section's name/aliases: casefold, collapse whitespace, and treat ``&`` as ``and``
    (so "People & Cultures" and "People and Cultures" join)."""
    s = (name or "").strip().casefold().replace("&", " and ")
    return _WS.sub(" ", s).strip()
