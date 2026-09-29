"""Hybrid entry pairing -- the GBF spine, run as its own pass BEFORE any meaning judgment.

For each gold entry we find the WOF entry that is the same entity, or ``None``. Two stages:
  1. **Python exact locks** (deterministic, no model): equal anchor slug, then equal normalized
     name. Each fires ONLY when the key is unique on both sides (an ambiguous key never false-locks;
     a certain match never wastes a model call -- a model can only add error to ``aldenburg <->
     aldenburg``).
  2. **Model residual** (voted, one call): the still-unmatched gold + WOF entries go to the judge,
     which matches same-entity pairs. A residual gold entry the model can't place stays unmatched.

Keeping pairing a separate pass means a pairing miss never hides inside a meaning verdict, and the
ordering check reuses the exact same map (zero extra calls). Layering: GBF does not import from
``evals/rubric/`` -- the tiny name-normalizer + slug-id are reimplemented here as local twins of
``rubric.resolve.section_key`` / ``entity_slug`` (see ``norm_name`` / ``_slug_id``).
"""

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from evals.common.parse import Entry, ParsedWOF
from evals.common.slugify import slugify

logger = logging.getLogger("evals.gbf.pair")

_WS = re.compile(r"\s+")


def norm_name(name: str) -> str:
    """Normalized join key for an exact-name lock (a local twin of ``rubric.resolve.section_key``):
    strip, casefold, treat ``&`` as ``and``, collapse whitespace. So "The Kraken Clan" folds only
    with an identically-worded name -- articles are NOT stripped here (that would over-merge e.g. a
    Location "Citadel" with an Organization "The Citadel")."""
    s = (name or "").strip().casefold().replace("&", " and ")
    return _WS.sub(" ", s).strip()


def _slug_id(name: str) -> str:
    """A criterion-id slug for a gold entry name (a local twin of ``rubric.resolve.entity_slug``):
    the shared golden-convention slugifier, with a stable ``unnamed-<hash>`` fallback for a
    letterless name so an id never ends in a schema-invalid trailing dot."""
    s = slugify(name)
    if s:
        return s
    return "unnamed-" + hashlib.sha1((name or "").encode("utf-8")).hexdigest()[:8]


def gold_slug(entry: Entry) -> str:
    """The slug identifying a gold entry: its authored anchor when it has one (unique in the gold,
    and the key the ranks YAML uses), else a slug of its name."""
    return entry.anchor_id or _slug_id(entry.name)


def real_entries(parsed: ParsedWOF) -> List[Entry]:
    """Every real content entry across the document, in section-then-document order, EXCLUDING the
    footnotes section (identity check -- it is also present in ``parsed.sections``) and grouping
    headings (``is_grouping`` labels like "Could Not Place", which are not entities)."""
    out: List[Entry] = []
    for sec in parsed.sections:
        if sec is parsed.footnotes_section:
            continue
        for e in sec.entries:
            if not e.is_grouping:
                out.append(e)
    return out


@dataclass
class PairMap:
    """The result of pairing. All lists are parallel by gold-entry index.

    ``matched[i]`` is the WOF ``Entry`` gold entry ``i`` paired to, or ``None``. ``residual_failed``
    is True when the model residual call errored -- pairing is then unreliable, so the runner turns
    the unmatched-residual entries into ``skipped`` (we couldn't judge, not "absent") and skips
    ordering. ``residual_indices`` are the gold indices that were sent to the model (so the runner
    can tell a model-said-no-match ``na`` from a call-errored ``skipped``)."""

    gold_entries: List[Entry]
    gold_slugs: List[str]
    matched: List[Optional[Entry]]
    residual_failed: bool = False
    residual_indices: Tuple[int, ...] = field(default_factory=tuple)

    def wof_id_to_slug(self) -> Dict[int, str]:
        """``{id(wof_entry): gold_slug}`` for the matched pairs -- ordering uses object identity to
        map a WOF History entry back to its gold slug (WOF entries are unhashable dataclasses, and
        first-wins pairing guarantees each WOF entry appears at most once)."""
        return {id(w): s for w, s in zip(self.matched, self.gold_slugs) if w is not None}


def _lock_by_key(key_fn, gold_es, wof_es, matched, used_wof):
    """Fill ``matched`` for any still-unmatched gold entry whose ``key_fn`` value is unique among
    still-unmatched gold AND among still-unmatched WOF (uniqueness-gated, so an ambiguous key is
    left for the model). Mutates ``matched`` + ``used_wof`` in place."""
    gold_unmatched = [i for i in range(len(gold_es)) if matched[i] is None]
    wof_unmatched = [j for j in range(len(wof_es)) if j not in used_wof]

    gold_by_key: Dict[str, List[int]] = {}
    for i in gold_unmatched:
        k = key_fn(gold_es[i])
        if k:
            gold_by_key.setdefault(k, []).append(i)
    wof_by_key: Dict[str, List[int]] = {}
    for j in wof_unmatched:
        k = key_fn(wof_es[j])
        if k:
            wof_by_key.setdefault(k, []).append(j)

    for k, gis in gold_by_key.items():
        wjs = wof_by_key.get(k)
        if len(gis) == 1 and wjs is not None and len(wjs) == 1:
            matched[gis[0]] = wof_es[wjs[0]]
            used_wof.add(wjs[0])


def pair(parsed_wof: ParsedWOF, gold: ParsedWOF, adj) -> PairMap:
    """Pair each gold entry to a WOF entry (or ``None``). Python exact locks first, then one voted
    model call over the residual. The residual call is guarded here (co-located with the pairing it
    serves): on a request-time error, ``residual_failed`` is set and the still-unmatched residual
    entries are left ``None`` for the runner to mark ``skipped`` (never a false ``na``)."""
    gold_es = real_entries(gold)
    wof_es = real_entries(parsed_wof)
    gold_slugs = [gold_slug(e) for e in gold_es]
    matched: List[Optional[Entry]] = [None] * len(gold_es)
    used_wof: set = set()

    # Stage 1: exact locks, anchor first (most certain), then normalized name.
    _lock_by_key(lambda e: (e.anchor_id or "").strip(), gold_es, wof_es, matched, used_wof)
    _lock_by_key(lambda e: norm_name(e.name), gold_es, wof_es, matched, used_wof)

    # Stage 2: model residual over what's left on both sides.
    residual_gold_idx = [i for i in range(len(gold_es)) if matched[i] is None]
    residual_wof_idx = [j for j in range(len(wof_es)) if j not in used_wof]
    residual_failed = False
    residual_indices: Tuple[int, ...] = ()

    if residual_gold_idx and residual_wof_idx:
        residual_indices = tuple(residual_gold_idx)
        gold_residual = [gold_es[i] for i in residual_gold_idx]
        wof_residual = [wof_es[j] for j in residual_wof_idx]
        try:
            mapping = adj.pair_residual(gold_residual, wof_residual)
        except Exception as exc:                     # noqa: BLE001 -- degrade, never abort
            logger.warning("[REVIEW] gbf residual pairing errored (%s); leaving %d gold entr(y/ies) "
                           "unpaired", exc, len(residual_gold_idx))
            residual_failed = True
        else:
            for local_g, local_w in mapping.items():
                if local_w is None:
                    continue
                if not (0 <= local_g < len(residual_gold_idx)) or not (0 <= local_w < len(residual_wof_idx)):
                    continue
                gi = residual_gold_idx[local_g]
                wj = residual_wof_idx[local_w]
                if matched[gi] is not None or wj in used_wof:
                    logger.warning("[REVIEW] gbf residual pairing proposed a conflicting match "
                                   "(gold #%d / wof #%d); keeping the first", gi, wj)
                    continue
                matched[gi] = wof_es[wj]
                used_wof.add(wj)

    unmatched = sum(1 for m in matched if m is None)
    extra_wof = len(wof_es) - len(used_wof)
    if extra_wof:
        logger.info("gbf pairing: %d WOF entr(y/ies) not present in the gold (ignored for scoring)",
                    extra_wof)
    logger.info("gbf pairing: %d/%d gold entries matched (%d unmatched)",
                len(gold_es) - unmatched, len(gold_es), unmatched)
    return PairMap(gold_es, gold_slugs, matched, residual_failed, residual_indices)


def majority_index(mapping_votes: List[Dict[int, Optional[int]]], n: int) -> Dict[int, Optional[int]]:
    """Fold per-index integer votes (residual pairing) into a per-index majority. For each of the
    ``n`` residual gold indices, take the most common non-negative ``wof_index`` across the votes;
    a tie, an all-``None`` column, or no votes -> ``None`` (leave unmatched) + a ``[REVIEW]`` log."""
    out: Dict[int, Optional[int]] = {}
    for i in range(n):
        col = [v.get(i) for v in mapping_votes]
        choices = [c for c in col if isinstance(c, int) and not isinstance(c, bool) and c >= 0]
        if not choices:
            out[i] = None
            continue
        counts = Counter(choices).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            logger.warning("[REVIEW] gbf residual pairing: tie for gold #%d; leaving unmatched", i)
            out[i] = None
        else:
            out[i] = counts[0][0]
    return out
