"""Rank-based, co-temporal-aware ordering check -- pure Python, no model call (its verdict rests on
the model-produced pairing, so the emitted item is still ``engine="model"``).

The gold's chronological ranks are the reference (``output/gbf_event_ranks.yaml`` -> ``{slug: rank}``;
co-temporal events SHARE a rank; rank-less events, e.g. "Could Not Place", are excluded). We read the
WOF's History events in document order, map each to its gold rank via the pairing map, and count
INVERSIONS: a pair the WOF orders opposite to the ranks. Same-rank (co-temporal) pairs never count --
swapping them is not an error.

The WOF History section is located by NORMALIZED name (FIX #3 spirit -- never an exact title-string
match), so a WOF that titles it "Timeline" still grades.
"""

import logging
from typing import Dict, List, Tuple

from evals.common.models import Engine, Evidence, Item, Status
from evals.common.parse import ParsedWOF
from evals.gbf.pair import PairMap, gold_slug, norm_name

logger = logging.getLogger("evals.gbf.ordering")

_ORDERING_ID = "gbf.ordering"
_ORDERING_DESC = "History events in chronological order"
_MAX_PAIRS_SHOWN = 8


def _status(n_ranked: int, inversions: int) -> Status:
    """0 inversions -> PASS; a small minority (``<= max(1, n//10)``) -> PARTIAL; else FAIL. A twin of
    ``evals.fc.fc_lint.status_from_counts`` (kept local so GBF doesn't reach into another eval)."""
    if inversions == 0:
        return Status.PASS
    if inversions <= max(1, n_ranked // 10):
        return Status.PARTIAL
    return Status.FAIL


def _history_titles(gold: ParsedWOF, ranks: Dict[str, int]) -> set:
    """Normalized titles that count as the History section: the always-accepted "history"/"timeline"
    (the renderer emits one of those), plus the normalized title of whatever gold section actually
    holds ranked events (so a gold that named it differently still anchors the lookup)."""
    titles = {"history", "timeline"}
    for s in gold.sections:
        if any(gold_slug(e) in ranks for e in s.entries if not e.is_grouping):
            titles.add(norm_name(s.title))
    return titles


def _wof_history_events(wof: ParsedWOF, titles: set) -> List:
    for s in wof.sections:
        if norm_name(s.title) in titles:
            return [e for e in s.entries if not e.is_grouping]
    return []


def score_ordering(gold: ParsedWOF, wof: ParsedWOF, pair_map: PairMap,
                   ranks: Dict[str, int]) -> Item:
    """Grade the WOF's History ordering against the gold ranks. ``na`` when fewer than 2 rankable
    events are present (nothing to order)."""
    id_to_slug = pair_map.wof_id_to_slug()
    events = _wof_history_events(wof, _history_titles(gold, ranks))

    # (entry, rank) for each WOF history event that paired to a ranked gold event, in document order.
    ranked: List[Tuple[object, int]] = []
    for e in events:
        slug = id_to_slug.get(id(e))
        if slug is None:
            continue
        rank = ranks.get(slug)
        if rank is None:
            continue
        ranked.append((e, rank))

    if len(ranked) < 2:
        return Item(id=_ORDERING_ID, description=_ORDERING_DESC, engine=Engine.MODEL,
                    status=Status.NA,
                    evidence=Evidence(lines=[], detail=f"only {len(ranked)} rankable event(s) present; "
                                                       "nothing to order"))

    inversions: List[Tuple[object, int, object, int]] = []
    for a in range(len(ranked)):
        ea, ra = ranked[a]
        for b in range(a + 1, len(ranked)):
            eb, rb = ranked[b]
            if ra != rb and ra > rb:              # a is before b in the WOF but ranks later
                inversions.append((ea, ra, eb, rb))

    n = len(ranked)
    status = _status(n, len(inversions))
    lines = sorted({e.lineno for e, _ in ranked if getattr(e, "lineno", None)})
    if not inversions:
        return Item(id=_ORDERING_ID, description=_ORDERING_DESC, engine=Engine.MODEL, status=status,
                    evidence=Evidence(lines=lines,
                                      detail=f"{n} events in chronological order (0 inversions)"))

    shown = inversions[:_MAX_PAIRS_SHOWN]
    parts = [f"expected {id_to_slug.get(id(eb))!r} before {id_to_slug.get(id(ea))!r}, "
             f"WOF had {id_to_slug.get(id(ea))!r} before {id_to_slug.get(id(eb))!r}"
             for ea, _ra, eb, _rb in shown]
    more = f" (+{len(inversions) - len(shown)} more)" if len(inversions) > len(shown) else ""
    inv_lines = sorted({ln for ea, _ra, eb, _rb in inversions
                        for ln in (getattr(ea, "lineno", None), getattr(eb, "lineno", None)) if ln})
    return Item(id=_ORDERING_ID, description=_ORDERING_DESC, engine=Engine.MODEL, status=status,
                evidence=Evidence(lines=inv_lines,
                                  detail=f"{len(inversions)} inversion(s) across {n} events: "
                                         + "; ".join(parts) + more))
