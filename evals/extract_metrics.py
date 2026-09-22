"""Pure metrics for the extractor A/B eval (evals/extract_ab.py).

This module answers ONE question with numbers: "does a candidate extractor model
drop entities or quotes relative to the trusted Sonnet baseline?" It is
deliberately PURE -- no I/O, no network, no logging, no SDK imports -- so the
harness can feed it data and the fast test suite can exercise every branch with
fabricated inputs. All the real API work, log capture, and printing lives in the
harness; everything here is just data in, verdicts out.

Two families of helper:
  * entity coverage -- ``normalize_name`` + ``match_entities`` diff the two runs'
    entity lists per type (did the candidate lose an entity?).
  * dropped-quote triage -- ``classify_dropped_quote`` sorts each quote the
    candidate's verbatim net dropped into "cosmetic" (a punctuation/spacing
    difference the production normalizer just doesn't fold yet -> recoverable with
    a one-line change) vs "reword" (the model actually paraphrased -> a real loss).

``format_report`` turns all of it into the stdout report, labelling the robust
object-derived metrics (PRIMARY) apart from the best-effort log-derived ones
(DIAGNOSTIC).
"""

import re
from dataclasses import dataclass, field
from typing import Callable


# The six extractor types, in the fixed order the report walks them (matches the
# orchestrator's extractor dict + the renderer's section order). Kept here so the
# harness and the report agree on one ordering.
ENTITY_TYPES = ("locations", "characters", "history", "organizations", "items", "people")


def normalize_name(name: str) -> str:
    """Fold an entity name to a comparison key: strip the ends, squeeze internal
    whitespace to single spaces, and casefold.

    Used ONLY for entity matching -- kept deliberately separate from the
    extractor's ``_normalize_for_match`` (that one is for verbatim quotes and is
    case-PRESERVING on purpose). Here case is noise ("Lake Mundi" == "lake mundi"
    is the same page), so we casefold; punctuation is preserved, since a real name
    difference like "St. Merrow" vs "St Merrow" should surface (the alias set
    absorbs those variants when the models actually record them).

    ``casefold`` rather than ``lower`` so non-ASCII names ("Straße") fold too.
    """
    return re.sub(r"\s+", " ", (name or "").strip()).casefold()


def _aggressive_norm(text: str) -> str:
    """Fold text to its bare alphanumeric skeleton: casefold, then delete EVERY
    non-alphanumeric character (spaces, punctuation, curly quotes, dashes -- all
    gone).

    This is intentionally MORE permissive than the production
    ``_normalize_for_match`` (which keeps case and word spacing). The gap between
    the two is exactly what makes the cosmetic-vs-reword split meaningful: if a
    dropped quote matches a source under this skeleton but NOT under the
    production normalizer, the words are identical and only punctuation/spacing
    the folder misses differs -- a cosmetic drop. Used only by
    ``classify_dropped_quote``.
    """
    return re.sub(r"[^0-9a-z]+", "", (text or "").casefold())


@dataclass
class EntityDiff:
    """One entity type's baseline-vs-candidate coverage diff.

    ``missed`` = entities the baseline found but the candidate did not (the
    candidate LOST them -- the number that matters). ``extra`` = entities only the
    candidate returned (a possible invention, or just a rename that surfaces as
    one missed + one extra -- Han eyeballs those). Both hold ORIGINAL display
    names (real casing), not the normalized keys, so the report reads naturally.
    """
    entity_type: str
    baseline_count: int
    candidate_count: int
    matched: int
    missed: list = field(default_factory=list)
    extra: list = field(default_factory=list)


def _name_and_aliases(entity) -> tuple:
    """Return ``(normalized_name, {normalized alias texts})`` for one lore entity.

    Reads the real lore-model shape: ``.name`` is a plain string and ``.aliases``
    is a ``list[Alias]`` whose elements carry ``.text``. Kept in one place so the
    two-phase matcher below stays readable.
    """
    nn = normalize_name(getattr(entity, "name", "") or "")
    alias_set = {
        normalize_name(a.text)
        for a in (getattr(entity, "aliases", None) or [])
        if getattr(a, "text", None)
    }
    return nn, alias_set


def match_entities(baseline: list, candidate: list, entity_type: str) -> EntityDiff:
    """Diff two runs' entity lists for one type and report matched / missed / extra.

    A baseline entity B matches a candidate entity C when, on normalized names,
    ANY of these holds (exact-on-normalized only -- NO fuzzy/edit-distance
    matching by design; a near-miss like "The Emperor" vs "Emperor Krieger" simply
    surfaces as one missed + one extra for Han to eyeball):
      * their names are equal, or
      * B's name is one of C's alias texts, or
      * C's name is one of B's alias texts.
    (Name-to-name and name-to-alias only -- deliberately not alias-to-alias.)

    The matching is a two-phase greedy pass over a CONSUMED-candidate pool, so one
    candidate can never be claimed by two different baselines (which would
    double-count a match):
      * Phase 1 -- strongest signal first: each still-unmatched baseline, in
        order, claims the first still-available candidate with an EQUAL name.
      * Phase 2 -- weaker alias membership: each remaining baseline claims the
        first still-available candidate linked by an alias either way.
    Doing exact-name before alias stops a weak alias match from stealing a
    candidate a later baseline needs by exact name.

    If two baselines would both match one candidate, the first in order claims it
    and the second falls to ``missed`` -- a deterministic, documented tradeoff that
    slightly over-states misses but never mis-answers "did the candidate drop this
    entity?". Empty either side is handled naturally (empty baseline -> everything
    is ``extra``; empty candidate -> everything is ``missed``).
    """
    # Precompute normalized name + alias set once per entity (avoids re-normalizing
    # inside the nested scan).
    base_keys = [_name_and_aliases(b) for b in baseline]
    cand_keys = [_name_and_aliases(c) for c in candidate]

    available = set(range(len(candidate)))   # candidate indices not yet claimed
    matched_baseline = set()                 # baseline indices that found a match

    def _claim(predicate):
        """For each unmatched baseline in order, claim the first available
        candidate satisfying ``predicate(base_idx, cand_idx)``."""
        for bi in range(len(baseline)):
            if bi in matched_baseline:
                continue
            for ci in sorted(available):
                if predicate(bi, ci):
                    matched_baseline.add(bi)
                    available.discard(ci)
                    break

    # Phase 1: exact normalized-name equality.
    _claim(lambda bi, ci: base_keys[bi][0] and base_keys[bi][0] == cand_keys[ci][0])
    # Phase 2: alias membership either direction (B's name in C's aliases, or C's
    # name in B's aliases). Guard against empty names matching an empty set.
    _claim(lambda bi, ci: (
        (base_keys[bi][0] and base_keys[bi][0] in cand_keys[ci][1])
        or (cand_keys[ci][0] and cand_keys[ci][0] in base_keys[bi][1])
    ))

    missed = [baseline[bi].name for bi in range(len(baseline)) if bi not in matched_baseline]
    extra = [candidate[ci].name for ci in range(len(candidate)) if ci in available]
    return EntityDiff(
        entity_type=entity_type,
        baseline_count=len(baseline),
        candidate_count=len(candidate),
        matched=len(matched_baseline),
        missed=missed,
        extra=extra,
    )


def classify_dropped_quote(
    dropped_quote: str,
    source_messages: list,
    current_normalizer: Callable[[str], str],
) -> str:
    """Sort one dropped candidate quote into ``"cosmetic"`` / ``"reword"`` /
    ``"unknown"``.

    Why a candidate quote gets dropped: the extractor's verbatim net
    (``_quote_is_verbatim``) requires each supporting quote to appear -- once both
    are run through the production normalizer -- as a SUBSTRING of some message in
    its file-pure batch. A quote that isn't verbatim anywhere has its whole detail
    dropped. So a paraphrasing model loses lore; it never corrupts it. This
    function tells apart the two reasons a quote failed that check:

      * ``"cosmetic"`` -- the words ARE present in a source message (matches under
        the aggressive skeleton), and the production normalizer rejected it only
        over punctuation/spacing/case, NOT the words themselves -- a model that
        copies faithfully, not a real loss. Usually a one-line
        ``_normalize_for_match`` fold recovers it; the exception is a CASE-only
        difference, which production rejects on purpose (case is treated as a real
        change), so eyeball the printed samples before concluding "just tweak the
        normalizer".
      * ``"reword"`` -- the words themselves are absent from every source even
        under the aggressive skeleton, so the model paraphrased or invented. A
        genuine lore loss.
      * ``"unknown"`` -- no source messages to compare against, or a blank quote,
        or the degenerate case where the quote matches under BOTH normalizers
        (which is inconsistent with a real drop -- see below). Flagged honestly
        rather than force-fit into a bucket.

    ``current_normalizer`` is INJECTED (the harness passes the real
    ``agents.base_extractor._normalize_for_match``; the unit test passes a trivial
    fake) so this function stays pure and independently testable.

    Both comparisons use SUBSTRING containment, NOT equality -- mirroring
    ``_quote_is_verbatim``'s substring check, because a supporting quote is a
    FRAGMENT of a longer message. An equality test would report "reword" for
    nearly every real drop and make the whole diagnostic useless.

    (The both-match "unknown" case can arise because the harness compares against
    ALL filtered messages, a superset of the one file-pure batch the drop actually
    happened in -- a quote can be verbatim in some OTHER file yet dropped for not
    being in its own batch. We don't claim that as cosmetic or reword.)
    """
    if not source_messages:
        return "unknown"

    q_agg = _aggressive_norm(dropped_quote)
    if not q_agg:
        return "unknown"                       # blank / whitespace-only quote

    agg_hit = any(q_agg in _aggressive_norm(src) for src in source_messages)
    if not agg_hit:
        return "reword"                        # words absent everywhere -> paraphrase

    q_cur = current_normalizer(dropped_quote)
    cur_hit = any(q_cur in current_normalizer(src) for src in source_messages)
    if not cur_hit:
        return "cosmetic"                      # words present, only punctuation/spacing differs

    return "unknown"                           # present under BOTH -> inconsistent with a real drop


def _fmt_table(header: list, rows: list) -> str:
    """Render a simple left/right-aligned text table. First column is
    left-justified (labels); the rest are right-justified (numbers). Pure string
    work -- the report's only layout helper."""
    all_rows = [header] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]
    out = []
    for r in all_rows:
        cells = [r[0].ljust(widths[0])] + [r[i].rjust(widths[i]) for i in range(1, len(r))]
        out.append("  ".join(cells).rstrip())
    return "\n".join(out)


def format_report(
    models: dict,
    per_type_diffs: list,
    quote_stats: dict,
    json_retry_stats: dict,
) -> str:
    """Build the full stdout report as one string (pure -- the harness prints it).

    Expected input shapes (the harness assembles these):
      * ``models`` -- ``{"baseline": <spec>, "candidate": <spec>,
        "filtered_messages": <int>}``.
      * ``per_type_diffs`` -- ``list[EntityDiff]``, one per type (report order).
      * ``quote_stats`` -- ``{"survival": {type: {"baseline": int,
        "candidate": int}}, "drops": {"cosmetic": int, "reword": int,
        "unknown": int}, "drop_samples": [(classification, quote_text), ...]}``.
      * ``json_retry_stats`` -- ``{"baseline": int, "candidate": int}``.

    Two tiers are labelled literally in the output: PRIMARY (robust -- read
    straight off the returned lore objects) and DIAGNOSTIC (best-effort -- parsed
    from WARNING logs, so cache-sensitive; see the caveats printed inline).
    """
    baseline = models.get("baseline", "?")
    candidate = models.get("candidate", "?")
    lines = []

    lines.append("=== Wikitizer A/B Extractor Eval ===")
    lines.append(f"Baseline : {baseline}")
    lines.append(f"Candidate: {candidate}")
    lines.append(f"Filtered messages (shared input): {models.get('filtered_messages', '?')}")
    lines.append("")

    # ---- PRIMARY tier -------------------------------------------------------- #
    lines.append("--- PRIMARY (robust; computed from returned lore objects) ---")
    lines.append("")
    lines.append("Entity coverage")
    cov_rows = []
    tot = {"base": 0, "cand": 0, "matched": 0, "missed": 0, "extra": 0}
    for d in per_type_diffs:
        cov_rows.append([
            d.entity_type, d.baseline_count, d.candidate_count,
            d.matched, len(d.missed), len(d.extra),
        ])
        tot["base"] += d.baseline_count
        tot["cand"] += d.candidate_count
        tot["matched"] += d.matched
        tot["missed"] += len(d.missed)
        tot["extra"] += len(d.extra)
    cov_rows.append(["TOTAL", tot["base"], tot["cand"], tot["matched"], tot["missed"], tot["extra"]])
    lines.append(_fmt_table(["type", "base", "cand", "matched", "missed", "extra"], cov_rows))
    lines.append("")

    # Named lists of what the candidate missed / invented, per type (only where
    # there's something to show).
    lines.append("Entities MISSED by candidate (in baseline, no candidate match):")
    any_missed = False
    for d in per_type_diffs:
        if d.missed:
            any_missed = True
            lines.append(f"  {d.entity_type}: " + "; ".join(d.missed))
    if not any_missed:
        lines.append("  (none)")
    lines.append("")
    lines.append("Entities EXTRA in candidate (no baseline match -- rename or invention):")
    any_extra = False
    for d in per_type_diffs:
        if d.extra:
            any_extra = True
            lines.append(f"  {d.entity_type}: " + "; ".join(d.extra))
    if not any_extra:
        lines.append("  (none)")
    lines.append("")

    # Surviving-quote counts per type (sum of len(supporting_quotes)).
    lines.append("Supporting-quote survival (sum of len(supporting_quotes) per type)")
    survival = quote_stats.get("survival", {})
    q_rows = []
    qb_tot = qc_tot = 0
    for d in per_type_diffs:
        s = survival.get(d.entity_type, {})
        b = s.get("baseline", 0)
        c = s.get("candidate", 0)
        q_rows.append([d.entity_type, b, c, c - b])
        qb_tot += b
        qc_tot += c
    q_rows.append(["TOTAL", qb_tot, qc_tot, qc_tot - qb_tot])
    lines.append(_fmt_table(["type", "base", "cand", "delta"], q_rows))
    lines.append("")

    # ---- DIAGNOSTIC tier ----------------------------------------------------- #
    lines.append("--- DIAGNOSTIC (best-effort; parsed from WARNING logs; cache-sensitive) ---")
    lines.append("")
    drops = quote_stats.get("drops", {})
    cosmetic = drops.get("cosmetic", 0)
    reword = drops.get("reword", 0)
    unknown = drops.get("unknown", 0)
    lines.append("Dropped quotes on the CANDIDATE run (paraphrase -> detail dropped):")
    lines.append(f"  cosmetic: {cosmetic}    reword: {reword}    unknown: {unknown}")
    lines.append("  (cosmetic = same words, only punctuation/spacing/case differs -- "
                 "usually a one-line _normalize_for_match fold recovers it, EXCEPT a "
                 "case-only diff, which production rejects on purpose; eyeball the "
                 "samples. reword = the model actually paraphrased.)")
    samples = quote_stats.get("drop_samples", [])
    if samples:
        lines.append("  samples:")
        for classification, quote_text in samples:
            lines.append(f"    [{classification}] {quote_text}")
    lines.append("")
    lines.append("JSON parse-retries (json_repair re-asks; ACCURATE ONLY on an "
                 "uncached/paid run -- 0 under a full cache):")
    lines.append(f"  baseline : {json_retry_stats.get('baseline', 0)}")
    lines.append(f"  candidate: {json_retry_stats.get('candidate', 0)}")

    return "\n".join(lines)
