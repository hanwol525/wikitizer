"""Phase 4.6 Part 2: turn the full lore into a RESTRICTED view that hides
excluded source files.

Two pure, LLM-free jobs:
  * validate_exclusions -- fail fast if an excluded name isn't one of the input
    files (a silent mismatch would leave secrets in the "restricted" doc).
  * filter_entities -- carve confidential-only facts and quotes out of the
    reconciled entities, dropping any entity left with nothing public.

History is NOT filtered here -- the orchestrator handles History by re-running its
extraction over the non-excluded messages (a secret can't leak because the
extractor never sees it). This module only touches the five fact-carrying entity
types (Location / Character / Organization / Item / PeopleAndCultures).

Names and aliases now carry provenance too (Part 3): ``name_sources`` on each entity
and ``Alias.source_files``, both tagged at extraction from the entry's file-pure
batch. So the carve also strips secret-only aliases and, when the canonical name
isn't provably public, RE-HEADS the entity to its first surviving public alias --
closing the entity-name leak an earlier review found. See ``_filter_entity``.
"""

import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Same convention as the agents/renderer/orchestrator: loud, human-actionable flags
# carry this prefix so the Phase 5 review pass can find them.
REVIEW_PREFIX = "[REVIEW]"


def validate_exclusions(exclude_sources: list[str], files: list[str]) -> None:
    """Raise ValueError if any name in `exclude_sources` isn't one of the input
    files. Match is on the BARE FILENAME, because that's what every message stores
    as its source (the parser sets source_file = Path(filepath).name). So the
    exclude list must be bare filenames ('secret.txt'), NOT paths ('logs/secret.txt').

    Fail loud and early: a name matching nothing would silently leave those messages
    in the restricted doc -- the exact leak this feature exists to prevent. This is
    STRICT on purpose -- even a path to a real file (e.g. 'logs/dm.txt' when 'dm.txt'
    is a real input) is an error, so you always see the exact valid names and confirm
    you're excluding what you think.
    """
    basenames = [Path(f).name for f in files]
    counts = Counter(basenames)
    dupes = sorted([name for name, c in counts.items() if c > 1])
    if dupes:
        raise ValueError(
            "Cannot build exclusions with duplicate input basenames (messages store only "
            f"the bare filename as source_file). Duplicates: {dupes}"
        )

    valid = set(basenames)          # the bare filenames messages carry
    unknown = [name for name in exclude_sources if name not in valid]
    if not unknown:
        return

    # Build a helpful message. For a bad name that's really a PATH to a real file
    # (the most likely slip, since `files` is a list of paths), point at the bare name.
    problems = []
    for name in unknown:
        base = Path(name).name
        if base != name and base in valid:
            problems.append(f"{name!r} looks like a path -- use the bare filename {base!r}")
        else:
            problems.append(f"{name!r} is not one of the input files")
    raise ValueError(
        "Cannot exclude sources that aren't input files. "
        + "; ".join(problems)
        + f". Valid filenames are: {sorted(valid)}"
    )


def _filter_entity(entity, excluded):
    """Return a COPY of `entity` with confidential-only facts, quotes and aliases
    removed and -- if its canonical name isn't provably public -- re-headed to a
    surviving public alias. Returns None if nothing public survives.

    `excluded` is a set of bare filenames. Rules:
      * Keep a Detail if AT LEAST ONE of its source_files is NOT excluded (the fact
        is also public). A Detail with no sources at all is dropped -- we can't prove
        it's public, and over-hiding is the safe direction for a secrecy feature.
      * Keep a Quote if its (single) source_file is NOT excluded.
      * Drop the entity if it has neither a surviving Detail nor a surviving Quote.
      * Keep an Alias on the same any-source-public rule as a Detail.
      * If `name` isn't provably public (no non-excluded entry in `name_sources`, or
        no sources at all), PROMOTE the first surviving alias to be the heading. The
        old secret name is dropped outright, not demoted to an alias.

    The promotion can't run out of options -- see INVARIANT 2 in the brief: a
    surviving public detail/quote belongs to a member from a public file, and
    `_combine_group` folds every losing member name into the alias pool, so a public
    alias is always there. We still handle the impossible case by dropping + logging
    [REVIEW], because a broken invariant must be loud rather than silently leaky.

    Builds a NEW object via model_copy so the reconciled originals (which the FULL
    doc still needs) are never mutated. One code path works for all five types
    uniformly: model_copy carries every other field through untouched, including a
    Character's is_pc / player_name. HistoryEvent never reaches here -- it has no
    `details`, and the orchestrator re-runs its extraction instead of carving it.
    """
    kept_details = [d for d in entity.details
                    if any(sf not in excluded for sf in d.source_files)]
    kept_quotes = [q for q in entity.supporting_quotes
                   if q.source_file not in excluded]
    if not kept_details and not kept_quotes:
        return None

    kept_aliases = [a for a in entity.aliases
                    if any(sf not in excluded for sf in a.source_files)]

    name = entity.name
    name_sources = entity.name_sources
    if not any(sf not in excluded for sf in name_sources):
        # The heading came solely from an excluded file (or carries no provenance at
        # all, which we treat identically -- can't prove it's public).
        if not kept_aliases:
            logger.warning(
                "%s No public name for entity %r after exclusion, and no public alias to "
                "re-head to -- dropping it. This should be impossible (a surviving public "
                "fact implies a public member name in the alias pool); check whether "
                "extractor batches are still file-pure.", REVIEW_PREFIX, entity.name,
            )
            return None
        promoted = kept_aliases[0]      # first-seen order
        logger.debug("Exclusion: re-heading %r -> %r (canonical was source-excluded)",
                     entity.name, promoted.text)
        name = promoted.text
        name_sources = promoted.source_files   # kept as-is, never rewritten
        kept_aliases = kept_aliases[1:]        # the new heading shouldn't alias itself

    return entity.model_copy(update={"name": name,
                                     "name_sources": name_sources,
                                     "aliases": kept_aliases,
                                     "details": kept_details,
                                     "supporting_quotes": kept_quotes})


def filter_entities(entities: list, excluded) -> list:
    """Filter a list of same-type entities; drop the ones left wholly confidential.
    Returns a new list of new objects (never mutates the inputs)."""
    out = []
    for e in entities:
        kept = _filter_entity(e, excluded)
        if kept is not None:
            out.append(kept)
    return out
