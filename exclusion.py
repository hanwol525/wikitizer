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

KNOWN LIMITATION -- entity NAME/ALIAS leak (confirmed by adversarial review, fix
DEFERRED to a follow-up brief). Carving scrubs ``details`` and ``supporting_quotes``,
which carry source provenance (``Detail.source_files`` / ``Quote.source_file``). An
entity's ``name`` and ``aliases`` are bare strings with NO provenance, so they can't
be scrubbed here: a SURVIVING entity (kept because it has any public fact/quote)
whose canonical name -- or an alias -- originated *solely* from an excluded file will
still show that name in the restricted doc's heading, anchor id, and cross-link
table. Two reachable paths: (a) the reconciler prefers a proper name as canonical,
so a public "the old fort" merged with a secret "Blackspire Keep" heads the merged
entity with the secret name; (b) the org/item/people fallback sets ``name`` to a
detail's text, which can be a secret fact. The leak-proof fix is to RE-EXTRACT the
five entity types over the non-excluded messages (exactly how History is handled
above), or to give names/aliases their own provenance and scrub/re-head here; both
are out of the current brief's scope. Tracked by the xfail regression test
``test_restricted_doc_leaks_secret_derived_entity_name_KNOWN_GAP`` in
tests/test_orchestrator.py, which flips to passing once the gap is closed.
"""

from collections import Counter
from pathlib import Path


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
    """Return a COPY of `entity` with confidential-only facts and confidential
    quotes removed -- or None if nothing public survives (drop the whole entity).

    `excluded` is a set of bare filenames. Rules:
      * Keep a Detail if AT LEAST ONE of its source_files is NOT excluded (the fact
        is also public). A Detail with no sources at all is dropped -- we can't prove
        it's public, and over-hiding is the safe direction for a secrecy feature.
      * Keep a Quote if its (single) source_file is NOT excluded.
      * Keep the entity if it still has any Detail OR any Quote; else drop it.

    Builds a NEW object via model_copy so the reconciled originals (which the FULL
    doc still needs) are never mutated. One code path works for all five types
    uniformly: model_copy carries every other field through untouched, including a
    Character's is_pc / player_name.
    """
    kept_details = [d for d in entity.details
                    if any(sf not in excluded for sf in d.source_files)]
    kept_quotes = [q for q in entity.supporting_quotes
                   if q.source_file not in excluded]
    if not kept_details and not kept_quotes:
        return None
    return entity.model_copy(update={"details": kept_details,
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
