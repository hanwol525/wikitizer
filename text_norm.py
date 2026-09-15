"""Canonical name-folding shared by the anchor layer (renderer/crosslink.slugify)
and the merge layer (agents/reconciler._name_key).

These two used to normalize names DIFFERENTLY: the cross-linker's ``slugify`` strips
apostrophes and ASCII-folds accents, but the reconciler's grouping key kept them -- so
a name like "Scoia'tael" split into several reconciler buckets (never merged) yet
collapsed to ONE crosslink anchor, producing duplicate pages AND dead links. Both now
delegate here, so the anchor and the grouping key can never drift apart again.

The load-bearing invariant: ``slug(a) == slug(b)`` IMPLIES ``name_key(a) == name_key(b)``.
The grouping key is defined in terms of the slug precisely so that guarantee holds by
construction -- two names that will become the same anchor always land in the same
merge bucket.

Pure, no I/O, no logging: ``re`` + ``unicodedata`` only.
"""

import re
import unicodedata


def slug(name: str) -> str:
    """Turn an entity name into an anchor id / merge slug.

    Pipeline, in order:
      1. ASCII-fold via NFKD: decompose, drop combining marks, keep only ASCII
         ("Théoden" -> "theoden", "Canción" -> "cancion").
      2. Lowercase.
      3. Whitespace runs -> hyphens.
      4. Keep internal hyphens; strip apostrophes and anything else outside
         [a-z0-9-] ("Mal'taav" -> "maltaav", "Half-Elf" -> "half-elf").
      5. Collapse hyphen runs to one; trim the ends.

    Returns "" for an all-punctuation / no-ASCII-letter name.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(
        ch for ch in decomposed
        if not unicodedata.combining(ch) and ord(ch) < 128
    )
    lowered = ascii_only.lower()
    hyphenated = re.sub(r"\s+", "-", lowered)
    kept = re.sub(r"[^a-z0-9-]", "", hyphenated)
    return re.sub(r"-+", "-", kept).strip("-")


def name_key(name: str) -> str:
    """A merge-grouping key that AGREES with ``slug`` on which names are "the same"
    (case, whitespace, apostrophes, accents, and stray punctuation all folded away)
    but keeps words SPACE-separated instead of hyphen-joined -- so a downstream
    article-strip ("the citadel" -> "citadel") and token split still work on words.

    Guarantees ``slug(a) == slug(b)`` => ``name_key(a) == name_key(b)`` because it is
    literally the slug with hyphens turned back into spaces. Returns "" for an
    all-punctuation name (callers that must never group junk together check for that).
    """
    return slug(name).replace("-", " ").strip()
