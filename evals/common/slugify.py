"""The golden-convention slug rules the FC eval pins, plus the tolerant oracle that
compares an actual anchor id against an entry name.

This is deliberately a TWIN of the production ``text_norm.slug`` -- NOT a reuse of it --
because the golden response and the FC checklist require a rule the production slug
breaks: **non-ASCII letters are preserved** (``Skjöldr Aldvarðr`` -> ``skjöldr-aldvarðr``),
whereas ``text_norm.slug`` ASCII-folds ("skjoldr-aldvarr"). Reusing it would false-fail
``fc.slugs.non-ascii``. The other rules match: lowercase, hyphen-join words, drop
apostrophes/quotes, flatten parentheticals.

The slug is only "as correct as the rules pinned here," so every rule is validated against
``output/gol-lore-full.md`` (see tests). The comparison is deliberately TOLERANT -- it
accepts either a leading-article drop or a ``-N`` disambiguation suffix -- because the
golden itself is inconsistent about the article (``The Citadel`` -> ``citadel-2`` drops it,
``The Dvergarim`` -> ``the-dvergarim`` keeps it), and the FC eval's whole ethos is to
under-flag before ever manufacturing a wrong finding.

Pure: ``re`` + ``unicodedata`` only.
"""

import re
import unicodedata

# Apostrophes and quotation marks (straight + curly, single + double) that are DELETED
# from a slug -- so "Crown's Nest" -> "crowns-nest" (not "crown-s-nest").
_QUOTE_CHARS = "'‘’\"“”"

# A leading English article that the golden sometimes drops from a slug/anchor.
_LEADING_ARTICLE_RE = re.compile(r"^\s*the\s+", re.IGNORECASE)

# A "<base>-<number>" disambiguation suffix (e.g. "citadel-2").
_SUFFIX_RE = re.compile(r"-\d+$")


def slugify(name: str) -> str:
    """Turn an entity/entry name into its expected anchor id under the golden convention.

    Pipeline, in order:
      1. NFC-normalize, so a decomposed accent (o + combining diaeresis) becomes a single
         precomposed letter and survives as one word char.
      2. DELETE (no hyphen in their place):
           - apostrophes/quotes (straight + curly), so "Crown's" -> "crowns"; and
           - non-ASCII PUNCTUATION/symbols like the en/em-dash and ellipsis, so
             "Maltraav–Kriega" -> "maltraavkriega" (matches the golden -- the en-dash is
             dropped, NOT turned into a hyphen). Non-ASCII LETTERS (``ö``/``ð``) are KEPT.
      3. Lowercase (Unicode-aware: ``Ð`` -> ``ð``).
      4. Replace every run of remaining separators (ASCII spaces, parens, commas, other ASCII
         punctuation, and underscores) with a single hyphen -- which is how a parenthetical
         "(The Great Well, The Pond)" flattens into the hyphenated slug. Unicode letters/digits
         and the ASCII hyphen are kept.
      5. Collapse hyphen runs and trim the ends.

    Returns "" for an all-punctuation / letterless name.
    """
    normalized = unicodedata.normalize("NFC", name)
    kept = []
    for ch in normalized:
        if ch in _QUOTE_CHARS:                          # apostrophes/quotes: dropped, no hyphen
            continue
        if ord(ch) > 127 and not ch.isalnum():          # non-ASCII punctuation (–, —, …): dropped
            continue
        kept.append(ch)
    lowered = "".join(kept).lower()
    # ``[^\w-]`` keeps Unicode letters/digits/underscore and the ASCII hyphen; everything else
    # (space, parens, commas, ...) becomes a separator. Then fold the surviving underscores.
    hyphenated = re.sub(r"[^\w-]+", "-", lowered, flags=re.UNICODE).replace("_", "-")
    return re.sub(r"-+", "-", hyphenated).strip("-")


def strip_leading_article(name: str) -> str:
    """Drop a single leading "The " (case-insensitive) -- the article the golden sometimes
    omits from a slug. Only used to widen the tolerant match, never to rewrite a name."""
    return _LEADING_ARTICLE_RE.sub("", name, count=1)


def slug_bases(name: str) -> set:
    """The set of acceptable slug BASES for a name: the slug of the name, and the slug of
    the name with any leading article removed. Empty slugs are dropped."""
    return {s for s in (slugify(name), slugify(strip_leading_article(name))) if s}


def anchor_matches_name(anchor: str, name: str) -> bool:
    """True when ``anchor`` is a plausible golden-convention slug of ``name``.

    Accepts an exact base match OR a base plus a ``-N`` disambiguation suffix, for EITHER
    the full name or the article-stripped name. This is the single tolerant slug oracle the
    slug checks lean on; it errs toward accepting (a real slug bug -- wrong chars, uppercase,
    a surviving apostrophe, a transliterated accent -- still fails because none of the bases
    will match, while a merely-stylistic article/suffix difference is forgiven).
    """
    bases = slug_bases(name)
    if not bases:
        return False
    if anchor in bases:
        return True
    stripped = _SUFFIX_RE.sub("", anchor)
    return stripped != anchor and stripped in bases


def has_non_ascii_letter(text: str) -> bool:
    """True if ``text`` contains a non-ASCII alphabetic character (e.g. ``ö``, ``ð``)."""
    return any(ord(ch) > 127 and ch.isalpha() for ch in text)
