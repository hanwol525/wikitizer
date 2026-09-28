"""Tests for evals/rubric/resolve.py -- ParsedWOF narrowing + the normalized section key.
Fully offline, fabricated WOFs."""

from evals.common.parse import parse_wof
from evals.rubric.resolve import (
    cited_quotes,
    entity_slug,
    entry_candidates,
    section_candidates,
    section_key,
)

WOF = """\
# Wiki

subtitle

---

## Locations

### <a id="alpha"></a>Alpha

Alpha is a fortress on a hill.[^1]

### <a id="beta"></a>Beta

Beta is a port town.[^2]

## Footnotes

[^1]: `Alpha the fortress` — DM, dm.txt
[^2]: `Beta the port` — DM, dm.txt
"""


def test_section_candidates_in_document_order():
    p = parse_wof(WOF)
    assert section_candidates(p) == ["Locations", "Footnotes"]


def test_entry_candidates_by_index():
    p = parse_wof(WOF)
    entries = entry_candidates(p, 1)                   # Locations is WOF section #1 (1-based)
    assert [e.name for e in entries] == ["Alpha", "Beta"]


def test_entry_candidates_absent_or_out_of_range():
    p = parse_wof(WOF)
    assert entry_candidates(p, None) == []
    assert entry_candidates(p, 99) == []


def test_cited_quotes_resolves_only_the_entrys_own_refs():
    p = parse_wof(WOF)
    alpha, beta = entry_candidates(p, 1)
    a_q = cited_quotes(p, alpha)
    b_q = cited_quotes(p, beta)
    assert [n for n, _ in a_q] == [1]                  # Alpha cites only [^1], not Beta's [^2]
    assert [n for n, _ in b_q] == [2]
    assert "Alpha the fortress" in a_q[0][1]


def test_cited_quotes_dedups_repeated_ref():
    wof = (
        "# W\n\nsub\n\n---\n\n## Locations\n\n"
        "### <a id=\"x\"></a>X\n\nA cites [^1] and again [^1].[^1]\n\n"
        "## Footnotes\n\n[^1]: `only once` — DM, dm.txt\n"
    )
    p = parse_wof(wof)
    entry = entry_candidates(p, 1)[0]
    result = cited_quotes(p, entry)
    assert len(result) == 1                            # [^1] cited three times -> one quote
    assert result[0][0] == 1 and "only once" in result[0][1]


def test_section_key_normalizes_ampersand_case_and_whitespace():
    assert section_key("People & Cultures") == section_key("people and cultures")
    assert section_key("  Locations  ") == "locations"
    assert section_key("People   &   Cultures") == "people and cultures"


def test_entity_slug_uses_shared_slugify():
    assert entity_slug("Lake Mundi") == "lake-mundi"
    assert entity_slug("Mal'taav") == "maltaav"


def test_entity_slug_fallback_for_letterless_name_is_schema_safe():
    import re
    s = entity_slug("@#$")                             # slugify() -> "" -> must fall back
    assert s and "." not in s and " " not in s
    assert s == entity_slug("@#$")                     # deterministic (both grading paths agree)
    # an id built from the fallback still matches the rubric schema's id pattern
    assert re.match(r"^rubric\.[^\s.]+(\.[^\s.]+)*$", f"rubric.presence.locations.{s}")
