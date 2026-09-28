"""Tests for evals/common/parse.py -- the ParsedWOF tokenizer. Fully offline, fabricated WOFs."""

from evals.common.parse import parse_wof

TITLED = """\
# The Lore of Somewhere

A short wiki about a made-up place.

---

## Locations

### <a id="alpha"></a>Alpha

Alpha is a town near [Beta](#beta).[^1]

### <a id="beta"></a>Beta

Beta borders [Alpha](#alpha).[^1][^2]

---

## History

- <a id="the-founding"></a>**The Founding** — ~200 years ago. It happened.[^2]

### Could Not Place

- <a id="the-vague-event"></a>**The Vague Event**. No date known.[^1]

---

## Footnotes

[^1]: `alpha quote` — Matt, log.txt
[^2]: `beta quote` / `second excerpt` — Sam, log.txt
"""

# No document title: sections are `#`, entries are `##`, no top matter before the first section.
NO_TITLE = """\
# Locations

## Alpha

Alpha is a place.

# Characters

## Bram

Bram is a person.
"""


def test_titled_document_structure():
    p = parse_wof(TITLED)
    assert p.opens_with_title is True
    assert p.title.text == "The Lore of Somewhere"
    assert p.h1_count == 1
    assert (p.section_level, p.entry_level) == (2, 3)
    assert p.subtitle is not None and "made-up place" in p.subtitle[1]
    assert [s.title for s in p.sections] == ["Locations", "History", "Footnotes"]


def test_no_title_document_structure():
    p = parse_wof(NO_TITLE)
    assert p.opens_with_title is False
    assert p.title is None
    assert (p.section_level, p.entry_level) == (1, 2)
    assert [s.title for s in p.sections] == ["Locations", "Characters"]


def test_entries_and_bullets_and_grouping():
    p = parse_wof(TITLED)
    loc = [s for s in p.sections if s.title == "Locations"][0]
    assert [e.name for e in loc.entries] == ["Alpha", "Beta"]
    assert all(e.anchor_id and not e.is_bullet for e in loc.entries)

    hist = [s for s in p.sections if s.title == "History"][0]
    names = [(e.name, e.is_bullet, e.is_grouping) for e in hist.entries]
    assert ("The Founding", True, False) in names
    assert ("Could Not Place", False, True) in names          # anchorless entry-level heading
    assert ("The Vague Event", True, False) in names


def test_anchor_link_footnote_collection():
    p = parse_wof(TITLED)
    assert {a.id for a in p.anchors} >= {"alpha", "beta", "the-founding", "the-vague-event"}
    assert p.non_canonical_anchors == []
    assert {ln.slug for ln in p.links} == {"alpha", "beta"}
    assert [d.n for d in p.fn_defs] == [1, 2]
    assert len(p.fn_refs) >= 4                                 # inline refs, def lines excluded


def test_footnotes_section_located_structurally_by_name_agnostic():
    # FIX #3: rename the footnotes section "Notes" -- it must STILL be found (it holds the defs).
    renamed = TITLED.replace("## Footnotes", "## Notes")
    p = parse_wof(renamed)
    assert p.footnotes_section is not None
    assert p.footnotes_section.title == "Notes"
    assert p.footnotes_section.level == p.section_level


def test_fenced_code_is_a_no_go_zone():
    text = """\
# Title

Subtitle line.

---

## Section

### <a id="real"></a>Real Entry

```
## Not A Section
### <a id="fake"></a>Not An Entry
```

Body.
"""
    p = parse_wof(text)
    # The heading + anchor inside the fence must be ignored.
    assert [s.title for s in p.sections] == ["Section"]
    assert {a.id for a in p.anchors} == {"real"}


def test_non_canonical_anchor_detected():
    text = '# T\n\nsub\n\n---\n\n## S\n\n### <a name="old"></a>Old\n'
    p = parse_wof(text)
    assert any("name=" in tok for _, tok in p.non_canonical_anchors)
