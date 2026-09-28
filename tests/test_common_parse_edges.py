"""Deeper coverage of evals/common/parse.py internals: title-detection variants, subtitle
edges, anchor-context classification, section body slicing, bullet-name fallbacks, and
fenced-code handling. Offline, fabricated WOFs.
"""

from evals.common.parse import parse_wof


def test_title_with_hr_but_no_subtitle():
    # A title followed immediately by an HR (no prose) is still a title (top matter = the HR),
    # but the subtitle is absent -- so the subtitle check can later fail rather than mis-read.
    text = "# Title\n\n---\n\n## Section\n\n### <a id=\"a\"></a>A\n\nBody.\n"
    p = parse_wof(text)
    assert p.opens_with_title is True
    assert p.subtitle is None
    assert p.section_level == 2


def test_no_title_when_first_heading_is_level_2():
    text = "## Section\n\n### Entry\n\nBody.\n"
    p = parse_wof(text)
    assert p.opens_with_title is False
    assert p.section_level == 2 and p.entry_level == 3   # min heading level is 2


def test_no_title_multi_hash_sections_with_double_hash_entries():
    text = ("# Locations\n\n## Alpha\n\nA.\n\n# Characters\n\n## Bram\n\nB.\n")
    p = parse_wof(text)
    assert p.opens_with_title is False                   # >1 H1, no top matter -> not a title
    assert [s.title for s in p.sections] == ["Locations", "Characters"]
    assert p.section_level == 1 and p.entry_level == 2


def test_title_detected_by_three_tier_ladder_even_without_top_matter():
    # "# T" / "## S" / "### A" is the three-tier titled ladder; even with no subtitle/HR between
    # the title and the first section, the level-3 heading is the tell -> titled. (This is the
    # Scenario B fix: a titled doc that merely forgot its subtitle is still classified titled, so
    # its missing-subtitle violation surfaces rather than being hidden as NA.)
    text = "# T\n\n## S\n\n### <a id=\"a\"></a>A\n\nBody.\n"
    p = parse_wof(text)
    assert p.opens_with_title is True
    assert p.subtitle is None                # no prose subtitle -> subtitle check will FAIL, not NA


def test_anchor_context_classification():
    text = """\
# T

sub

---

## Locations

### <a id="alpha"></a>Alpha

Alpha mentions <a id="inline-note"></a> a thing.[^1]

## History

- <a id="ev"></a>**Event** — happened.[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""
    p = parse_wof(text)
    ctx = {a.id: a.context for a in p.anchors}
    assert ctx["alpha"] == "heading"
    assert ctx["inline-note"] == "inline"
    assert ctx["ev"] == "bullet"


def test_section_body_slicing_is_per_section():
    text = """\
# T

sub

---

## One

### <a id="a"></a>A

Body A.

---

## Two

### <a id="b"></a>B

Body B.
"""
    p = parse_wof(text)
    one = [s for s in p.sections if s.title == "One"][0]
    two = [s for s in p.sections if s.title == "Two"][0]
    assert [e.name for e in one.entries] == ["A"]
    assert [e.name for e in two.entries] == ["B"]


def test_bullet_name_fallback_without_bold():
    text = ("# T\n\nsub\n\n---\n\n## History\n\n"
            "- <a id=\"ev\"></a>Event without bold — happened.[^1]\n\n"
            "---\n\n## Footnotes\n\n[^1]: `q` — Matt, log.txt\n")
    p = parse_wof(text)
    hist = [s for s in p.sections if s.title == "History"][0]
    ev = [e for e in hist.entries if e.is_bullet][0]
    # No **bold**, so anchor_id stays None (flagged by anchor-bold) but a name is still recovered.
    assert ev.anchor_id is None
    assert "Event without bold" in ev.name


def test_indented_sub_bullet_is_not_an_entry():
    text = ("# T\n\nsub\n\n---\n\n## History\n\n"
            "- <a id=\"ev\"></a>**Event** — happened.[^1]\n"
            "  - a nested detail bullet\n\n"
            "---\n\n## Footnotes\n\n[^1]: `q` — Matt, log.txt\n")
    p = parse_wof(text)
    hist = [s for s in p.sections if s.title == "History"][0]
    bullets = [e for e in hist.entries if e.is_bullet]
    assert len(bullets) == 1 and bullets[0].name == "Event"    # the nested bullet is not an entry


def test_unclosed_fence_swallows_following_headings():
    text = "# T\n\nsub\n\n---\n\n## S\n\n### <a id=\"a\"></a>A\n\n```\n## Not A Section\n"
    p = parse_wof(text)
    assert [s.title for s in p.sections] == ["S"]
    assert {a.id for a in p.anchors} == {"a"}


def test_kramdown_anchor_is_non_canonical():
    text = "# T\n\nsub\n\n---\n\n## S\n\n### Heading {#kram}\n\nbody\n"
    p = parse_wof(text)
    assert any("{#kram}" in tok for _, tok in p.non_canonical_anchors)


def test_fn_ref_positions_recorded_for_adjacency():
    text = ("# T\n\nsub\n\n---\n\n## S\n\n### <a id=\"a\"></a>A\n\nStacked.[^1][^2]\n\n"
            "---\n\n## Footnotes\n\n[^1]: `x` — M, l.txt\n[^2]: `y` — M, l.txt\n")
    p = parse_wof(text)
    stacked_line = [r for r in p.fn_refs if r.n in (1, 2)]
    by_line = {}
    for r in stacked_line:
        by_line.setdefault(r.lineno, []).append(r)
    line_refs = sorted(next(v for v in by_line.values() if len(v) == 2), key=lambda r: r.start)
    assert line_refs[0].end == line_refs[1].start        # directly adjacent [^1][^2]
