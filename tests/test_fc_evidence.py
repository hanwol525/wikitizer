"""Coverage of the evidence a check attaches (line numbers, expected/found), and golden
shape-locks. Evidence is required on every item and drives the human review, so it's worth
asserting its content -- not just the pass/fail status. Offline.
"""

from pathlib import Path

import pytest

from evals.fc import fc_lint as L
from evals.fc.parse import parse_wof

GOOD = """\
# Title

Subtitle prose here.

---

## Locations

### <a id="alpha"></a>Alpha

Alpha is near [Beta](#beta).[^1]

### <a id="beta"></a>Beta

Beta is a place.[^1][^2]

---

## Footnotes

[^1]: `alpha quote` — Matt, log.txt
[^2]: `beta one` / `beta two` — Sam, log.txt
"""


def item(text, fn):
    return fn(parse_wof(text))


# --- evidence content on fails ---------------------------------------------- #

def test_links_resolve_evidence_points_at_broken_line_and_target():
    bad = GOOD.replace("[Beta](#beta)", "[Beta](#nowhere)")
    ev = item(bad, L.check_links_resolve).evidence
    assert ev.lines == [11]                                # the Alpha body line
    assert "#nowhere" in (ev.found or "")


def test_anchors_unique_evidence_lists_dup_id_and_lines():
    dup = GOOD.replace('<a id="beta"></a>', '<a id="alpha"></a>')
    ev = item(dup, L.check_anchors_unique).evidence
    assert "alpha" in ev.detail
    assert len(ev.lines) >= 2                              # both occurrences' lines


def test_kebab_evidence_reports_expected_found_style():
    bad = GOOD.replace('id="alpha"', 'id="Alpha_1"')
    ev = item(bad, L.check_slugs_kebab).evidence
    assert "Alpha_1" in (ev.found or "")


def test_non_ascii_evidence_shows_transliteration_arrow():
    bad = GOOD.replace('<a id="alpha"></a>Alpha', '<a id="skjoldr"></a>Skjöldr')
    ev = item(bad, L.check_slugs_non_ascii).evidence
    assert "Skjöldr" in (ev.found or "") and "skjoldr" in (ev.found or "")


def test_single_h1_evidence_reports_count():
    two = GOOD.replace("## Footnotes", "# Stray H1\n\n## Footnotes")
    ev = item(two, L.check_title_single_h1).evidence
    assert ev.expected == "1 H1" and "2" in (ev.found or "")


def test_alphabetized_evidence_names_the_out_of_order_pair():
    swapped = GOOD.replace(
        '### <a id="alpha"></a>Alpha\n\nAlpha is near [Beta](#beta).[^1]\n\n### <a id="beta"></a>Beta\n\nBeta is a place.[^1][^2]',
        '### <a id="beta"></a>Beta\n\nBeta is a place.[^1][^2]\n\n### <a id="alpha"></a>Alpha\n\nAlpha is near [Beta](#beta).[^1]',
    )
    ev = item(swapped, L.check_entries_alphabetized).evidence
    assert "Beta" in (ev.found or "") and "Alpha" in (ev.found or "")


def test_pass_evidence_is_terse():
    ev = item(GOOD, L.check_anchors_unique).evidence
    assert ev.lines == [] and "0 collisions" in ev.detail


# --- golden shape-locks ----------------------------------------------------- #

_GOLDEN = Path(__file__).resolve().parent.parent / "output" / "gol-lore-full.md"


@pytest.mark.skipif(not _GOLDEN.exists(), reason="golden file not present (gitignored)")
def test_golden_shape_counts_are_locked():
    p = parse_wof(_GOLDEN.read_text(encoding="utf-8"))
    assert p.opens_with_title and p.title.text == "The Lore of Gol"
    assert [s.title for s in p.sections] == [
        "Locations", "History", "People & Cultures", "Organizations",
        "Characters", "Items", "Footnotes",
    ]
    assert len(p.anchors) == 72
    assert [d.n for d in p.fn_defs] == list(range(1, 118))   # 1..117, in order
    assert p.non_canonical_anchors == []
    # The single grouping heading and the PC asterisk markers are present.
    groupings = [e.name for s in p.sections for e in s.entries if e.is_grouping]
    assert groupings == ["Could Not Place"]
    starred = [e.name for s in p.sections for e in s.entries
               if e.name.rstrip().endswith("*") and not e.name.rstrip().endswith("**")]
    assert len(starred) == 5                                  # the five player characters


@pytest.mark.skipif(not _GOLDEN.exists(), reason="golden file not present (gitignored)")
def test_golden_candidate_extraction_shape():
    p = parse_wof(_GOLDEN.read_text(encoding="utf-8"))
    groups = L.grouping_candidates(p)
    assert [g["label"] for g in groups] == ["Could Not Place"]
    assert L.tbd_candidates(p)                                # one per entry, non-empty
    figs = L.figure_candidates(p)
    assert figs and all("figure" in f and "sentence" in f for f in figs)
