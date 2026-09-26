"""Tests for evals/fc/fc_lint.py -- the 24 mechanical checks + candidate extraction.

Fully offline, fabricated WOFs. A small well-formed ``GOOD`` fixture is mutated per check to
force pass / fail / partial / na. Includes the review-fix regressions: FIX #2 (inline-anchor
must not fail a grouping heading), FIX #3 (fndef.* work under a renamed footnotes section),
FIX #4 (figure_candidates excludes refs/anchor/link/slug digits).
"""

from evals.fc import fc_lint as L
from evals.common.parse import parse_wof

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

NO_TITLE = """\
# Locations

## Alpha

Alpha is a place.
"""


def st(text, fn):
    return fn(parse_wof(text)).status.value


def it(text, fn):
    return fn(parse_wof(text))


# --- a sanity floor: GOOD passes or n/a's every mechanical check ------------- #

def test_good_fixture_all_pass_or_na():
    items = L.run_lint(parse_wof(GOOD))
    assert len(items) == 24
    bad = [(i.id, i.status.value) for i in items if i.status.value not in ("pass", "na")]
    assert bad == [], bad


# --- title ------------------------------------------------------------------ #

def test_single_h1_pass_fail_na():
    assert st(GOOD, L.check_title_single_h1) == "pass"
    two = GOOD.replace("## Footnotes", "# Stray H1\n\n## Footnotes")
    assert st(two, L.check_title_single_h1) == "fail"
    assert st(NO_TITLE, L.check_title_single_h1) == "na"


def test_subtitle_pass_fail_na():
    assert st(GOOD, L.check_title_subtitle) == "pass"
    no_sub = "# Title\n\n---\n\n## Locations\n\n### <a id=\"alpha\"></a>Alpha\n\nBody.\n"
    assert st(no_sub, L.check_title_subtitle) == "fail"
    assert st(NO_TITLE, L.check_title_subtitle) == "na"


# --- sections --------------------------------------------------------------- #

def test_sections_consistent_depth():
    assert st(GOOD, L.check_sections_consistent_depth) == "pass"
    # An HR-preceded heading at the wrong level is an inconsistent "section divider".
    bad = GOOD.replace("---\n\n## Footnotes", "---\n\n### Footnotes")
    assert it(bad, L.check_sections_consistent_depth).status.value in ("fail", "partial")


def test_sections_no_anchor():
    assert st(GOOD, L.check_sections_no_anchor) == "pass"
    bad = GOOD.replace("## Locations", '## <a id="locations"></a>Locations')
    assert it(bad, L.check_sections_no_anchor).status.value in ("fail", "partial")


def test_layout_hr_separators():
    assert st(GOOD, L.check_layout_hr) == "pass"
    # Drop the rule before Footnotes.
    bad = GOOD.replace("\n---\n\n## Footnotes", "\n\n## Footnotes")
    assert it(bad, L.check_layout_hr).status.value in ("fail", "partial")


# --- entries ---------------------------------------------------------------- #

def test_entries_depth():
    assert st(GOOD, L.check_entries_depth) == "pass"
    bad = GOOD.replace("### <a id=\"beta\"></a>Beta", "#### <a id=\"beta\"></a>Beta")
    assert it(bad, L.check_entries_depth).status.value in ("fail", "partial")


def test_entries_inline_anchor_pass_and_fail():
    assert st(GOOD, L.check_entries_inline_anchor) == "pass"
    # Anchor present but NOT leading the name -> mis-positioned (FIX #2 territory).
    bad = GOOD.replace('### <a id="alpha"></a>Alpha', '### Alpha <a id="alpha"></a>')
    assert it(bad, L.check_entries_inline_anchor).status.value in ("fail", "partial")


def test_entries_inline_anchor_ignores_grouping_heading():
    # FIX #2: a legitimate anchorless grouping heading must NOT fail inline-anchor (that's the
    # model check's job). inline-anchor only judges anchors that exist.
    with_group = GOOD.replace(
        "---\n\n## Footnotes",
        "### Could Not Place\n\n- <a id=\"x\"></a>**X** — a thing.[^1]\n\n---\n\n## Footnotes",
    )
    assert st(with_group, L.check_entries_inline_anchor) == "pass"


def test_entries_alphabetized():
    assert st(GOOD, L.check_entries_alphabetized) == "pass"
    swapped = GOOD.replace(
        '### <a id="alpha"></a>Alpha\n\nAlpha is near [Beta](#beta).[^1]\n\n### <a id="beta"></a>Beta\n\nBeta is a place.[^1][^2]',
        '### <a id="beta"></a>Beta\n\nBeta is a place.[^1][^2]\n\n### <a id="alpha"></a>Alpha\n\nAlpha is near [Beta](#beta).[^1]',
    )
    assert it(swapped, L.check_entries_alphabetized).status.value in ("fail", "partial")


def test_alphabetized_ignores_leading_the():
    text = """\
# T

sub

---

## S

### <a id="apple"></a>Apple

x[^1]

### <a id="the-banana"></a>The Banana

y[^1]

### <a id="cherry"></a>Cherry

z[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""
    # Apple, (The) Banana, Cherry -> in order once "The" is ignored.
    assert st(text, L.check_entries_alphabetized) == "pass"


# --- anchors & slugs -------------------------------------------------------- #

def test_anchors_html_form():
    assert st(GOOD, L.check_anchors_html_form) == "pass"
    bad = GOOD.replace('<a id="alpha"></a>', '<a name="alpha"></a>')
    assert it(bad, L.check_anchors_html_form).status.value in ("fail", "partial")


def test_anchors_unique():
    assert st(GOOD, L.check_anchors_unique) == "pass"
    dup = GOOD.replace('<a id="beta"></a>', '<a id="alpha"></a>')
    assert it(dup, L.check_anchors_unique).status.value in ("fail", "partial")


def test_slugs_kebab_case():
    assert st(GOOD, L.check_slugs_kebab) == "pass"
    bad = GOOD.replace('id="alpha"', 'id="Alpha_1"')
    assert it(bad, L.check_slugs_kebab).status.value in ("fail", "partial")


def test_slugs_punctuation():
    assert st(GOOD, L.check_slugs_punctuation) == "pass"
    bad = GOOD.replace('id="alpha"', "id=\"al'pha\"")
    assert it(bad, L.check_slugs_punctuation).status.value in ("fail", "partial")


def test_slugs_non_ascii():
    assert st(GOOD, L.check_slugs_non_ascii) == "na"       # no non-ASCII names in GOOD
    ok = GOOD.replace('<a id="alpha"></a>Alpha', '<a id="skjöldr"></a>Skjöldr')
    assert st(ok, L.check_slugs_non_ascii) == "pass"
    bad = GOOD.replace('<a id="alpha"></a>Alpha', '<a id="skjoldr"></a>Skjöldr')
    assert it(bad, L.check_slugs_non_ascii).status.value in ("fail", "partial")


def test_slugs_disambiguation():
    assert st(GOOD, L.check_slugs_disambiguation) == "na"   # no cross-section clash in GOOD
    clash = """\
# T

sub

---

## Locations

### <a id="citadel"></a>Citadel

A place.[^1]

---

## Organizations

### <a id="citadel-2"></a>The Citadel

A body.[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""
    assert st(clash, L.check_slugs_disambiguation) == "pass"


# --- cross-links ------------------------------------------------------------ #

def test_links_format():
    assert st(GOOD, L.check_links_format) == "pass"
    bad = GOOD.replace("[Beta](#beta)", "[Beta] (#beta)")   # space between ] and (
    assert it(bad, L.check_links_format).status.value in ("fail", "partial")


def test_links_resolve():
    assert st(GOOD, L.check_links_resolve) == "pass"
    bad = GOOD.replace("[Beta](#beta)", "[Beta](#nowhere)")
    assert it(bad, L.check_links_resolve).status.value in ("fail", "partial")


# --- list entries ----------------------------------------------------------- #

def _with_history(bullets):
    return GOOD.replace(
        "---\n\n## Footnotes",
        "## History\n\n" + bullets + "\n\n---\n\n## Footnotes",
    )


def test_list_anchor_bold_na_pass_fail():
    assert st(GOOD, L.check_list_anchor_bold) == "na"       # no bullet sections
    ok = _with_history("- <a id=\"the-founding\"></a>**The Founding** — it happened.[^1]")
    assert st(ok, L.check_list_anchor_bold) == "pass"
    bad = _with_history("- <a id=\"the-founding\"></a>The Founding — no bold.[^1]")
    assert it(bad, L.check_list_anchor_bold).status.value in ("fail", "partial")


# --- footnote refs ---------------------------------------------------------- #

def test_fnref_format():
    assert st(GOOD, L.check_fnref_format) == "pass"
    bad = GOOD.replace("Alpha is near [Beta](#beta).[^1]", "Alpha is near [Beta](#beta).[^one]")
    assert it(bad, L.check_fnref_format).status.value in ("fail", "partial")


def test_fnref_adjacent():
    assert st(GOOD, L.check_fnref_adjacent) == "pass"       # GOOD has [^1][^2] adjacent
    bad = GOOD.replace("Beta is a place.[^1][^2]", "Beta is a place.[^1], [^2]")
    assert it(bad, L.check_fnref_adjacent).status.value in ("fail", "partial")


# --- footnote defs ---------------------------------------------------------- #

def test_fndef_collected_and_fix3_renamed_section():
    assert st(GOOD, L.check_fndef_collected) == "pass"
    # FIX #3: the footnotes section renamed "Notes" is still located structurally.
    renamed = GOOD.replace("## Footnotes", "## Notes")
    assert st(renamed, L.check_fndef_collected) == "pass"


def test_fndef_numeric_order():
    assert st(GOOD, L.check_fndef_numeric_order) == "pass"
    bad = GOOD.replace("[^2]: `beta one`", "[^0]: `beta one`")   # 1 then 0 -> not ascending
    assert it(bad, L.check_fndef_numeric_order).status.value in ("fail", "partial")


def test_fndef_format():
    assert st(GOOD, L.check_fndef_format) == "pass"
    bad = GOOD.replace("[^1]: `alpha quote` — Matt, log.txt", "[^1]: just some prose, no quote")
    assert it(bad, L.check_fndef_format).status.value in ("fail", "partial")


def test_fndef_multi_quote_sep():
    assert st(GOOD, L.check_fndef_multi_quote_sep) == "pass"   # GOOD [^2] uses ' / '
    # "and" between two same-speaker excerpts is an accepted connective (golden [^117]).
    and_variant = GOOD.replace("`beta one` / `beta two`", "`beta one` and `beta two`")
    assert st(and_variant, L.check_fndef_multi_quote_sep) == "pass"
    # Two excerpts jammed together with no separator is the real violation.
    jammed = GOOD.replace("`beta one` / `beta two`", "`beta one``beta two`")
    assert it(jammed, L.check_fndef_multi_quote_sep).status.value in ("fail", "partial")


# --- other markers ---------------------------------------------------------- #

def test_markers_asterisk_na_pass_fail():
    assert st(GOOD, L.check_markers_asterisk) == "na"       # no '*' markers
    ok = GOOD.replace(">Alpha\n", ">Alpha*\n").replace(">Beta\n", ">Beta*\n")
    assert st(ok, L.check_markers_asterisk) == "pass"
    bad = GOOD.replace(">Alpha\n", ">Alpha *\n")            # space before the '*'
    assert it(bad, L.check_markers_asterisk).status.value in ("fail", "partial")


# --- status_from_counts ----------------------------------------------------- #

def test_status_from_counts_thresholds():
    from evals.fc.models import Status
    assert L.status_from_counts(20, 0) == Status.PASS
    assert L.status_from_counts(20, 1) == Status.PARTIAL     # a lone straggler (1 <= 20//10)
    assert L.status_from_counts(20, 3) == Status.FAIL        # 3 > 2
    assert L.status_from_counts(5, 1) == Status.PARTIAL      # max(1, 0) == 1
    assert L.status_from_counts(5, 2) == Status.FAIL


# --- candidate extraction --------------------------------------------------- #

def test_grouping_candidates():
    with_group = GOOD.replace(
        "---\n\n## Footnotes",
        "## History\n\n### Could Not Place\n\n- <a id=\"x\"></a>**X** — a thing.[^1]\n\n---\n\n## Footnotes",
    )
    cands = L.grouping_candidates(parse_wof(with_group))
    assert len(cands) == 1
    assert cands[0]["label"] == "Could Not Place"
    assert cands[0]["entries_beneath"] == ["X"]


def test_figure_candidates_excludes_refs_and_slug_digits():
    # FIX #4: '[^1]' and the '2' inside 'krieger-imperium-2' / '#beta2' must NOT surface as
    # figures; the prose '200' must.
    text = """\
# T

sub

---

## Locations

### <a id="krieger-imperium-2"></a>Realm

It rose about 200 years ago.[^1] See [Realm](#krieger-imperium-2).

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""
    figs = L.figure_candidates(parse_wof(text))
    all_figs = {f["figure"] for f in figs}
    assert "200" in all_figs
    assert "1" not in all_figs        # the [^1] ref was stripped
    assert "2" not in all_figs        # the slug digit was stripped


def test_tbd_candidates_one_per_entry():
    cands = L.tbd_candidates(parse_wof(GOOD))
    names = {c["name"] for c in cands}
    assert names == {"Alpha", "Beta"}
    assert all("body" in c for c in cands)
