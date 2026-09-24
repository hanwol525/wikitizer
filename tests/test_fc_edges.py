"""Additional FC edge-case coverage: the partial-vs-fail scoring distinction on sized
fixtures, every remaining `na` branch, a disambiguation FAIL, malformed links, and
multi-candidate adjudicator behavior. Offline.
"""

import json

from evals.fc import fc_lint as L
from evals.fc.adjudicator import Adjudicator
from evals.fc.models import Status
from evals.fc.parse import parse_wof


def st(text, fn):
    return fn(parse_wof(text)).status.value


# --- na branches: a plain doc with no anchors/links/footnotes ---------------- #

PLAIN = """\
# T

sub

---

## Section

### Alpha

Just prose. No anchors, links, or footnotes here.
"""


def test_na_branches_on_plain_doc():
    assert st(PLAIN, L.check_anchors_html_form) == "na"
    assert st(PLAIN, L.check_anchors_unique) == "na"
    assert st(PLAIN, L.check_slugs_kebab) == "na"
    assert st(PLAIN, L.check_slugs_punctuation) == "na"
    assert st(PLAIN, L.check_slugs_non_ascii) == "na"
    assert st(PLAIN, L.check_slugs_disambiguation) == "na"
    assert st(PLAIN, L.check_links_format) == "na"
    assert st(PLAIN, L.check_links_resolve) == "na"
    assert st(PLAIN, L.check_list_anchor_bold) == "na"
    assert st(PLAIN, L.check_fnref_format) == "na"
    assert st(PLAIN, L.check_fnref_adjacent) == "na"
    assert st(PLAIN, L.check_fndef_collected) == "na"
    assert st(PLAIN, L.check_fndef_numeric_order) == "na"
    assert st(PLAIN, L.check_fndef_format) == "na"
    assert st(PLAIN, L.check_fndef_multi_quote_sep) == "na"
    assert st(PLAIN, L.check_markers_asterisk) == "na"


# --- partial vs fail: enough items that the small-minority threshold bites ---- #

def _many_entries(n, broken_link_count=0):
    """A Locations section with n anchored entries; the first `broken_link_count` link to a
    missing anchor. Names A00.. keep them alphabetical."""
    entries = []
    for i in range(n):
        name = f"E{i:02d}"
        target = "#missing" if i < broken_link_count else "#e00"
        entries.append(f"### <a id=\"e{i:02d}\"></a>{name}\n\nBody links [x]({target}).[^1]\n")
    return ("# T\n\nsub\n\n---\n\n## Locations\n\n" + "\n".join(entries)
            + "\n---\n\n## Footnotes\n\n[^1]: `q` — Matt, log.txt\n")


def test_links_resolve_partial_then_fail():
    # 12 links, 1 broken -> 1 <= 12//10 -> PARTIAL.
    assert st(_many_entries(12, broken_link_count=1), L.check_links_resolve) == "partial"
    # 12 links, 5 broken -> FAIL.
    assert st(_many_entries(12, broken_link_count=5), L.check_links_resolve) == "fail"


def test_links_resolve_single_small_fixture_is_partial_not_fail():
    # A tiny fixture (few links) with one break: max(1, total//10) == 1 -> PARTIAL, still
    # scores as a fail but stays visible.
    assert st(_many_entries(2, broken_link_count=1), L.check_links_resolve) == "partial"


# --- disambiguation FAIL ---------------------------------------------------- #

def test_disambiguation_fail_when_clash_not_suffixed():
    # "Citadel" in two sections sharing the SAME anchor -> not disambiguated -> fail.
    clash = """\
# T

sub

---

## Locations

### <a id="citadel"></a>Citadel

A place.[^1]

---

## Organizations

### <a id="citadel"></a>Citadel

A body.[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""
    assert parse_wof(clash) is not None
    assert st(clash, L.check_slugs_disambiguation) in ("fail", "partial")


# --- malformed links -------------------------------------------------------- #

def test_links_format_empty_text_and_empty_slug():
    empty_text = _many_entries(2).replace("[x](#e00)", "[](#e00)")
    assert st(empty_text, L.check_links_format) in ("fail", "partial")
    empty_slug = _many_entries(2).replace("[x](#e00)", "[x](#)")
    assert st(empty_slug, L.check_links_format) in ("fail", "partial")


# --- fnref.adjacent na when refs never stack -------------------------------- #

def test_fnref_adjacent_na_when_no_stacking():
    text = ("# T\n\nsub\n\n---\n\n## S\n\n### <a id=\"a\"></a>A\n\nOne ref only.[^1]\n"
            "\n---\n\n## Footnotes\n\n[^1]: `q` — Matt, log.txt\n")
    assert st(text, L.check_fnref_adjacent) == "na"


# --- markers.asterisk: a mix of marked and unmarked is fine (consistency-only) ---- #

def test_markers_asterisk_partial_when_one_has_a_space():
    # Two marked entries, one with a stray space before '*' -> the minority is flagged.
    text = ("# T\n\nsub\n\n---\n\n## Characters\n\n"
            "### <a id=\"a\"></a>Aaa*\n\nx[^1]\n\n"
            "### <a id=\"b\"></a>Bbb *\n\ny[^1]\n\n"
            "---\n\n## Footnotes\n\n[^1]: `q` — Matt, log.txt\n")
    assert st(text, L.check_markers_asterisk) in ("fail", "partial")


# --- adjudicator: multi-candidate mixed verdicts ---------------------------- #

class Fake:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, system, user):
        return self.replies.pop(0)


def test_adjudicator_multi_candidate_lists_only_offenders():
    cands = [
        {"figure": "1424", "sentence": "the year is 1424", "lineno": 1},
        {"figure": "200", "sentence": "about 200 years ago", "lineno": 2},
        {"figure": "12", "sentence": "12 houses exactly", "lineno": 3},
    ]
    reply = json.dumps({"verdicts": ["exact_ok", "approximate_unmarked", "exact_ok"]})
    item = Adjudicator(Fake([reply] * 3), votes=3).judge_approx_tilde(cands)
    assert item.status == Status.FAIL
    assert "200" in (item.evidence.found or "")
    assert item.evidence.lines == [2]              # only the flagged candidate's line


def test_adjudicator_single_vote_no_split_flag():
    # votes=1 -> a single vote decides, never a "split".
    item = Adjudicator(Fake([json.dumps({"verdicts": ["ok"]})]), votes=1).judge_tbd(
        [{"name": "A", "body": "real content", "lineno": 5}])
    assert item.status == Status.PASS and "[REVIEW]" not in item.evidence.detail
