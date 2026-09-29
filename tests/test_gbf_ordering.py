"""Tests for evals/gbf/ordering.py -- rank-based, co-temporal-aware inversion count. Fully offline
(pairing is by exact anchor, so no model call is needed)."""

from evals.common.models import Status
from evals.common.parse import parse_wof
from evals.gbf.adjudicator import GbfAdjudicator
from evals.gbf.ordering import score_ordering
from evals.gbf.pair import pair

# ev-b and ev-c are CO-TEMPORAL (share rank 2); swapping them must not count as an inversion.
_RANKS = {"ev-a": 1, "ev-b": 2, "ev-c": 2, "ev-d": 3}

# A Locations section with a level-3 entity makes parse_wof detect the title, so ``## History``
# parses as a SECTION (not an entry) -- exactly as a real renderer WOF is shaped.
_GOLD = """# Gold

## Locations

### <a id="place"></a>Place

A place.

## History

- <a id="ev-a"></a>**Event A** — the first age.
- <a id="ev-b"></a>**Event B** — the second age.
- <a id="ev-c"></a>**Event C** — also the second age.
- <a id="ev-d"></a>**Event D** — the third age.
- <a id="ev-e"></a>**Event E** — undated, could not be placed.
"""


class _NoModel:
    model = "x"; provider = "openrouter"; temperature = 0.6; thinking = False; base_url = None
    def complete(self, system, user):                    # pragma: no cover - never called
        raise AssertionError("ordering tests must not call the model")


def _score(wof_md, ranks=None):
    gold = parse_wof(_GOLD)
    wof = parse_wof(wof_md)
    pm = pair(wof, gold, GbfAdjudicator(_NoModel(), votes=1))
    return score_ordering(gold, wof, pm, ranks if ranks is not None else _RANKS)


def _hist(*bullets, title="History"):
    # The Locations section (with a level-3 entity) triggers title detection so ``## {title}``
    # parses as a section; "Place" also pairs by anchor with the gold.
    body = "\n".join(bullets)
    return ("# WOF\n\n## Locations\n\n### <a id=\"place\"></a>Place\n\nA place.\n\n"
            f"## {title}\n\n{body}\n")


_A = '- <a id="ev-a"></a>**Event A** — first.'
_B = '- <a id="ev-b"></a>**Event B** — second.'
_C = '- <a id="ev-c"></a>**Event C** — second too.'
_D = '- <a id="ev-d"></a>**Event D** — third.'


def test_in_order_passes():
    item = _score(_hist(_A, _B, _C, _D))
    assert item.id == "gbf.ordering"
    assert item.status == Status.PASS
    assert "0 inversions" in item.evidence.detail


def test_non_cotemporal_reversal_fails():
    # D (rank 3) placed first -> inverts against A, B, C (3 inversions, n=4 -> fail)
    item = _score(_hist(_D, _A, _B, _C))
    assert item.status == Status.FAIL
    assert "inversion" in item.evidence.detail
    assert "ev-d" in item.evidence.detail and "ev-a" in item.evidence.detail


def test_single_inversion_is_partial():
    # swap only D and C-position: A, B, D, C -> D(3) before C(2) = 1 inversion; n=4 -> partial
    item = _score(_hist(_A, _B, _D, _C))
    assert item.status == Status.PARTIAL


def test_cotemporal_swap_adds_no_inversion():
    # B and C share rank 2; swapping them is NOT an inversion.
    item = _score(_hist(_A, _C, _B, _D))
    assert item.status == Status.PASS
    assert "0 inversions" in item.evidence.detail


def test_fewer_than_two_ranked_events_is_na():
    item = _score(_hist(_A))
    assert item.status == Status.NA
    assert "nothing to order" in item.evidence.detail


def test_rankless_event_is_excluded():
    # Event E has no rank; including it must not affect the (in-order) A,B,C,D verdict.
    e = '- <a id="ev-e"></a>**Event E** — undated.'
    item = _score(_hist(_A, _B, e, _C, _D))
    assert item.status == Status.PASS


def test_history_located_by_normalized_name_timeline():
    # A WOF that titles the section "Timeline" (not "History") must still grade.
    item = _score(_hist(_A, _B, _C, _D, title="Timeline"))
    assert item.status == Status.PASS


def test_missing_history_section_is_na():
    wof = "# WOF\n\n## Locations\n\n### <a id=\"somewhere\"></a>Somewhere\n\nA place.\n"
    item = _score(wof)
    assert item.status == Status.NA
