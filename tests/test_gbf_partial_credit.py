"""GBF's partial-vs-fail differentiation (both halves of the change). Fully offline.

Two things this guards:
  * The GBF summary gives ``partial`` HALF credit (GBF-only; FC/rubric keep partial=fail). This is the
    real eval output, not just the bake-off.
  * The bake-off agreement is FAIL-AWARE: ``pass``/``partial`` are one "faithful, no contradiction"
    class and only ``fail`` is distinct, so a judge partial where a label expected pass is NOT a miss.
"""

from evals.common.models import Engine, Evidence, Item, Status
from evals.gbf.scoring import build_gbf_summary

# The bake-off helpers are pure functions; importing them here exercises the fail-aware metric offline
# (importing the module has no side effects -- its integration marker only gates its OWN tests).
from tests.test_gbf_bakeoff import _agreement, _contradiction_recall, _faith_class


def _it(cid, status):
    return Item(id=cid, engine=Engine.MODEL, status=status, evidence=Evidence(detail="d"))


# --- half credit in the real GBF summary ------------------------------------ #

def test_partial_earns_half_credit_with_fractional_display():
    # faithfulness: pass=6 partial=3 fail=1 -> applicable=10, credit = 6 + 0.5*3 = 7.5 -> 0.75.
    items = ([_it(f"gbf.faithfulness.p{i}", Status.PASS) for i in range(6)]
             + [_it(f"gbf.faithfulness.h{i}", Status.PARTIAL) for i in range(3)]
             + [_it("gbf.faithfulness.f0", Status.FAIL),
                _it("gbf.ordering", Status.PASS)])
    s = build_gbf_summary(items)
    assert s.faithfulness.n_pass == 6 and s.faithfulness.partial == 3 and s.faithfulness.fail == 1
    assert s.faithfulness.applicable == 10
    assert s.faithfulness.score == 0.75
    assert s.faithfulness.score_display == "7.5/10"


def test_no_partials_display_stays_integer():
    # pass=3 fail=1 -> credit=3 -> "3/4" (no fractional suffix when there are no partials).
    items = [_it("gbf.faithfulness.a", Status.PASS), _it("gbf.faithfulness.b", Status.PASS),
             _it("gbf.faithfulness.c", Status.PASS), _it("gbf.faithfulness.d", Status.FAIL),
             _it("gbf.ordering", Status.PASS)]
    s = build_gbf_summary(items)
    assert s.faithfulness.score == 0.75 and s.faithfulness.score_display == "3/4"


def test_all_pass_is_full_credit():
    items = [_it("gbf.faithfulness.a", Status.PASS), _it("gbf.faithfulness.b", Status.PASS),
             _it("gbf.ordering", Status.PASS)]
    s = build_gbf_summary(items)
    assert s.faithfulness.score == 1.0 and s.faithfulness.score_display == "2/2"


def test_ordering_partial_also_earns_half_credit():
    # ordering is a single-item bucket; a partial there scores 0.5/1, not 0/1.
    s = build_gbf_summary([_it("gbf.faithfulness.a", Status.PASS), _it("gbf.ordering", Status.PARTIAL)])
    assert s.ordering.partial == 1 and s.ordering.applicable == 1
    assert s.ordering.score == 0.5 and s.ordering.score_display == "0.5/1"


def test_empty_bucket_scores_none_and_zero_display():
    # na-only faithfulness -> nothing applicable -> None / "0/0" (unchanged by the weighting).
    s = build_gbf_summary([_it("gbf.faithfulness.a", Status.NA), _it("gbf.ordering", Status.NA)])
    assert s.faithfulness.applicable == 0 and s.faithfulness.score is None
    assert s.faithfulness.score_display == "0/0"


def test_na_and_skipped_excluded_from_credit():
    # na/skipped never touch the numerator or denominator; partial still earns half of what's applicable.
    items = [_it("gbf.faithfulness.a", Status.PASS), _it("gbf.faithfulness.b", Status.PARTIAL),
             _it("gbf.faithfulness.c", Status.NA), _it("gbf.faithfulness.d", Status.SKIPPED),
             _it("gbf.ordering", Status.PASS)]
    s = build_gbf_summary(items)
    # applicable = pass(1) + partial(1) = 2; credit = 1 + 0.5 = 1.5 -> 0.75
    assert s.faithfulness.applicable == 2 and s.faithfulness.na == 1
    assert s.faithfulness.score == 0.75 and s.faithfulness.score_display == "1.5/2"
    assert s.complete is False   # a skipped item flips complete


# --- fail-aware bake-off agreement ------------------------------------------ #

def test_faith_class_pass_and_partial_are_one_class():
    assert _faith_class("pass") == "ok"
    assert _faith_class("partial") == "ok"
    assert _faith_class("fail") == "fail"
    # a pairing miss on a present entity must never fold into "ok"
    assert _faith_class("na") == "na"
    assert _faith_class("skipped") == "skipped"
    assert _faith_class(None) == "missing"


def test_partial_vs_pass_is_agreement_not_a_miss():
    expected = {"v.md": {"ordering": "pass", "faithfulness": {"a": "pass", "b": "fail", "c": "partial"}}}
    graded = {"v.md": {"gbf.ordering": "pass",
                       "gbf.faithfulness.a": "partial",   # ok-vs-ok  -> hit + slip
                       "gbf.faithfulness.b": "fail",      # exact fail -> hit
                       "gbf.faithfulness.c": "pass"}}     # ok-vs-ok  -> hit + slip
    o_hit, o_tot, f_hit, f_tot, s_hit, s_tot, misses, slips = _agreement(graded, expected)
    assert (o_hit, o_tot) == (1, 1)
    assert (f_hit, f_tot) == (3, 3) and not misses      # fail-aware: everything agrees
    assert (s_hit, s_tot) == (1, 3)                     # strict: only the exact 'fail' matches
    assert len(slips) == 2                              # the two partial<->pass differences


def test_contradiction_confusion_is_a_real_miss():
    # A hallucinated contradiction (fail where pass expected) and a missed one (pass where fail
    # expected) are both fail-aware misses; a pairing na on a present entity is a miss too.
    expected = {"v.md": {"faithfulness": {"a": "pass", "b": "fail", "c": "pass"}}}
    graded = {"v.md": {"gbf.faithfulness.a": "fail",    # hallucinated contradiction -> miss
                       "gbf.faithfulness.b": "pass",    # missed contradiction       -> miss
                       "gbf.faithfulness.c": "na"}}      # pairing miss on present entity -> miss
    _o_hit, _o_tot, f_hit, f_tot, _s_hit, _s_tot, misses, _slips = _agreement(graded, expected)
    assert f_hit == 0 and f_tot == 3 and len(misses) == 3


def test_contradiction_recall_counts_only_fail_labels():
    # Recall = of entries LABELED fail, how many the judge returned fail. pass/partial labels are ignored.
    expected = {"c.md": {"faithfulness": {"a": "fail", "b": "fail", "c": "pass"}}}
    both = {"c.md": {"gbf.faithfulness.a": "fail", "gbf.faithfulness.b": "fail",
                     "gbf.faithfulness.c": "pass"}}
    hit, tot, missed = _contradiction_recall(both, expected)
    assert (hit, tot) == (2, 2) and missed == []             # 'c' (pass label) is not counted


def test_contradiction_recall_flags_a_missed_contradiction():
    expected = {"c.md": {"faithfulness": {"a": "fail", "b": "fail"}}}
    # judge downgrades one real contradiction to a mere partial -> not caught
    graded = {"c.md": {"gbf.faithfulness.a": "fail", "gbf.faithfulness.b": "partial"}}
    hit, tot, missed = _contradiction_recall(graded, expected)
    assert (hit, tot) == (1, 2) and len(missed) == 1 and "not caught" in missed[0].lower()


def test_contradiction_recall_zero_when_no_fail_labels():
    expected = {"v.md": {"faithfulness": {"a": "pass", "b": "partial"}}}
    graded = {"v.md": {"gbf.faithfulness.a": "pass", "gbf.faithfulness.b": "partial"}}
    assert _contradiction_recall(graded, expected) == (0, 0, [])
