"""Offline consistency guard for the GBF adversarial fixtures (tests/fixtures/gbf/).

The full grading of these fixtures needs the gitignored gold + a live judge (test_gbf_bakeoff.py),
so on a fresh clone nothing else touches them. This test runs with no gold and no model: it just
proves each fixture parses, carries the entries expected.yaml references, and has a real History
sequence -- catching a fixture typo (a renamed anchor, a dropped entry) that would otherwise only
surface in the triple-gated bake-off.
"""

from pathlib import Path

import pytest
import yaml

from evals.common.parse import parse_wof
from evals.gbf.pair import gold_slug, real_entries

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gbf"
_EXPECTED = _FIXTURES / "expected.yaml"


def _expected():
    return yaml.safe_load(_EXPECTED.read_text(encoding="utf-8"))


def test_expected_yaml_lists_the_four_variants():
    exp = _expected()
    assert set(exp) == {"faithful.md", "contradictions.md", "omissions.md", "reordered.md"}
    for spec in exp.values():
        assert "ordering" in spec and "faithfulness" in spec


@pytest.mark.parametrize("variant", ["faithful.md", "contradictions.md", "omissions.md", "reordered.md"])
def test_fixture_parses_and_contains_expected_entries(variant):
    text = (_FIXTURES / variant).read_text(encoding="utf-8")
    parsed = parse_wof(text)
    slugs = {gold_slug(e) for e in real_entries(parsed)}
    expected_slugs = set((_expected()[variant].get("faithfulness") or {}))
    missing = expected_slugs - slugs
    assert not missing, f"{variant}: expected.yaml references entries absent from the fixture: {missing}"


def test_history_events_are_consistent_across_variants():
    # Every variant should carry the same 12 History event anchors (only their ORDER / prose differs).
    def hist_anchors(variant):
        parsed = parse_wof((_FIXTURES / variant).read_text(encoding="utf-8"))
        hist = next(s for s in parsed.sections if s.title.strip().lower() == "history")
        return {e.anchor_id for e in hist.entries if not e.is_grouping}

    base = hist_anchors("faithful.md")
    assert len(base) == 12
    for variant in ("contradictions.md", "omissions.md", "reordered.md"):
        assert hist_anchors(variant) == base, f"{variant} History anchors differ from faithful.md"
