"""Golden-file test: run the FC linter over output/gol-lore-full.md and assert every
mechanical check is pass/na -- the acceptance bar.

The golden WOF is gitignored PII (a real-log wiki), so this `skipif`s when it's absent (a
fresh clone), mirroring tests/fixtures/real_messages.py. Offline: no model, no key.
"""

from pathlib import Path

import pytest

from evals.fc.fc_lint import MECHANICAL_IDS, run_lint
from evals.fc.parse import parse_wof
from evals.fc.runner import grade

_GOLDEN = Path(__file__).resolve().parent.parent / "output" / "gol-lore-full.md"
pytestmark = pytest.mark.skipif(not _GOLDEN.exists(),
                                reason="output/gol-lore-full.md not present (gitignored)")


def _text():
    return _GOLDEN.read_text(encoding="utf-8")


def test_golden_all_mechanical_pass_or_na():
    items = run_lint(parse_wof(_text()))
    assert len(items) == 24
    offenders = [(i.id, i.status.value, i.evidence.detail) for i in items
                 if i.status.value not in ("pass", "na")]
    assert offenders == [], offenders


def test_golden_has_every_mechanical_id_with_evidence():
    items = run_lint(parse_wof(_text()))
    assert {i.id for i in items} == set(MECHANICAL_IDS)
    assert all(i.evidence.detail for i in items)


def test_golden_no_model_run_summary_math():
    result = grade(_text(), "gol-lore-full.md", use_model=False)
    s = result.summary
    assert s.skipped == 3 and s.complete is False
    assert s.applicable == s.n_pass + s.partial + s.fail
    assert s.fail == 0 and s.partial == 0             # golden is clean on the mechanical side
    assert s.score == 1.0 and s.score_display.endswith("/24")
