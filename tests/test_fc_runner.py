"""Tests for evals/fc/runner.py -- the orchestration seam. Offline (fake model client)."""

import json
import re
from datetime import datetime, timezone

from evals.fc.fc_lint import MECHANICAL_IDS, MODEL_IDS
from evals.fc.models import Status
from evals.fc.runner import grade

NOW = datetime(2026, 9, 23, 14, 32, 10, tzinfo=timezone.utc)

WOF = """\
# Title

Subtitle prose.

---

## Locations

### <a id="alpha"></a>Alpha

Alpha near [Beta](#beta).[^1]

### <a id="beta"></a>Beta

Beta.[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""


class BenignFake:
    """Returns the benign label for each check, sized to the request."""

    model, provider, temperature, thinking, base_url = (
        "fake/qwen", "openrouter", 0.6, False, "https://x/api/v1")

    def complete(self, system, user):
        n = len(re.findall(r"(?m)^\d+\.", user))
        label = ("group_label" if "group_label" in system
                 else "exact_ok" if "approximate_unmarked" in system else "ok")
        return json.dumps({"verdicts": [label] * n})


def test_grade_no_model_emits_three_skipped():
    result = grade(WOF, "wiki.md", use_model=False, now=NOW)
    model_items = [i for i in result.items if i.engine.value == "model"]
    assert len(model_items) == 3 and all(i.status == Status.SKIPPED for i in model_items)
    assert result.summary.complete is False
    assert [g.engine for g in result.graders] == ["mechanical"]


def test_grade_with_model_runs_all_three():
    result = grade(WOF, "wiki.md", votes=3, use_model=True, model_client=BenignFake(), now=NOW)
    model_items = [i for i in result.items if i.engine.value == "model"]
    # Benign answers -> a check with candidates passes; one with none (no grouping heading in
    # this WOF) is na. Neither is skipped, so the run stays complete and a model grader is recorded.
    assert len(model_items) == 3 and all(i.status in (Status.PASS, Status.NA) for i in model_items)
    assert result.summary.skipped == 0 and result.summary.complete is True
    assert [g.engine for g in result.graders] == ["mechanical", "model"]
    grader = [g for g in result.graders if g.engine == "model"][0]
    assert grader.name == "fake/qwen" and grader.votes == 3


def test_grade_orders_items_by_canonical_id():
    result = grade(WOF, "wiki.md", use_model=False, now=NOW)
    ids = [i.id for i in result.items]
    assert ids == MECHANICAL_IDS + MODEL_IDS
    assert len(ids) == 27


def test_grade_use_model_true_but_no_client_skips():
    result = grade(WOF, "wiki.md", use_model=True, model_client=None, now=NOW)
    assert result.summary.skipped == 3 and result.summary.complete is False


def test_grade_uses_injected_now():
    result = grade(WOF, "wiki.md", use_model=False, now=NOW)
    assert result.graded_at == NOW
