"""Golden test: run GBF --no-model over the real gold + ranks and check the shape is schema-valid.

Offline (no model calls, no key). Skips when the gitignored gold or ranks file is absent (a fresh
clone). The acceptance bar here is only "the criterion universe enumerates and the envelope is valid";
the model-graded quality bars live in the integration + bake-off suites.
"""

import json
from pathlib import Path

import pytest

from evals.gbf.emit import result_to_json
from evals.gbf.reference import load_gold, load_ranks
from evals.gbf.runner import grade_file

_ROOT = Path(__file__).resolve().parent.parent
_GOLD = _ROOT / "output" / "gol-lore-full.md"
_RANKS = _ROOT / "output" / "gbf_event_ranks.yaml"

pytestmark = pytest.mark.skipif(
    not (_GOLD.exists() and _RANKS.exists()),
    reason="output/gol-lore-full.md or output/gbf_event_ranks.yaml not present (gitignored)",
)


def test_no_model_golden_is_all_skipped_and_schema_valid():
    jsonschema = pytest.importorskip("jsonschema")
    gold = load_gold(_GOLD)
    ranks = load_ranks(_RANKS)
    result = grade_file(_GOLD, gold, ranks, use_model=False)

    assert result.eval == "gbf"
    assert result.summary.complete is False
    assert result.summary.faithfulness.score_display == "0/0"
    assert result.summary.ordering.score_display == "0/0"
    assert all(it.status.value == "skipped" for it in result.items)
    # exactly one ordering item + one faithfulness item per real gold entry
    assert sum(1 for it in result.items if it.id == "gbf.ordering") == 1
    assert sum(1 for it in result.items if it.id.startswith("gbf.faithfulness.")) == len(result.items) - 1

    schema = json.loads((_ROOT / "gbf-result.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(result_to_json(result)), schema)


def test_every_ranked_slug_exists_in_the_gold():
    # Guard the ranks file against drift: every ranked slug must be a real gold event anchor.
    from evals.gbf.pair import gold_slug, real_entries
    gold = load_gold(_GOLD)
    ranks = load_ranks(_RANKS)
    gold_slugs = {gold_slug(e) for e in real_entries(gold)}
    missing = [s for s in ranks if s not in gold_slugs]
    assert not missing, f"ranks reference slugs absent from the gold: {missing}"
