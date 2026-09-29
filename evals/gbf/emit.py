"""Build and serialize the final ``GbfResult``.

Like the rubric, GBF has NO always-on mechanical grader, so ``graders`` always carries a single
``ModelGrader`` (to satisfy the schema's ``minItems: 1``); whether the model actually ran is carried
by the derived ``summary.complete`` + per-item ``skipped`` status, not by grader presence.

Serialization mirrors the rubric's ``exclude_none`` + re-add-nullable-score subtlety, re-adding
BOTH nullable-but-required sub-score ``score`` fields.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from evals.common.models import ModelGrader
from evals.gbf.models import GbfResult, Item
from evals.gbf.scoring import build_gbf_summary


def build_result(wof: str, items: List[Item], now: datetime, model_cfg: dict) -> GbfResult:
    graders = [ModelGrader(**model_cfg)]
    summary = build_gbf_summary(items)             # derives ``complete`` from the items
    return GbfResult(wof=wof, graded_at=now, graders=graders, summary=summary, items=items)


def result_to_json(result: GbfResult) -> str:
    """The schema-valid JSON string (2-space indent, non-ASCII preserved).

    ``exclude_none=True`` omits absent optionals (``description`` / ``expected`` / ``found`` /
    ``endpoint``) rather than emitting invalid ``null``s; the two required-but-nullable ``score``
    fields (schema type ``["number","null"]``) are re-added afterward so they are always present
    even when ``None`` (0 applicable items in a bucket)."""
    data = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    summ = data.setdefault("summary", {})
    summ.setdefault("faithfulness", {})["score"] = result.summary.faithfulness.score
    summ.setdefault("ordering", {})["score"] = result.summary.ordering.score
    return json.dumps(data, indent=2, ensure_ascii=False)


def write_result(result: GbfResult, out_path) -> str:
    """Write the result JSON to ``out_path`` (utf-8) and return the JSON string. Creates the parent
    directory if needed."""
    text = result_to_json(result)
    out = Path(out_path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    return text
