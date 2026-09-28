"""Build and serialize the final ``FCResult``.

The ``graders`` list always carries the mechanical grader; it carries the MODEL grader only
when the model actually ran (≥1 model item is not ``skipped``) -- so a fully-skipped run is
honest about what executed, while the schema's ``minItems: 1`` stays satisfied by the
mechanical entry alone.

Serialization uses ``by_alias=True`` so ``Summary.n_pass`` emits the schema key ``"pass"``.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from evals.common.models import MechanicalGrader, ModelGrader, Status
from evals.common.scoring import build_summary
from evals.fc import FC_LINT_TOOL, FC_LINT_VERSION
from evals.fc.models import FCItem, FCResult


def build_result(wof: str, items: List[FCItem], now: datetime,
                 model_cfg: Optional[dict] = None) -> FCResult:
    graders = [MechanicalGrader(tool=FC_LINT_TOOL, version=FC_LINT_VERSION)]
    model_ran = any(it.engine.value == "model" and it.status != Status.SKIPPED for it in items)
    if model_ran and model_cfg is not None:
        graders.append(ModelGrader(**model_cfg))
    summary = build_summary(items)
    return FCResult(wof=wof, graded_at=now, graders=graders, summary=summary, items=items)


def result_to_json(result: FCResult) -> str:
    """The schema-valid JSON string for a result (2-space indent, non-ASCII preserved).

    ``exclude_none=True`` OMITS absent optionals (``description`` / ``expected`` / ``found`` /
    ``endpoint``) rather than emitting them as explicit ``null`` -- the schema types those as
    plain strings, so a ``null`` would be INVALID. The one required-but-nullable field,
    ``summary.score`` (schema type ``["number","null"]``), is re-added afterward so it is always
    present even when the score is ``None`` (0 applicable checks)."""
    data = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    data.setdefault("summary", {})["score"] = result.summary.score
    return json.dumps(data, indent=2, ensure_ascii=False)


def write_result(result: FCResult, out_path) -> str:
    """Write the result JSON to ``out_path`` (utf-8) and return the JSON string. Creates the
    parent directory if needed so ``--out some/new/dir/r.json`` doesn't crash after the result
    was already computed."""
    text = result_to_json(result)
    out = Path(out_path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    return text
