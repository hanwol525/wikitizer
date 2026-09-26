"""Orchestration: parse a WOF, run the 24 mechanical checks + the 3 model checks (or 3
``skipped`` items when the model is off/unavailable), score, and build the ``FCResult``.

This is the seam the CLI (``__main__``) and the tests both drive. ``now`` and ``model_client``
are injected so a test can pin a deterministic timestamp and feed a fake model.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evals.common import DEFAULT_PROVIDER, DEFAULT_TEMPERATURE
from evals.common.models import Engine, Evidence, Status
from evals.common.parse import parse_wof
from evals.fc.adjudicator import Adjudicator, GROUPING_ID, TBD_ID, TILDE_ID
from evals.fc.emit import build_result
from evals.fc.fc_lint import (
    CRITERION_TEXT,
    MECHANICAL_IDS,
    MODEL_IDS,
    figure_candidates,
    grouping_candidates,
    run_lint,
    tbd_candidates,
)
from evals.fc.models import FCItem, FCResult

logger = logging.getLogger("evals.fc.runner")

_ID_ORDER = {cid: i for i, cid in enumerate(MECHANICAL_IDS + MODEL_IDS)}


def _skipped_item(cid: str, detail: str = "model disabled or unavailable") -> FCItem:
    return FCItem(id=cid, description=CRITERION_TEXT.get(cid), engine=Engine.MODEL,
                  status=Status.SKIPPED, evidence=Evidence(lines=[], detail=detail))


def _skipped_model_items() -> list:
    return [_skipped_item(cid) for cid in MODEL_IDS]


def _safe_model_check(cid: str, fn) -> FCItem:
    """Run one model check; a request-time client error (auth, timeout, a 5xx the SDK
    exhausted its retries on) degrades JUST that check to a skipped ``[REVIEW]`` item instead
    of aborting the whole run -- so grade() still emits a valid FCResult with the 24 mechanical
    results intact (the "degrade toward less-complete, never corrupt" policy)."""
    try:
        return fn()
    except Exception as exc:                       # noqa: BLE001 -- any transport/SDK error degrades
        logger.warning("[REVIEW] model check %s errored (%s); skipping it", cid, exc)
        return _skipped_item(cid, detail=f"[REVIEW] model check errored: {exc}")


def _model_cfg(model_client, votes: int) -> dict:
    return {
        "name": getattr(model_client, "model", "unknown"),
        "provider": getattr(model_client, "provider", DEFAULT_PROVIDER),
        "temperature": float(getattr(model_client, "temperature", DEFAULT_TEMPERATURE)),
        "thinking": bool(getattr(model_client, "thinking", False)),
        # Record the EFFECTIVE vote count (the adjudicator clamps to >=1), not a requested 0/neg.
        "votes": max(1, int(votes)),
        "endpoint": getattr(model_client, "base_url", None),
    }


def grade(wof_text: str, wof_name: str, votes: int = 3, use_model: bool = True,
          model_client=None, now: Optional[datetime] = None) -> FCResult:
    now = now or datetime.now(timezone.utc)
    parsed = parse_wof(wof_text)
    items = run_lint(parsed)

    model_cfg = None
    if use_model and model_client is not None:
        adj = Adjudicator(model_client, votes=votes)   # construction is network-free
        items.append(_safe_model_check(
            GROUPING_ID, lambda: adj.judge_grouping(grouping_candidates(parsed))))
        items.append(_safe_model_check(
            TILDE_ID, lambda: adj.judge_approx_tilde(figure_candidates(parsed))))
        items.append(_safe_model_check(
            TBD_ID, lambda: adj.judge_tbd(tbd_candidates(parsed))))
        model_cfg = _model_cfg(model_client, votes)
    else:
        items.extend(_skipped_model_items())

    items.sort(key=lambda it: _ID_ORDER.get(it.id, 999))
    return build_result(wof_name, items, now, model_cfg)


def grade_file(path, votes: int = 3, use_model: bool = True, model_client=None,
               now: Optional[datetime] = None) -> FCResult:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return grade(text, path.name, votes=votes, use_model=use_model,
                 model_client=model_client, now=now)
