"""Rubric eval CLI:

    python -m evals.rubric <wof_path> [--out result.json] [--votes N] [--no-model] [--rubric PATH]

Grades one Wiki Output File against the campaign rubric and emits a schema-valid JSON result to
stdout (and to ``--out`` when given). It is all-model: with ``--no-model`` or absent OpenAI-compat
credentials, every criterion is emitted as ``skipped`` and ``summary.complete`` is false.
``parse_args`` / ``main`` take ``argv`` explicitly so the offline tests drive them without touching
``sys.argv``.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import yaml
from pydantic import ValidationError

from evals.common import DEFAULT_VOTES
from evals.common.model_client import build_model_client
from evals.rubric import DEFAULT_RUBRIC_PATH, ENV_VOTES, MODEL_PREFIX
from evals.rubric.emit import result_to_json, write_result
from evals.rubric.loader import load_rubric
from evals.rubric.runner import grade_file

logger = logging.getLogger("evals.rubric")


def _default_votes() -> int:
    raw = os.environ.get(ENV_VOTES)
    try:
        return int(raw) if raw else DEFAULT_VOTES
    except ValueError:
        return DEFAULT_VOTES


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evals.rubric",
        description="Grade one Wiki Output File against the campaign rubric (presence + sourcing).",
    )
    parser.add_argument("wof_path", metavar="WOF",
                        help="Path to the Wiki Output File (markdown) to grade.")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Also write the JSON result to this path (utf-8).")
    parser.add_argument("--votes", type=int, default=None, metavar="N",
                        help="Self-consistency votes per model call (default: "
                             f"${ENV_VOTES} or {DEFAULT_VOTES}). Kept odd is recommended.")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip all model checks (emit every criterion as 'skipped').")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC_PATH, metavar="PATH",
                        help=f"Path to the rubric YAML checklist (default: {DEFAULT_RUBRIC_PATH}).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Pull .env so the SDK finds LLM_OPENAI_BASE_URL / LLM_OPENAI_API_KEY (same as main.py).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:                                  # dotenv is optional for this offline-first tool
        pass

    # The rubric IS the checklist -- a missing/invalid file is a clean CLI error, not a traceback.
    try:
        rubric = load_rubric(args.rubric)
    except FileNotFoundError:
        logger.error("Rubric file not found: %s. It is the checklist and is required "
                     "(pass --rubric or create %s).", args.rubric, DEFAULT_RUBRIC_PATH)
        return 2
    except (yaml.YAMLError, ValidationError) as exc:
        logger.error("Rubric file is invalid (%s): %s", args.rubric, exc)
        return 2

    # An empty checklist grades nothing (and would emit a schema-invalid items:[] result), so
    # treat it as a config error -- same spirit as the missing-file hard stop.
    if not any([rubric.sections, rubric.locations, rubric.characters, rubric.organizations,
                rubric.people_and_cultures, rubric.events]):
        logger.error("Rubric %s defines no criteria (empty checklist); nothing to grade.", args.rubric)
        return 2

    votes = args.votes if args.votes is not None else _default_votes()

    model_client: Optional[object] = None
    use_model = not args.no_model
    if use_model:
        try:
            model_client = build_model_client(prefix=MODEL_PREFIX)
        except Exception as exc:                       # noqa: BLE001
            logger.warning("[REVIEW] model client unavailable (%s); every criterion will be "
                           "skipped.", exc)
            model_client = None
        if model_client is None:
            logger.info("No usable model backend; every rubric criterion will be skipped.")
            use_model = False
    else:
        logger.info("--no-model: every rubric criterion will be skipped.")

    result = grade_file(args.wof_path, rubric, votes=votes, use_model=use_model,
                        model_client=model_client)

    text = result_to_json(result)
    print(text)
    if args.out:
        write_result(result, args.out)
        logger.info("Wrote result to %s", args.out)

    s = result.summary
    logger.info("Rubric presence %s, sourcing %s, complete=%s",
                s.presence.score_display, s.sourcing.score_display, s.complete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
