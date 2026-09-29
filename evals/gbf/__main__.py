"""GBF eval CLI:

    python -m evals.gbf <wof_path> [--out result.json] [--votes N] [--no-model]
                        [--gold PATH] [--ranks PATH]

Grades one Wiki Output File against the gold reference for faithfulness + ordering and emits a
schema-valid JSON result to stdout (and to ``--out`` when given). It is all-model: with ``--no-model``
or absent OpenAI-compat credentials, every criterion is emitted as ``skipped`` and
``summary.complete`` is false. ``parse_args`` / ``main`` take ``argv`` explicitly so the offline
tests drive them without touching ``sys.argv``.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import yaml

from evals.common.model_client import build_model_client
from evals.gbf import (
    DEFAULT_GBF_MODEL,
    DEFAULT_GBF_VOTES,
    DEFAULT_GOLD_PATH,
    DEFAULT_MAX_WORKERS,
    DEFAULT_RANKS_PATH,
    ENV_MAX_WORKERS,
    ENV_VOTES,
    MODEL_PREFIX,
)
from evals.gbf.emit import result_to_json, write_result
from evals.gbf.reference import load_gold, load_ranks
from evals.gbf.runner import grade_file

logger = logging.getLogger("evals.gbf")


def _default_votes() -> int:
    raw = os.environ.get(ENV_VOTES)
    try:
        return int(raw) if raw else DEFAULT_GBF_VOTES
    except ValueError:
        return DEFAULT_GBF_VOTES


def _default_workers() -> int:
    raw = os.environ.get(ENV_MAX_WORKERS)
    try:
        return int(raw) if raw else DEFAULT_MAX_WORKERS
    except ValueError:
        return DEFAULT_MAX_WORKERS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evals.gbf",
        description="Grade one Wiki Output File against the gold (faithfulness + ordering).",
    )
    parser.add_argument("wof_path", metavar="WOF",
                        help="Path to the Wiki Output File (markdown) to grade.")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Also write the JSON result to this path (utf-8).")
    parser.add_argument("--votes", type=int, default=None, metavar="N",
                        help=f"Self-consistency votes per model call (default: ${ENV_VOTES} or "
                             f"{DEFAULT_GBF_VOTES}). Kept odd is recommended.")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip all model checks (emit every criterion as 'skipped').")
    parser.add_argument("--gold", default=DEFAULT_GOLD_PATH, metavar="PATH",
                        help=f"Path to the gold reference WOF (default: {DEFAULT_GOLD_PATH}).")
    parser.add_argument("--ranks", default=DEFAULT_RANKS_PATH, metavar="PATH",
                        help=f"Path to the event-ranks YAML (default: {DEFAULT_RANKS_PATH}).")
    parser.add_argument("--max-workers", type=int, default=None, metavar="N",
                        help=f"Parallel faithfulness calls (default: ${ENV_MAX_WORKERS} or "
                             f"{DEFAULT_MAX_WORKERS}; 1 = sequential).")
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

    # The gold + ranks ARE the reference -- a missing/invalid file is a clean CLI error, not a traceback.
    try:
        gold = load_gold(args.gold)
    except FileNotFoundError:
        logger.error("Gold reference not found: %s. It is the faithfulness/ordering reference and is "
                     "required (pass --gold or create %s).", args.gold, DEFAULT_GOLD_PATH)
        return 2
    try:
        ranks = load_ranks(args.ranks)
    except FileNotFoundError:
        logger.error("Event-ranks file not found: %s. It is the ordering ground truth and is required "
                     "(pass --ranks or create %s).", args.ranks, DEFAULT_RANKS_PATH)
        return 2
    except (ValueError, yaml.YAMLError) as exc:
        logger.error("Event-ranks file is invalid (%s): %s", args.ranks, exc)
        return 2

    votes = args.votes if args.votes is not None else _default_votes()
    max_workers = args.max_workers if args.max_workers is not None else _default_workers()

    model_client: Optional[object] = None
    use_model = not args.no_model
    if use_model:
        try:
            model_client = build_model_client(prefix=MODEL_PREFIX, default_model=DEFAULT_GBF_MODEL)
        except Exception as exc:                       # noqa: BLE001
            logger.warning("[REVIEW] model client unavailable (%s); every criterion will be skipped.", exc)
            model_client = None
        if model_client is None:
            logger.info("No usable model backend; every GBF criterion will be skipped.")
            use_model = False
    else:
        logger.info("--no-model: every GBF criterion will be skipped.")

    result = grade_file(args.wof_path, gold, ranks, votes=votes, use_model=use_model,
                        model_client=model_client, max_workers=max_workers)

    text = result_to_json(result)
    print(text)
    if args.out:
        write_result(result, args.out)
        logger.info("Wrote result to %s", args.out)

    s = result.summary
    logger.info("GBF faithfulness %s, ordering %s, complete=%s",
                s.faithfulness.score_display, s.ordering.score_display, s.complete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
