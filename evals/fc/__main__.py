"""FC eval CLI:

    python -m evals.fc <wof_path> [--out result.json] [--votes N] [--no-model]

Grades one Wiki Output File and emits a schema-valid JSON result to stdout (and to ``--out``
when given). The 3 model checks run only when the model is enabled AND the OpenAI-compat
credentials are present; otherwise they are emitted as ``skipped`` and ``summary.complete``
is false. ``parse_args`` / ``main`` take ``argv`` explicitly so the offline tests drive them
without touching ``sys.argv``.
"""

import argparse
import logging
import os
import sys
from typing import Optional

from evals.fc import DEFAULT_VOTES, ENV_VOTES
from evals.fc.emit import result_to_json, write_result
from evals.fc.model_client import build_model_client
from evals.fc.runner import grade_file

logger = logging.getLogger("evals.fc")


def _default_votes() -> int:
    raw = os.environ.get(ENV_VOTES)
    try:
        return int(raw) if raw else DEFAULT_VOTES
    except ValueError:
        return DEFAULT_VOTES


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evals.fc",
        description="Grade one Wiki Output File against the FC formatting checklist.",
    )
    parser.add_argument("wof_path", metavar="WOF",
                        help="Path to the Wiki Output File (markdown) to grade.")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Also write the JSON result to this path (utf-8).")
    parser.add_argument("--votes", type=int, default=None, metavar="N",
                        help="Self-consistency votes per model candidate (default: "
                             f"${ENV_VOTES} or {DEFAULT_VOTES}). Kept odd is recommended.")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip the 3 model checks (emit them as 'skipped').")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Pull .env so the SDK finds LLM_OPENAI_BASE_URL / LLM_OPENAI_API_KEY (same as main.py).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:                              # dotenv is optional for this offline-first tool
        pass

    votes = args.votes if args.votes is not None else _default_votes()

    model_client: Optional[object] = None
    use_model = not args.no_model
    if use_model:
        # Guard construction: absent credentials return None, but a missing `openai` SDK (the
        # Anthropic-only setup) or a bad base_url raises here -- degrade to the skipped path
        # rather than crashing the CLI with a traceback and no result.
        try:
            model_client = build_model_client()
        except Exception as exc:                    # noqa: BLE001
            logger.warning("[REVIEW] model client unavailable (%s); the 3 model checks will be "
                           "skipped.", exc)
            model_client = None
        if model_client is None:
            logger.info("No usable model backend; the 3 model checks will be skipped.")
            use_model = False
    else:
        logger.info("--no-model: the 3 model checks will be skipped.")

    result = grade_file(args.wof_path, votes=votes, use_model=use_model,
                        model_client=model_client)

    text = result_to_json(result)
    print(text)
    if args.out:
        write_result(result, args.out)
        logger.info("Wrote result to %s", args.out)

    s = result.summary
    logger.info("FC score %s (applicable=%d, pass=%d partial=%d fail=%d na=%d skipped=%d) complete=%s",
                s.score_display, s.applicable, s.n_pass, s.partial, s.fail, s.na, s.skipped,
                s.complete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
