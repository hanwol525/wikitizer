"""Dev A/B harness for the six extractor agents: baseline (Sonnet) vs a cheaper
candidate, over the SAME real filtered messages. Prints a coverage + quote report
so Han can decide, on real data, whether the extractor swap loses lore.

    python -m evals.extract_ab --logs logs/ \
        --candidate openrouter:deepseek/deepseek-v4-flash

THIS MAKES REAL, PAID API CALLS and reads Han's real (gitignored) logs. It is a
standalone dev script -- NOT part of the pytest suite (its filename doesn't match
`test_*.py`, so pytest never collects it) and it changes NO pipeline code. It only
composes existing seams: the config loaders, `Orchestrator._build_agents` (the
same construction seam the orchestrator tests use), `parse_messages`, the noise
filter, and each extractor's `.extract`. The candidate swap itself is pure `.env`
config (per-role `WIKITIZER_<ROLE>_MODEL`); this harness just sets that env var
for the EXTRACT role around each run.

The report has two tiers:
  * PRIMARY  -- entity coverage + surviving-quote counts, read straight off the
                returned lore objects. Robust; unaffected by the dev cache.
  * DIAGNOSTIC -- candidate dropped-quote count (cosmetic vs reword) + json-repair
                retry counts, parsed from WARNING logs. Best-effort and
                cache-sensitive (see the caveats printed in the report).

COUPLING NOTE: the DIAGNOSTIC tier reads two specific WARNING call sites --
`_resolve_quote`'s "Quote not found verbatim..." drop (agents/base_extractor.py)
and `call_claude_json`'s "Claude returned invalid JSON..." retry (agents/base.py).
If either log template changes, `_ExtractLogCapture` below needs updating. The
PRIMARY tier is object-derived and is unaffected by any log change.
"""

import argparse
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the harness runnable both as `python -m evals.extract_ab` and as
# `python evals/extract_ab.py` by ensuring the repo root (which holds
# orchestrator.py, speaker_map.py, the `evals` and `agents` packages, ...) is on
# sys.path. A dev-script convenience; the pipeline never does this.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from orchestrator import Orchestrator, PipelineConfig
from speaker_map import load_speaker_map
from player_map import load_player_map
from renderer.crosslink import load_crosslink_words
from parsers.ingest import parse_messages
from agents.noise_filter import select_for_extraction
from agents.base_extractor import _normalize_for_match

from evals.extract_metrics import (
    ENTITY_TYPES,
    classify_dropped_quote,
    format_report,
    match_entities,
)

# Per-install config defaults -- mirror main.py's constants so the eval reads the
# same gitignored config files main.py does.
CROSSLINK_WORDS_PATH = "config/crosslink_words.json"
DEFAULT_SPEAKER_MAP_PATH = "config/speaker_map.json"
DEFAULT_PLAYER_MAP_PATH = "config/player_map.json"

# The two env vars the run mutates per model; snapshotted and restored so the
# process is left as it was found (matters if the harness is ever imported).
_MUTATED_ENV = ("WIKITIZER_EXTRACT_MODEL", "WIKITIZER_OPENAI_REASONING")

logger = logging.getLogger("evals.extract_ab")


def _is_openai_spec(spec: str) -> bool:
    """True for a spec routed to the OpenAI-compat endpoint (OpenRouter/GLM/etc.).
    Mirrors `_OPENAI_PREFIXES` in agents/llm_client.py."""
    return spec.startswith(("openrouter:", "openai:"))


def _require_openai_env(specs) -> None:
    """Fail cheap (before ANY paid call) if either run's spec routes to the
    OpenAI-compat endpoint but its credentials aren't set.

    Checked for BOTH specs up front on purpose: the baseline runs first and is
    usually an Anthropic model that skips this check, so validating inside the
    per-run loop would bill the whole baseline extraction (the dominant cost)
    before the candidate's missing key ever surfaced.
    """
    needs = [s for s in specs if _is_openai_spec(s)]
    if needs and (not os.environ.get("LLM_OPENAI_BASE_URL")
                  or not os.environ.get("LLM_OPENAI_API_KEY")):
        raise SystemExit(
            "A model spec routes to the OpenAI-compat endpoint "
            f"({', '.join(needs)}), but LLM_OPENAI_BASE_URL and/or "
            "LLM_OPENAI_API_KEY are not set.\n"
            "Add them to .env, e.g.:\n"
            "    LLM_OPENAI_BASE_URL=https://openrouter.ai/api/v1\n"
            "    LLM_OPENAI_API_KEY=<your key>"
        )


class _ExtractLogCapture(logging.Handler):
    """Captures the two WARNING call sites the DIAGNOSTIC tier needs, reading the
    structured `record.args`/`record.msg` directly (no string formatting), and
    attributes each dropped quote to whichever extractor type is currently running
    (`current_type`, set by the caller before each `.extract()`).

    Attribution is clean ONLY because the harness runs the extractors ONE AT A
    TIME in a single thread -- every record between two `current_type` assignments
    belongs to the current type. (This is why the harness bypasses the
    orchestrator's ThreadPoolExecutor.) `emit` can never raise into the run.
    """

    def __init__(self):
        # WARNING level so the INFO "recovered from its true message" line and
        # other progress chatter are ignored automatically.
        super().__init__(level=logging.WARNING)
        self.current_type = None
        self.drops = defaultdict(list)   # entity_type -> [dropped quote text, ...]
        self.json_retries = 0

    def emit(self, record):
        try:
            msg = record.msg
            args = record.args if isinstance(record.args, tuple) else ()
            # The real paraphrase-drop: base_extractor's _resolve_quote logs this
            # exact template with args (source_id, quote_text, detail_suffix) when
            # a quote is verbatim in NEITHER its cited message NOR anywhere in its
            # batch. The funcName + template check isolates it from the three
            # sibling warnings (non-string / non-int / out-of-range id) and the
            # ambiguous-multi-match line (which RECOVERS, not drops).
            if (record.name == "agents.base_extractor"
                    and getattr(record, "funcName", "") == "_resolve_quote"
                    and isinstance(msg, str)
                    and msg.startswith("Quote not found verbatim")):
                if len(args) >= 2:
                    self.drops[self.current_type].append(args[1])   # quote_text
            # A json_repair parse-retry: base's call_claude_json logs this per
            # failed parse before it re-asks or raises.
            elif (record.name == "agents.base"
                    and isinstance(msg, str)
                    and msg.startswith("Claude returned invalid JSON")):
                self.json_retries += 1
        except Exception:   # a diagnostic handler must NEVER break the paid run
            pass


def parse_args(argv=None):
    """Parse the harness CLI. `argv` is a param (not read from sys.argv) so the
    no-paid paths stay easy to drive by hand."""
    parser = argparse.ArgumentParser(
        prog="extract_ab",
        description="A/B the six extractor agents (baseline vs candidate) over real logs.",
    )
    parser.add_argument(
        "--logs", required=True, metavar="DIR",
        help="Directory of exported chat logs (gitignored). All *.txt in it are ingested.",
    )
    parser.add_argument(
        "--candidate", default="openrouter:deepseek/deepseek-v4-flash", metavar="SPEC",
        help="Model spec for the candidate EXTRACT run (default: %(default)s). "
             "CONFIRM the exact slug on OpenRouter's model list -- this default may be "
             "a placeholder. An openrouter:/openai: prefix routes to the OpenAI-compat "
             "endpoint (needs LLM_OPENAI_BASE_URL + LLM_OPENAI_API_KEY in .env).",
    )
    parser.add_argument(
        "--baseline", default="claude-sonnet-4-6", metavar="SPEC",
        help="Model spec for the trusted baseline EXTRACT run (default: %(default)s).",
    )
    parser.add_argument(
        "--speaker-map", default=DEFAULT_SPEAKER_MAP_PATH, metavar="PATH",
        help="Path to the speaker map JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--player-map", default=DEFAULT_PLAYER_MAP_PATH, metavar="PATH",
        help="Path to the declared-party JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--report-file", default=None, metavar="PATH",
        help="Optional: also write the report (markdown) to this path.",
    )
    return parser.parse_args(argv)


def _build_config(args) -> PipelineConfig:
    """Load the gitignored config files into a PipelineConfig -- the same three
    loaders main.py uses (there is no build_config helper to import, so we mirror
    main.py:283-288 here). `current_year`/`player_map` don't affect extraction
    coverage, but the characters extractor is wired with `player_map` in
    _build_agents, so we pass the real one."""
    return PipelineConfig(
        speaker_map=load_speaker_map(args.speaker_map),
        crosslink_words=load_crosslink_words(CROSSLINK_WORDS_PATH),
        current_year=None,
        player_map=load_player_map(args.player_map),
    )


def run(args) -> str:
    """Do the whole A/B and return the report string (also printed by main)."""
    # --- enumerate logs first (fail cheap on an empty dir, before any config
    # load or paid call) --------------------------------------------------------
    files = sorted(str(p) for p in Path(args.logs).glob("*.txt"))
    if not files:
        raise SystemExit(f"No *.txt logs found in {args.logs!r}; nothing to extract.")
    logger.info("Ingesting %d log file(s) from %s", len(files), args.logs)

    # Validate OpenAI-compat credentials for BOTH runs up front, before the first
    # paid call (the noise filter) or the baseline extraction -- a missing key must
    # not cost a full baseline run before it surfaces. (main() has already loaded
    # .env, so os.environ reflects it here.)
    _require_openai_env([args.baseline, args.candidate])

    config = _build_config(args)

    saved = {k: os.environ.get(k) for k in _MUTATED_ENV}
    try:
        # --- build the shared filtered input ONCE, under clean EXTRACT env -------
        for k in _MUTATED_ENV:
            os.environ.pop(k, None)
        orch0 = Orchestrator(cache=True)
        noise_filter, *_ = orch0._build_agents(config)

        all_messages = []
        for filepath in files:
            all_messages.extend(parse_messages(filepath, config.speaker_map))
        classified = noise_filter.classify(all_messages)
        filtered = select_for_extraction(classified)
        source_texts = [m.content for m in filtered]
        logger.info("Filtered to %d message(s) for extraction (shared by both runs).",
                    len(filtered))

        # No messages survived the funnel -> extractors would short-circuit empty
        # (no paid call). Report zeros honestly and stop.
        if not filtered:
            report = format_report(
                models={"baseline": args.baseline, "candidate": args.candidate,
                        "filtered_messages": 0},
                per_type_diffs=[match_entities([], [], t) for t in ENTITY_TYPES],
                quote_stats={"survival": {}, "drops": {}, "drop_samples": []},
                json_retry_stats={"baseline": 0, "candidate": 0},
            )
            return report

        # --- run each model over the shared filtered list -----------------------
        results = {}     # label -> {type -> [entities]}
        qcounts = {}     # label -> {type -> surviving-quote count}
        drops = {}       # label -> {type -> [dropped quote text]}
        retries = {}     # label -> int

        for label, spec in (("baseline", args.baseline), ("candidate", args.candidate)):
            # Set the EXTRACT model BEFORE the fresh Orchestrator so its first
            # resolve("EXTRACT") builds the right client.
            os.environ["WIKITIZER_EXTRACT_MODEL"] = spec
            # Credentials were validated up front by _require_openai_env; here we
            # only pick the reasoning mode. For an OpenAI-compat candidate, turn
            # reasoning OFF (V4/GLM non-thinking mode -- we want extraction, not
            # deliberation); it's captured at client-build time inside _build_agents.
            if _is_openai_spec(spec):
                os.environ["WIKITIZER_OPENAI_REASONING"] = "off"
            else:
                os.environ.pop("WIKITIZER_OPENAI_REASONING", None)

            logger.info("=== %s run: WIKITIZER_EXTRACT_MODEL=%s ===", label, spec)
            orch = Orchestrator(cache=True)   # fresh resolver -> clean provider cache
            _nf, extractors, _rec, _prose = orch._build_agents(config)

            handler = _ExtractLogCapture()
            base_ext_logger = logging.getLogger("agents.base_extractor")
            base_logger = logging.getLogger("agents.base")
            base_ext_logger.addHandler(handler)
            base_logger.addHandler(handler)
            try:
                results[label] = {}
                qcounts[label] = {}
                # Run the six extractors ONE AT A TIME (not the executor) so log
                # capture attributes cleanly and deterministically to each type.
                for etype in ENTITY_TYPES:
                    handler.current_type = etype     # attribute BEFORE the call
                    ents = extractors[etype].extract(filtered)
                    results[label][etype] = ents
                    qcounts[label][etype] = sum(len(e.supporting_quotes) for e in ents)
                    logger.info("  %-14s -> %3d entities, %3d surviving quotes",
                                etype, len(ents), qcounts[label][etype])
            finally:
                base_ext_logger.removeHandler(handler)
                base_logger.removeHandler(handler)
            drops[label] = dict(handler.drops)
            retries[label] = handler.json_retries
    finally:
        # Leave the process env exactly as we found it.
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # --- diff (PRIMARY) + classify the candidate's drops (DIAGNOSTIC) -----------
    per_type_diffs = [
        match_entities(results["baseline"][t], results["candidate"][t], t)
        for t in ENTITY_TYPES
    ]
    survival = {
        t: {"baseline": qcounts["baseline"][t], "candidate": qcounts["candidate"][t]}
        for t in ENTITY_TYPES
    }
    # Classify every candidate drop against the shared filtered messages, passing
    # the REAL production normalizer so "cosmetic" means exactly "a fold the real
    # _normalize_for_match doesn't do yet".
    classified_drops = [
        (classify_dropped_quote(qt, source_texts, _normalize_for_match), qt)
        for t in ENTITY_TYPES
        for qt in drops.get("candidate", {}).get(t, [])
    ]
    tally = Counter(c for c, _ in classified_drops)
    quote_stats = {
        "survival": survival,
        "drops": {"cosmetic": tally.get("cosmetic", 0),
                  "reword": tally.get("reword", 0),
                  "unknown": tally.get("unknown", 0)},
        "drop_samples": classified_drops[:12],
    }
    json_retry_stats = {"baseline": retries["baseline"], "candidate": retries["candidate"]}

    return format_report(
        models={"baseline": args.baseline, "candidate": args.candidate,
                "filtered_messages": len(filtered)},
        per_type_diffs=per_type_diffs,
        quote_stats=quote_stats,
        json_retry_stats=json_retry_stats,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    # Show pipeline progress + [REVIEW]/warnings on stderr during the (long, paid)
    # run; the report goes to stdout at the end, so a `> report.md` redirect keeps
    # them separate.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Pull .env so the SDKs find ANTHROPIC_API_KEY / LLM_OPENAI_* (same as main.py).
    load_dotenv()

    report = run(args)
    print(report)
    if args.report_file:
        Path(args.report_file).write_text(report + "\n", encoding="utf-8")
        logger.info("Wrote report to %s", args.report_file)


if __name__ == "__main__":
    main()
