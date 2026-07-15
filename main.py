"""Pipeline entry point + the command-line front door.

Parses the CLI, loads config + the API key from the environment, builds the
orchestrator, runs the pipeline over the given chat-log files, and writes the
finished wiki (plus a restricted copy when --exclude-sources is given) to disk.

    python main.py --files logs/*.txt
    python main.py --files logs/*.txt --exclude-sources dm.txt
    python main.py --files logs/*.txt --output /tmp/wiki.md --exclude-sources dm.txt

The restricted copy's path is DERIVED from --output (see restricted_path), so the
two docs always land together. SPEAKER_MAP_PATH / CROSSLINK_WORDS_PATH stay module
constants on purpose: they're per-install config, not per-run knobs.
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from orchestrator import Orchestrator, PipelineConfig
from speaker_map import load_speaker_map
from renderer.crosslink import load_crosslink_words

DEFAULT_OUTPUT_PATH = "output/wiki.md"   # output/ is gitignored -- a real-log wiki carries PII

# Per-install config, not per-run knobs -- deliberately NOT CLI args.
SPEAKER_MAP_PATH = "config/speaker_map.json"
CROSSLINK_WORDS_PATH = "config/crosslink_words.json"


def parse_args(argv=None):
    """Parse the command line into a Namespace: files, output, exclude_sources.

    `argv` is a parameter rather than read from sys.argv so tests can call
    parse_args(["--files", "a.txt"]) directly with no monkeypatching. argparse falls
    back to sys.argv[1:] when it's None, which is what main() passes in production.

    --files is REQUIRED with no default: the old hardcoded FILES pointed at
    gitignored logs, so a default would make a fresh clone fail confusingly. It stays
    a NAMED arg rather than a positional for two reasons -- a nargs="+" positional
    sitting next to a nargs="+" optional (--exclude-sources) is a real argparse
    ambiguity, and both take filenames, so naming them keeps which-is-which obvious.
    """
    parser = argparse.ArgumentParser(
        prog="wikitizer",
        description="Turn exported D&D chat logs into a structured wiki.",
    )
    parser.add_argument(
        "--files", nargs="+", required=True, metavar="PATH",
        help="Chat-log files to ingest. Shell globs work: --files logs/*.txt",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, metavar="PATH",
        help="Where to write the full wiki (default: %(default)s). With "
             "--exclude-sources, the restricted copy is written alongside it as "
             "<name>_restricted<ext>.",
    )
    parser.add_argument(
        "--exclude-sources", nargs="+", default=[], metavar="FILENAME",
        help="Bare filenames (NOT paths) to hide from a second, restricted wiki -- "
             "e.g. --exclude-sources dm.txt keeps the confidential DM log out of the "
             "players' copy. Omit for no restricted doc. A name that isn't one of "
             "--files is a hard error (exclusion.validate_exclusions, inside run()).",
    )
    args = parser.parse_args(argv)
    # Fail cheap: an existing-directory --output would only blow up at write_text time,
    # AFTER the whole paid pipeline ran -- so reject it up front, same "fail before any
    # paid call" spirit as config loading and validate_exclusions. parser.error exits 2
    # like every other CLI mistake. This guards only the one common, cheaply-checkable
    # case (habitually pointing --output at the gitignored output/ DIR); the broader
    # class of write failures -- permissions, disk-full -- isn't pre-checkable and stays
    # a post-run error.
    if Path(args.output).is_dir():
        parser.error(f"--output must be a file path, not a directory: {args.output!r}")
    return args


def restricted_path(output) -> Path:
    """Where the restricted wiki goes, derived from the full wiki's path:
    output/wiki.md -> output/wiki_restricted.md -- which is EXACTLY the old hardcoded
    RESTRICTED_OUTPUT_PATH, so the default behaviour is unchanged by the CLI.

    Derived rather than given its own arg so the two docs always land TOGETHER
    wherever you point --output. A fixed path would mean `--output /tmp/wiki.md`
    silently dropping the players' copy into output/ where you weren't looking.

    Uses with_name, NOT with_stem: with_stem is a later pathlib addition and this
    project is pinned to 3.9.6, while with_name has been there since pathlib shipped
    in 3.4. Identical result, no version question.
    """
    p = Path(output)
    return p.with_name(p.stem + "_restricted" + p.suffix)


def main(argv=None) -> None:
    """Run the pipeline. `argv` is threaded to parse_args so tests can drive main()
    without touching sys.argv; production calls main() with None."""
    args = parse_args(argv)

    # Turn logging ON. Python's logging is SILENT until configured, so without this
    # every [REVIEW] flag and warning the pipeline emits would go nowhere. After
    # parse_args on purpose: an argparse error or --help writes to stderr and exits
    # on its own, with no logging needed.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Pull .env into the environment so the Anthropic SDK finds ANTHROPIC_API_KEY.
    load_dotenv()

    # Load BOTH config files up front. If either is missing/malformed this raises
    # HERE -- before any paid LLM call -- which is exactly what we want (fail cheap).
    config = PipelineConfig(
        speaker_map=load_speaker_map(SPEAKER_MAP_PATH),
        crosslink_words=load_crosslink_words(CROSSLINK_WORDS_PATH),
    )

    # A bad --exclude-sources name raises ValueError from inside run(), before any
    # paid call. The CLI deliberately does NOT re-check it: validate_exclusions lives
    # in run() so every caller inherits the guard, not just this one.
    output = Orchestrator().run(args.files, config, exclude_sources=args.exclude_sources)

    log = logging.getLogger(__name__)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output.full, encoding="utf-8")   # utf-8: names may have accents
    log.info("Wrote full wiki (%d chars) to %s", len(output.full), out)

    # `restricted is None` means no exclusions were requested. An empty STRING would
    # mean they were, but nothing public survived -- we still write that (an empty
    # wiki is a real answer) and note it, so it's never a silent surprise.
    if output.restricted is not None:
        rout = restricted_path(args.output)
        # A no-op today (the derived path is always a sibling of --output, whose
        # parent we just made), but idempotent and it keeps this block readable
        # standalone.
        rout.parent.mkdir(parents=True, exist_ok=True)
        rout.write_text(output.restricted, encoding="utf-8")
        if output.restricted:
            log.info("Wrote restricted wiki (%d chars) to %s", len(output.restricted), rout)
        else:
            log.warning("Restricted wiki is EMPTY (every source was excluded?); wrote %s anyway", rout)


if __name__ == "__main__":
    main()
