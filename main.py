"""Pipeline entry point.

Loads config + the API key from the environment, builds the orchestrator, runs
the pipeline on a hardcoded list of chat-log files, and writes the finished wiki
(plus a restricted copy when EXCLUDE_SOURCES is set) to disk. A later brief will
replace the hardcoded FILES / OUTPUT_PATH / EXCLUDE_SOURCES with an argparse CLI
(--files, --output, --exclude-sources) -- the exclusion MECHANISM already works
(4.6 Part 2); only the command-line front door is outstanding. Until then, edit
the values below (same convention as scripts/build_speaker_map.py).
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from orchestrator import Orchestrator, PipelineConfig
from speaker_map import load_speaker_map
from renderer.crosslink import load_crosslink_words

# --- Edit these for now; a later brief makes them CLI args. Point FILES at your
#     real logs (logs/ is gitignored). ---
FILES = [
    "logs/group_chat.txt",
    "logs/dm.txt",
]
OUTPUT_PATH = "output/wiki.md"   # output/ is gitignored -- a real-log wiki carries PII

# Bare filenames (NOT paths) to hide from a second, "restricted" wiki -- e.g.
# ["dm.txt"] keeps the confidential DM file out of the players' copy. Empty list =
# no restricted doc. (Brief 3 turns this into a --exclude-sources CLI arg.)
EXCLUDE_SOURCES = []
RESTRICTED_OUTPUT_PATH = "output/wiki_restricted.md"   # only written when EXCLUDE_SOURCES is non-empty

SPEAKER_MAP_PATH = "config/speaker_map.json"
CROSSLINK_WORDS_PATH = "config/crosslink_words.json"


def main() -> None:
    # Turn logging ON. Python's logging is SILENT until configured, so without this
    # every [REVIEW] flag and warning the pipeline emits would go nowhere.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Pull .env into the environment so the Anthropic SDK finds ANTHROPIC_API_KEY.
    load_dotenv()

    # Load BOTH config files up front. If either is missing/malformed this raises
    # HERE -- before any paid LLM call -- which is exactly what we want (fail cheap).
    config = PipelineConfig(
        speaker_map=load_speaker_map(SPEAKER_MAP_PATH),
        crosslink_words=load_crosslink_words(CROSSLINK_WORDS_PATH),
    )

    output = Orchestrator().run(FILES, config, exclude_sources=EXCLUDE_SOURCES)

    log = logging.getLogger(__name__)

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output.full, encoding="utf-8")   # utf-8: names may have accents
    log.info("Wrote full wiki (%d chars) to %s", len(output.full), out)

    # `restricted is None` means no exclusions were requested. An empty STRING would
    # mean they were, but nothing public survived -- we still write that (an empty
    # wiki is a real answer) and note it, so it's never a silent surprise.
    if output.restricted is not None:
        rout = Path(RESTRICTED_OUTPUT_PATH)
        rout.parent.mkdir(parents=True, exist_ok=True)
        rout.write_text(output.restricted, encoding="utf-8")
        if output.restricted:
            log.info("Wrote restricted wiki (%d chars) to %s", len(output.restricted), rout)
        else:
            log.warning("Restricted wiki is EMPTY (every source was excluded?); wrote %s anyway", rout)


if __name__ == "__main__":
    main()
