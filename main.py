"""Pipeline entry point.

Loads config + the API key from the environment, builds the orchestrator, runs
the pipeline on a hardcoded list of chat-log files, and writes the finished wiki
to disk. Phase 4.6 will replace the hardcoded FILES / OUTPUT_PATH with an argparse
CLI (--files, --output, --exclude-sources); until then, edit the list below (same
convention as scripts/build_speaker_map.py).
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from orchestrator import Orchestrator, PipelineConfig
from speaker_map import load_speaker_map
from renderer.crosslink import load_crosslink_words

# --- Edit these two for now; 4.6 makes them CLI args. Point FILES at your real
#     logs (logs/ is gitignored). ---
FILES = [
    "logs/group_chat.txt",
    "logs/dm.txt",
]
OUTPUT_PATH = "output/wiki.md"   # output/ is gitignored -- a real-log wiki carries PII

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

    wiki_markdown = Orchestrator().run(FILES, config)

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(wiki_markdown, encoding="utf-8")   # utf-8: names may have accents
    logging.getLogger(__name__).info("Wrote wiki (%d chars) to %s", len(wiki_markdown), out)


if __name__ == "__main__":
    main()
