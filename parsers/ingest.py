"""Ingestion dispatch: pick the right parser for a chat-log file.

Two input formats are supported (dual input):
  * ``legacy``   -- the copy-pasted iMessage `.txt` (``parsers/chat_parser``): a
    participant line then a row of 100 dashes, footers, alignment.
  * ``imessage`` -- a structured ReagentX **imessage-exporter** TXT export
    (``parsers/imessage_export_parser``): each message is a timestamp line, then a
    sender line, then the body.

Both emit the same ``list[Message]``, so everything downstream is format-agnostic.
``parse_messages`` is the single entry point the orchestrator calls; ``"auto"``
(the default) sniffs the file so a user rarely needs the ``--input-format`` flag.
"""

import logging

from parsers.chat_parser import read_clean, parse_chat_log
from parsers.imessage_export_parser import parse_imessage_export, _timestamp_or_none

logger = logging.getLogger(__name__)

# How many leading non-blank lines to sniff when auto-detecting the format.
_SNIFF_LINES = 5


def detect_format(filepath: str) -> str:
    """Sniff a file's first few non-blank lines and return ``"imessage"`` or
    ``"legacy"``. The two are trivially distinguishable: an imessage-exporter export
    opens with a message **timestamp line** ("May 17, 2022  5:29:42 PM"), while the
    legacy copy-paste opens with a participant line followed by a row of dashes.
    Defaults to ``"legacy"`` when neither signal is seen (the safe, pre-existing path).
    """
    seen = 0
    for line in read_clean(filepath):
        if not line.strip():
            continue
        if _timestamp_or_none(line) is not None:
            return "imessage"
        # A run of dashes (the ~100-dash separator) is the legacy header's tell.
        if set(line.strip()) == {"-"} and len(line.strip()) >= 10:
            return "legacy"
        seen += 1
        if seen >= _SNIFF_LINES:
            break
    return "legacy"


def parse_messages(filepath: str, speaker_map: dict, input_format: str = "auto") -> list:
    """Parse one chat-log file into a ``list[Message]`` using the chosen format.

    ``input_format`` is ``"auto"`` (sniff via :func:`detect_format`), ``"imessage"``,
    or ``"legacy"``. Any other value is an error (surfaced to the caller).
    """
    fmt = detect_format(filepath) if input_format == "auto" else input_format
    if fmt == "imessage":
        logger.info("Parsing %s as an imessage-exporter TXT export.", filepath)
        return parse_imessage_export(filepath, speaker_map)
    if fmt == "legacy":
        return parse_chat_log(filepath, speaker_map)
    raise ValueError(
        f"Unknown input_format {input_format!r}; expected 'auto', 'imessage', or 'legacy'."
    )
