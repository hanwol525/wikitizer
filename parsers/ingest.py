"""Ingestion entry point: parse a chat-log file into ``list[Message]``.

The pipeline ingests **imessage-exporter TXT exports** (ReagentX imessage-exporter,
``imessage-exporter -f txt ...``): each message is a timestamp line, then a sender line,
then the body. ``parse_messages`` is the single entry point the orchestrator calls; it
delegates to :func:`parsers.imessage_export_parser.parse_imessage_export`. (This thin seam
is kept so a future format could be added back behind one call site.)
"""

import logging

from parsers.imessage_export_parser import parse_imessage_export

logger = logging.getLogger(__name__)


def parse_messages(filepath: str, speaker_map: dict) -> list:
    """Parse one imessage-exporter TXT export into a ``list[Message]``."""
    logger.info("Parsing %s as an imessage-exporter TXT export.", filepath)
    return parse_imessage_export(filepath, speaker_map)
