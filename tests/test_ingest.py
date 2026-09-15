"""Tests for parsers/ingest.py -- the imessage-only parse_messages entry point.

Just confirms the thin seam delegates to the imessage-exporter parser. Offline, no API.
"""

from parsers.ingest import parse_messages
from parsers.imessage_export_parser import parse_imessage_export


SPEAKER_MAP = {"+15551230000": "Matt", "exporter": "Hannah"}

IMESSAGE_LINES = [
    "May 17, 2022  5:29:42 PM",
    "Me",
    "Welcome to the campaign.",
    "",
    "May 17, 2022  5:30:00 PM",
    "+15551230000",
    "Eglon lies to the north.",
]


def write(tmp_path, lines, name="chat.txt"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_parse_messages_delegates_to_imessage_parser(tmp_path):
    f = write(tmp_path, IMESSAGE_LINES)
    assert parse_messages(f, SPEAKER_MAP) == parse_imessage_export(f, SPEAKER_MAP)
