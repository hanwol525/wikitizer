"""Tests for parsers/ingest.py -- format detection + the parse_messages dispatcher.

Confirms the two input formats are auto-distinguished and routed to the right parser,
that an explicit input_format overrides detection, and that a bad format errors.
Offline, no API, no PII.
"""

import pytest

from parsers.ingest import detect_format, parse_messages
from parsers.chat_parser import parse_chat_log
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

# Minimal valid legacy group export: participant header (leads with a comma), the
# dashes row, a seeding lone timestamp, one body line, then a complete phone footer.
LEGACY_LINES = [
    ",+15551230000",
    "-" * 100,
    "01/01/2024 12:00:00",
    "Eglon lies to the north.",
    "+15551230000 01/01/2024 12:00:05",
]


def write(tmp_path, lines, name="chat.txt"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# --- detect_format ---------------------------------------------------------- #
def test_detect_imessage_by_leading_timestamp(tmp_path):
    assert detect_format(write(tmp_path, IMESSAGE_LINES)) == "imessage"


def test_detect_legacy_by_dashes_row(tmp_path):
    assert detect_format(write(tmp_path, LEGACY_LINES)) == "legacy"


def test_detect_defaults_to_legacy_when_ambiguous(tmp_path):
    assert detect_format(write(tmp_path, ["just some prose", "and more prose"])) == "legacy"


# --- parse_messages routing ------------------------------------------------- #
def test_auto_routes_imessage(tmp_path):
    f = write(tmp_path, IMESSAGE_LINES)
    assert parse_messages(f, SPEAKER_MAP, "auto") == parse_imessage_export(f, SPEAKER_MAP)


def test_auto_routes_legacy(tmp_path):
    f = write(tmp_path, LEGACY_LINES)
    assert parse_messages(f, SPEAKER_MAP, "auto") == parse_chat_log(f, SPEAKER_MAP)


def test_explicit_imessage_overrides_detection(tmp_path):
    # A file that would auto-detect legacy, forced through the imessage parser, yields
    # the imessage parser's result (here: no messages, since it isn't that format).
    f = write(tmp_path, LEGACY_LINES)
    assert parse_messages(f, SPEAKER_MAP, "imessage") == parse_imessage_export(f, SPEAKER_MAP)


def test_default_input_format_is_auto(tmp_path):
    f = write(tmp_path, IMESSAGE_LINES)
    assert parse_messages(f, SPEAKER_MAP) == parse_imessage_export(f, SPEAKER_MAP)


def test_bad_input_format_raises(tmp_path):
    f = write(tmp_path, IMESSAGE_LINES)
    with pytest.raises(ValueError):
        parse_messages(f, SPEAKER_MAP, "bogus")
