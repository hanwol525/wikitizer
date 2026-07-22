"""Tests for parsers/imessage_export_parser.py -- the ReagentX imessage-exporter
TXT ingestion path.

Fixtures are inline imessage-exporter-format strings written to `tmp_path` (the
parser reads real files via read_clean). Offline, no API, no PII. The legacy
copy-paste parser and its tests are untouched (dual input).
"""

from datetime import datetime

import pytest

from parsers.imessage_export_parser import (
    parse_imessage_export,
    _timestamp_or_none,
    _resolve_sender,
    _strip_annotations,
    _normalize_phone,
    _is_attachment_line,
)


SPEAKER_MAP = {"+16303463392": "Sam", "+15551230000": "Matt", "exporter": "Hannah"}


def write(tmp_path, lines, name="chat.txt"):
    """Write imessage-exporter-style lines to a temp file (plain \\n, utf-8)."""
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def parse(tmp_path, lines, speaker_map=None):
    return parse_imessage_export(write(tmp_path, lines), speaker_map or SPEAKER_MAP)


# --- timestamp parsing ------------------------------------------------------ #
def test_timestamp_two_spaces_and_pm():
    assert _timestamp_or_none("May 17, 2022  5:29:42 PM") == datetime(2022, 5, 17, 17, 29, 42)


def test_timestamp_single_digit_day_and_hour_am():
    assert _timestamp_or_none("Jan 5, 2023  9:05:03 AM") == datetime(2023, 1, 5, 9, 5, 3)


def test_timestamp_tolerates_single_space():
    assert _timestamp_or_none("Dec 31, 2021 11:59:59 PM") == datetime(2021, 12, 31, 23, 59, 59)


def test_non_timestamp_lines_are_none():
    assert _timestamp_or_none("Me") is None
    assert _timestamp_or_none("The city of Eglon") is None
    # an announcement line (timestamp + trailing text) is NOT a bare timestamp
    assert _timestamp_or_none("May 17, 2022  5:35:00 PM Sam named the conversation X") is None
    # a regex-shaped but impossible/bogus date fails the real parse
    assert _timestamp_or_none("Xyz 1, 2020  1:00:00 PM") is None


# --- basic message shape + sender resolution -------------------------------- #
def test_two_messages_me_and_handle(tmp_path):
    msgs = parse(tmp_path, [
        "May 17, 2022  5:29:42 PM",
        "Me",
        "Welcome to the campaign.",
        "",
        "May 17, 2022  5:30:00 PM",
        "+16303463392",
        "The city of Eglon lies to the north.",
    ])
    assert len(msgs) == 2
    assert msgs[0].sender == "Hannah"                      # "Me" -> exporter
    assert msgs[0].content == "Welcome to the campaign."
    assert msgs[0].timestamp == datetime(2022, 5, 17, 17, 29, 42)
    assert msgs[0].source_file == "chat.txt"               # bare filename
    assert msgs[1].sender == "Sam"                         # phone -> speaker_map
    assert msgs[1].content == "The city of Eglon lies to the north."


def test_multiline_body_preserved(tmp_path):
    msgs = parse(tmp_path, [
        "May 17, 2022  5:40:00 PM",
        "Sam",
        "Line one of the tale.",
        "Line two continues it.",
    ])
    assert len(msgs) == 1
    assert msgs[0].content == "Line one of the tale.\nLine two continues it."


def test_contact_name_sender_passes_through(tmp_path):
    # A resolved contact name (not "Me", not in the map) is used verbatim.
    msgs = parse(tmp_path, ["May 17, 2022  5:41:00 PM", "Colin", "Aerin draws her bow."])
    assert msgs[0].sender == "Colin"


# --- sender helpers --------------------------------------------------------- #
def test_resolve_sender_me_is_exporter():
    assert _resolve_sender("Me", SPEAKER_MAP, "chat.txt") == "Hannah"
    assert _resolve_sender("me", SPEAKER_MAP, "chat.txt") == "Hannah"   # case-insensitive


def test_resolve_sender_phone_normalization():
    # A prettily formatted number still matches an E.164 speaker-map key.
    assert _resolve_sender("+1 (630) 346-3392", SPEAKER_MAP, "chat.txt") == "Sam"


def test_resolve_sender_unknown_handle_warns_and_falls_through(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="parsers.imessage_export_parser"):
        out = _resolve_sender("+19998887777", SPEAKER_MAP, "chat.txt")
    assert out == "+19998887777"
    assert any("Unknown handle" in r.getMessage() for r in caplog.records)


def test_normalize_phone_rejects_a_name():
    assert _normalize_phone("Colin") is None
    assert _normalize_phone("+1 (630) 346-3392") == "+16303463392"


# --- annotation stripping --------------------------------------------------- #
def test_tapbacks_block_is_stripped(tmp_path):
    msgs = parse(tmp_path, [
        "May 17, 2022  5:31:00 PM",
        "Sam",
        "Kriggy is the crown prince.",
        "Tapbacks:",
        "    Loved by Hannah",
        "    Liked by Matt",
    ])
    assert len(msgs) == 1
    assert msgs[0].content == "Kriggy is the crown prince."


def test_pure_tapback_message_is_dropped(tmp_path):
    msgs = parse(tmp_path, [
        "May 17, 2022  5:32:00 PM",
        "Matt",
        "Tapbacks:",
        "    Loved by Sam",
    ])
    assert msgs == []                                      # nothing but a tapback -> no content


def test_attachment_path_stripped_caption_kept(tmp_path):
    msgs = parse(tmp_path, [
        "May 17, 2022  5:33:00 PM",
        "Sam",
        "/Users/sam/Library/Messages/Attachments/ab/12/map.jpeg",
        "Here's the map of the region.",
    ])
    assert len(msgs) == 1
    assert msgs[0].content == "Here's the map of the region."


def test_transcription_line_stripped(tmp_path):
    msgs = parse(tmp_path, [
        "May 17, 2022  5:33:30 PM",
        "Sam",
        "/Users/sam/Attachments/audio.caf",
        "Transcription: hello there",
        "Real text after.",
    ])
    assert msgs[0].content == "Real text after."


def test_reply_and_read_receipt_annotations_dropped(tmp_path):
    msgs = parse(tmp_path, [
        "May 17, 2022  5:34:00 PM",
        "Sam",
        "The council convened at dawn.",
        "This message responded to an earlier message.",
        "(Read by Hannah after 3 minutes)",
    ])
    assert msgs[0].content == "The council convened at dawn."


def test_announcement_line_dropped(tmp_path):
    # An announcement sits between two real messages; it is neither a boundary nor lore.
    msgs = parse(tmp_path, [
        "May 17, 2022  5:34:00 PM",
        "Sam",
        "Real lore here.",
        "",
        "May 17, 2022  5:35:00 PM Matt named the conversation The Party",
        "",
        "May 17, 2022  5:36:00 PM",
        "Hannah",
        "More lore.",
    ])
    assert [m.content for m in msgs] == ["Real lore here.", "More lore."]
    assert all("named the conversation" not in m.content for m in msgs)


def test_deleted_and_unsent_markers_dropped(tmp_path):
    msgs = parse(tmp_path, [
        "May 17, 2022  5:37:00 PM",
        "Sam",
        "This message was deleted from the conversation!",
    ])
    assert msgs == []


# --- resilience ------------------------------------------------------------- #
def test_leading_junk_before_first_message_ignored(tmp_path):
    msgs = parse(tmp_path, [
        "some stray header line",
        "",
        "May 17, 2022  5:38:00 PM",
        "Sam",
        "Actual lore.",
    ])
    assert len(msgs) == 1 and msgs[0].content == "Actual lore."


def test_empty_file_returns_empty(tmp_path):
    assert parse(tmp_path, [""]) == []


# --- from a REAL imessage-exporter export (pinned the format tokens) --------- #
def test_read_receipt_on_timestamp_line_is_a_boundary():
    # The serious real-format finding: a read receipt is appended to the timestamp
    # line. The message must still be detected, and the receipt parsed away.
    assert _timestamp_or_none(
        "May 24, 2020 10:15:11 AM (Read by you after 1 minute, 6 seconds)"
    ) == datetime(2020, 5, 24, 10, 15, 11)


def test_message_with_read_receipt_header_is_parsed(tmp_path):
    msgs = parse(tmp_path, [
        "May 24, 2020 10:15:11 AM (Read by you after 1 minute, 6 seconds)",
        "Colin",
        "Aerin nocks an arrow.",
    ])
    assert len(msgs) == 1
    assert msgs[0].sender == "Colin"
    assert msgs[0].content == "Aerin nocks an arrow."      # receipt text nowhere in content
    assert "Read by you" not in msgs[0].content


def test_multiword_contact_name_sender(tmp_path):
    msgs = parse(tmp_path, ["May 24, 2020 10:19:08 AM", "Helen Corey", "Luv u so much."])
    assert msgs[0].sender == "Helen Corey"


def test_is_attachment_line_real_backup_and_mac_paths():
    # iPhone-backup source: deep RELATIVE path, spaces, no extension.
    assert _is_attachment_line(
        "Library/Application Support/MobileSync/Backup/"
        "00008110-001A02340CA0201E/8f/8f4889c3fb46a9e62ccab9654312f14282681311"
    )
    # Mac source: absolute path into the Messages attachments store.
    assert _is_attachment_line("/Users/h/Library/Messages/Attachments/ab/12/pic.jpeg")
    # Prose with a slash or two is NOT an attachment (anti-false-strip).
    assert not _is_attachment_line("Mornings r getting colder and/or wetter.")
    assert not _is_attachment_line("we did rock/paper/scissors")


def test_backup_attachment_path_stripped_caption_kept(tmp_path):
    msgs = parse(tmp_path, [
        "Sep 11, 2020  6:23:21 AM",
        "Me",
        "Library/Application Support/MobileSync/Backup/00008110-001A02340CA0201E/8f/"
        "8f4889c3fb46a9e62ccab9654312f14282681311",
        "Mornings r getting colder. Thankfully I have this blanket.",
    ])
    assert len(msgs) == 1
    assert msgs[0].content == "Mornings r getting colder. Thankfully I have this blanket."
    assert "MobileSync" not in msgs[0].content


# The exact paste the user provided, end to end. Locks the whole real format down.
REAL_SAMPLE = [
    "May 24, 2020 10:14:22 AM",
    "Me",
    "Woke up still in shock that I’m probably getting the Kia so I just wanted to "
    "say thank you thank you thank you again ❤️❤️❤️❤️",
    "",
    "May 24, 2020 10:15:11 AM (Read by you after 1 minute, 6 seconds)",
    "Palouie",
    "❤️\U0001f618\U0001f1fa\U0001f1f8",
    "",
    "May 24, 2020 10:19:08 AM (Read by you after 5 minutes, 17 seconds)",
    "Helen Corey",
    "Luv u so much\U0001f495\U0001f60d",
    "",
    "May 24, 2020 10:24:38 AM",
    "Me",
    "\U0001f60d\U0001f60dluv u 2 \U0001f618",
    "",
    "Sep 11, 2020  6:23:21 AM",
    "Me",
    "Library/Application Support/MobileSync/Backup/00008110-001A02340CA0201E/8f/"
    "8f4889c3fb46a9e62ccab9654312f14282681311",
    "Mornings r getting colder. Thankfully I have this blanket that Boopah made to stay "
    "warm (been with me for all 4 years now!) ❤️\U0001f970 Luv u guys",
    "",
    "Sep 11, 2020  6:26:21 AM (Read by you after 9 minutes, 52 seconds)",
    "Helen Corey",
    "Awwww \U0001f970 Still hot in Nash but days r getting shorter. Hope ur studies r "
    "going well. Luv & miss ur coffee visits.\U0001f60d",
    "Tapbacks:",
    "    Loved by Me",
    "",
    "Sep 11, 2020  6:36:09 AM (Read by you after 4 seconds)",
    "Palouie",
    "Hi Hannah ",
    "Glad u r staying warm. Just got back from my morning walk. I didn’t see ",
    "Mom deer and 2 fawns this morning.",
    "Still warm here. ",
    "Study well and have fun",
    "Love ❤️ xoxo \U0001f618 ",
    "Pa Lou",
    "Tapbacks:",
    "    Loved by Me",
]


def test_real_sample_end_to_end(tmp_path):
    msgs = parse(tmp_path, REAL_SAMPLE, speaker_map={"exporter": "Hannah"})
    assert len(msgs) == 7                                          # every message recovered
    assert [m.sender for m in msgs] == [
        "Hannah", "Palouie", "Helen Corey", "Hannah", "Hannah", "Helen Corey", "Palouie",
    ]
    # timestamps parsed from the core, ignoring any read-receipt suffix
    assert msgs[0].timestamp == datetime(2020, 5, 24, 10, 14, 22)
    assert msgs[1].timestamp == datetime(2020, 5, 24, 10, 15, 11)   # this one had a receipt
    assert msgs[4].timestamp == datetime(2020, 9, 11, 6, 23, 21)    # single-digit hour, 2 spaces
    # no read-receipt text or attachment path leaked into any body
    assert all("Read by you" not in m.content for m in msgs)
    assert all("MobileSync" not in m.content for m in msgs)
    assert all("Tapbacks:" not in m.content for m in msgs)
    assert all("Loved by Me" not in m.content for m in msgs)
    # the attachment message keeps only its caption
    assert msgs[4].content.startswith("Mornings r getting colder.")
    # the multi-line message is preserved intact (first + last body line survive)
    assert msgs[6].content.startswith("Hi Hannah")
    assert msgs[6].content.rstrip().endswith("Pa Lou")
