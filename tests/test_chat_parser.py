"""Tests for parsers/chat_parser.py (Phase 2.1: boundary detection + sender
attribution).

Covers the five cases the project breakdown asks for (single exporter message,
single phone-prefixed message, multi-line message, a reaction left intact at this
stage, two messages back to back) plus the export quirks catalogued in
FORMAT_NOTES.md and baked into the parser during the 2.1 session: the ``\\r\\r\\n``
line-ending trap, footer-not-header sender attribution, the opening lone
timestamp + carry-forward, 1-on-1 alignment, truncated-final-message recovery,
and the negative/error paths.

Every log is written with real ``\\r\\r\\n`` line endings as raw UTF-8 bytes, so
the tests exercise read_clean's carriage-return stripping the same way the real
exports do. tmp_path is pytest's auto-cleaning temp dir, so no real log is ever
touched.
"""

import logging
from datetime import datetime

import pytest

from parsers.chat_parser import parse_chat_log, read_clean


# A realistic header: the second line of every export is exactly 100 dashes.
DASHES = "-" * 100

# Alice/Bob are phone-prefixed participants; Hannah is the exporter (bare footer).
SPEAKER_MAP = {
    "+15555550101": "Alice",
    "+15555550102": "Bob",
    "exporter": "Hannah",
}


def write_log(tmp_path, lines, name="dndgroup.txt"):
    """Write ``lines`` to a temp log using real ``\\r\\r\\n`` endings.

    Raw bytes (not write_text) so universal-newline handling can't pre-clean the
    carriage returns -- we want read_clean to do that, exactly as in production.
    UTF-8 because names/quotes/emoji are non-ASCII.
    """
    path = tmp_path / name
    path.write_bytes("\r\r\n".join(lines).encode("utf-8"))
    return str(path)


# --- read_clean: the \r\r\n trap (FORMAT_NOTES, "read this first" #1) ---------

def test_read_clean_strips_all_carriage_returns(tmp_path):
    # \r\r\n must collapse to one clean break. Plain text-mode reading would turn
    # each \r\r\n into TWO \n and yield phantom blanks: ["a", "", "b", "", ""].
    path = tmp_path / "x.txt"
    path.write_bytes(b"a\r\r\nb\r\r\n")
    assert read_clean(str(path)) == ["a", "b", ""]


def test_carriage_returns_do_not_create_phantom_blank_lines(tmp_path):
    # A two-line body must come back as "line one\nline two", NOT with a phantom
    # blank line wedged between them.
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "line one",
        "line two",
        "+15555550101 01/02/2024 10:05:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert msgs[0].content == "line one\nline two"


# --- the five breakdown cases ------------------------------------------------

def test_single_exporter_message(tmp_path):
    # Exporter = a bare footer (no phone). Maps through the "exporter" key.
    log = write_log(tmp_path, [
        ", +15555550101, +15555550102", DASHES,
        "01/02/2024 10:00:00",
        "hello from the exporter",
        "01/02/2024 10:05:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert len(msgs) == 1
    assert msgs[0].sender == "Hannah"
    assert msgs[0].content == "hello from the exporter"
    assert msgs[0].source_file == "dndgroup.txt"  # Path(filepath).name


def test_single_phone_prefixed_message(tmp_path):
    log = write_log(tmp_path, [
        ", +15555550101, +15555550102", DASHES,
        "01/02/2024 10:00:00",
        "hey what's the lore on Lake Mundi",
        "+15555550101 01/02/2024 10:05:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert len(msgs) == 1
    assert msgs[0].sender == "Alice"
    assert msgs[0].content == "hey what's the lore on Lake Mundi"


def test_multiline_message(tmp_path):
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "line one of the message",
        "line two of the message",
        "line three",
        "+15555550101 01/02/2024 10:05:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert len(msgs) == 1
    assert msgs[0].content == "line one of the message\nline two of the message\nline three"


def test_two_messages_back_to_back(tmp_path):
    # Boundary detection: each footer closes exactly one message.
    log = write_log(tmp_path, [
        ", +15555550101, +15555550102", DASHES,
        "01/02/2024 10:00:00",
        "first message",
        "+15555550101 01/02/2024 10:05:00",
        "second message",
        "+15555550102 01/02/2024 10:06:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert len(msgs) == 2
    assert (msgs[0].sender, msgs[0].content) == ("Alice", "first message")
    assert (msgs[1].sender, msgs[1].content) == ("Bob", "second message")


def test_parser_leaves_reactions_in_content(tmp_path):
    # Phase 2.1 scope is boundary + sender ONLY; reaction noise stays in content
    # for the Phase 2.2 reaction_filter to strip later.
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "Loved “real message”",
        "+15555550101 01/02/2024 10:05:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert msgs[0].content == "Loved “real message”"


# --- sender attribution: footers, not headers (FORMAT_NOTES Pattern 2) -------

def test_footer_phone_names_the_message_above_it(tmp_path):
    # If this were read as a header naming the message BELOW, A would be Bob's.
    log = write_log(tmp_path, [
        ", +15555550101, +15555550102", DASHES,
        "01/02/2024 10:00:00",
        "message body A",
        "+15555550101 01/02/2024 10:05:00",  # +101 (Alice) names A, the block above
        "message body B",
        "+15555550102 01/02/2024 10:06:00",  # +102 (Bob) names B, the block above
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert (msgs[0].sender, msgs[0].content) == ("Alice", "message body A")
    assert (msgs[1].sender, msgs[1].content) == ("Bob", "message body B")


def test_unknown_phone_falls_back_to_raw_number_with_warning(tmp_path, caplog):
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "msg from a stranger",
        "+15559999999 01/02/2024 10:05:00",  # not in the speaker map
    ])
    with caplog.at_level(logging.WARNING):
        msgs = parse_chat_log(log, SPEAKER_MAP)
    assert msgs[0].sender == "+15559999999"  # raw number, not dropped
    assert "+15559999999" in caplog.text


# --- 1-on-1 mode: no phones, attribute by body alignment (Pattern 3) ---------

def test_oneonone_attributes_sender_by_alignment(tmp_path):
    # Single phone in the header (no leading comma) => 1-on-1 mode. Footers are
    # all bare, so the sender comes from indentation: left (col 0) = the other
    # person, right (indented past the threshold) = the exporter.
    indent = " " * 30  # well past EXPORTER_ALIGN_THRESHOLD (20)
    log = write_log(tmp_path, [
        "+15555550101", DASHES,
        "01/02/2024 10:00:00",
        "I'm not quite that meticulous",                 # left-aligned -> other
        indent + "01/02/2024 10:05:00",
        indent + "Do any cultures correspond to real ones",  # right-aligned -> exporter
        indent + "01/02/2024 10:06:00",
    ], name="dm_convo.txt")
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert (msgs[0].sender, msgs[0].content) == ("Alice", "I'm not quite that meticulous")
    # Right-aligned -> exporter; the alignment whitespace is stripped from content.
    assert (msgs[1].sender, msgs[1].content) == ("Hannah", "Do any cultures correspond to real ones")


# --- timestamp semantics: opening lone ts seeds, footer ts carries forward ----

def test_timestamps_carry_forward_from_footers(tmp_path):
    # The opening lone timestamp belongs to msg 1; a footer's timestamp belongs
    # to the NEXT message (the footer's PHONE names the message above it, but its
    # TIMESTAMP pairs with the one below).
    log = write_log(tmp_path, [
        ", +15555550101, +15555550102", DASHES,
        "01/02/2024 10:00:00",
        "first",
        "+15555550101 01/02/2024 10:05:00",
        "second",
        "+15555550102 01/02/2024 10:06:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert msgs[0].timestamp == datetime(2024, 1, 2, 10, 0, 0)   # opening lone ts
    assert msgs[1].timestamp == datetime(2024, 1, 2, 10, 5, 0)   # carried from msg1's footer


def test_message_without_leading_timestamp_borrows_footer_timestamp(tmp_path, caplog):
    # Not seen in the real exports, but the parser must not crash: with no lone
    # timestamp seeding it, the message borrows its own footer's timestamp and
    # logs a warning.
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "body with no leading timestamp",
        "+15555550101 01/02/2024 10:05:00",
    ])
    with caplog.at_level(logging.WARNING):
        msgs = parse_chat_log(log, SPEAKER_MAP)
    assert msgs[0].sender == "Alice"
    assert msgs[0].timestamp == datetime(2024, 1, 2, 10, 5, 0)
    assert "No leading timestamp" in caplog.text


# --- multi-line bodies: blank lines are paragraph breaks, not boundaries ------

def test_internal_blank_lines_preserved_as_paragraph_breaks(tmp_path):
    # Pattern 6: never split a message on a blank line; only footers end messages.
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "First paragraph here",
        "",
        "Second paragraph after a blank line",
        "+15555550101 01/02/2024 10:05:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert len(msgs) == 1
    assert msgs[0].content == "First paragraph here\n\nSecond paragraph after a blank line"


def test_single_character_message_is_not_dropped(tmp_path):
    # Pattern 11: a one-char body (e.g. "?") is a legitimate message.
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "?",
        "+15555550101 01/02/2024 10:05:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert len(msgs) == 1
    assert msgs[0].content == "?"


# --- truncated final message recovery (real exports end mid-message) ----------

def test_truncated_final_message_with_dangling_phone_recovered(tmp_path):
    # The real exports stop mid-message with the last footer cut off right after
    # the phone. That dangling phone still names the sender; the message is
    # recovered, not dropped.
    log = write_log(tmp_path, [
        ", +15555550101, +15555550102", DASHES,
        "01/02/2024 10:00:00",
        "first message body",
        "+15555550101 01/02/2024 10:05:00",
        "second truncated message",
        "+15555550102",  # footer cut off after the phone (no timestamp)
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert len(msgs) == 2
    assert msgs[1].sender == "Bob"
    assert msgs[1].content == "second truncated message"
    assert msgs[1].timestamp == datetime(2024, 1, 2, 10, 5, 0)  # carried-forward ts


def test_truncated_final_message_without_phone_is_exporter(tmp_path):
    # No dangling phone on the final block => it was the exporter's, same rule as
    # a bare footer inside the loop.
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "real one",
        "+15555550101 01/02/2024 10:05:00",
        "final exporter line with no footer at all",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert len(msgs) == 2
    assert msgs[1].sender == "Hannah"
    assert msgs[1].content == "final exporter line with no footer at all"


# --- photo / URL noise is left for later layers ------------------------------

def test_parser_keeps_photo_token_and_urls_in_content(tmp_path):
    # Phase 2.1 does not strip [photo] (that's 2.2) and never strips URLs (that's
    # deferred all the way to the Phase 3 LLM noise filter).
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "[photo]",
        "https://dnd5e.wikidot.com/feat:spear-mastery",
        "01/02/2024 10:06:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert msgs[0].content == "[photo]\nhttps://dnd5e.wikidot.com/feat:spear-mastery"


# --- unicode ------------------------------------------------------------------

def test_utf8_accents_and_emoji_preserved(tmp_path):
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "01/02/2024 10:00:00",
        "Café señor — déjà vu \U0001f602\U0001f97a",
        "+15555550101 01/02/2024 10:05:00",
    ])
    msgs = parse_chat_log(log, SPEAKER_MAP)
    assert msgs[0].content == "Café señor — déjà vu \U0001f602\U0001f97a"


# --- negative / error paths --------------------------------------------------

def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_chat_log(str(tmp_path / "does_not_exist.txt"), SPEAKER_MAP)


def test_impossible_timestamp_raises_value_error(tmp_path):
    # The parser deliberately does not validate timestamps: a footer that matches
    # the regex shape but is an impossible date (month 13) lets strptime's
    # ValueError bubble up rather than silently mangling the data.
    log = write_log(tmp_path, [
        ", +15555550101", DASHES,
        "13/45/2024 10:00:00",  # impossible date, used as the opening lone timestamp
        "some body",
        "+15555550101 01/02/2024 10:05:00",
    ])
    with pytest.raises(ValueError):
        parse_chat_log(log, SPEAKER_MAP)


def test_fewer_than_two_lines_returns_empty_with_warning(tmp_path, caplog):
    log = write_log(tmp_path, ["only one line"])
    with caplog.at_level(logging.WARNING):
        result = parse_chat_log(log, SPEAKER_MAP)
    assert result == []
    assert "fewer than 2 lines" in caplog.text


def test_empty_file_returns_empty(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    assert parse_chat_log(str(path), SPEAKER_MAP) == []
