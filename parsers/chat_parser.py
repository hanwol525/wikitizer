"""Phase 2.1: message boundary detection + sender attribution.

Turns one raw exported chat log into a ``list[Message]``. This layer is pure
Python (no LLM) and deliberately does NOT yet filter reactions, ``[photo]``
lines, or URLs -- that's Phase 2.2, so all of it stays in the message content
for now.

See ``FORMAT_NOTES.md`` for the full catalogue of export quirks. The
load-bearing facts this module relies on:

  * Lines end with ``\\r\\r\\n``; we must read raw bytes and strip the ``\\r``
    ourselves, or universal-newline handling injects a phantom blank line after
    every real line (see :func:`read_clean`).
  * The ``+phone  timestamp`` / bare ``timestamp`` line is a FOOTER, not a
    header: the phone names the message *above* it, and a footer with no phone
    was sent by the exporter.
  * The 1-on-1 file has no phone numbers anywhere, so sender attribution there
    falls back to left/right body alignment (right = exporter, left = the other
    person).
"""

import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.message import Message

logger = logging.getLogger(__name__)

# Footers, not headers. Check PHONE_FOOTER first: BARE_FOOTER would otherwise
# also match the timestamp half of a phone footer. Anchor with ``\s*$`` (not
# ``$``) so trailing whitespace / stray carriage returns don't defeat the match.
# Capture groups: PHONE_FOOTER -> (phone, timestamp); BARE_FOOTER -> (timestamp,).
PHONE_FOOTER = re.compile(r"^(\+\d+)\s+(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s*$")
BARE_FOOTER = re.compile(r"^\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s*$")

# A phone number alone on a line, with no timestamp after it. This is what a
# footer looks like when the export was truncated mid-footer -- and the real
# logs ALL end this way, with the very last message's footer cut off right after
# the phone. We only consult this when recovering that trailing message (see the
# end of parse_chat_log); a normal, complete phone footer always carries its
# timestamp and is matched by PHONE_FOOTER inside the loop.
DANGLING_PHONE = re.compile(r"^(\+\d+)\s*$")

TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

# 1-on-1 mode only: a body indented at least this many columns is the exporter
# (right-aligned); column 0 is the other person. This is the single genuinely
# heuristic knob in the parser. The real dm file shows exporter lines starting
# around column 30-47 and the other person at column 0, so 20 splits them
# cleanly with room to spare. Nudge it if a 1-on-1 line lands on the wrong name.
EXPORTER_ALIGN_THRESHOLD = 20


def read_clean(path: str) -> list[str]:
    """Read a log as UTF-8 raw bytes and strip ALL carriage returns.

    Never use a text-mode read here: universal-newline handling turns each
    ``\\r\\r\\n`` into two ``\\n``, giving a phantom blank line after every real
    line. UTF-8 matters too (accented names, curly quotes, emoji).
    """
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8")
    return raw.replace("\r", "").split("\n")


def _footer_timestamp(m_phone, m_bare) -> str:
    """Pull the raw timestamp string out of whichever footer matched."""
    return m_phone.group(2) if m_phone else m_bare.group(1)


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _resolve_group_sender(m_phone, m_bare, speaker_map: dict, source_name: str) -> str:
    """Group mode: the footer phone is the exact sender; a bare footer is the
    exporter. An unknown phone falls back to the raw number (and a warning)."""
    if m_phone:
        phone = m_phone.group(1)
        name = speaker_map.get(phone)
        if name is not None:
            return name
        logger.warning("Unknown phone number %s in %s; using the raw number as the sender", phone, source_name)
        return phone
    return speaker_map.get("exporter", "exporter")


def _resolve_oneonone_sender(buffer: list, other_number: Optional[str], speaker_map: dict) -> str:
    """1-on-1 mode: no phones exist, so attribute by body alignment.

    Right-aligned (indented past the threshold) = exporter; left-aligned
    (column 0) = the other person. Measured on the first non-blank body line and
    BEFORE the body is stripped, so the alignment signal is still intact.
    """
    lead = 0
    for line in buffer:
        if line.strip():
            lead = _leading_spaces(line)
            break
    if lead >= EXPORTER_ALIGN_THRESHOLD:
        return speaker_map.get("exporter", "exporter")
    return speaker_map.get(other_number, other_number)


def _clean_body(buffer: list) -> str:
    """Strip each line (leading alignment + trailing space are both noise once
    we've used them), keep internal blank lines as paragraph breaks, then trim
    leading/trailing blank lines off the whole message."""
    return "\n".join(line.strip() for line in buffer).strip()


def _build_message(sender: str, ts_str: str, content: str, source_name: str) -> Message:
    """Assemble a Message, parsing the raw footer timestamp string."""
    return Message(
        sender=sender,
        timestamp=datetime.strptime(ts_str, TIMESTAMP_FORMAT),
        content=content,
        source_file=source_name,
    )


def _recover_group_final_sender(buffer: list, speaker_map: dict, source_name: str):
    """Attribute the truncated final message of a GROUP file.

    The real exports end mid-message with the last footer cut off. If the buffer
    ends with a phone-only line (a footer that lost its timestamp), that phone
    names the sender and is peeled off the body. Otherwise the final message had
    no phone footer at all, which -- exactly as in the main loop -- means the
    exporter sent it. Returns ``(sender, body_lines)``.
    """
    for i in range(len(buffer) - 1, -1, -1):
        if not buffer[i].strip():
            continue  # skip trailing blank lines to find the real last line
        m = DANGLING_PHONE.match(buffer[i])
        if m:
            phone = m.group(1)
            name = speaker_map.get(phone)
            if name is None:
                logger.warning("Unknown phone number %s in %s; using the raw number as the sender", phone, source_name)
                name = phone
            return name, buffer[:i]
        break  # last real line is body, not a dangling phone => no phone footer
    return speaker_map.get("exporter", "exporter"), buffer


def parse_chat_log(filepath: str, speaker_map: dict) -> list[Message]:
    """Parse one exported chat log into a ``list[Message]``.

    Phase 2.1 scope is boundary detection + sender attribution only; reaction /
    ``[photo]`` / URL noise is intentionally left IN the content for Phase 2.2.
    """
    lines = read_clean(filepath)
    source_name = Path(filepath).name
    if len(lines) < 2:
        logger.warning("%s has fewer than 2 lines; no header block to skip", source_name)
        return []

    # Step 5 -- detect the file's mode from line 1. A group chat leads with a
    # comma (the exporter's own empty slot, then the other numbers); a 1-on-1
    # file is a single bare +phone with no leading comma.
    header = lines[0]
    if header.lstrip().startswith(","):
        is_group = True
        other_number: Optional[str] = None
    else:
        is_group = False
        other_number = header.strip()

    messages: list[Message] = []
    buffer: list[str] = []
    pending_ts: Optional[str] = None  # timestamp seeding the message currently being built

    for line in lines[2:]:  # skip the participant line + the row of 100 dashes
        m_phone = PHONE_FOOTER.match(line)
        m_bare = BARE_FOOTER.match(line) if not m_phone else None

        if m_phone or m_bare:
            ts = _footer_timestamp(m_phone, m_bare)
            content = _clean_body(buffer)
            if not content:
                # Empty / all-blank buffer => this footer is the file's opening
                # lone timestamp (or a stray). Seed the next message's timestamp
                # and emit nothing.
                pending_ts = ts
                buffer = []
                continue

            # Close the message the buffer holds. The footer's PHONE names *this*
            # message; its TIMESTAMP belongs to the NEXT one, so carry it forward.
            if is_group:
                sender = _resolve_group_sender(m_phone, m_bare, speaker_map, source_name)
            else:
                sender = _resolve_oneonone_sender(buffer, other_number, speaker_map)

            if pending_ts is None:
                # No opening lone timestamp seeded this message (not seen in the
                # real exports, but don't crash on it) -- borrow this footer's
                # own timestamp so we still build a valid Message.
                logger.warning("No leading timestamp for a message in %s; using its footer's timestamp", source_name)
                pending_ts = ts

            messages.append(_build_message(sender, pending_ts, content, source_name))
            pending_ts = ts
            buffer = []
        else:
            buffer.append(line)

    # A well-formed message ends with a footer, but every real export ends
    # mid-message with that final footer truncated -- so the buffer almost always
    # still holds one last, fully recoverable message here. Flush it rather than
    # lose it: its timestamp is the carried-forward pending_ts, and a truncated
    # phone-only line (group mode) or the body alignment (1-on-1) still names the
    # sender. Only warn-and-drop when it's genuinely unrecoverable -- no timestamp
    # was ever seeded, or the leftover is whitespace-only.
    leftover = _clean_body(buffer)
    if leftover and pending_ts is not None:
        if is_group:
            sender, body_lines = _recover_group_final_sender(buffer, speaker_map, source_name)
        else:
            sender, body_lines = _resolve_oneonone_sender(buffer, other_number, speaker_map), buffer
        content = _clean_body(body_lines)
        if content:
            messages.append(_build_message(sender, pending_ts, content, source_name))
    elif leftover:
        logger.warning("Dropping %d trailing line(s) with no closing footer and no seeded timestamp in %s", len(buffer), source_name)

    return messages
