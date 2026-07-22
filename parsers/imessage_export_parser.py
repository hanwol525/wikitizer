"""Ingestion path for ReagentX **imessage-exporter** TXT exports.

The legacy `chat_parser` handles the messy copy-pasted iMessage `.txt` (footer
regexes + left/right alignment guessing). This module handles the *structured*
TXT that the `imessage-exporter` CLI (`-f txt`) writes, where every message is::

    May 17, 2022  5:29:42 PM        <- timestamp line (the message boundary)
    Me                              <- sender line ("Me" = the exporter, else a handle/name)
    the message body                <- one or more body lines
    Tapbacks:                       <- optional; each reaction indented 4 spaces
        Loved by Sam
                                    <- blank line separates messages

The big win over the legacy parser is a DETERMINISTIC per-message sender (its own
line), so there is no alignment heuristic. Reactions, attachments, replies, read
receipts, and group announcements are all STRUCTURALLY annotated, so we strip them
here rather than pattern-matching curly-quote reaction text downstream.

Output is the same `list[Message]` seam every parser produces, so the orchestrator
loop is unchanged (it routes here via `parsers/ingest.py`).

The format is CONFIRMED against a real export. Two findings the templates alone didn't
show: a **read receipt is appended to the timestamp line** in parentheses
("... 10:15:11 AM (Read by you after 1 minute, 6 seconds)"), so the boundary regex
allows a trailing `(...)` and parses only the timestamp core; and an **attachment path
from an iPhone-backup source is a deep RELATIVE path with spaces and no extension**
("Library/Application Support/MobileSync/Backup/.../<hash>"), handled by
`_is_attachment_line`. Senders arrive as resolved contact names ("Helen Corey"), which
`_resolve_sender` passes through unchanged.
"""

import re
import logging
from datetime import datetime
from pathlib import Path

from models.message import Message
from parsers.chat_parser import read_clean, _clean_body

logger = logging.getLogger(__name__)

# A message boundary: the timestamp line, e.g. "May 17, 2022  5:29:42 PM".
# imessage-exporter prints it with a NON-abbreviated day/hour (no leading zero), so a
# single-digit hour gets an extra space ("...2020  6:23:21 AM") -- we tolerate any run
# of whitespace and confirm with a real parse in _timestamp_or_none. CRUCIALLY, a
# read receipt is appended to this SAME line in parentheses:
#   "May 24, 2020 10:15:11 AM (Read by you after 1 minute, 6 seconds)"
# so the line is the timestamp CORE plus zero or more trailing "(...)" parentheticals.
# _timestamp_or_none captures the core (group 1) and parses just that. An announcement
# line ("<timestamp> <who> named the conversation ...") has trailing text that ISN'T a
# parenthetical, so it does not match -- it's dropped separately during stripping.
_TIMESTAMP_CORE = r"[A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} (?:AM|PM)"
_TIMESTAMP_LINE_RE = re.compile(r"^(" + _TIMESTAMP_CORE + r")(?:\s+\([^)]*\))*$")
_TIMESTAMP_FORMAT = "%b %d, %Y %I:%M:%S %p"  # whitespace collapsed to single spaces first

# An announcement line = a timestamp with trailing NON-parenthetical text (group
# renamed / joined / left, "unsent a message!"). The [^(\s] guard keeps it disjoint
# from a timestamp+read-receipt line, which is a real message boundary, not an announcement.
_ANNOUNCEMENT_RE = re.compile(r"^" + _TIMESTAMP_CORE + r"\s+[^(\s]")

# A standalone read-receipt line (kept defensively; real exports put the receipt on
# the timestamp line, handled above).
_READ_RECEIPT_RE = re.compile(r"^\(Read by .+\)$")

# Attachment path line. CONFIRMED against a real export: imessage-exporter writes the
# raw attachment path, which for an iPhone-backup source is a DEEP RELATIVE path with
# spaces and NO extension, e.g.
#   Library/Application Support/MobileSync/Backup/<UDID>/8f/8f4889c3fb46a9e62ccab...
# and for a Mac source is an absolute "/Users/.../Library/Messages/Attachments/..." path.
# We detect: an absolute/home/relative path prefix, a known attachment-location marker,
# or a deep path (4+ segments) whose final segment is a long hash or a name.ext -- so a
# stray prose line with a slash or two is NOT mistaken for an attachment.
_ATTACHMENT_MARKERS = ("MobileSync/Backup/", "/Attachments/", "Messages/Attachments/")
_DEEP_PATH_RE = re.compile(
    r"^[\w .~+-]+(?:/[\w .~+-]+){2,}/"
    r"(?:[A-Za-z0-9][\w-]{15,}|[\w .~+-]+\.[A-Za-z0-9]{1,5})$"
)

# Metadata lines that are never lore.
_REPLY_ANNOTATION = "This message responded to an earlier message."
_DELETED_ANNOTATION = "This message was deleted from the conversation!"
_UNSENT_RE = re.compile(r"\bunsent this message part\b")

# imessage-exporter's default self-label for the exporter's own messages. (A run
# with `--custom-name` overrides it; document that, or add the custom name to the
# speaker map, since we only special-case "Me" here.)
_SELF_LABEL = "me"


def _timestamp_or_none(line: str):
    """Return the parsed `datetime` if `line` is a message timestamp line (optionally
    carrying a trailing `(Read by ...)` read receipt), else None. Parses only the
    captured timestamp CORE, so the receipt never reaches strptime. Also rejects an
    almost-timestamp (a regex hit that isn't a real date, e.g. 'Xyz 1, 2020 ...')."""
    m = _TIMESTAMP_LINE_RE.match(line.strip())
    if not m:
        return None
    try:
        return datetime.strptime(re.sub(r"\s+", " ", m.group(1)), _TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _is_plausible_sender(line: str) -> bool:
    """The line after a timestamp is the sender. Guard the boundary handshake: a
    real sender line is non-blank and is NOT itself a timestamp."""
    return bool(line.strip()) and _timestamp_or_none(line) is None


def _normalize_phone(handle: str):
    """Reduce a handle to a bare `+digits` key so a prettily-formatted number
    ('+1 (630) 346-3392') still matches an E.164 speaker-map key ('+16303463392').
    Returns None when the handle isn't phone-ish (a contact name has no digits)."""
    digits = re.sub(r"[^\d+]", "", handle)
    return digits if len(re.sub(r"\D", "", digits)) >= 7 else None


def _resolve_sender(sender_raw: str, speaker_map: dict, source_name: str) -> str:
    """Map an imessage-exporter sender line to a display name. 'Me' is the exporter
    (reuse the reserved 'exporter' speaker-map key). Otherwise it's a handle or an
    already-resolved contact name: try a direct hit, then a phone-normalized hit,
    then fall through to the raw value (a real name, or an unmapped handle + a warning)."""
    s = sender_raw.strip()
    if s.lower() == _SELF_LABEL:
        return speaker_map.get("exporter", "exporter")
    if s in speaker_map:
        return speaker_map[s]
    norm = _normalize_phone(s)
    if norm is not None:
        if norm in speaker_map:
            return speaker_map[norm]
        logger.warning("Unknown handle %r in %s; using it as the sender", s, source_name)
    return s


def _is_attachment_line(stripped: str) -> bool:
    """True if a line is an imessage-exporter attachment path (to strip). Confirmed
    shapes: an iPhone-backup source writes a deep RELATIVE path
    ('Library/Application Support/MobileSync/Backup/.../<hash>', no extension); a Mac
    source writes an absolute '/Users/.../Library/Messages/Attachments/...' path.
    We match an explicit path prefix, a known attachment-location marker, or a deep
    (4+ segment) path ending in a long hash or a name.ext -- so a stray prose line
    with one or two slashes is NOT mistaken for a path (anti-false-strip)."""
    if stripped.startswith(("/", "~/", "./", "Attachments/")):
        return True
    if any(marker in stripped for marker in _ATTACHMENT_MARKERS):
        return True
    return bool(_DEEP_PATH_RE.match(stripped))


def _strip_annotations(body_lines: list) -> list:
    """Drop imessage-exporter's structural annotations from a message's body lines,
    keeping only real content. Removes: the indented `Tapbacks:` block, attachment
    path lines (+ their `Transcription:` follow-on), reply/read-receipt/announcement
    lines, and deleted/unsent markers. Blank lines are kept (paragraph breaks);
    `_clean_body` trims the edges afterward."""
    kept: list = []
    in_tapbacks = False
    for raw in body_lines:
        stripped = raw.strip()
        if in_tapbacks:
            # Tapback entries are 4-space indented; the block ends at the first
            # non-indented, non-blank line (which we then process normally).
            if not stripped or raw.startswith("    "):
                continue
            in_tapbacks = False
        if stripped == "Tapbacks:":
            in_tapbacks = True
            continue
        if not stripped:
            kept.append(raw)
            continue
        if (_ANNOUNCEMENT_RE.match(stripped)
                or stripped == _REPLY_ANNOTATION
                or stripped == _DELETED_ANNOTATION
                or _READ_RECEIPT_RE.match(stripped)
                or _UNSENT_RE.search(stripped)
                or _is_attachment_line(stripped)
                or stripped.startswith("Transcription:")):
            continue
        kept.append(raw)
    return kept


def parse_imessage_export(filepath: str, speaker_map: dict) -> list:
    """Parse one imessage-exporter TXT export into a `list[Message]`.

    Boundary detection is a two-line handshake: a timestamp line whose next line is
    a plausible sender opens a message; its body runs until the next such boundary.
    This ignores announcement lines (timestamp + trailing text) as boundaries; they
    are dropped as annotations. A message whose body is nothing but stripped
    annotations (a pure tapback/attachment) yields no content and is dropped.
    """
    lines = read_clean(filepath)
    source_name = Path(filepath).name

    messages: list = []
    n = len(lines)
    i = 0
    while i < n:
        ts = _timestamp_or_none(lines[i])
        if ts is not None and i + 1 < n and _is_plausible_sender(lines[i + 1]):
            sender_raw = lines[i + 1].strip()
            j = i + 2
            body: list = []
            while j < n:
                if (_timestamp_or_none(lines[j]) is not None
                        and j + 1 < n and _is_plausible_sender(lines[j + 1])):
                    break
                body.append(lines[j])
                j += 1
            content = _clean_body(_strip_annotations(body))
            if content:
                sender = _resolve_sender(sender_raw, speaker_map, source_name)
                messages.append(Message(sender=sender, timestamp=ts,
                                        content=content, source_file=source_name))
            i = j
        else:
            i += 1

    if not messages:
        logger.warning("%s parsed to zero messages (not an imessage-exporter TXT export?)",
                       source_name)
    return messages
