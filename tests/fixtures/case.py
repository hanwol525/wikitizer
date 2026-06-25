"""Shared ``Case`` dataclass for the Phase 3.10 extractor integration fixtures.

Committed and intentionally chat-free: the real bodies live in the gitignored
``real_messages.py`` and the fabricated ones in the committed
``synthetic_messages.py``. Both import ``Case`` (and the ``build_message``
helper) from here so the shape is defined in exactly one place.
"""

from dataclasses import dataclass, field
from datetime import datetime

from models.message import Message

# A single fixed, valid placeholder. The extractors never read the timestamp, so
# its value is irrelevant to these tests -- it just has to be a real ``datetime``.
PLACEHOLDER_TIMESTAMP = datetime(2024, 1, 1, 12, 0, 0)


def build_message(sender: str, content: str, source_file: str) -> Message:
    """Build a fixture ``Message`` with the shared placeholder timestamp."""
    return Message(
        sender=sender,
        timestamp=PLACEHOLDER_TIMESTAMP,
        content=content,
        source_file=source_file,
    )


@dataclass
class Case:
    """One hand-picked message plus what to check against each extractor.

    ``expect`` is keyed by extractor short-name -- one of ``"locations"``,
    ``"characters"``, ``"history"``, ``"organizations"``, ``"items"``,
    ``"people"`` -- and each value is a dict of loose, LLM-tolerant checks:

      * ``"expect": ["X", ...]``   -- every X must match some entity (name/alias,
        case-insensitive substring).
      * ``"expect_any": ["X", ...]`` -- at least ONE of the Xs must match some
        entity. (Used where the chat could name a thing several ways, e.g.
        "marsh tribes" vs "Dagger Swamp tribes".)
      * ``"reject": ["X", ...]``   -- no entity may match any X (the "wrong
        bucket" boundary guard).
      * ``"scope": "world"|"regional"|"personal"`` -- History only; some returned
        event must carry that scope.
      * ``"empty": True``          -- the extractor must return [].
      * ``"min_count": N``         -- at least N entities returned.

    An empty dict (``{}``) for an extractor means *eyeball-only*: run it and write
    the report, but assert nothing.
    """

    id: str
    message: Message
    expect: dict = field(default_factory=dict)
