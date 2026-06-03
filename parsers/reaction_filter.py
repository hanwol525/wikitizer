"""Phase 2.2: strip export-artifact noise (reactions + ``[photo]`` tokens).

This layer runs *after* :func:`parsers.chat_parser.parse_chat_log` and *before*
any LLM. It is pure Python -- it does NOT parse and does NOT call an LLM; both
of those belong to other layers.

Three facts drive the design (see ``FORMAT_NOTES.md``, Patterns 5/7/8/10):

  * ``chat_parser`` has already merged every multi-line message into a single
    ``Message.content`` -- including a reaction whose closing curly quote got
    stranded on its own line (Pattern 7). So each message's content is checked
    as **one string**; we do NOT scan line by line hunting for boundaries.
  * Reaction quotes in the export are **curly** ``“`` (U+201C), not the
    straight ``"``. A straight quote here would match zero reactions and let
    every one slip through as fake content -- a silent failure (Pattern 10).
  * The function is **pure**: it returns a new list and never mutates the input
    list or the ``Message`` objects in it.

The reaction vocabulary below was verified exhaustively against all three real
logs (``dndgroup.txt``, ``royalty.txt``, ``dm convo.txt``): every tapback verb
that actually occurs is covered, in both English and the Spanish forms the
exporter's locale leaked in.
"""

import re

from models.message import Message

# Quoted-text reactions: a fixed opener + the (possibly wrapped) quoted message.
# Match by prefix -- the parser already gathered the wrap into one content
# string, so the opener sits at the very start of ``.content`` (Pattern 7).
# ``“`` is the opening curly quote.
_QUOTED_REACTION_OPENERS = (
    "Loved “", "Liked “", "Laughed at “",
    "Emphasized “", "Disliked “", "Removed a laugh from “",
    "Le encantó “", "Le dio risa “", "Exclamó por “",
)

# Image reactions: no quoted text, so match the whole stripped line exactly.
_IMAGE_REACTIONS = (
    "Loved an image", "Liked an image", "Laughed at an image",
    "Emphasized an image", "Disliked an image",
    "Le encantó una imagen",
)

# Emoji reactions: the emoji in the middle changes every time, so a fixed
# string can't catch them -- needs a regex. ``“`` is the curly quote again.
_EMOJI_REACTION = re.compile(
    r"^Reacted .+? to “"                 # Reacted <emoji> to “...”
    r"|^Reacted .+? to an image$"        # Reacted <emoji> to an image
    r"|^Reaccionó con .+? a “"           # Reaccionó con <emoji> a “...”
    r"|^Reaccionó con .+? a una imagen$" # Reaccionó con <emoji> a una imagen
)


def _is_reaction(content: str) -> bool:
    """True if the whole message is a tapback/reaction pseudo-message.

    Checking only the front works *because* the parser already merged the
    wrapped quote + stranded closing-quote line into one ``.content`` -- the
    opener is right at the start.
    """
    text = content.strip()
    if text in _IMAGE_REACTIONS:
        return True
    if any(text.startswith(opener) for opener in _QUOTED_REACTION_OPENERS):
        return True
    if _EMOJI_REACTION.match(text):
        return True
    return False


def _strip_photo_token(content: str) -> str:
    """Drop any line that is exactly ``[photo]``; keep everything else.

    A caption under a photo is sometimes real lore (Pattern 8), so don't
    blanket-drop the message -- just remove the token. Bare ``[photo]`` ->
    returns ``""`` (caller drops it); ``[photo]`` + caption -> returns just the
    caption.
    """
    kept_lines = [ln for ln in content.split("\n") if ln.strip() != "[photo]"]
    return "\n".join(kept_lines).strip()


def filter_reactions(messages: list[Message]) -> list[Message]:
    """Return a new list with reaction / ``[photo]`` noise removed.

    Pure: never mutates the input list or the ``Message`` objects in it.
    """
    kept: list[Message] = []
    for msg in messages:
        if _is_reaction(msg.content):
            continue                                   # drop the whole reaction
        cleaned = _strip_photo_token(msg.content)
        if not cleaned:
            continue                                   # was a bare [photo]
        if cleaned != msg.content:
            # content changed (stripped a [photo] token) -> clone with the new
            # content rather than editing the caller's Message in place.
            msg = msg.model_copy(update={"content": cleaned})
        kept.append(msg)
    return kept
