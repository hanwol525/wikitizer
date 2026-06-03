"""Tests for parsers/reaction_filter.py (Phase 2.2: strip reaction/[photo] noise).

filter_reactions runs after the parser, so each reaction is already a single
``Message.content`` (the parser merged any wrapped lines, including a stranded
closing curly quote). These tests cover every opener vocabulary the 2.2 session
verified against the real logs -- English and Spanish, quoted/image/emoji forms --
plus the curly-quote requirement (Pattern 10), the [photo] caption rule
(Pattern 8), the URL-is-not-noise decision, and the function's purity contract.

The reactions below use the curly “ (U+201C) / ” (U+201D) the exporter actually
produces, NOT straight quotes -- that distinction is itself under test.
"""

from datetime import datetime

import pytest

from parsers.chat_parser import parse_chat_log
from parsers.reaction_filter import filter_reactions
from models.message import Message


def make_message(content, sender="Alice", source_file="dndgroup.txt"):
    return Message(
        sender=sender,
        timestamp=datetime(2024, 1, 2, 10, 0, 0),
        content=content,
        source_file=source_file,
    )


def contents(messages):
    return [m.content for m in messages]


# --- reactions are dropped: quoted text, every opener (EN + ES) ---------------

@pytest.mark.parametrize("content", [
    "Loved “Dungeon Master here, who wants to play”",
    "Liked “a short message”",
    "Laughed at “a short message”",
    "Emphasized “a short message”",
    "Disliked “a short message”",
    "Removed a laugh from “a short message”",
    "Le encantó “All hail emperor Terrasque Krieger”",  # Spanish: Loved
    "Le dio risa “a short message”",                    # Spanish: Laughed at
    "Exclamó por “a short message”",                    # Spanish: Emphasized
])
def test_quoted_text_reactions_are_dropped(content):
    assert filter_reactions([make_message(content)]) == []


# --- reactions are dropped: image reactions (no quoted text) ------------------

@pytest.mark.parametrize("content", [
    "Loved an image",
    "Liked an image",
    "Laughed at an image",
    "Emphasized an image",
    "Disliked an image",
    "Le encantó una imagen",  # Spanish: Loved an image
])
def test_image_reactions_are_dropped(content):
    assert filter_reactions([make_message(content)]) == []


# --- reactions are dropped: emoji reactions (emoji varies, needs the regex) ---

@pytest.mark.parametrize("content", [
    "Reacted \U0001f92b to “Is Silvery Barbs banned in this campaign”",  # EN, quoted
    "Reacted \U0001f602 to an image",                                    # EN, image
    "Reaccionó con \U0001f602 a “Mundi probably got shape humanoid”",    # ES, quoted
    "Reaccionó con \U0001f97a a una imagen",                            # ES, image
])
def test_emoji_reactions_are_dropped(content):
    assert filter_reactions([make_message(content)]) == []


def test_multiline_reaction_with_stranded_closing_quote_dropped():
    # Pattern 7: the parser already merged the wrap + the lone closing-quote line
    # into one content. The opener still sits at the very start, so it's caught.
    content = (
        "Loved “If you don’t want to answer yet that’s fine. I’ll be lore\n"
        "dumping world info tomorrow. You can decide after that if you’d like\n"
        "”"
    )
    assert filter_reactions([make_message(content)]) == []


# --- curly-quote requirement (FORMAT_NOTES Pattern 10) -----------------------

def test_straight_quote_lookalike_is_not_treated_as_a_reaction():
    # The openers require the curly “ (U+201C). A line with a STRAIGHT quote is
    # not what the export produces for a reaction, so it's kept as real content.
    # Regression guard: don't "simplify" the curly quote to a straight one, or
    # every real (curly) reaction would slip through as fake content.
    msg = make_message('Loved "this has straight quotes"')
    assert contents(filter_reactions([msg])) == ['Loved "this has straight quotes"']


def test_message_starting_with_reaction_word_but_no_quote_is_kept():
    # "Loved " without the opening curly quote is ordinary human content.
    msg = make_message("Loved the session tonight, that boss fight was unreal")
    assert contents(filter_reactions([msg])) == [
        "Loved the session tonight, that boss fight was unreal"
    ]


# --- [photo] handling (Pattern 8) --------------------------------------------

def test_bare_photo_token_is_dropped():
    assert filter_reactions([make_message("[photo]")]) == []


def test_photo_with_caption_keeps_the_caption():
    # The caption can be real lore, so strip only the [photo] token and keep the
    # rest.
    msg = make_message("[photo]\nVery simple map of the continent of Gol")
    assert contents(filter_reactions([msg])) == [
        "Very simple map of the continent of Gol"
    ]


def test_inline_photo_word_in_a_sentence_is_kept():
    # Only a line that is EXACTLY "[photo]" is stripped; the token appearing
    # inside real text is left alone.
    msg = make_message("check out this [photo] of the map I drew")
    assert contents(filter_reactions([msg])) == [
        "check out this [photo] of the map I drew"
    ]


# --- URLs are intentionally NOT filtered here (deferred to Phase 3) ----------

def test_urls_are_not_filtered():
    # A bare link is human-typed content, not an export artifact; judging it is
    # the Phase 3 LLM noise filter's job, not this layer's.
    msg = make_message("https://www.dndbeyond.com/characters/000000000")
    assert contents(filter_reactions([msg])) == [
        "https://www.dndbeyond.com/characters/000000000"
    ]


# --- ordinary content passes through unchanged -------------------------------

def test_real_messages_pass_through_untouched():
    msgs = [
        make_message("The royal family is human yeah"),
        make_message("Lake Mundi is a massive central lake divided into three rings"),
    ]
    assert contents(filter_reactions(msgs)) == [
        "The royal family is human yeah",
        "Lake Mundi is a massive central lake divided into three rings",
    ]


def test_empty_list_returns_empty_list():
    assert filter_reactions([]) == []


def test_order_is_preserved_across_a_mixed_list():
    msgs = [
        make_message("real message one"),
        make_message("Loved “real message one”"),   # reaction, dropped
        make_message("real message two"),
        make_message("[photo]"),                     # bare photo, dropped
        make_message("[photo]\nkept caption"),       # caption kept
    ]
    assert contents(filter_reactions(msgs)) == [
        "real message one",
        "real message two",
        "kept caption",
    ]


# --- purity contract: never mutate the input ---------------------------------

def test_filter_is_pure_and_does_not_mutate_input():
    original = [
        make_message("a normal message"),
        make_message("[photo]\nA real caption worth keeping"),
        make_message("Loved “a normal message”"),
    ]
    result = filter_reactions(original)

    # Input list and its Message objects are untouched.
    assert len(original) == 3
    assert original[1].content == "[photo]\nA real caption worth keeping"

    # A message whose content changed is returned as a CLONE, not edited in place.
    caption_msg = next(m for m in result if m.content == "A real caption worth keeping")
    assert caption_msg is not original[1]
    # ...but its other fields are carried over from the original.
    assert caption_msg.sender == original[1].sender
    assert caption_msg.timestamp == original[1].timestamp

    # An unchanged message is passed through as the SAME object (no needless copy).
    assert result[0] is original[0]


# --- integration: parse then filter ------------------------------------------

def test_parse_then_filter_removes_reactions_end_to_end(tmp_path):
    dashes = "-" * 100
    path = tmp_path / "dndgroup.txt"
    path.write_bytes("\r\r\n".join([
        ", +15555550101, +15555550102", dashes,
        "01/02/2024 10:00:00",
        "Dungeon Master here, who wants to play",
        "+15555550101 01/02/2024 10:05:00",            # real msg -> Alice
        "Loved “Dungeon Master here, who wants to play”",
        "+15555550102 01/02/2024 10:06:00",            # reaction -> Bob
        "[photo]",
        "Very simple map of Gol",
        "01/02/2024 10:07:00",                         # photo+caption -> exporter
    ]).encode("utf-8"))

    speaker_map = {"+15555550101": "Alice", "+15555550102": "Bob", "exporter": "Hannah"}
    parsed = parse_chat_log(str(path), speaker_map)
    assert len(parsed) == 3  # parser keeps the reaction at this stage

    filtered = filter_reactions(parsed)
    assert contents(filtered) == [
        "Dungeon Master here, who wants to play",
        "Very simple map of Gol",
    ]
    assert filtered[0].sender == "Alice"
    assert filtered[1].sender == "Hannah"  # the photo+caption was the exporter's
