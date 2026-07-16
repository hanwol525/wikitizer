"""Phase A1: the renderer's detail-join punctuation fix.

The extractors write each `Detail.text` as a short fragment with NO trailing period,
so the old bare-space join produced run-ons ("A massive lake Home to three rings").
`_smart_join_details` now appends a terminal period to any fact that doesn't already
end in sentence punctuation, then space-joins -- and the entity body source prefers
the prose agent's `entity.prose` when set, falling back to this join.

Offline, pure -- no API key, no integration marker (render_wiki + _smart_join_details
are pure).
"""

from models.lore import Character, Detail, Location, Quote
from renderer.markdown import _smart_join_details, render_wiki


def det(text, *source_files):
    return Detail(text=text, source_files=list(source_files))


# --- _smart_join_details (the pure helper) --------------------------------- #
def test_period_less_fragments_get_a_period_then_space_join():
    out = _smart_join_details([det("A hidden valley"), det("Home to Elrond")])
    assert out == "A hidden valley. Home to Elrond."


def test_already_punctuated_fragments_are_left_alone_no_double_period():
    # A fact already ending in . ! or ? must NOT gain a second terminal mark.
    out = _smart_join_details([det("A hidden valley."), det("Who lives here?"),
                               det("Beware!")])
    assert out == "A hidden valley. Who lives here? Beware!"


def test_empty_and_whitespace_only_details_are_skipped():
    out = _smart_join_details([det("Real fact"), det("   "), det(""), det("Another")])
    assert out == "Real fact. Another."


def test_empty_list_joins_to_empty_string():
    assert _smart_join_details([]) == ""


# --- through the real renderer --------------------------------------------- #
def test_rendered_entity_body_has_periods_between_facts():
    loc = Location(name="Rivendell", details=[det("A hidden valley"),
                                              det("Home to Elrond")])
    out = render_wiki([loc], [], [], [], [], [])
    assert "A hidden valley. Home to Elrond." in out
    # No run-on (the exact bug): the two facts never abut without a period.
    assert "valley Home" not in out


# --- prose field takes precedence over the join ---------------------------- #
def test_entity_prose_is_used_verbatim_over_the_joined_details():
    # When the prose agent has run, the polished paragraph is the body; the raw
    # details are NOT re-joined into the output.
    loc = Location(
        name="Rivendell",
        details=[det("A hidden valley"), det("Home to Elrond")],
        prose="A hidden valley in the mountains, home to Elrond and the High Elves.",
    )
    out = render_wiki([loc], [], [], [], [], [])
    assert "A hidden valley in the mountains, home to Elrond and the High Elves." in out
    # The un-polished join must NOT also appear.
    assert "A hidden valley. Home to Elrond." not in out


def test_entity_prose_none_falls_back_to_smart_join():
    loc = Location(name="Bree", details=[det("A town of Men")], prose=None)
    out = render_wiki([loc], [], [], [], [], [])
    assert "A town of Men." in out


def test_prose_body_still_gets_footnote_markers_and_crosslinks():
    # The prose path must not lose footnotes: quotes still hang at the body end.
    q = Quote(text="Bree is near Rivendell", speaker="M", source_file="g.txt")
    bree = Character(name="Barliman", details=[det("An innkeeper")],
                     prose="An innkeeper at Bree, near Rivendell.",
                     supporting_quotes=[q])
    rivendell = Location(name="Rivendell", details=[det("A valley")])
    out = render_wiki([rivendell], [bree], [], [], [], [])
    assert "[^1]" in out                       # footnote marker present on the prose body
    assert "[Rivendell](#rivendell)" in out    # crosslink still fires inside prose
