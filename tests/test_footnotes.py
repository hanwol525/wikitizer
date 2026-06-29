"""Unit tests for renderer/footnotes.py (Phase 4.2: the footnote registry).

Plain pytest -- no network, no API key, no integration marker. This component is
pure Python over already-validated ``Quote`` objects, so every assertion below is
deterministic and exact. The two fidelity edges (asterisks survive, newlines
collapse) get isolated tests because they're the most likely to regress and the
most important to catch; the holistic block test (#8) covers code-span wrapping,
the attribution tail, ordering, and structure together.
"""

from models.lore import Quote
from renderer.footnotes import FootnoteRegistry


def q(text, speaker="dm", source="dndgroup.txt"):
    return Quote(text=text, speaker=speaker, source_file=source)


# 1 --------------------------------------------------------------------------
def test_sequential_one_based_numbering():
    reg = FootnoteRegistry()
    assert reg.add(q("alpha")) == 1
    assert reg.add(q("beta")) == 2
    assert reg.add(q("gamma")) == 3


# 2 --------------------------------------------------------------------------
def test_dedup_returns_same_number_and_does_not_burn_one():
    # The strongest dedup assertion: a repeat add must return the SAME number AND
    # not consume the next one -- so the following distinct quote is 2, not 3.
    reg = FootnoteRegistry()
    assert reg.add(q("the city fell")) == 1
    assert reg.add(q("the city fell")) == 1
    assert reg.add(q("a distinct fact")) == 2


# 3 --------------------------------------------------------------------------
def test_edge_space_does_not_split_a_footnote():
    # A stray trailing space must not split one quote into two footnotes. (The
    # INTERNAL-whitespace twin case -- newline vs space -- is #17.)
    reg = FootnoteRegistry()
    assert reg.add(q("the city fell")) == 1
    assert reg.add(q("the city fell ")) == 1


# 4 --------------------------------------------------------------------------
def test_speaker_is_part_of_the_key():
    # Same text + same source but DIFFERENT speaker => two footnotes. Two people
    # attesting an identical line is more evidence, so each gets its own number.
    reg = FootnoteRegistry()
    assert reg.add(q("the city fell", speaker="player_a")) == 1
    assert reg.add(q("the city fell", speaker="dm")) == 2


# 5 --------------------------------------------------------------------------
def test_source_is_part_of_the_key():
    # Same text + same speaker but DIFFERENT source => two footnotes.
    reg = FootnoteRegistry()
    assert reg.add(q("the city fell", source="group_a.txt")) == 1
    assert reg.add(q("the city fell", source="group_b.txt")) == 2


# 6 --------------------------------------------------------------------------
def test_keys_on_field_values_not_object_identity():
    # Two SEPARATE Quote instances with identical fields must collapse to one
    # footnote -- the key is the field-value triple, not object identity.
    reg = FootnoteRegistry()
    first = Quote(text="same fact", speaker="dm", source_file="dndgroup.txt")
    second = Quote(text="same fact", speaker="dm", source_file="dndgroup.txt")
    assert first is not second
    assert reg.add(first) == 1
    assert reg.add(second) == 1
    assert reg.add(q("other fact")) == 2


# 7 --------------------------------------------------------------------------
def test_empty_registry_renders_empty_string():
    assert FootnoteRegistry().render_definitions() == ""


# 8 --------------------------------------------------------------------------
def test_full_definitions_block_exact_string():
    # Holistic: heading, one blank line, code spans, em-dash + comma tail,
    # ascending order, and NO trailing newline.
    reg = FootnoteRegistry()
    reg.add(q("first fact"))
    reg.add(q("second fact"))
    expected = (
        "## Footnotes\n"
        "\n"
        "[^1]: `first fact` — dm, dndgroup.txt\n"
        "[^2]: `second fact` — dm, dndgroup.txt"
    )
    assert reg.render_definitions() == expected


# 9 --------------------------------------------------------------------------
def test_asterisks_survive_literally_inside_code_span():
    # The load-bearing fidelity case: roleplay asterisks must not be eaten as
    # markdown emphasis -- the code span keeps them literal.
    reg = FootnoteRegistry()
    reg.add(q("*draws his sword* you shall not pass"))
    rendered = reg.render_definitions()
    assert "`*draws his sword* you shall not pass`" in rendered


# 10 -------------------------------------------------------------------------
def test_internal_newline_collapses_to_one_space():
    # A footnote definition is line-based; an internal newline must be flattened
    # so one multi-line quote stays one definition.
    reg = FootnoteRegistry()
    reg.add(q("line one\nline two"))
    rendered = reg.render_definitions()
    assert "`line one line two`" in rendered
    # The definition line itself must carry no newline mid-quote.
    definition_line = [ln for ln in rendered.split("\n") if ln.startswith("[^1]:")][0]
    assert "\n" not in definition_line


# 11 -------------------------------------------------------------------------
def test_definitions_follow_insertion_order_not_text_sort():
    # Proves there is no accidental alphabetical sort: "zebra" added first is
    # [^1], "apple" added second is [^2].
    reg = FootnoteRegistry()
    reg.add(q("zebra"))
    reg.add(q("apple"))
    rendered = reg.render_definitions()
    assert "[^1]: `zebra`" in rendered
    assert "[^2]: `apple`" in rendered


# 12 -------------------------------------------------------------------------
def test_empty_text_tripwire():
    # Pins the "empty can't reach here and isn't rejected" assumption: a
    # whitespace-only quote strips to "" but still gets a number -- no crash, no
    # special-casing. If someone later "hardens" add() to reject/return-None on
    # empty text, this test trips and forces a conscious decision about it.
    reg = FootnoteRegistry()
    assert reg.add(q("   ")) == 1


# --- hardening tests added after an adversarial review (Phase 4.2) -----------
# These are ADDITIVE: the original 12 above are untouched. 13-14 are regression
# tests for the code-span backtick fix; 15-16 close two coverage gaps the review
# surfaced (a weaker flatten and an accidental sort both survived the first 12).


# 13 -------------------------------------------------------------------------
def test_backtick_in_quote_widens_fence_so_asterisks_survive():
    # Regression for the code-span backtick gap: a quote containing a literal
    # backtick must NOT break the span. With a fixed single-backtick wrapper the
    # open fence fuses with the inner backtick, spilling "*drew his blade*"
    # OUTSIDE the span where markdown italicizes it. The fence must widen to two
    # backticks so the WHOLE quote -- inner backticks and asterisks alike -- stays
    # inside one code span.
    reg = FootnoteRegistry()
    reg.add(q("he said `hi` and *drew his blade*"))
    rendered = reg.render_definitions()
    assert "``he said `hi` and *drew his blade*``" in rendered


# 14 -------------------------------------------------------------------------
def test_leading_and_trailing_backtick_are_space_padded():
    # A quote that STARTS or ENDS with a backtick would fuse with the fence; the
    # invisible space pad (CommonMark strips one leading + one trailing space
    # inside a span) keeps the fence and the edge backtick distinct.
    reg = FootnoteRegistry()
    assert reg.add(q("`start")) == 1
    assert reg.add(q("end`")) == 2
    rendered = reg.render_definitions()
    assert "[^1]: `` `start ``" in rendered
    assert "[^2]: `` end` ``" in rendered


# 15 -------------------------------------------------------------------------
def test_tab_and_multispace_collapse_to_single_space():
    # Pins the FULL whitespace-collapse contract (" ".join(text.split())), not
    # just the single-newline case #10 covers: a weaker text.replace("\n", " ")
    # leaves tabs and double-spaces intact and is killed here.
    reg = FootnoteRegistry()
    reg.add(q("a\tb   c\nd"))
    rendered = reg.render_definitions()
    assert "`a b c d`" in rendered
    definition_line = [ln for ln in rendered.split("\n") if ln.startswith("[^1]:")][0]
    assert "\t" not in definition_line
    assert "  " not in definition_line


# 16 -------------------------------------------------------------------------
def test_definition_lines_are_in_insertion_order_not_sorted():
    # Strengthens #11, which only checks substring membership: an accidental
    # sorted() on the render loop keeps both substrings present but FLIPS their
    # order, so #11 passes on broken output. This asserts the [^1] line actually
    # PRECEDES the [^2] line, catching the sort.
    reg = FootnoteRegistry()
    reg.add(q("zebra"))
    reg.add(q("apple"))
    rendered = reg.render_definitions()
    assert rendered.index("[^1]: `zebra`") < rendered.index("[^2]: `apple`")


# 17 -------------------------------------------------------------------------
def test_internal_whitespace_twins_dedup_to_one_footnote():
    # THE regression guard for the normalization fix (and the _normalize_quote_text
    # refactor): two quotes differing ONLY by internal whitespace -- a newline vs a
    # space -- must collapse to ONE footnote. The pre-fix .strip()-only key split
    # these into [^1] and [^2] yet rendered them identically (the exact
    # duplicate-looking-footnote bug). This FAILS on a .strip()-only key and passes
    # on the whitespace-collapse key, so it pins the fix instead of decorating it.
    reg = FootnoteRegistry()
    assert reg.add(q("line one\nline two")) == 1
    assert reg.add(q("line one line two")) == 1   # same key after collapse
    assert reg.add(q("a distinct fact")) == 2     # and it didn't burn a number


# 18 -------------------------------------------------------------------------
def test_normalization_touches_only_whitespace_not_case_or_punctuation():
    # Companion guard to #17. _normalize_quote_text is now the SINGLE point where
    # quote text is mutated (and render trusts the stored text), so it must touch
    # ONLY whitespace -- never case or punctuation -- or it would silently corrupt
    # the verbatim footnote this whole component exists to preserve. #17 pins the
    # whitespace dimension; this pins the content dimension. A normalizer that
    # collapses whitespace correctly but ALSO lowercases or drops punctuation
    # (e.g. " ".join(t.lower().split())) passes #1-#17 -- every quote there is
    # already lowercase and punctuation-free -- but FAILS here.
    reg = FootnoteRegistry()
    reg.add(q("The City FELL! it did."))
    # rendered verbatim: case + the "!" survive inside the code span
    assert "`The City FELL! it did.`" in reg.render_definitions()
    # and case is part of the dedup key (matches the extractor's "case is a real
    # change" stance): two case-different quotes are two footnotes, not one.
    assert reg.add(q("Gol")) == 2
    assert reg.add(q("gol")) == 3
