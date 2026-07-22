"""Alphabetical section ordering ignores a leading article (bucket G).

"The Deathsworn" should file under D, not T, while the article stays in the heading.
Offline, pure.
"""

from models.lore import Detail, Location
from renderer.markdown import _alpha_key, render_wiki


def _loc(name):
    return Location(name=name, details=[Detail(text="A place.", source_files=["g.txt"])])


# --- _alpha_key ------------------------------------------------------------- #
def test_alpha_key_strips_leading_articles():
    assert _alpha_key("The Deathsworn") == "deathsworn"
    assert _alpha_key("A Fellowship") == "fellowship"
    assert _alpha_key("An Order") == "order"


def test_alpha_key_leaves_non_article_prefixes():
    # "The"/"A"/"An" only strip when followed by whitespace -- not a word that starts with them.
    assert _alpha_key("Theodore") == "theodore"
    assert _alpha_key("Atlas") == "atlas"
    assert _alpha_key("Deathsworn") == "deathsworn"


# --- through render_wiki ---------------------------------------------------- #
def test_render_wiki_orders_ignoring_leading_the():
    md = render_wiki([_loc("The Ant"), _loc("Bee")], [], [], [], [], [])
    # With article-aware ordering, "The Ant" (files under A) precedes "Bee" (B); a raw
    # name sort would put "The Ant" (T) last.
    assert md.index("The Ant") < md.index("Bee")
