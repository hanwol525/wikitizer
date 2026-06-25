"""COMMITTED synthetic fixtures for the Phase 3.10 extractor integration suite.

These five messages (#16-#20 from the Phase 3.10 message list) are *fabricated*
to hit boundary seams the real chat doesn't isolate cleanly -- creature-kind vs
named creature, regional-scope history, individual vs collective, a people vs
their state, and a character name-collision. They contain no real chat, so they
are safe to publish and let anyone with an API key run the synthetic subset.

The real campaign messages (#1-#15) live in the gitignored ``real_messages.py``.

Bodies are copied VERBATIM from the source list -- curly apostrophes/quotes
(' " " - ...) and line breaks are load-bearing (one extractor seam folds curly
punctuation), so do not straighten or reflow them.
"""

from tests.fixtures.case import Case, build_message

SOURCE_FILE = "synthetic_fixtures.txt"


def _msg(sender: str, content: str):
    return build_message(sender, content, SOURCE_FILE)


CASES = [
    # 16. Creature-kind vs single named creature: "direwolves" (a kind) -> People
    # & Cultures; "Greymaw" (one named individual) -> Character. The boundary the
    # characters prompt got its special clarifying line for.
    Case(
        id="kind_vs_creature",
        message=_msg(
            "player_b",
            "direwolves in Gol arent just big wolves btw — theyre pack hunters the size of a horse and way smarter than they’ve got any right to be. the one that nearly took my arm off last session, Greymaw, he’s the alpha of the pack out past Eglon. still got that milky scar over one eye from it",
        ),
        expect={
            # "direwol" not "direwolf": the extractor correctly returns the kind as
            # "Direwolves", and "direwolf" isn't a substring of that (…wolves != …wolf).
            "people": {"expect": ["direwol"], "reject": ["Greymaw"]},
            "characters": {"expect": ["Greymaw"], "reject": ["direwolf"]},
        },
    ),
    # 17. History with REGIONAL scope: confined to the Dagger Swamp -- bigger than
    # one family (not personal), not world-spanning (not world). The only fixture
    # that exercises the third Scope value.
    Case(
        id="hist_swamp",
        message=_msg(
            "dm",
            "Oh the Dagger Swamp had its own messy history way before the Imperium ever showed up. There was a feud between the upper and lower marsh tribes — something like 30 years of it — over who controlled the salt flats. It only really settled when the two biggest tribes intermarried. Pretty contained though; nobody outside the swamp paid it much mind.",
        ),
        expect={
            "history": {"min_count": 1, "scope": "regional"},
        },
    ),
    # 18. Individual vs collective: "Sludge" (a named person) -> Character; "the
    # Dagger Swamp tribes" (a people) -> People & Cultures. Isolates what's tangled
    # inside #15. People could be named "marsh tribes" or "Dagger Swamp tribes" --
    # hence expect_any.
    Case(
        id="individual_vs_collective",
        message=_msg(
            "player_a",
            "Sludge is that marsh witch CJ keeps bringing up, right? She’s one of the Dagger Swamp tribes — those bog folk who never bent the knee to the Imperium. Super insular, live way out in the wetlands, the whole talking-to-swamp-spirits thing.",
        ),
        expect={
            "characters": {"expect": ["Sludge"], "reject": ["marsh", "Dagger Swamp"]},
            "people": {"expect_any": ["marsh", "Dagger Swamp"], "reject": ["Sludge"]},
        },
    ),
    # 19. A people vs their formal state: "the Kriegans / people of Kriega" -> P&C;
    # "the Imperium" (Emperor, chancellors, navy) -> Organization. Isolates what's
    # tangled inside #12/#15.
    Case(
        id="people_vs_state",
        message=_msg(
            "dm",
            "Worth keeping the Imperium and the Kriegan people separate in your head. The Kriegans themselves are just folk — farmers, sailors, that sort of thing — with their own customs and a generally magic-friendly culture. The Imperium is the apparatus: the Emperor, the lord chancellors, the navy. Plenty of ordinary Kriegans actually can’t stand it.",
        ),
        expect={
            "people": {"expect": ["Kriegan"], "reject": ["Imperium"]},
            "organizations": {"expect": ["Imperium"], "reject": ["Kriegan"]},
        },
    ),
    # 20. Character name-collision: "Ryan the innkeeper" is an NPC that shares a
    # real player's first name (Ryan is in the test roster), so the extractor
    # should KEEP it but flag the collision in the logs (eyeball that). The
    # is_pc/player_name tag shows in the report; only the name match is asserted.
    Case(
        id="opt_name_collision",
        message=_msg(
            "dm",
            "Ryan the innkeeper at Crown’s Nest is the one who gave us the job — no relation to our Ryan obviously lol, just a coincidence",
        ),
        expect={
            "characters": {"expect": ["Ryan"]},
        },
    ),
]
