"""Step-1 config builder: declare the party (config/player_map.json) off the speaker map.

Who plays which character -- and each character's last name + pronouns -- is ground truth
only the group knows, so the pipeline REQUIRES this file (build it before running main.py).
It reads the speaker map (build that first), lists each real person, and captures their
player-character(s): a main name (heading), an optional last name (pairs with the main +
alias names for recognition -- "Kriggy"/"Krigius"/"Kriggy Krieger"/"Krigius Krieger" all
fold), optional aliases, and pronouns (default they/them). A person may have several PCs.

Run it after the speaker map, then point the pipeline at the result::

    python scripts/build_imessage_speaker_map.py logs/*.txt
    python scripts/build_player_map.py                  # answer the prompts
    python main.py --files logs/*.txt

Flow: ``player_roster() -> prompt_for_characters() -> save`` -- same injectable-I/O shape as
the speaker-map builders, so it maps cleanly onto a UI form later.
"""

import sys
import json
import argparse
from pathlib import Path

# Allow `python scripts/this.py` to import the top-level modules (sys.path[0] is scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speaker_map import load_speaker_map  # noqa: E402
from player_map import (  # noqa: E402
    DEFAULT_PLAYER_MAP_PATH, DEFAULT_PRONOUNS, save_player_map,
)

DEFAULT_SPEAKER_MAP_PATH = "config/speaker_map.json"


def player_roster(speaker_map) -> list:
    """The distinct real people to ask about -- the speaker map's VALUES (incl. the exporter),
    de-duplicated case-insensitively, original casing + order preserved. Blanks dropped."""
    seen, out = set(), []
    for name in (speaker_map or {}).values():
        n = (name or "").strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _parse_list(s: str) -> list:
    """Comma-separated free text -> a list of non-empty trimmed strings."""
    return [part.strip() for part in (s or "").split(",") if part.strip()]


def _parse_pronouns(s: str) -> list:
    """'she/her' / 'they, them' / '' -> a pronoun list; empty defaults to they/them."""
    import re
    parts = [p.strip() for p in re.split(r"[\s/,]+", s or "") if p.strip()]
    return parts or list(DEFAULT_PRONOUNS)


def prompt_for_characters(roster, input_fn=input, print_fn=print) -> list:
    """Per person, capture zero or more player-characters. Returns a list of entry dicts
    ``{player, main_name, last_name, aliases, pronouns}``. Non-players (the DM, a guest) are
    skippable. ``input_fn``/``print_fn`` are injectable so this is testable without stdin."""
    entries = []
    print_fn("\nDeclare each player's character(s). Enter a character's main name, or '-' to "
             "skip someone who isn't a player (the DM, a guest).\n")
    for person in roster:
        first = True
        while True:
            if first:
                main = input_fn(f"  Does {person} play a character? [main name, or '-' to skip]: ").strip()
                if main == "-":
                    break
            else:
                main = input_fn(f"  Another character for {person}? [main name, or Enter to move on]: ").strip()
            if not main:
                break
            last = input_fn("    Last name (Enter for none): ").strip()
            aliases = _parse_list(input_fn("    Other names/aliases (comma-separated, Enter for none): "))
            pronouns = _parse_pronouns(input_fn("    Pronouns (e.g. she/her; Enter for they/them): "))
            entries.append({
                "player": person,
                "main_name": main,
                "last_name": last or None,
                "aliases": aliases,
                "pronouns": pronouns,
            })
            first = False
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="build_player_map",
        description="Declare the party (config/player_map.json) off the speaker map: per "
                    "person, capture their character(s) -- main name, last name, aliases, pronouns.",
    )
    parser.add_argument("--speaker-map", default=DEFAULT_SPEAKER_MAP_PATH, metavar="PATH",
                        help="The speaker map to read the person roster from (default: %(default)s).")
    parser.add_argument("-o", "--output", default=DEFAULT_PLAYER_MAP_PATH, metavar="PATH",
                        help="Where to write the player map (default: %(default)s).")
    args = parser.parse_args(argv)

    try:
        speaker_map = load_speaker_map(args.speaker_map)
    except FileNotFoundError:
        raise SystemExit(
            f"Speaker map not found: {args.speaker_map}\n"
            f"Build it first:\n"
            f"    python scripts/build_imessage_speaker_map.py logs/*.txt\n"
            f"Or point --speaker-map at yours, e.g. "
            f"--speaker-map config/speaker_map.imessage.json"
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Speaker map {args.speaker_map} is not valid JSON: {exc}")
    roster = player_roster(speaker_map)
    if not roster:
        print("No people found in the speaker map -- build it first "
              "(scripts/build_imessage_speaker_map.py).")
        return
    entries = prompt_for_characters(roster)
    if not entries:
        print("No characters declared; nothing saved.")
        return

    print("\nAbout to save:")
    print(json.dumps(entries, ensure_ascii=False, indent=2))
    if input("\nLook good? (y/n): ").strip().lower() != "y":
        print("Canceled, nothing saved.")
        return
    save_player_map(entries, args.output)
    print(f"Saved {len(entries)} character(s) to {args.output}")


if __name__ == "__main__":
    main()
