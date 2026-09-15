"""Pipeline entry point + the command-line front door.

Parses the CLI, loads config + the API key from the environment, builds the
orchestrator, runs the pipeline over the given chat-log files, and writes the
finished wiki (plus a restricted copy when --exclude-sources is given) to disk.

    python main.py --files logs/*.txt
    python main.py --files logs/*.txt --exclude-sources dm.txt
    python main.py --files logs/*.txt --output /tmp/wiki.md --exclude-sources dm.txt

The restricted copy's path is DERIVED from --output (see restricted_path), so the
two docs always land together. SPEAKER_MAP_PATH / CROSSLINK_WORDS_PATH / PLAYER_MAP_PATH
stay module constants on purpose: they're per-install config, not per-run knobs.

--confirm-players builds/updates the declared party from the characters discovered in a
run, writing back to the file it read (--player-map); the saved party takes effect on the
NEXT run.
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from orchestrator import Orchestrator, PipelineConfig
from speaker_map import load_speaker_map
from renderer.crosslink import load_crosslink_words
from player_map import declared_characters, load_player_map, save_player_map

DEFAULT_OUTPUT_PATH = "output/wiki.md"   # output/ is gitignored -- a real-log wiki carries PII

# Per-install config, not per-run knobs -- deliberately NOT CLI args.
SPEAKER_MAP_PATH = "config/speaker_map.json"
CROSSLINK_WORDS_PATH = "config/crosslink_words.json"
PLAYER_MAP_PATH = "config/player_map.json"


def parse_args(argv=None):
    """Parse the command line into a Namespace: files, output, exclude_sources.

    `argv` is a parameter rather than read from sys.argv so tests can call
    parse_args(["--files", "a.txt"]) directly with no monkeypatching. argparse falls
    back to sys.argv[1:] when it's None, which is what main() passes in production.

    --files is REQUIRED with no default: the old hardcoded FILES pointed at
    gitignored logs, so a default would make a fresh clone fail confusingly. It stays
    a NAMED arg rather than a positional for two reasons -- a nargs="+" positional
    sitting next to a nargs="+" optional (--exclude-sources) is a real argparse
    ambiguity, and both take filenames, so naming them keeps which-is-which obvious.
    """
    parser = argparse.ArgumentParser(
        prog="wikitizer",
        description="Turn exported D&D chat logs into a structured wiki.",
    )
    parser.add_argument(
        "--files", nargs="+", required=True, metavar="PATH",
        help="Chat-log files to ingest. Shell globs work: --files logs/*.txt",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, metavar="PATH",
        help="Where to write the full wiki (default: %(default)s). With "
             "--exclude-sources, the restricted copy is written alongside it as "
             "<name>_restricted<ext>.",
    )
    parser.add_argument(
        "--exclude-sources", nargs="+", default=[], metavar="FILENAME",
        help="Bare filenames (NOT paths) to hide from a second, restricted wiki -- "
             "e.g. --exclude-sources dm.txt keeps the confidential DM log out of the "
             "players' copy. Omit for no restricted doc. A name that isn't one of "
             "--files is a hard error (exclusion.validate_exclusions, inside run()).",
    )
    parser.add_argument(
        "--speaker-map", default=SPEAKER_MAP_PATH, metavar="PATH",
        help="Path to the speaker map JSON (default: %(default)s). Override to run a "
             "different setup without touching the default (built by "
             "scripts/build_imessage_speaker_map.py).",
    )
    parser.add_argument(
        "--player-map", default=PLAYER_MAP_PATH, metavar="PATH",
        help="Path to the declared-party JSON (default: %(default)s). Override to point "
             "at a different party file (same reason as --speaker-map).",
    )
    parser.add_argument(
        "--current-year", type=int, default=None, metavar="YEAR",
        help="The campaign's present-day reference year (e.g. 1424). Lets the timeline "
             "resolve present-relative dates ('200 years ago' -> 1224). Omit to let the "
             "pipeline auto-detect it from the lore (or leave such events undated if none "
             "is stated); a value here OVERRIDES the auto-detected one.",
    )
    parser.add_argument(
        "--cache", action="store_true",
        help="Reuse cached LLM responses for identical extractor/noise-filter calls "
             "(a gitignored .llm_cache/ dir; never expires). Skips re-billing the same "
             "calls when you re-run the SAME logs while debugging. The reconciler and "
             "timeline calls always re-run. Off by default so a normal run never serves "
             "a stale answer.",
    )
    parser.add_argument(
        "--confirm-players", action="store_true",
        help="After the run, interactively confirm/correct which real person plays each "
             "discovered character, and save the answers back to --player-map. Who "
             "plays a character can't be inferred reliably, so this is how you declare it. "
             "The saved party takes effect on the NEXT run (it drives extraction + merge). "
             "Off by default (a normal run is non-interactive).",
    )
    parser.add_argument(
        "--no-player-map", action="store_true",
        help="Run WITHOUT a declared party. The player map is normally REQUIRED (build it "
             "with scripts/build_player_map.py); this is a deliberate opt-out for a "
             "party-less/test run, and PCs may then duplicate or be mis-attributed.",
    )
    args = parser.parse_args(argv)
    # Fail cheap: an existing-directory --output would only blow up at write_text time,
    # AFTER the whole paid pipeline ran -- so reject it up front, same "fail before any
    # paid call" spirit as config loading and validate_exclusions. parser.error exits 2
    # like every other CLI mistake. This guards only the one common, cheaply-checkable
    # case (habitually pointing --output at the gitignored output/ DIR); the broader
    # class of write failures -- permissions, disk-full -- isn't pre-checkable and stays
    # a post-run error.
    if Path(args.output).is_dir():
        parser.error(f"--output must be a file path, not a directory: {args.output!r}")
    return args


def restricted_path(output) -> Path:
    """Where the restricted wiki goes, derived from the full wiki's path:
    output/wiki.md -> output/wiki_restricted.md -- which is EXACTLY the old hardcoded
    RESTRICTED_OUTPUT_PATH, so the default behaviour is unchanged by the CLI.

    Derived rather than given its own arg so the two docs always land TOGETHER
    wherever you point --output. A fixed path would mean `--output /tmp/wiki.md`
    silently dropping the players' copy into output/ where you weren't looking.

    Uses with_name, NOT with_stem: with_stem is a later pathlib addition and this
    project is pinned to 3.9.6, while with_name has been there since pathlib shipped
    in 3.4. Identical result, no version question.
    """
    p = Path(output)
    return p.with_name(p.stem + "_restricted" + p.suffix)


def confirm_player_map(pcs, existing, input_fn=input, print_fn=print) -> list:
    """Interactively confirm/correct the player of each discovered PC and return the
    updated declared party in the CANONICAL LIST form -- one entry per CHARACTER,
    ``{"player", "main_name", "last_name", "aliases", "pronouns"}`` -- merged with
    ``existing``.

    One entry per CHARACTER, not the old ``{player: [names]}`` bag, because that shape
    was LOSSY on every round-trip: a player's second PC silently became an alias of the
    first, and ``last_name``/``pronouns`` were dropped for EVERY entry. The pronoun loss
    was the worst of it -- the loader defaults missing pronouns to they/them and the
    prose pass then actively rewrites that character's pronouns on the next run. So this
    path now preserves whatever ``scripts/build_player_map.py`` built and only edits the
    one thing it asks about: who plays each character.

    Pure except for the injected ``input_fn``/``print_fn`` (defaults to builtins), so it
    unit-tests without real stdin. Per character: Enter keeps its current player, a typed
    name (re)assigns it, ``-`` skips it. A discovered PC is matched to an existing entry
    by that entry's RECOGNITION names (so "Kriggy Krieger" finds the Kriggy/Krieger entry),
    and then only its ``player`` and ``aliases`` are touched -- the declared ``main_name``
    stays the heading and ``last_name``/``pronouns`` ride through untouched. A PC matching
    no entry becomes a new one. A reassignment also detaches those names from every OTHER
    entry, so the party can never hold one name under two people.

    ``existing`` may be the canonical list or any of the old dict forms -- both normalize
    through ``declared_characters``.
    """
    entries = [
        {"player": dc.player, "main_name": dc.main_name, "last_name": dc.last_name,
         "aliases": list(dc.aliases), "pronouns": list(dc.pronouns)}
        for dc in declared_characters(existing)
    ]

    def _recognition(entry):
        # The cross-product ({main_name} U aliases) x {"", last_name}. Computed through
        # declared_characters so the matching here can never drift from the matching the
        # extractor/reconciler do -- one implementation, in player_map.
        return set(declared_characters([entry])[0].recognition)

    def _find(names_lower):
        for entry in entries:
            if names_lower & _recognition(entry):
                return entry
        return None

    def _detach(keep, names_lower):
        """Drop `names_lower` from every OTHER entry so a reassignment can't leave one
        character under two players. An other entry whose own `main_name` is one of these
        names IS this character declared twice (a hand-edited config), so it goes away --
        but `keep` inherits its `last_name` when it has none, so the collapse can't lose
        a recognition form. Pronouns aren't inherited: the loader has already defaulted
        an absent list to they/them, so 'absent' isn't distinguishable here."""
        survivors = []
        for e in entries:
            if e is keep:
                survivors.append(e)
                continue
            if e["main_name"].strip().lower() in names_lower:
                keep["last_name"] = keep["last_name"] or e["last_name"]
                continue
            e["aliases"] = [a for a in e["aliases"] if a.strip().lower() not in names_lower]
            survivors.append(e)
        entries[:] = survivors

    if not pcs:
        print_fn("No player characters were discovered in this run; nothing to confirm.")
        return entries

    print_fn("Confirm who plays each character (Enter = keep, type a name to set, '-' = skip):")
    for c in pcs:
        aliases = [a.text for a in c.aliases]
        alias_str = f" (aka {', '.join(aliases)})" if aliases else ""
        current = c.player_name or "unset"
        answer = input_fn(f"  {c.name}{alias_str} -- player [{current}]: ").strip()
        if answer == "-":
            continue
        player = answer if answer else c.player_name
        if not player:
            continue                       # blank + no current player -> nothing to assign
        names = [n for n in [c.name, *aliases] if n and n.strip()]
        if not names:
            continue
        names_lower = {n.strip().lower() for n in names}

        entry = _find(names_lower)
        if entry is None:
            # A newly discovered character: its extraction-time pronouns are the best
            # starting point (empty -> the loader defaults them to they/them).
            entry = {"player": player, "main_name": names[0], "last_name": None,
                     "aliases": [], "pronouns": list(getattr(c, "pronouns", None) or [])}
            entries.append(entry)
        else:
            entry["player"] = player       # the ONLY field the confirmation rewrites

        known = _recognition(entry)
        for n in names:                    # fold in any name the entry doesn't know yet
            if n.strip().lower() not in known:
                entry["aliases"].append(n)
                known.add(n.strip().lower())
        _detach(entry, names_lower)

    return [e for e in entries if e["main_name"].strip()]


def main(argv=None) -> None:
    """Run the pipeline. `argv` is threaded to parse_args so tests can drive main()
    without touching sys.argv; production calls main() with None."""
    args = parse_args(argv)

    # Turn logging ON. Python's logging is SILENT until configured, so without this
    # every [REVIEW] flag and warning the pipeline emits would go nowhere. After
    # parse_args on purpose: an argparse error or --help writes to stderr and exits
    # on its own, with no logging needed.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Pull .env into the environment so the Anthropic SDK finds ANTHROPIC_API_KEY.
    load_dotenv()

    # Load the config files up front -- before any paid LLM call (fail cheap). A missing /
    # malformed file becomes a friendly, actionable SystemExit, NOT a raw traceback.
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
    try:
        # The declared party (gitignored). Missing -> [] (the gate below decides); a
        # malformed shape raises ValueError -> a friendly rebuild hint here.
        player_map = load_player_map(args.player_map)
    except ValueError as exc:
        raise SystemExit(f"Player map {args.player_map} is malformed: {exc}\n"
                         f"Rebuild it: python scripts/build_player_map.py")

    config = PipelineConfig(
        speaker_map=speaker_map,
        crosslink_words=load_crosslink_words(CROSSLINK_WORDS_PATH),
        current_year=args.current_year,
        player_map=player_map,
    )

    # Player/character disambiguation is only as good as the declared party, so the map is a
    # HARD, step-1 requirement -- refuse to run without it (before any paid call), unless the
    # user consciously opts out with --no-player-map. When the map IS present it's the source
    # of truth (the extractor drops any LLM player guess for an undeclared character).
    if not config.player_map and not args.no_player_map:
        raise SystemExit(
            f"No declared party found ({args.player_map} is missing or empty).\n"
            f"Build it first:\n"
            f"    python scripts/build_imessage_speaker_map.py logs/*.txt\n"
            f"    python scripts/build_player_map.py\n"
            f"Or pass --no-player-map to run without one (PCs may duplicate or be mis-attributed)."
        )
    if not config.player_map:
        logging.getLogger(__name__).warning(
            "[REVIEW] Running with NO declared party (--no-player-map); character/player "
            "disambiguation is disabled and PCs may duplicate or be mis-attributed.",
        )

    # A bad --exclude-sources name raises ValueError from inside run(), before any
    # paid call. The CLI deliberately does NOT re-check it: validate_exclusions lives
    # in run() so every caller inherits the guard, not just this one.
    output = Orchestrator(cache=args.cache).run(
        args.files, config, exclude_sources=args.exclude_sources)

    log = logging.getLogger(__name__)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output.full, encoding="utf-8")   # utf-8: names may have accents
    log.info("Wrote full wiki (%d chars) to %s", len(output.full), out)

    # `restricted is None` means no exclusions were requested. An empty STRING would
    # mean they were, but nothing public survived -- we still write that (an empty
    # wiki is a real answer) and note it, so it's never a silent surprise.
    if output.restricted is not None:
        rout = restricted_path(args.output)
        # A no-op today (the derived path is always a sibling of --output, whose
        # parent we just made), but idempotent and it keeps this block readable
        # standalone.
        rout.parent.mkdir(parents=True, exist_ok=True)
        rout.write_text(output.restricted, encoding="utf-8")
        if output.restricted:
            log.info("Wrote restricted wiki (%d chars) to %s", len(output.restricted), rout)
        else:
            log.warning("Restricted wiki is EMPTY (every source was excluded?); wrote %s anyway", rout)

    # --confirm-players: build/update the declared party from the discovered PCs. The
    # saved map takes effect on the NEXT run (it drives extraction + merge, which have
    # already happened this run) -- said plainly so the unchanged output isn't a surprise.
    if args.confirm_players:
        updated = confirm_player_map(output.characters, config.player_map)
        # Write back to the file we READ (args.player_map), not the module default --
        # saving to the default would clobber the standard party with answers built from
        # a different one whenever --player-map is used.
        save_player_map(updated, args.player_map)
        log.info("Saved the declared party to %s -- re-run to apply it.", args.player_map)


if __name__ == "__main__":
    main()
