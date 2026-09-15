"""The declared party: a user-owned list of the group's player-characters.

Who plays which character is a fact only the group knows -- no automatic heuristic can
recover it reliably (a chat where lore is narrated by a few people makes "who voices a
character" a bad proxy for "who plays it"). So the party is USER config, the same category
as ``config/speaker_map.json``: gitignored (real names are PII), NOT hardcoded into source,
built up front (see ``scripts/build_player_map.py``) as the ground-truth anchor for
player<->character links, name recognition, and pronouns.

Canonical shape (``config/player_map.json``) -- a LIST of character entries::

    [
      {"player": "Sam",    "main_name": "Kriggy", "last_name": "Krieger",     "aliases": ["Krigius"], "pronouns": ["he","him"]},
      {"player": "Conrad", "main_name": "Aerin",  "last_name": "Wakestrider", "aliases": [],          "pronouns": ["they","them"]}
    ]

Per entry: ``player`` (a speaker_map value) · ``main_name`` (required -- the DISPLAY/heading
name, a first name OR a nickname) · ``last_name`` (nullable; RECOGNITION-ONLY, never displayed)
· ``aliases`` (nullable) · ``pronouns`` (nullable -> defaults ``["they","them"]``). Two entries
may share a ``player`` (a player with more than one PC).

Recognition (the extractor's lookup + the reconciler's declared-merge floor) matches on the
CROSS-PRODUCT ``{main_name UNION aliases} x {"", last_name}`` -- so "Kriggy", "Krigius",
"Kriggy Krieger", and "Krigius Krieger" all fold, while the heading stays ``main_name``.

Back-compat: the old dict forms ``{player: {main_name, aliases}}`` / ``{player: [names]}`` /
``{player: name}`` still load -- each becomes one entry per player (``last_name=None``, default
pronouns). A dict value MAY also carry ``last_name``/``pronouns`` (one PC per player then).

The loader is TOLERANT of a missing file (-> ``[]``) but STRICT on a parsed-but-wrong shape --
a silent wrong shape would quietly disable the whole feature.
"""

import json
from dataclasses import dataclass
from typing import Optional


DEFAULT_PLAYER_MAP_PATH = "config/player_map.json"
DEFAULT_PRONOUNS = ["they", "them"]


@dataclass
class DeclaredCharacter:
    """One declared player-character, fully normalized. ``recognition`` is the lowercased,
    de-duplicated, ORDER-PRESERVING (bases first, then base+last_name forms) name set the
    reconciler/extractor match against; ``player``/``main_name``/``aliases`` keep original
    casing for display; ``pronouns`` is always non-empty (defaulted)."""
    player: str
    main_name: str
    last_name: Optional[str]
    aliases: list
    pronouns: list
    recognition: list


def _coerce_names(value) -> list:
    """Normalize a dict-form player's VALUE into a recognition list ``[main_name, *aliases]``
    (main_name FIRST, original casing, non-empty). Accepts the object form
    ``{"main_name", "aliases"}``, a flat ``list[str]`` (first = main_name), or a bare ``str``.
    Raises ValueError on any other shape or a non-string name. (Back-compat helper for the
    old dict-keyed config; the new list form is parsed by ``_as_entry_list``.)"""
    if isinstance(value, dict):
        main = value.get("main_name")
        if not isinstance(main, str) or not main.strip():
            raise ValueError("a player_map object needs a non-empty string 'main_name'.")
        aliases = value.get("aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(
                f"player_map 'aliases' must be a list of strings, got {type(aliases).__name__}."
            )
        raw = [main] + aliases
    elif isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raise ValueError(
            f"player_map values must be an object, a string, or a list, got {type(value).__name__}."
        )
    names = []
    for n in raw:
        if not isinstance(n, str):
            raise ValueError(f"player_map character names must be strings, got {type(n).__name__}.")
        if n.strip():
            names.append(n)
    return names


def _as_entry_list(player_map) -> list:
    """Normalize ANY supported shape into the canonical list of entry-dicts
    ``[{player, main_name, last_name?, aliases?, pronouns?}, ...]``. A top-level LIST is the
    new form (each item validated as an object); a top-level DICT is the old
    ``{player: value}`` form (each value coerced, a dict value carried through so it can also
    supply last_name/pronouns). Raises on a broken top-level type."""
    if not player_map:
        return []
    if isinstance(player_map, list):
        for e in player_map:
            if not isinstance(e, dict):
                raise ValueError(f"player_map list entries must be objects, got {type(e).__name__}.")
        return list(player_map)
    if isinstance(player_map, dict):
        out = []
        for player, value in player_map.items():
            if not isinstance(player, str) or not player.strip():
                raise ValueError(f"player_map keys (players) must be non-empty strings, got {player!r}.")
            if isinstance(value, dict):
                entry = dict(value)          # may carry last_name/pronouns too
                entry["player"] = player
                out.append(entry)
            else:
                names = _coerce_names(value)  # list/str -> [main_name, *aliases]
                if names:
                    out.append({"player": player, "main_name": names[0], "aliases": names[1:]})
        return out
    raise ValueError(f"player_map must be a list or an object, got {type(player_map).__name__}.")


def _recognition_list(main_name: str, aliases: list, last_name: Optional[str]) -> list:
    """Lowercased, de-duplicated recognition names: every base (main_name + aliases) FIRST,
    then each base paired with last_name. So Kriggy/[Krigius]/Krieger ->
    ['kriggy', 'krigius', 'kriggy krieger', 'krigius krieger']. Only FULL 'base last' forms
    are added -- never a bare last_name -- so a different surname-sharer never folds."""
    bases, seen = [], set()
    for b in [main_name, *aliases]:
        bl = b.strip().lower()
        if bl and bl not in seen:
            seen.add(bl)
            bases.append(bl)
    rec = list(bases)
    ln = last_name.strip().lower() if last_name else ""
    if ln:
        for b in bases:
            full = f"{b} {ln}"
            if full not in seen:
                seen.add(full)
                rec.append(full)
    return rec


def declared_characters(player_map) -> list:
    """Parse ANY supported config shape into ``list[DeclaredCharacter]`` -- the single source
    of truth for the declared party. Computes the recognition cross-product and defaults empty
    pronouns to they/them. Raises ValueError on a broken entry (missing/blank main_name)."""
    out = []
    for e in _as_entry_list(player_map):
        player = (e.get("player") or "").strip()
        main_name = e.get("main_name")
        if not isinstance(main_name, str) or not main_name.strip():
            raise ValueError("each declared character needs a non-empty string 'main_name'.")
        main_name = main_name.strip()
        raw_last = e.get("last_name")
        last_name = raw_last.strip() if isinstance(raw_last, str) and raw_last.strip() else None
        raw_aliases = e.get("aliases") or []
        raw_pronouns = e.get("pronouns") or []
        if not isinstance(raw_aliases, list):
            raise ValueError(f"'aliases' must be a list of strings, got {type(raw_aliases).__name__}.")
        if not isinstance(raw_pronouns, list):
            raise ValueError(f"'pronouns' must be a list of strings, got {type(raw_pronouns).__name__}.")
        aliases = [a.strip() for a in raw_aliases if isinstance(a, str) and a.strip()]
        pronouns = [p.strip() for p in raw_pronouns if isinstance(p, str) and p.strip()]
        out.append(DeclaredCharacter(
            player=player,
            main_name=main_name,
            last_name=last_name,
            aliases=aliases,
            pronouns=pronouns or list(DEFAULT_PRONOUNS),
            recognition=_recognition_list(main_name, aliases, last_name),
        ))
    return out


def load_player_map(path: str = DEFAULT_PLAYER_MAP_PATH):
    """Load the declared party as the canonical list of entry-dicts. Missing file -> ``[]``
    (no party). A file that PARSED but is the wrong shape RAISES (validated by running it
    through ``declared_characters``), so a broken config fails before any paid call rather
    than silently disabling the feature."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    entries = _as_entry_list(data)
    declared_characters(entries)      # validate (raises on a broken entry); result discarded
    return entries


def save_player_map(mapping, path: str = DEFAULT_PLAYER_MAP_PATH) -> None:
    """Write the declared party to disk in the canonical LIST form. Accepts either the list
    form or an old dict-keyed map (normalized via ``_as_entry_list``). ``ensure_ascii=False``
    so accented names stay readable. Entries without a usable main_name are skipped; pronouns
    are written as given (empty -> the loader defaults them to they/them)."""
    out = []
    for e in _as_entry_list(mapping):
        main = e.get("main_name")
        if not isinstance(main, str) or not main.strip():
            continue
        raw_last = e.get("last_name")
        out.append({
            "player": (e.get("player") or "").strip(),
            "main_name": main.strip(),
            "last_name": raw_last.strip() if isinstance(raw_last, str) and raw_last.strip() else None,
            "aliases": [a.strip() for a in (e.get("aliases") or []) if isinstance(a, str) and a.strip()],
            "pronouns": [p.strip() for p in (e.get("pronouns") or []) if isinstance(p, str) and p.strip()],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_character_lookup(player_map) -> dict:
    """Invert the party into ``{recognition_name_lower: player}`` for the extractor -- every
    cross-product name (incl. 'Kriggy Krieger') maps to its player. Player keeps its original
    casing. If the same name is declared under two players, the last one wins (harmless
    user-config edge case)."""
    lookup = {}
    for dc in declared_characters(player_map):
        for name in dc.recognition:
            lookup[name] = dc.player
    return lookup


def build_pronoun_lookup(player_map) -> dict:
    """``{recognition_name_lower: pronouns}`` for the extractor, so a detail about a declared
    character can be authored with that character's declared pronouns."""
    lookup = {}
    for dc in declared_characters(player_map):
        for name in dc.recognition:
            lookup[name] = dc.pronouns
    return lookup


def declared_groups_with_players(player_map) -> list:
    """Each declared character as a 3-tuple ``(player_lower, main_name, [recognition_lower])``,
    in config order (multi-PC -> a player appears in several tuples). Thin wrapper over
    :func:`declared_characters`; kept for the reconciler's declared-merge floor and its tests.
    The recognition list now INCLUDES the last-name cross-product."""
    return [(dc.player.strip().lower(), dc.main_name, dc.recognition)
            for dc in declared_characters(player_map)]


def declared_groups(player_map) -> list:
    """The recognition name-lists for the reconciler's declared-merge floor -- one per declared
    character (see :func:`declared_groups_with_players`). Membership is order-independent (the
    caller set-ifies for lookup); the heading defaults to that character's ``main_name``."""
    return [names for _, _, names in declared_groups_with_players(player_map)]
