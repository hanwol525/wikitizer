"""The declared party: a user-owned ``{player: [character name, alias, ...]}`` map.

Who plays which character is a fact only the group knows -- no automatic heuristic can
recover it reliably (a chat where lore is narrated by a few people makes "who voices a
character" a bad proxy for "who plays it"). So the party is USER config, the same
category as ``config/speaker_map.json``: gitignored (real names are PII), and NOT
hardcoded into source, which keeps the app conversation-agnostic.

Shape (``config/player_map.json``)::

    {"Sam": ["Kriggy", "Krigius Krieger", "Ambrose Chamberlain"], "Hannah": ["CJ"]}

Each key is a real player; its value is that player's character plus any aliases. The
characters extractor stamps ``player_name`` authoritatively from this, and the reconciler
merges the names grouped under one player as a single character.

The loader is TOLERANT of a missing file (returns ``{}`` -> the app runs with no party
declared, falling back to the LLM's guess), but STRICT on a file that parses to the wrong
shape -- a silent wrong shape would quietly disable the whole feature.
"""

import json


DEFAULT_PLAYER_MAP_PATH = "config/player_map.json"


def _coerce_names(value) -> list:
    """A player's value may be a single name (str) or a list of names; normalize to a
    list of non-empty strings. Raises ValueError on any other shape."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(
            f"player_map values must be a string or a list of strings, got {type(value).__name__}."
        )
    names = []
    for n in value:
        if not isinstance(n, str):
            raise ValueError(f"player_map character names must be strings, got {type(n).__name__}.")
        if n.strip():
            names.append(n)
    return names


def load_player_map(path: str = DEFAULT_PLAYER_MAP_PATH) -> dict:
    """Load the declared party. Missing file -> ``{}`` (no party declared). A file that
    PARSED but is the wrong shape RAISES (a non-dict top level, or a value that isn't a
    string / list-of-strings) -- silently accepting it would disable the feature without
    telling anyone. Returns a ``{player: [name, ...]}`` dict with values normalized to
    lists of non-empty strings."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"player_map.json must be a JSON object, got {type(data).__name__}.")

    out = {}
    for player, names in data.items():
        if not isinstance(player, str) or not player.strip():
            raise ValueError(f"player_map keys (players) must be non-empty strings, got {player!r}.")
        out[player] = _coerce_names(names)
    return out


def save_player_map(mapping: dict, path: str = DEFAULT_PLAYER_MAP_PATH) -> None:
    """Write the declared party back to disk (used by ``--confirm-players``). Values are
    normalized to lists; ``ensure_ascii=False`` so accented names stay readable."""
    normalized = {player: _coerce_names(names) for player, names in mapping.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_character_lookup(player_map: dict) -> dict:
    """Invert the party into a normalized ``{character_name_lower: player}`` lookup for
    the extractor. Keys are ``strip().lower()`` (matching the roster convention). If the
    same character name is (mis)declared under two players, the last one wins -- a
    user-config edge case, harmless to resolve deterministically."""
    lookup = {}
    for player, names in player_map.items():
        for name in _coerce_names(names):
            lookup[name.strip().lower()] = player
    return lookup


def declared_groups_with_players(player_map: dict) -> list:
    """Like :func:`declared_groups`, but pairs each group with the (lowercased) PLAYER who
    owns it: ``[(player_lower, [name_lower, ...]), ...]`` for every player listing >= 1 name,
    in player_map order. The owning player key lets the reconciler fold a character
    mis-named after the player -- the player minted as their own PC -- into that player's
    declared character. Same normalization/order as :func:`declared_groups`."""
    out = []
    for player, names in player_map.items():
        seen = set()
        ordered = []
        for n in _coerce_names(names):
            key = n.strip().lower()
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        if ordered:
            out.append((player.strip().lower(), ordered))
    return out


def declared_groups(player_map: dict) -> list:
    """The character-identity groups for the reconciler's declared-merge floor: one
    normalized, ORDER-PRESERVING name-list per player key that lists at least one name.
    Each group's names are aliases of ONE character (the user's declaration), so the
    reconciler merges any entries whose name/alias falls in the same group. Order is
    kept (first-listed wins) so the merge can prefer the user's first name as the
    heading; membership is order-independent (the caller sets-ifies for lookup)."""
    return [names for _, names in declared_groups_with_players(player_map)]
