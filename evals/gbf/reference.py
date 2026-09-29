"""GBF's two reference inputs: the gold WOF and the event-rank map.

Both are gitignored campaign content and both are hard-error-if-missing -- the gold IS the
faithfulness/ordering reference and the ranks ARE the ordering ground truth, so a missing file is a
config error the CLI surfaces cleanly (``return 2``), never a silent skip. Committed fixtures
(``output/gol-lore-full.md`` is the real gold the FC/rubric golden tests already lean on; the ranks
have a fabricated ``example_event_ranks.yaml`` next to this package) stand in for offline tests.

Pure loading only; the parsing/matching lives downstream (``pair.py`` / ``ordering.py``).
"""

from pathlib import Path
from typing import Dict

import yaml

from evals.common.parse import ParsedWOF, parse_wof


def load_gold(path) -> ParsedWOF:
    """Read the gold WOF (utf-8) and return its ``ParsedWOF``. Raises ``FileNotFoundError`` when
    absent -- it is the reference GBF grades against."""
    text = Path(path).read_text(encoding="utf-8")   # missing file -> FileNotFoundError
    return parse_wof(text)


def load_ranks(path) -> Dict[str, int]:
    """Read the event-rank YAML -> ``{gold-event-slug: rank}``. Co-temporal events share a rank;
    events omitted here (e.g. anything under "Could Not Place") are excluded from the ordering
    check by their absence.

    Raises ``FileNotFoundError`` when absent (the ranks ARE the ordering ground truth) and
    ``ValueError`` on a wrong shape (not a mapping, an empty/blank key, or a non-int rank) so a
    typo'd file fails loudly rather than silently disabling ordering."""
    with Path(path).open(encoding="utf-8") as f:    # missing file -> FileNotFoundError
        data = yaml.safe_load(f)
    if data is None:                                 # an empty file -> no ranked events
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"event-ranks file must be a mapping of slug -> rank, got {type(data).__name__}")
    ranks: Dict[str, int] = {}
    for key, value in data.items():
        slug = str(key).strip()
        if not slug:
            raise ValueError("event-ranks file has an empty slug key")
        # A YAML bool is an int subclass -- reject it so ``True: 1`` can't masquerade as a rank.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"rank for {slug!r} must be an integer, got {value!r}")
        ranks[slug] = value
    return ranks
