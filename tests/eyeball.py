import textwrap
from models.lore import Character, HistoryEvent

WIDTH = 76  # wrap long quotes/descriptions so they stay readable


def _wrap(text, pad):
    return textwrap.fill(
        text, width=WIDTH, initial_indent=pad, subsequent_indent=pad + "  "
    )


def _quotes_block(quotes):
    if not quotes:
        return "     quotes: (none — verbatim check dropped them all)\n"
    out = ["     quotes:\n"]
    for q in quotes:
        out.append(_wrap(f'- "{q.text}"', "       ") + "\n")
        out.append(f"           — {q.speaker}  ({q.source_file})\n")
    return "".join(out)


def format_entities(entities, title):
    """Return a human-eyeball report for one extractor's output.

    Returns a string so you can print() it or write it to a file. Pairs each
    entity's claimed facts with the quotes that back it, and surfaces the field
    most likely to be wrong per type (PC/player for characters, scope/position
    for history).
    """
    out = [
        "=" * 64 + "\n",
        f"{title.upper()}  ({len(entities)} found)\n",
        "=" * 64 + "\n\n",
    ]

    for i, e in enumerate(entities, start=1):
        header = f"{i}. {e.name}"
        if isinstance(e, Character):
            header += f"   [PC · player: {e.player_name or '??'}]" if e.is_pc else "   [NPC]"
        out.append(header + "\n")
        out.append(f"   aka: {', '.join(e.aliases) if e.aliases else '—'}\n")

        if isinstance(e, HistoryEvent):
            pos = e.chronological_position
            pos_str = "(unplaced)" if pos is None else str(pos)
            # .value dodges the str(Enum) quirk in 3.9 (str(Scope.WORLD) -> "Scope.WORLD")
            out.append(f"   scope: {e.scope.value}     position: {pos_str}\n")
            out.append("   description:\n")
            out.append(_wrap(e.description, "     ") + "\n")
        else:  # Location / Character / Organization / Item / PeopleAndCultures all carry `details`
            out.append("   facts:\n")
            if e.details:
                for d in e.details:
                    out.append(_wrap("- " + d, "     ") + "\n")
            else:
                out.append("     (none)\n")

        out.append(_quotes_block(e.supporting_quotes))
        out.append("\n")

    return "".join(out)
