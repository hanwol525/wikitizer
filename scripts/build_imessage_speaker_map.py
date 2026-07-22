"""One-off helper to build a NAME-keyed speaker map for imessage-exporter exports.

Sibling of ``scripts/build_speaker_map.py`` (which discovers PHONE numbers in the legacy
copy-paste logs). imessage-exporter resolves contacts to NAMES, so this discovers the
distinct sender names instead and builds a ``{sender_name: canonical_name, "exporter":
you}`` map -- whose VALUES become the anti-conflation roster (``speaker_map.values()``),
so they should line up with the player keys in ``config/player_map.json``.

Run it directly on your exported campaign files, then point the pipeline at the map::

    python scripts/build_imessage_speaker_map.py logs/imessage/*.txt -o config/speaker_map.imessage.json
    python main.py --files logs/imessage/*.txt --input-format imessage \\
        --speaker-map config/speaker_map.imessage.json

Flow: ``discover_senders() -> prompt_for_names() -> save()`` -- same shape as the legacy
script, so it maps cleanly onto a UI later (swap the prompt step for a form).
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

# Allow `python scripts/this.py` to import the package: run that way, sys.path[0] is the
# scripts/ dir, not the repo root, so the `parsers` package wouldn't be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parsers.imessage_export_parser import parse_imessage_export  # noqa: E402

DEFAULT_OUTPUT = "config/speaker_map.imessage.json"


def discover_senders(filepaths) -> Counter:
    """Count messages per distinct sender across the imessage-exporter files.

    Reuses the real parser with an EMPTY speaker map, so ``Me`` resolves to the literal
    ``"exporter"`` (the reserved key) and every other sender passes through as its raw
    contact name -- and read receipts / tapbacks / attachments are already handled
    correctly. Returns a ``Counter`` of ``{sender: message_count}``.
    """
    counts: Counter = Counter()
    for filepath in filepaths:
        for msg in parse_imessage_export(filepath, {}):
            counts[msg.sender] += 1
    return counts


def prompt_for_names(counts, input_fn=input, print_fn=print) -> dict:
    """Show each discovered sender + message count and ask for the canonical name to
    record. The reserved ``"exporter"`` pseudo-sender (your own ``Me`` messages) is asked
    for specially. Pressing Enter keeps a contact name as-is. Returns a
    ``{sender: canonical}`` map with an ``"exporter"`` entry (``input_fn``/``print_fn``
    are injectable so this is testable without real stdin)."""
    counts = Counter(counts)                       # copy: don't mutate the caller's Counter
    mapping = {}
    exporter_count = counts.pop("exporter", 0)
    print_fn(f"\nFound {len(counts)} other sender(s) across your export(s).\n")
    for sender, n in counts.most_common():
        print_fn(f"  Sender:   {sender}")
        print_fn(f"  Messages: {n}")
        name = input_fn(f"  Canonical name for {sender!r} (Enter to keep as-is): ").strip()
        mapping[sender] = name or sender
    you = input_fn(f"\nAnd you -- the exporter ({exporter_count} 'Me' messages)? Your name: ").strip()
    if you:
        mapping["exporter"] = you
    return mapping


def save(mapping, path) -> None:
    """Write the name-keyed map (utf-8, ``ensure_ascii=False`` so accented names stay readable)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="build_imessage_speaker_map",
        description="Discover senders in imessage-exporter TXT exports and build a "
                    "name-keyed speaker map.",
    )
    parser.add_argument("files", nargs="+", metavar="PATH",
                        help="imessage-exporter .txt export files (shell globs work).")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, metavar="PATH",
                        help="Where to write the map (default: %(default)s).")
    args = parser.parse_args(argv)

    counts = discover_senders(args.files)
    if not counts:
        print("No messages parsed -- are these imessage-exporter TXT exports?")
        return
    mapping = prompt_for_names(counts)

    print("\nAbout to save:")
    print(json.dumps(mapping, ensure_ascii=False, indent=2))
    if input("\nLook good? (y/n): ").strip().lower() != "y":
        print("Canceled, nothing saved.")
        return
    save(mapping, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
