"""One-off helper to build config/speaker_map.json by hand.

NOT part of the main pipeline. Run it directly when you want to (re)build the
speaker map for a new set of chat logs:

    python scripts/build_speaker_map.py

Flow: discover() -> prompt() -> save(). That shape maps cleanly onto a real UI
later -- swap the prompt step for a form; discovery and the saved JSON stay
identical.

NOTE: update the `files` list in main() to point at your real chat logs before
running. The defaults below are placeholders.
"""

import re
import json
from pathlib import Path
from collections import defaultdict

# ^ anchors to the start of each line (re.MULTILINE), so we match a phone
# number only when it begins a message header, not mid-message.
PHONE_HEADER = re.compile(r"^(\+\d+)\s+\d{2}/\d{2}/\d{4}", re.MULTILINE)


def discover_phone_numbers(filepaths: list[str]) -> dict[str, list[str]]:
    """Find every unique phone number and grab sample messages for each.

    Returns {phone_number: [sample_text, ...]}. Regex-only on purpose -- we
    don't need the real parser just to list who's in the chat.
    """
    samples = defaultdict(list)
    for filepath in filepaths:
        text = Path(filepath).read_text(encoding="utf-8")
        for match in PHONE_HEADER.finditer(text):
            phone = match.group(1)
            start = match.end()
            sample = text[start:start + 200].strip()
            samples[phone].append(sample)
    return dict(samples)


def prompt_for_names(samples: dict[str, list[str]]) -> dict[str, str]:
    """Show each phone number with a sample message and ask who it is."""
    speaker_map = {}
    print(f"\nFound {len(samples)} unique phone numbers across your chat logs.\n")
    for phone, sample_messages in sorted(samples.items()):
        print(f"  Phone:    {phone}")
        print(f"  Messages: {len(sample_messages)}")
        print(f"  Sample:   {sample_messages[0][:120]}...")
        name = input(f"  Name for {phone} (Enter to skip): ").strip()
        if name:
            speaker_map[phone] = name
        print()
    exporter = input("Finally, who exported the chat (i.e. you)? ").strip()
    speaker_map["exporter"] = exporter
    return speaker_map


def main():
    files = [
        "logs/royalty.txt",
        "logs/dndgroup.txt",
        "logs/dm convo.txt",
    ]
    samples = discover_phone_numbers(files)
    speaker_map = prompt_for_names(samples)

    print("\nAbout to save:")
    print(json.dumps(speaker_map, indent=2))
    if input("\nLook good? (y/n): ").strip().lower() != "y":
        print("Canceled, nothing saved.")
        return

    out_path = Path("config/speaker_map.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(speaker_map, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
