"""Tests for scripts/build_imessage_speaker_map.py -- the imessage sender discovery.

`discover_senders` reuses the real parser, so `Me` -> "exporter" and contact names pass
through; `prompt_for_names` is driven by injectable input/print so it runs without stdin.
Offline, no API, no PII.
"""

from scripts.build_imessage_speaker_map import discover_senders, prompt_for_names, save
from collections import Counter


def write(tmp_path, lines, name="chat.txt"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


SAMPLE = [
    "May 24, 2020 10:14:22 AM",
    "Me",
    "hi there",
    "",
    "May 24, 2020 10:15:11 AM (Read by you after 1 minute, 6 seconds)",
    "Palouie",
    "hello",
    "",
    "May 24, 2020 10:16:11 AM",
    "Palouie",
    "and again",
    "",
    "May 24, 2020 10:17:11 AM",
    "Helen Corey",
    "luv u",
]


def fake_input(answers):
    it = iter(answers)
    return lambda prompt="": next(it)


# --- discover_senders ------------------------------------------------------- #
def test_discover_counts_senders_me_becomes_exporter(tmp_path):
    counts = discover_senders([write(tmp_path, SAMPLE)])
    assert counts == Counter({"exporter": 1, "Palouie": 2, "Helen Corey": 1})


def test_discover_across_multiple_files(tmp_path):
    a = write(tmp_path, SAMPLE, name="a.txt")
    b = write(tmp_path, ["May 24, 2020 10:20:00 AM", "Palouie", "third"], name="b.txt")
    counts = discover_senders([a, b])
    assert counts["Palouie"] == 3
    assert counts["exporter"] == 1


def test_discover_empty_when_not_imessage(tmp_path):
    # A legacy-shaped file yields no imessage messages -> empty Counter.
    legacy = write(tmp_path, [",+15551230000", "-" * 100, "hi", "+15551230000 01/01/2024 12:00:05"])
    assert discover_senders([legacy]) == Counter()


# --- prompt_for_names ------------------------------------------------------- #
def test_prompt_keeps_names_and_sets_exporter():
    counts = Counter({"exporter": 1, "Palouie": 2, "Helen Corey": 1})
    # most_common order: Palouie (2), then Helen Corey (1). Enter (keep) for both, then exporter name.
    mapping = prompt_for_names(counts, input_fn=fake_input(["", "", "Hannah"]),
                               print_fn=lambda *a, **k: None)
    assert mapping == {"Palouie": "Palouie", "Helen Corey": "Helen Corey", "exporter": "Hannah"}


def test_prompt_can_canonicalize_a_contact_name():
    counts = Counter({"exporter": 1, "Palouie": 2})
    mapping = prompt_for_names(counts, input_fn=fake_input(["Lou", "Hannah"]),
                               print_fn=lambda *a, **k: None)
    assert mapping["Palouie"] == "Lou"          # renamed to the canonical roster name
    assert mapping["exporter"] == "Hannah"


def test_prompt_does_not_mutate_caller_counter():
    counts = Counter({"exporter": 1, "Palouie": 2})
    prompt_for_names(counts, input_fn=fake_input(["", "Hannah"]), print_fn=lambda *a, **k: None)
    assert counts == Counter({"exporter": 1, "Palouie": 2})   # "exporter" not popped off the original


def test_prompt_blank_exporter_leaves_it_unset():
    counts = Counter({"Palouie": 1})
    mapping = prompt_for_names(counts, input_fn=fake_input(["", ""]),
                               print_fn=lambda *a, **k: None)
    assert "exporter" not in mapping            # blank -> not recorded


# --- save ------------------------------------------------------------------- #
def test_save_writes_name_keyed_json(tmp_path):
    import json
    out = tmp_path / "sub" / "speaker_map.imessage.json"
    save({"Palouie": "Lou", "exporter": "Hannah"}, str(out))
    assert json.loads(out.read_text(encoding="utf-8")) == {"Palouie": "Lou", "exporter": "Hannah"}
