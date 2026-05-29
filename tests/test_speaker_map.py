import json

import pytest

from speaker_map import load_speaker_map
from scripts.build_speaker_map import discover_phone_numbers


# --- loader -----------------------------------------------------------------

def test_load_speaker_map(tmp_path):
    # tmp_path is a pytest built-in fixture: a fresh temp dir that auto-cleans
    # after the test, so we never touch the real config file.
    config_file = tmp_path / "speaker_map.json"
    config_file.write_text('{"+15555550100": "Alice", "exporter": "Bob"}')
    result = load_speaker_map(str(config_file))
    assert result == {"+15555550100": "Alice", "exporter": "Bob"}


def test_load_speaker_map_missing_file():
    with pytest.raises(FileNotFoundError):
        load_speaker_map("does_not_exist.json")


def test_load_speaker_map_malformed_json(tmp_path):
    config_file = tmp_path / "speaker_map.json"
    config_file.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_speaker_map(str(config_file))


# --- discovery (build script) ----------------------------------------------

def test_discover_finds_phone_numbers(tmp_path):
    chat = tmp_path / "chat.txt"
    chat.write_text(
        "+15555550100 01/02/2024 10:30:00\n"
        "Hey what's the lore on Lake Mundi\n"
        "+16156187446 01/02/2024 10:31:00\n"
        "It's a massive central lake\n"
    )
    samples = discover_phone_numbers([str(chat)])
    assert set(samples.keys()) == {"+15555550100", "+16156187446"}
    assert "Hey what's the lore" in samples["+15555550100"][0]


def test_discover_ignores_exporter_lines(tmp_path):
    # Exporter messages have a bare timestamp (no phone prefix) and must not
    # be picked up as phone numbers.
    chat = tmp_path / "chat.txt"
    chat.write_text(
        "01/02/2024 10:30:00\n"
        "This is the exporter talking\n"
        "+15555550100 01/02/2024 10:31:00\n"
        "And this is someone else\n"
    )
    samples = discover_phone_numbers([str(chat)])
    assert list(samples.keys()) == ["+15555550100"]


def test_discover_collects_samples_across_files(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("+15555550100 01/02/2024 10:30:00\nmessage in file a\n")
    b.write_text("+15555550100 03/04/2024 11:00:00\nmessage in file b\n")
    samples = discover_phone_numbers([str(a), str(b)])
    assert len(samples["+15555550100"]) == 2
