import json

import pytest

from speaker_map import load_speaker_map


# --- loader (format-agnostic JSON: contact names or E.164 numbers -> a name) ---

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
