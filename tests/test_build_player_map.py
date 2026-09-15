"""The mandatory step-1 party builder (scripts/build_player_map.py, run-1.17 §B).

Reads the speaker map's people, prompts per person for their character(s) -- main name,
last name, aliases, pronouns -- and emits the canonical list form. Injectable input/print
so it runs without stdin. Offline, no API, no PII.
"""

import pytest

from scripts.build_player_map import (
    main, player_roster, prompt_for_characters, _parse_list, _parse_pronouns,
)


def _scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_player_roster_dedups_case_insensitively_and_keeps_order():
    sm = {"+1": "Sam", "+2": "Conrad", "exporter": "Hannah", "+3": "sam", "+4": "  "}
    assert player_roster(sm) == ["Sam", "Conrad", "Hannah"]


def test_player_roster_empty():
    assert player_roster({}) == []


def test_parse_pronouns_forms_and_default():
    assert _parse_pronouns("she/her") == ["she", "her"]
    assert _parse_pronouns("they, them") == ["they", "them"]
    assert _parse_pronouns("  ze / zir ") == ["ze", "zir"]
    assert _parse_pronouns("") == ["they", "them"]      # default


def test_parse_list():
    assert _parse_list("Krigius, Kriggy K") == ["Krigius", "Kriggy K"]
    assert _parse_list("") == []


def test_prompt_multi_pc_and_skip():
    # Sam: two PCs; Conrad: skipped (not a player).
    answers = [
        "Kriggy", "Krieger", "Krigius", "he/him",   # Sam's 1st PC
        "Baldric", "", "", "",                       # Sam's 2nd PC (blank last/aliases/pronouns)
        "",                                          # no more for Sam
        "-",                                         # skip Conrad
    ]
    out = prompt_for_characters(["Sam", "Conrad"],
                                input_fn=_scripted(answers), print_fn=lambda *a, **k: None)
    assert out == [
        {"player": "Sam", "main_name": "Kriggy", "last_name": "Krieger",
         "aliases": ["Krigius"], "pronouns": ["he", "him"]},
        {"player": "Sam", "main_name": "Baldric", "last_name": None,
         "aliases": [], "pronouns": ["they", "them"]},          # blank pronouns -> default
    ]


def test_prompt_first_blank_skips_person():
    out = prompt_for_characters(["Sam"], input_fn=_scripted([""]), print_fn=lambda *a, **k: None)
    assert out == []


def test_prompt_dash_skips_person():
    out = prompt_for_characters(["Sam"], input_fn=_scripted(["-"]), print_fn=lambda *a, **k: None)
    assert out == []


def test_main_missing_speaker_map_exits_cleanly():
    # A missing speaker map must give a friendly, actionable message -- not a raw traceback.
    with pytest.raises(SystemExit) as exc:
        main(["--speaker-map", "does/not/exist.json"])
    assert "Speaker map not found" in str(exc.value)


def test_main_bad_json_speaker_map_exits_cleanly(tmp_path):
    bad = tmp_path / "sm.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["--speaker-map", str(bad)])
    assert "not valid JSON" in str(exc.value)
