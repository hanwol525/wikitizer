"""Tests for main.py (Phase 4.6 Part 4: the argparse CLI).

main.py had no tests before this brief. Everything here is offline: restricted_path is
pure, parse_args does only a cheap is_dir stat on --output, and the main() tests stub
every I/O boundary main touches
(Orchestrator, load_dotenv, both config loaders) so nothing hits the network, no API
key is needed, and no real config file is read -- config/speaker_map.json is
gitignored, so a fresh clone has none and an unstubbed main() would raise
FileNotFoundError.
"""

import logging

import pytest

import main as main_mod
from main import parse_args, restricted_path, DEFAULT_OUTPUT_PATH
from orchestrator import WikiOutput


# --- parse_args -------------------------------------------------------------

def test_parse_args_minimal_defaults():
    args = parse_args(["--files", "a.txt"])
    assert args.files == ["a.txt"]
    assert args.output == DEFAULT_OUTPUT_PATH     # "output/wiki.md"
    assert args.exclude_sources == []             # omitted -> no restricted doc


def test_parse_args_multiple_files():
    args = parse_args(["--files", "a.txt", "b.txt", "c.txt"])
    assert args.files == ["a.txt", "b.txt", "c.txt"]


def test_parse_args_output_override():
    assert parse_args(["--files", "a.txt", "--output", "/tmp/w.md"]).output == "/tmp/w.md"


def test_parse_args_exclude_sources_dashes_become_underscores():
    # argparse maps --exclude-sources -> args.exclude_sources
    args = parse_args(["--files", "a.txt", "--exclude-sources", "dm.txt", "solo.txt"])
    assert args.exclude_sources == ["dm.txt", "solo.txt"]


def test_parse_args_files_is_required():
    with pytest.raises(SystemExit):      # argparse errors out (exit 2)
        parse_args([])


def test_parse_args_exclude_sources_with_no_values_errors():
    # nargs="+": passing the flag with nothing is a mistake, not a silent no-op.
    with pytest.raises(SystemExit):
        parse_args(["--files", "a.txt", "--exclude-sources"])


def test_parse_args_unknown_flag_errors():
    with pytest.raises(SystemExit):
        parse_args(["--files", "a.txt", "--nope"])


def test_parse_args_output_that_is_a_directory_errors(tmp_path):
    # Fail cheap: an existing-dir --output would otherwise crash at write time, AFTER
    # the whole paid run. Rejected up front (exit 2), like other CLI mistakes.
    with pytest.raises(SystemExit):
        parse_args(["--files", "a.txt", "--output", str(tmp_path)])


# --- restricted_path --------------------------------------------------------

def test_restricted_path_matches_the_old_hardcoded_default():
    # The derivation must reproduce Part 2's RESTRICTED_OUTPUT_PATH exactly, so the
    # default behaviour is unchanged by the CLI. This is the anchor test.
    assert str(restricted_path(DEFAULT_OUTPUT_PATH)) == "output/wiki_restricted.md"


def test_restricted_path_follows_output_wherever_it_points():
    assert str(restricted_path("/tmp/mywiki.md")) == "/tmp/mywiki_restricted.md"


def test_restricted_path_handles_a_path_with_no_extension():
    assert str(restricted_path("wiki")) == "wiki_restricted"


# --- main() -----------------------------------------------------------------

@pytest.fixture
def stub_pipeline(monkeypatch):
    """Neuter every I/O boundary main() touches, leaving only main()'s own logic
    under test. Call it with the WikiOutput you want run() to return; it gives back
    a `calls` dict recording what run() received."""
    calls = {}

    def _install(result):
        class _StubOrchestrator:
            def __init__(self, *a, **kw):
                pass

            def run(self, files, config, exclude_sources=None):
                calls["files"] = files
                calls["exclude_sources"] = exclude_sources
                return result

        monkeypatch.setattr(main_mod, "Orchestrator", _StubOrchestrator)
        monkeypatch.setattr(main_mod, "load_dotenv", lambda: None)
        monkeypatch.setattr(main_mod, "load_speaker_map", lambda p: {"exporter": "H"})
        monkeypatch.setattr(main_mod, "load_crosslink_words",
                            lambda p: {"require_article": [], "never_link": []})
        # load_player_map is tolerant (missing -> {}), but stub it so the tests stay
        # hermetic even if a real config/player_map.json exists on the dev's machine.
        monkeypatch.setattr(main_mod, "load_player_map", lambda p: {})
        return calls

    return _install


def test_main_writes_full_only_without_exclusions(tmp_path, stub_pipeline):
    stub_pipeline(WikiOutput(full="FULL WIKI"))
    out = tmp_path / "wiki.md"
    main_mod.main(["--files", "logs/a.txt", "--output", str(out)])
    assert out.read_text(encoding="utf-8") == "FULL WIKI"
    assert not (tmp_path / "wiki_restricted.md").exists()   # none requested -> none written


def test_main_writes_both_docs_at_the_derived_path(tmp_path, stub_pipeline):
    calls = stub_pipeline(WikiOutput(full="FULL", restricted="RESTRICTED"))
    out = tmp_path / "wiki.md"
    main_mod.main(["--files", "logs/a.txt", "logs/dm.txt", "--output", str(out),
                   "--exclude-sources", "dm.txt"])
    assert out.read_text(encoding="utf-8") == "FULL"
    rout = tmp_path / "wiki_restricted.md"                  # derived, beside --output
    assert rout.read_text(encoding="utf-8") == "RESTRICTED"
    # and the parsed args reached run() unchanged
    assert calls["files"] == ["logs/a.txt", "logs/dm.txt"]
    assert calls["exclude_sources"] == ["dm.txt"]


def test_main_writes_an_empty_restricted_doc_with_a_warning(tmp_path, stub_pipeline, caplog):
    # "" is NOT None: exclusions WERE requested but nothing public survived. Still
    # write it (an empty wiki is a real answer) and say so loudly. This branch has
    # been untested since Part 2.
    stub_pipeline(WikiOutput(full="FULL", restricted=""))
    out = tmp_path / "wiki.md"
    with caplog.at_level(logging.WARNING):
        main_mod.main(["--files", "logs/a.txt", "--output", str(out),
                       "--exclude-sources", "a.txt"])
    rout = tmp_path / "wiki_restricted.md"
    assert rout.exists() and rout.read_text(encoding="utf-8") == ""
    assert "EMPTY" in caplog.text


def test_main_creates_a_missing_output_directory(tmp_path, stub_pipeline):
    stub_pipeline(WikiOutput(full="FULL"))
    out = tmp_path / "deep" / "nested" / "wiki.md"
    main_mod.main(["--files", "logs/a.txt", "--output", str(out)])
    assert out.read_text(encoding="utf-8") == "FULL"


def test_main_falls_back_to_the_default_output_path(tmp_path, stub_pipeline, monkeypatch):
    # chdir into tmp_path so the RELATIVE default ("output/wiki.md") lands in a temp
    # dir instead of the repo.
    stub_pipeline(WikiOutput(full="FULL"))
    monkeypatch.chdir(tmp_path)
    main_mod.main(["--files", "logs/a.txt"])
    assert (tmp_path / "output" / "wiki.md").read_text(encoding="utf-8") == "FULL"


# --- --confirm-players ------------------------------------------------------

def test_parse_args_confirm_players_flag():
    assert parse_args(["--files", "a.txt"]).confirm_players is False
    assert parse_args(["--files", "a.txt", "--confirm-players"]).confirm_players is True


def test_main_confirm_players_builds_and_saves_config(tmp_path, stub_pipeline, monkeypatch):
    # Wiring test: with the flag set, main() runs the confirm helper on the discovered
    # PCs and saves the result. The confirm logic itself lives in test_confirm_players.py,
    # so here we stub it and assert main() calls save_player_map with its output.
    stub_pipeline(WikiOutput(full="FULL", characters=["<pc objects>"]))
    saved = {}
    monkeypatch.setattr(main_mod, "confirm_player_map",
                        lambda pcs, existing: {"Sam": ["Kriggy"]})
    monkeypatch.setattr(main_mod, "save_player_map",
                        lambda mapping, path: saved.update(mapping=mapping, path=path))
    main_mod.main(["--files", "logs/a.txt", "--output", str(tmp_path / "wiki.md"),
                   "--confirm-players"])
    assert saved["mapping"] == {"Sam": ["Kriggy"]}
    assert saved["path"] == main_mod.PLAYER_MAP_PATH


def test_main_does_not_confirm_without_the_flag(tmp_path, stub_pipeline, monkeypatch):
    stub_pipeline(WikiOutput(full="FULL"))
    called = {"save": False}
    monkeypatch.setattr(main_mod, "save_player_map",
                        lambda *a, **k: called.__setitem__("save", True))
    main_mod.main(["--files", "logs/a.txt", "--output", str(tmp_path / "wiki.md")])
    assert called["save"] is False   # no --confirm-players -> config left untouched
