"""Tests for evals/gbf/__main__.py -- the CLI. Fully offline (uses --no-model so no client is built;
missing/invalid gold/ranks are clean errors)."""

import json
from pathlib import Path

import pytest

from evals.gbf import DEFAULT_GBF_MODEL, DEFAULT_GOLD_PATH, DEFAULT_RANKS_PATH
from evals.gbf import __main__ as cli

main = cli.main
parse_args = cli.parse_args

_WOF = ("# WOF\n\n## Locations\n\n### <a id=\"aldenburg\"></a>Aldenburg\n\nA town.\n\n"
        "## History\n\n- <a id=\"ev-a\"></a>**Event A** — first.\n"
        "- <a id=\"ev-b\"></a>**Event B** — second.\n")
_GOLD = _WOF
_RANKS = "ev-a: 1\nev-b: 2\n"


def _files(tmp_path):
    wof = tmp_path / "wof.md"; wof.write_text(_WOF, encoding="utf-8")
    gold = tmp_path / "gold.md"; gold.write_text(_GOLD, encoding="utf-8")
    ranks = tmp_path / "ranks.yaml"; ranks.write_text(_RANKS, encoding="utf-8")
    return wof, gold, ranks


def test_parse_args_defaults():
    ns = parse_args(["some.md"])
    assert ns.wof_path == "some.md"
    assert ns.out is None and ns.votes is None and ns.no_model is False
    assert ns.gold == DEFAULT_GOLD_PATH and ns.ranks == DEFAULT_RANKS_PATH
    assert ns.max_workers is None                        # unset -> _default_workers() decides


def test_parse_args_max_workers():
    assert parse_args(["some.md", "--max-workers", "3"]).max_workers == 3


def test_default_workers_honors_env(monkeypatch):
    from evals.gbf import DEFAULT_MAX_WORKERS, ENV_MAX_WORKERS
    monkeypatch.delenv(ENV_MAX_WORKERS, raising=False)
    assert cli._default_workers() == DEFAULT_MAX_WORKERS
    monkeypatch.setenv(ENV_MAX_WORKERS, "2")
    assert cli._default_workers() == 2
    monkeypatch.setenv(ENV_MAX_WORKERS, "not-an-int")   # bad value -> default
    assert cli._default_workers() == DEFAULT_MAX_WORKERS


def test_main_no_model_writes_schema_valid_json(tmp_path, capsys):
    jsonschema = pytest.importorskip("jsonschema")
    wof, gold, ranks = _files(tmp_path)
    out = tmp_path / "g.json"
    rc = main(["--no-model", "--gold", str(gold), "--ranks", str(ranks), "--out", str(out), str(wof)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    schema = json.loads((Path(__file__).resolve().parent.parent / "gbf-result.schema.json")
                        .read_text(encoding="utf-8"))
    jsonschema.validate(data, schema)
    assert data["eval"] == "gbf" and data["summary"]["complete"] is False
    # stdout also carries the JSON
    assert '"eval": "gbf"' in capsys.readouterr().out


def test_main_no_model_records_gbf_default_judge(tmp_path):
    wof, gold, ranks = _files(tmp_path)
    out = tmp_path / "g.json"
    main(["--no-model", "--gold", str(gold), "--ranks", str(ranks), "--out", str(out), str(wof)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["graders"][0]["name"] == DEFAULT_GBF_MODEL


def test_main_unset_env_model_uses_gbf_default(tmp_path, monkeypatch):
    # Credentials set but WIKITIZER_GBF_MODEL unset/blank -> the pinned GBF judge, NOT the shared
    # Qwen DEFAULT_MODEL. Capture the client main() builds; grade offline so nothing hits the network.
    monkeypatch.setenv("LLM_OPENAI_BASE_URL", "https://x/api/v1")
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "k")
    monkeypatch.setenv("WIKITIZER_GBF_MODEL", "")      # blank also beats a stray .env value
    seen = {}
    real_grade = cli.grade_file

    def spy_grade(wof_path, gold, ranks, **kw):
        seen["client"] = kw["model_client"]
        kw.update(use_model=False, model_client=None)
        return real_grade(wof_path, gold, ranks, **kw)

    monkeypatch.setattr(cli, "grade_file", spy_grade)
    wof, gold, ranks = _files(tmp_path)
    assert main(["--gold", str(gold), "--ranks", str(ranks), str(wof)]) == 0
    assert seen["client"] is not None and seen["client"].model == DEFAULT_GBF_MODEL


def test_main_missing_gold_returns_2(tmp_path):
    wof, _gold, ranks = _files(tmp_path)
    rc = main(["--no-model", "--gold", str(tmp_path / "nope.md"), "--ranks", str(ranks), str(wof)])
    assert rc == 2


def test_main_missing_ranks_returns_2(tmp_path):
    wof, gold, _ranks = _files(tmp_path)
    rc = main(["--no-model", "--gold", str(gold), "--ranks", str(tmp_path / "nope.yaml"), str(wof)])
    assert rc == 2


def test_main_invalid_ranks_returns_2(tmp_path):
    wof, gold, _ranks = _files(tmp_path)
    bad = tmp_path / "bad.yaml"; bad.write_text("- just\n- a list\n", encoding="utf-8")
    rc = main(["--no-model", "--gold", str(gold), "--ranks", str(bad), str(wof)])
    assert rc == 2


def test_main_malformed_yaml_ranks_returns_2(tmp_path):
    # Syntactically invalid YAML must be a clean CLI error (return 2), not an uncaught traceback.
    wof, gold, _ranks = _files(tmp_path)
    bad = tmp_path / "bad.yaml"; bad.write_text("ev-a: [1, 2\n", encoding="utf-8")   # unbalanced bracket
    rc = main(["--no-model", "--gold", str(gold), "--ranks", str(bad), str(wof)])
    assert rc == 2
