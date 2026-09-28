"""Tests for evals/rubric/__main__.py -- the CLI. Fully offline (a BenignFake stands in for
the model client; no network, no key)."""

import json
import re
from pathlib import Path

import pytest

from evals.rubric import __main__ as cli

_ROOT = Path(__file__).resolve().parent.parent

RUBRIC_YAML = "sections:\n  - Locations\nlocations:\n  - Aldenburg\n"
WOF = ("# W\n\nsub\n\n---\n\n## Locations\n\n"
       "### <a id=\"a\"></a>Aldenburg\n\nA town on the river.[^1]\n\n"
       "## Footnotes\n\n[^1]: `Aldenburg is a town on the river` — B, dm.txt\n")


class BenignFake:
    """Sizes its {"results":[...]} reply to the number of numbered rows in the prompt (1 for
    the single-entity support call) and answers positively on every field, so any rubric
    check runs cleanly end-to-end."""

    model = "qwen/qwen3-8b"
    provider = "openrouter"
    temperature = 0.6
    thinking = False
    base_url = None
    _ROW = re.compile(r"^\s*\d+\.", re.M)

    def complete(self, system, user):
        n = len(self._ROW.findall(user)) or 1
        rec = {"present": True, "match": 1, "pc_stated": True, "asterisk": True,
               "supported": True, "span": "a supporting span"}
        return json.dumps({"results": [dict(rec) for _ in range(n)]})


def _write(tmp_path, wof_text=WOF, rubric_text=RUBRIC_YAML):
    wof = tmp_path / "wiki.md"
    wof.write_text(wof_text, encoding="utf-8")
    rub = tmp_path / "rubric.yaml"
    rub.write_text(rubric_text, encoding="utf-8")
    return wof, rub


def test_parse_args_defaults_and_flags():
    ns = cli.parse_args(["wiki.md"])
    assert ns.wof_path == "wiki.md" and ns.out is None and ns.votes is None
    assert ns.no_model is False and ns.rubric.endswith("rubric.yaml")
    ns2 = cli.parse_args(["w.md", "--out", "o.json", "--votes", "5", "--no-model",
                          "--rubric", "r.yaml"])
    assert ns2.out == "o.json" and ns2.votes == 5 and ns2.no_model is True and ns2.rubric == "r.yaml"


def test_main_no_model_all_skipped(tmp_path):
    wof, rub = _write(tmp_path)
    out = tmp_path / "r.json"
    rc = cli.main([str(wof), "--no-model", "--rubric", str(rub), "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["eval"] == "rubric"
    assert data["summary"]["complete"] is False
    assert data["summary"]["presence"]["score_display"] == "0/0"
    assert data["summary"]["sourcing"]["score_display"] == "0/0"
    assert data["items"] and all(it["status"] == "skipped" for it in data["items"])
    assert data["graders"][0]["engine"] == "model"


def test_main_model_path_schema_valid(tmp_path, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    wof, rub = _write(tmp_path)
    out = tmp_path / "r.json"
    monkeypatch.setattr(cli, "build_model_client", lambda prefix="RUBRIC": BenignFake())
    rc = cli.main([str(wof), "--rubric", str(rub), "--out", str(out), "--votes", "1"])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["complete"] is True
    schema = json.loads((_ROOT / "rubric-result.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(data, schema)


def test_main_missing_rubric_is_clean_error(tmp_path):
    wof, _ = _write(tmp_path)
    rc = cli.main([str(wof), "--rubric", str(tmp_path / "nope.yaml"), "--no-model"])
    assert rc == 2                                     # clean error, not a traceback


def test_main_falls_back_to_skipped_without_a_client(tmp_path, monkeypatch):
    wof, rub = _write(tmp_path)
    monkeypatch.setattr(cli, "build_model_client", lambda prefix="RUBRIC": None)
    rc = cli.main([str(wof), "--rubric", str(rub)])
    assert rc == 0                                     # no client -> skipped path, still a clean exit


def test_main_empty_rubric_is_clean_error(tmp_path):
    # an empty checklist grades nothing (and would emit a schema-invalid items:[] result)
    wof, _ = _write(tmp_path)
    rub = tmp_path / "empty.yaml"
    rub.write_text("", encoding="utf-8")
    rc = cli.main([str(wof), "--rubric", str(rub), "--no-model"])
    assert rc == 2                                     # clean config error, not an off-spec result
