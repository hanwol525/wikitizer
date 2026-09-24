"""Tests for evals/fc/__main__.py -- the CLI. Fully offline.

The `--no-model` path needs no client. The model path monkeypatches build_model_client to
return a fake that answers each check with the benign label for however many candidates it
was sent, so `main` runs end-to-end without the network.
"""

import json

from evals.fc import __main__ as cli
from evals.fc.models import FCResult

WOF = """\
# Title

Subtitle prose.

---

## Locations

### <a id="alpha"></a>Alpha

Alpha near [Beta](#beta) about 200 years ago.[^1]

### <a id="beta"></a>Beta

Beta.[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""


class CountingFake:
    """Answers each candidate with the benign label for the check (inferred from the system
    prompt), sized to the number of "N." rows in the user message."""

    model, provider, temperature, thinking, base_url = (
        "fake/qwen", "openrouter", 0.6, False, "https://x/api/v1")

    def complete(self, system, user):
        import re
        n = len(re.findall(r"(?m)^\d+\.", user))
        if "group_label" in system:
            label = "group_label"
        elif "approximate_unmarked" in system:
            label = "exact_ok"
        else:
            label = "ok"
        return json.dumps({"verdicts": [label] * n})


def _write(tmp_path, text=WOF):
    p = tmp_path / "wiki.md"
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_args_defaults_and_flags():
    ns = cli.parse_args(["some.md"])
    assert ns.wof_path == "some.md" and ns.out is None and ns.no_model is False
    ns2 = cli.parse_args(["some.md", "--out", "r.json", "--votes", "5", "--no-model"])
    assert ns2.out == "r.json" and ns2.votes == 5 and ns2.no_model is True


def test_main_no_model_emits_skipped(tmp_path, capsys):
    wof = _write(tmp_path)
    out = tmp_path / "res.json"
    rc = cli.main([str(wof), "--no-model", "--out", str(out)])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    FCResult.model_validate(printed)                     # schema-valid via pydantic
    assert printed["summary"]["skipped"] == 3
    assert printed["summary"]["complete"] is False
    assert [g["engine"] for g in printed["graders"]] == ["mechanical"]
    # And it was written to --out identically.
    assert json.loads(out.read_text(encoding="utf-8"))["wof"] == "wiki.md"


def test_main_model_path_runs_offline_via_monkeypatch(tmp_path, capsys, monkeypatch):
    wof = _write(tmp_path)
    monkeypatch.setattr(cli, "build_model_client", lambda: CountingFake())
    rc = cli.main([str(wof), "--votes", "3"])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    FCResult.model_validate(printed)
    assert printed["summary"]["skipped"] == 0
    assert printed["summary"]["complete"] is True
    assert [g["engine"] for g in printed["graders"]] == ["mechanical", "model"]
    # The benign fake answers never flag anything; a check with candidates -> pass, a check with
    # none (this WOF has no grouping heading) -> na. Either way, no model fail and nothing skipped.
    model_items = [i for i in printed["items"] if i["engine"] == "model"]
    assert len(model_items) == 3 and all(i["status"] in ("pass", "na") for i in model_items)


def test_main_falls_back_to_skipped_when_no_credentials(tmp_path, capsys, monkeypatch):
    wof = _write(tmp_path)
    monkeypatch.setattr(cli, "build_model_client", lambda: None)   # no OpenAI-compat creds
    rc = cli.main([str(wof)])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"]["skipped"] == 3 and printed["summary"]["complete"] is False
