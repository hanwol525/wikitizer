"""Tests for evals/gbf/runner.py -- end-to-end grading + the degrade paths. Fully offline (a
content-routing FakeModelClient answers pairing vs faithfulness calls; ordering is pure Python)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals.common.models import Status
from evals.common.parse import parse_wof
from evals.gbf.runner import grade

_NOW = datetime(2026, 9, 23, 14, 32, 10, tzinfo=timezone.utc)

_RANKS = {"ev-a": 1, "ev-b": 2}

_GOLD = """# Gold

## Locations

### <a id="aldenburg"></a>Aldenburg

A town in the north, ruled by Baron Aldric.

### <a id="citadel"></a>Citadel

An ancient fortress that anchors the world's dating.

## History

- <a id="ev-a"></a>**Event A** — the first age.
- <a id="ev-b"></a>**Event B** — the second age.
"""


def _claim(text, verdict, span=""):
    return {"claim": text, "verdict": verdict, "span": span}


def _affirmed():
    return json.dumps({"claims": [_claim("core fact", "affirmed", "same fact")], "extra": []})


def _contradicted():
    return json.dumps({"claims": [_claim("ruled by Baron Aldric", "contradicted",
                                         "ruled by a council")], "extra": []})


class RoutingFake:
    model = "fake/deepseek"; provider = "openrouter"; temperature = 0.6; thinking = False; base_url = None

    def __init__(self, pair_reply=None, faith_by_name=None, faith_default=None,
                 raise_on_pair=False, raise_faith_names=()):
        self.pair_reply = pair_reply
        self.faith_by_name = faith_by_name or {}
        self.faith_default = faith_default or _affirmed()
        self.raise_on_pair = raise_on_pair
        self.raise_faith_names = tuple(raise_faith_names)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if "GOLD entries (unmatched)" in user:              # residual pairing call
            if self.raise_on_pair:
                raise RuntimeError("pair boom")
            return self.pair_reply if self.pair_reply is not None else json.dumps({"results": []})
        for nm in self.raise_faith_names:                   # faithfulness call
            if f"GOLD entry: {nm!r}" in user:
                raise RuntimeError("faith boom")
        for nm, rep in self.faith_by_name.items():
            if f"GOLD entry: {nm!r}" in user:
                return rep
        return self.faith_default


def _by_id(result):
    return {it.id: it for it in result.items}


def test_happy_path_grades_all(tmp_path):
    gold = parse_wof(_GOLD)
    fake = RoutingFake(faith_by_name={"Citadel": _contradicted()})   # rest default-affirmed
    result = grade(_GOLD, "wof.md", gold, _RANKS, votes=1, model_client=fake, now=_NOW)
    items = _by_id(result)
    # WOF == gold here: everything locks by anchor -> NO residual pairing call.
    assert not any("GOLD entries (unmatched)" in u for _s, u in fake.calls)
    assert items["gbf.faithfulness.aldenburg"].status == Status.PASS
    assert items["gbf.faithfulness.citadel"].status == Status.FAIL
    assert items["gbf.faithfulness.ev-a"].status == Status.PASS
    assert items["gbf.ordering"].status == Status.PASS
    assert result.summary.faithfulness.score_display == "3/4"     # aldenburg/ev-a/ev-b pass, citadel fail
    assert result.summary.ordering.score_display == "1/1"
    assert result.summary.complete is True


def test_no_model_all_skipped():
    gold = parse_wof(_GOLD)
    result = grade(_GOLD, "wof.md", gold, _RANKS, votes=1, use_model=False, now=_NOW)
    assert all(it.status == Status.SKIPPED for it in result.items)
    assert result.summary.complete is False
    assert result.summary.faithfulness.score_display == "0/0"
    assert result.summary.ordering.score_display == "0/0"


def test_absent_entity_is_na():
    # A WOF missing Citadel entirely: no residual WOF entry to match it -> na (not fail).
    wof = ("# WOF\n\n## Locations\n\n### <a id=\"aldenburg\"></a>Aldenburg\n\nA town.\n\n"
           "## History\n\n- <a id=\"ev-a\"></a>**Event A** — first.\n"
           "- <a id=\"ev-b\"></a>**Event B** — second.\n")
    gold = parse_wof(_GOLD)
    fake = RoutingFake()
    result = grade(wof, "wof.md", gold, _RANKS, votes=1, model_client=fake, now=_NOW)
    items = _by_id(result)
    assert items["gbf.faithfulness.citadel"].status == Status.NA
    assert "absent" in items["gbf.faithfulness.citadel"].evidence.detail


def test_degraded_pairing_skips_faithfulness_and_ordering():
    # A WOF whose entries neither anchor- nor name-match the gold -> everything is residual; the
    # pairing call errors -> all faithfulness skipped + ordering skipped + complete False.
    wof = ("# WOF\n\n## Locations\n\n### <a id=\"xyz\"></a>Xyzabad\n\nA far city.\n\n"
           "## History\n\n- <a id=\"qqq\"></a>**Some Battle** — long ago.\n")
    gold = parse_wof(_GOLD)
    fake = RoutingFake(raise_on_pair=True)
    result = grade(wof, "wof.md", gold, _RANKS, votes=1, model_client=fake, now=_NOW)
    items = _by_id(result)
    assert all(items[f"gbf.faithfulness.{s}"].status == Status.SKIPPED
               for s in ("aldenburg", "citadel", "ev-a", "ev-b"))
    assert items["gbf.ordering"].status == Status.SKIPPED
    assert result.summary.complete is False


def test_degraded_single_faithfulness_skips_only_that_item():
    gold = parse_wof(_GOLD)
    fake = RoutingFake(raise_faith_names=("Citadel",))
    result = grade(_GOLD, "wof.md", gold, _RANKS, votes=1, model_client=fake, now=_NOW)
    items = _by_id(result)
    assert items["gbf.faithfulness.citadel"].status == Status.SKIPPED
    assert "[REVIEW]" in items["gbf.faithfulness.citadel"].evidence.detail
    assert items["gbf.faithfulness.aldenburg"].status == Status.PASS   # others proceed
    assert items["gbf.ordering"].status == Status.PASS
    assert result.summary.complete is False


def test_parallel_equals_sequential():
    # Concurrency must change nothing but wall-clock: the same WOF graded with 1 vs 6 workers yields
    # identical items (id -> status). The RoutingFake is content-routed, so it's order-independent.
    gold = parse_wof(_GOLD)
    seq = grade(_GOLD, "wof.md", gold, _RANKS, votes=1,
                model_client=RoutingFake(faith_by_name={"Citadel": _contradicted()}),
                now=_NOW, max_workers=1)
    par = grade(_GOLD, "wof.md", gold, _RANKS, votes=1,
                model_client=RoutingFake(faith_by_name={"Citadel": _contradicted()}),
                now=_NOW, max_workers=6)
    assert {it.id: it.status for it in seq.items} == {it.id: it.status for it in par.items}
    assert seq.summary.faithfulness.score_display == par.summary.faithfulness.score_display


def test_per_entity_error_under_parallelism_skips_only_that_item():
    gold = parse_wof(_GOLD)
    fake = RoutingFake(raise_faith_names=("Citadel",))
    result = grade(_GOLD, "wof.md", gold, _RANKS, votes=1, model_client=fake, now=_NOW, max_workers=6)
    items = _by_id(result)
    assert items["gbf.faithfulness.citadel"].status == Status.SKIPPED
    assert items["gbf.faithfulness.aldenburg"].status == Status.PASS   # others still graded
    assert items["gbf.ordering"].status == Status.PASS
    assert result.summary.complete is False


def test_result_schema_valid():
    jsonschema = pytest.importorskip("jsonschema")
    from evals.gbf.emit import result_to_json
    schema_path = Path(__file__).resolve().parent.parent / "gbf-result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    gold = parse_wof(_GOLD)
    result = grade(_GOLD, "wof.md", gold, _RANKS, votes=1, model_client=RoutingFake(), now=_NOW)
    jsonschema.validate(json.loads(result_to_json(result)), schema)


def test_non_ascii_entity_slug_builds_valid_id_and_schema():
    # A non-ASCII entity name (Skjöldr Aldvarðr) must flow through faithfulness id-building + schema
    # validation: the slug PRESERVES non-ASCII letters and the id pattern was widened to admit it.
    jsonschema = pytest.importorskip("jsonschema")
    from evals.gbf.emit import result_to_json
    gold_md = ("# Gold\n\n## Characters\n\n### <a id=\"skjöldr-aldvarðr\"></a>Skjöldr Aldvarðr\n\n"
               "A dwarf of the Aldward family, bodyguard to the prince.\n\n"
               "## History\n\n- <a id=\"ev-a\"></a>**Event A** — the first age.\n"
               "- <a id=\"ev-b\"></a>**Event B** — the second age.\n")
    gold = parse_wof(gold_md)
    result = grade(gold_md, "wof.md", gold, _RANKS, votes=1, model_client=RoutingFake(), now=_NOW)
    assert "gbf.faithfulness.skjöldr-aldvarðr" in {it.id for it in result.items}
    schema = json.loads((Path(__file__).resolve().parent.parent / "gbf-result.schema.json")
                        .read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(result_to_json(result)), schema)
