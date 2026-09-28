"""Regression tests for the fixes from the adversarial multi-agent review of evals/fc/.

Each test pins a defect the review confirmed, so a future regression re-breaks a test. Offline.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals.fc import fc_lint as L
from evals.fc.emit import build_result, result_to_json, write_result
from evals.fc.models import Engine, Evidence, FCItem, FCResult, Status
from evals.common.parse import parse_wof
from evals.fc.runner import grade

NOW = datetime(2026, 9, 23, 14, 32, 10, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parent.parent

GOOD = """\
# Title

Subtitle prose here.

---

## Locations

### <a id="alpha"></a>Alpha

Alpha near [Beta](#beta).[^1]

### <a id="beta"></a>Beta

Beta.[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""


def st(text, fn):
    return fn(parse_wof(text)).status.value


# --- finding 1 (HIGH): request-time API error degrades, never crashes -------- #

class RaisingClient:
    model, provider, temperature, thinking, base_url = (
        "x/qwen", "openrouter", 0.6, False, "https://x/api/v1")

    def complete(self, system, user):
        raise RuntimeError("401 Unauthorized")   # the SDK does NOT retry auth errors


# A WOF that gives ALL THREE model checks candidates (a grouping heading, a numeric figure, and
# entries) so the raising client is actually invoked for each -> each degrades independently.
WOF_ALL_MODEL_CANDIDATES = """\
# T

sub

---

## History

- <a id="ev"></a>**The Event** — about 200 years ago. It happened.[^1]

### Could Not Place

- <a id="vague"></a>**The Vague One**. Unknown.[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""


def test_request_time_error_degrades_to_skipped_not_crash():
    result = grade(WOF_ALL_MODEL_CANDIDATES, "wiki.md", use_model=True,
                   model_client=RaisingClient(), now=NOW)
    mech = [i for i in result.items if i.engine == Engine.MECHANICAL]
    model = [i for i in result.items if i.engine == Engine.MODEL]
    # The whole run survives: 24 mechanical results emitted, all 3 model checks degrade to skipped
    # (each was invoked and each raised), the result is valid, and it's marked provisional.
    assert len(mech) == 24
    assert len(model) == 3 and all(i.status == Status.SKIPPED for i in model)
    assert all("[REVIEW]" in i.evidence.detail for i in model)
    assert result.summary.complete is False
    assert [g.engine for g in result.graders] == ["mechanical"]   # all skipped -> no model grader
    # And it is a fully valid FCResult (would have been a bare traceback before the fix).
    assert isinstance(result, FCResult) and result.summary.applicable == mech_applicable(mech)


def mech_applicable(mech):
    return sum(1 for i in mech if i.status in (Status.PASS, Status.PARTIAL, Status.FAIL))


# --- finding 2 (MEDIUM): main() guards a broken model client ---------------- #

def test_main_degrades_when_build_model_client_raises(tmp_path, capsys, monkeypatch):
    from evals.fc import __main__ as cli
    wof = tmp_path / "wiki.md"
    wof.write_text(GOOD, encoding="utf-8")

    def boom():
        raise ImportError("No module named 'openai'")

    monkeypatch.setattr(cli, "build_model_client", boom)
    rc = cli.main([str(wof)])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"]["skipped"] == 3 and printed["summary"]["complete"] is False


# --- finding 3 (MEDIUM): non-leading anchor on a section header fails --------- #

def test_sections_no_anchor_catches_non_leading_anchor():
    # anchor_id parses None (anchor not leading), but the header line still carries an anchor.
    bad = GOOD.replace("## Locations", '## Locations <a id="locations"></a>')
    assert st(bad, L.check_sections_no_anchor) in ("fail", "partial")
    # And the leading form is still caught.
    lead = GOOD.replace("## Locations", '## <a id="locations"></a>Locations')
    assert st(lead, L.check_sections_no_anchor) in ("fail", "partial")


# --- finding 4 (MEDIUM): opens_with_title robustness ------------------------ #

UNTITLED_BULLETED_FIRST = """\
# History

- <a id="the-founding"></a>**The Founding** — ~200 years ago. It happened.[^1]

## Could Not Place

- <a id="the-vague"></a>**The Vague Event**. No date known.[^1]

# Footnotes

[^1]: `q` — Matt, log.txt
"""


def test_untitled_bulleted_first_section_not_misread_as_titled():
    p = parse_wof(UNTITLED_BULLETED_FIRST)
    assert p.opens_with_title is False            # a bullet is NOT top matter
    # And the compliant untitled doc is not false-failed on the title/section checks.
    assert st(UNTITLED_BULLETED_FIRST, L.check_title_single_h1) == "na"


def test_titled_missing_subtitle_still_flags_missing_subtitle():
    # Scenario B: titled ladder (# / ## / ###) but no subtitle and no HR -> the depth-3 tell
    # keeps it titled, so the missing subtitle FAILS rather than hiding as NA.
    text = "# The Lore\n\n## Locations\n\n### <a id=\"a\"></a>Alpha\n\nBody.[^1]\n\n## Footnotes\n\n[^1]: `q` — M, l.txt\n"
    p = parse_wof(text)
    assert p.opens_with_title is True
    assert st(text, L.check_title_subtitle) == "fail"


# --- finding 7 (LOW): detail bullets in an ATX entry aren't phantom entries --- #

def test_detail_bullets_inside_atx_entry_are_not_entries():
    text = """\
# T

sub

---

## Locations

### <a id="alpha"></a>Alpha

Features:
- a lake
- a hill

More prose.[^1]

---

## Footnotes

[^1]: `q` — Matt, log.txt
"""
    p = parse_wof(text)
    loc = [s for s in p.sections if s.title == "Locations"][0]
    assert [e.name for e in loc.entries] == ["Alpha"]         # no phantom 'a lake' / 'a hill'
    assert st(text, L.check_list_anchor_bold) == "na"        # no real bullet-entry sections


# --- finding 6 (LOW): footnotes must be the LAST section -------------------- #

def test_fndef_collected_fails_when_footnotes_not_last():
    text = """\
# T

sub

---

## Footnotes

[^1]: `q` — Matt, log.txt

---

## Locations

### <a id="a"></a>Alpha

Body.[^1]
"""
    assert st(text, L.check_fndef_collected) in ("fail", "partial")


def test_fndef_collected_fails_when_defs_split_across_sections():
    text = """\
# T

sub

---

## Notes

[^1]: `q` — Matt, log.txt

---

## More Notes

[^2]: `r` — Sam, log.txt
"""
    # footnotes_section = first section with defs; [^2] is outside it -> fail.
    assert st(text, L.check_fndef_collected) in ("fail", "partial")


def test_fndef_collected_fails_when_def_before_any_section():
    # A def line in the top matter (before any section) -> footnotes_section is None -> fail.
    text = "# T\n\nsub\n\n[^1]: `q` — Matt, log.txt\n\n---\n\n## Locations\n\n### <a id=\"a\"></a>A\n\nBody.[^1]\n"
    assert st(text, L.check_fndef_collected) == "fail"


# --- findings 5 + uncertain: schema is committed and emission validates ------ #

def test_schema_file_is_committed():
    assert (REPO / "fc-result.schema.json").exists()


def test_emission_omits_null_optionals_but_keeps_required_score():
    # exclude_none: an item with no expected/found emits neither key; a None score stays present.
    items = [FCItem(id="fc.anchors.unique-ids", engine=Engine.MECHANICAL, status=Status.NA,
                    evidence=Evidence(lines=[], detail="no anchors"))]
    result = build_result("w.md", items, NOW, None)   # 0 applicable -> score None
    data = json.loads(result_to_json(result))
    ev = data["items"][0]["evidence"]
    assert "expected" not in ev and "found" not in ev and "description" not in data["items"][0]
    assert "score" in data["summary"] and data["summary"]["score"] is None


def test_emitted_result_validates_against_committed_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((REPO / "fc-result.schema.json").read_text(encoding="utf-8"))
    items = [
        FCItem(id="fc.anchors.unique-ids", engine=Engine.MECHANICAL, status=Status.PASS,
               evidence=Evidence(lines=[], detail="0 collisions")),
        FCItem(id="fc.links.resolve", engine=Engine.MECHANICAL, status=Status.FAIL,
               evidence=Evidence(lines=[3], detail="broken", expected="#a", found="#b")),
        FCItem(id="fc.markers.tbd", engine=Engine.MODEL, status=Status.NA,
               evidence=Evidence(lines=[], detail="no entries")),
    ]
    cfg = {"name": "qwen/qwen3-8b", "provider": "openrouter", "temperature": 0.6,
           "thinking": False, "votes": 3, "endpoint": None}
    # A NA model item still counts as "ran" for grader-recording, so a model grader is present.
    result = build_result("w.md", items, NOW, cfg)
    jsonschema.validate(json.loads(result_to_json(result)), schema)


# --- finding 9 (LOW): --out to a missing directory doesn't crash ------------- #

def test_write_result_creates_missing_parent(tmp_path):
    items = [FCItem(id="fc.anchors.unique-ids", engine=Engine.MECHANICAL, status=Status.PASS,
                    evidence=Evidence(lines=[], detail="ok"))]
    result = build_result("w.md", items, NOW, None)
    out = tmp_path / "new" / "deep" / "r.json"
    write_result(result, out)
    assert json.loads(out.read_text(encoding="utf-8"))["wof"] == "w.md"


# --- finding 10 (LOW): ModelGrader.votes records the effective (clamped) count -- #

def test_votes_clamped_in_model_grader():
    class Fake:
        model, provider, temperature, thinking, base_url = ("x", "openrouter", 0.6, False, "u")

        def complete(self, system, user):
            return json.dumps({"verdicts": ["ok"] * len(
                [ln for ln in user.splitlines() if ln[:2].strip().rstrip(".").isdigit()])})

    result = grade(GOOD, "w.md", votes=0, use_model=True, model_client=Fake(), now=NOW)
    model_graders = [g for g in result.graders if g.engine == "model"]
    if model_graders:                                  # present iff a model item wasn't skipped
        assert model_graders[0].votes == 1            # clamped up from the requested 0


# --- finding 5 end-to-end: the golden emission validates against the schema --- #

_GOLDEN = REPO / "output" / "gol-lore-full.md"


@pytest.mark.skipif(not _GOLDEN.exists(), reason="golden file not present (gitignored)")
def test_golden_no_model_emission_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from evals.fc.runner import grade_file
    schema = json.loads((REPO / "fc-result.schema.json").read_text(encoding="utf-8"))
    result = grade_file(_GOLDEN, use_model=False, now=NOW)
    jsonschema.validate(json.loads(result_to_json(result)), schema)
