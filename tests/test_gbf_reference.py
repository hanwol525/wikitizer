"""Tests for evals/gbf/reference.py -- load_gold / load_ranks. Fully offline."""

from pathlib import Path

import pytest

import evals.gbf
from evals.common.parse import ParsedWOF
from evals.gbf.reference import load_gold, load_ranks

_EXAMPLE_RANKS = Path(evals.gbf.__file__).with_name("example_event_ranks.yaml")


def test_load_gold_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gold(tmp_path / "nope.md")


def test_load_gold_parses(tmp_path):
    p = tmp_path / "g.md"
    p.write_text("# Title\n\n## Locations\n\n### <a id=\"aldenburg\"></a>Aldenburg\n\nA town.\n",
                 encoding="utf-8")
    parsed = load_gold(p)
    assert isinstance(parsed, ParsedWOF)
    assert any(s.title == "Locations" for s in parsed.sections)


def test_load_ranks_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ranks(tmp_path / "nope.yaml")


def test_load_ranks_example_parses():
    ranks = load_ranks(_EXAMPLE_RANKS)
    assert isinstance(ranks, dict)
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in ranks.items())
    assert ranks["the-founding-of-aldenburg"] == 1
    # co-temporal events share a rank
    assert ranks["the-war-of-ashes"] == ranks["the-siege-of-mistfen"]


def test_load_ranks_empty_file_is_empty(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_ranks(p) == {}


def test_load_ranks_rejects_non_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ranks(p)


def test_load_ranks_rejects_non_int_rank(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("the-founding: first\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ranks(p)


def test_load_ranks_rejects_bool_rank(tmp_path):
    # a YAML bool is an int subclass; it must not masquerade as a rank
    p = tmp_path / "bool.yaml"
    p.write_text("the-founding: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ranks(p)
