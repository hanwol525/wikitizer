# `evals/fc/` — the FC (Formatting-Checklist) eval

Grades one **Wiki Output File** (WOF) against the structural/formatting conventions the
golden response established — heading depths, inline anchors, kebab-case slugs, resolving
cross-links, footnote shape, `[TBD]`/`*` markers — and emits **one JSON result** conforming
to `fc-result.schema.json`.

```bash
python -m evals.fc <wof_path> [--out result.json] [--votes N] [--no-model]

# grade the golden file, no model calls:
python -m evals.fc output/gol-lore-full.md --no-model --out /tmp/fc.json
```

The JSON prints to stdout (and to `--out` when given); progress + `[REVIEW]` logs go to stderr.

## The "LLM-decides / Python-assembles" split

- **24 mechanical checks** (`fc_lint.py`) run in pure Python — deterministic structural
  linting. Python owns parsing (`parse.py`), scoring (`scoring.py`), and the output contract
  (`models.py` / `emit.py`).
- **3 model checks** run through a small LLM adjudicator (`adjudicator.py`) on a *tiny
  pre-extracted candidate set* (the model never sees the whole file): `fc.entries.grouping-no-anchor`
  (group label vs an entry that lost its anchor), `fc.list-entries.approx-tilde` (an
  approximate figure that should carry `~`), `fc.markers.tbd` (an unmarked stub).

**IDs are the join key.** Every criterion is a dotted `fc.*` slug; results reference the id,
never the prose. The authoritative id → prose map lives in `output/patterns-checklist.md` —
the linter hardcodes the ids and never reads the checklist at runtime.

## The result contract

`fc-result.schema.json` + the Pydantic v2 models in `models.py` are the source of truth (a
mismatch fails at emit time). Status is a **5-value** enum — `pass` / `partial` / `fail` /
`na` / `skipped`:

- **applicable-only denominator:** `applicable = pass + partial + fail` (`na` and `skipped`
  excluded); `score = pass / applicable` (`null` when nothing applies). `partial` scores as a
  fail but is counted separately.
- **`skipped` ⇒ provisional:** any `skipped` item forces `summary.complete = false`, so a run
  with the model disabled is visibly incomplete.

Every item carries `evidence` (terse on a pass, specific — line numbers + the offending value
— on a fail/partial).

## The model (adjudicator)

Default judge: **Qwen3-8B via OpenRouter**, **thinking OFF**, **non-greedy** (temp `0.6`;
Qwen3 discourages greedy decoding), forced JSON output. It reuses the pipeline's existing
OpenAI-compat `.env` path — set:

```
LLM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_OPENAI_API_KEY=<your OpenRouter key>
```

and (all optional, swappable) `WIKITIZER_FC_MODEL` (default `qwen/qwen3-8b` — confirm the
exact provider slug), `WIKITIZER_FC_TEMPERATURE` (default `0.6`), `WIKITIZER_FC_VOTES`
(default `3`, kept odd). Each candidate set is judged `--votes` times and a per-candidate
majority is taken; a wrong-length vote is discarded, an all-malformed set fails closed, and a
split vote logs `[REVIEW]`.

**Skipped path:** with `--no-model`, or when the OpenAI-compat credentials are absent, the 3
model items are emitted as `skipped` and `complete` is `false`.

## Scope (what this is NOT)

FC takes a WOF path in and emits one result out. The `wiki.md → wiki-[datetime].md` rename,
cross-eval orchestration, the final-grade synthesizer, and the sibling rubric/GBF evals are
separate layers. The result envelope is shared (`eval` discriminates fc/rubric/gbf), but only
FC is built here.

**Deferred checklist criteria.** The brief's table defines the FC id set as exactly these 24
mechanical + 3 model checks. A few structural clauses in `output/patterns-checklist.md` are
intentionally **not** implemented (and so are silently "not graded", not passed): cross-section
"see also" parentheticals; reuse of one footnote number per source / no duplicate definitions;
editorial square-bracket insertions inside a quote (`wo[u]ld`); and appending additional sources
after the primary attribution ("Also:" / "See also:"). Add them as new `fc.*` ids if a future
rubric wants them.

## Tests

Committed in the top-level `tests/` (`tests/test_fc_*.py`), offline (a `FakeModelClient`
feeds canned verdicts). `tests/test_fc_golden.py` runs the linter over `output/gol-lore-full.md`
(gitignored PII — the test `skipif`s when it's absent) and asserts all 24 mechanical checks
are `pass`/`na` — the acceptance bar. `tests/test_fc_integration.py` (`@pytest.mark.integration`,
`skipif LLM_OPENAI_API_KEY` absent) hits the real judge.
