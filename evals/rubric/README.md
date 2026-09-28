# `evals/rubric/` — the rubric (presence/sourcing) eval

Grades one **Wiki Output File** (WOF) against a campaign **rubric**: does it contain the
sections, entities, PC markings, and footnote-support the rubric requires? Emits **one JSON
result** conforming to `rubric-result.schema.json`.

```bash
python -m evals.rubric <wof_path> [--out result.json] [--votes N] [--no-model] [--rubric PATH]

# grade the golden file with no model calls (offline; every criterion -> skipped):
python -m evals.rubric output/gol-lore-full.md --no-model --out /tmp/rubric.json
```

The JSON prints to stdout (and to `--out` when given); progress + `[REVIEW]` logs go to stderr.

## The "LLM-decides / Python-assembles" split

This eval is **all-model** — every criterion's verdict comes from the judge model — but the
model never sees raw markdown. Python does the narrowing:

- **`parse.py` / `resolve.py`** turn the WOF into a `ParsedWOF` and extract tiny, numbered
  candidate sets: the section titles, a section's entry names, an entry's raw heading + clipped
  body, and (for sourcing) **only the quotes an entry itself cites**.
- **`adjudicator.py`** judges each criterion over those candidates with a tight single-purpose
  prompt, forced JSON, and self-consistency voting (majority over odd `--votes`; a wrong-length
  reply is discarded as an abstain, an all-malformed set fails closed with `[REVIEW]`, a split
  vote resolves toward the strict label + `[REVIEW]`).
- The model returns **indices** (which WOF section / which entry) and booleans; Python resolves
  the indices back to objects and assembles the items — a fuzzy name the model returns never has
  to be re-parsed. Only a real JSON `true` counts as a positive vote (a string `"false"`/`"no"`
  fails closed), and a `present: true` whose `match` isn't a valid index FAILs with `[REVIEW]`
  rather than passing with nothing mapped.

**Per-entity items** (not per-criterion): a category is batched into one call and the model
returns a verdict per required entity. Player characters get **two** items each (`pc-stated` +
`pc-marked`).

## The result contract

`rubric-result.schema.json` + the Pydantic v2 models in `evals/rubric/models.py` are the source
of truth (a mismatch fails at emit time). It reuses the shared envelope from `evals/common`
(`Status` / `Evidence` / `Item` / `ModelGrader`) and specializes only `summary`:

- **Split sub-scores, no combined score.** `summary` = `{complete, presence, sourcing}`, each
  sub-score an applicable-only `{pass, partial, fail, na, applicable, score, score_display}`.
- **`complete` is run-level and derived** (`skipped == 0`): a degraded run (a check errored to
  `skipped`) reports `complete=false`, never a false "full".
- **IDs are the join key** — every criterion is a dotted `rubric.*` slug (`rubric.sections.*`,
  `rubric.presence.<category>.*`, `rubric.pc-stated.*`, `rubric.pc-marked.*`,
  `rubric.sourcing.<category>.*`); results reference the id, never the prose. The
  `rubric.sourcing.*` prefix buckets an item into sourcing; everything else is presence.

## The rubric file (`output/rubric.yaml`)

The rubric IS the checklist, so it lives in **gitignored `output/rubric.yaml`** and a **missing
file is a hard error** (not a skip). A fabricated **`example_rubric.yaml`** is committed here as
the schema reference and the offline-test fixture. A bare string coerces to `{name: …}`;
`aliases` covers other names + acceptable alternative phrasings; `pc: true` marks a player
character (under `characters`).

## The model (adjudicator)

Default judge: **Qwen3-8B via OpenRouter**, **thinking OFF**, **non-greedy** (temp `0.6`), forced
JSON — the same shared `build_model_client` path as FC, with the `RUBRIC` env prefix:

```
LLM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_OPENAI_API_KEY=<your OpenRouter key>
```

and (all optional, swappable) `WIKITIZER_RUBRIC_MODEL`, `WIKITIZER_RUBRIC_TEMPERATURE`,
`WIKITIZER_RUBRIC_VOTES` (default `3`, kept odd).

**Sourcing** feeds the model only an entity's **own cited quotes** and demands a supporting span:
present + backed → pass; present + cites-but-unsupported → fail; present + cites-nothing → fail
(no call); **absent → `na`** (its presence item already failed — don't double-penalize).

**Skipped path:** with `--no-model`, or when the OpenAI-compat credentials are absent, every
criterion is emitted as `skipped` and `complete` is `false` (both sub-scores `0/0`).

## Scope (what this is NOT)

The rubric takes a WOF path + a rubric in and emits one result out. The `wiki.md → wiki-[datetime].md`
rename, cross-eval orchestration, the final-grade synthesizer, and the sibling FC/GBF evals are
separate layers. The result envelope is shared (`eval` discriminates fc/rubric/gbf).

## Tests

Committed in the top-level `tests/` (`tests/test_rubric_*.py`), offline (a per-file
`FakeModelClient` feeds canned JSON). `tests/test_rubric_golden.py` runs `--no-model` over
`output/gol-lore-full.md` (skips when the golden or `output/rubric.yaml` is absent).
`tests/test_rubric_integration.py` (`@pytest.mark.integration`, `skipif LLM_OPENAI_API_KEY` or the
golden/rubric files absent) hits the real judge. The shared scorer's `score_block` is tested in
`tests/test_common_scoring.py`.
