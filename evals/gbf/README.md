# `evals/gbf/` — the GBF (Golden-Baseline-File) eval

Grades one **Wiki Output File** (WOF) against the **gold** (`output/gol-lore-full.md`): for the
entries the WOF actually contains, is its **prose faithful in meaning** to the gold, and are its
**History events in the right chronological order**? Emits **one JSON result** conforming to
`gbf-result.schema.json`.

```bash
python -m evals.gbf <wof_path> [--out result.json] [--votes N] [--no-model] [--gold PATH] [--ranks PATH] [--max-workers N]

# grade the golden file with no model calls (offline; every criterion -> skipped):
python -m evals.gbf output/gol-lore-full.md --no-model --out /tmp/gbf.json
```

The per-entity faithfulness calls are independent and network-bound, so they fan out over a thread
pool (`--max-workers`, default **6**; `1` = sequential; also `WIKITIZER_GBF_MAX_WORKERS`). The live
judge is slow per call (~20s on long entries), so the default parallelism is what keeps a full-gold
run in minutes rather than over an hour — results are identical to sequential (items are id-sorted).

The JSON prints to stdout (and to `--out` when given); progress + `[REVIEW]` logs go to stderr.

## The "LLM-decides / Python-assembles" split

GBF is **all-model** and **parses BOTH files** (WOF + gold), but the model never sees raw markdown.
Python does the narrowing and the aggregation:

- **Hybrid pairing (`pair.py`), its own pass before any meaning judgment.** Python locks the
  unambiguous matches (equal anchor slug, then equal normalized name — each *uniqueness-gated*, so a
  certain match never wastes a model call and an ambiguous name never false-locks). Only the residual
  goes to the model (`GbfAdjudicator.pair_residual`, voted). A gold entry with no WOF match → `na`
  faithfulness (an omission is the rubric's job, not GBF's to fail). Keeping pairing separate means a
  pairing miss never hides inside a meaning verdict, and ordering reuses the same map (zero new calls).
- **Faithfulness (`adjudicator.py`), one gold/WOF pair per call, voted (default 5×).** The judge
  extracts the gold entry's key claims and marks each `affirmed` / `contradicted` / `omitted` in the
  WOF, plus lists WOF-only claims as `extra`. Python case/whitespace-normalizes each verdict, then
  folds each vote to a label (any contradiction → `fail`; any other off-menu or missing verdict →
  the vote abstains; all affirmed → `pass`; else → `partial`), majority-votes the labels (tie/all-malformed →
  `fail` + `[REVIEW]`, strict), and names the offending claims in the evidence. `[EXTRA]` notes are
  **orthogonal to the status** — they surface but never move the score.
- **Ordering (`ordering.py`), pure Python, no model call.** The gold's `{slug: rank}` map is the
  reference; the WOF's History events are mapped to ranks via the pairing map and scored by
  **inversion count** — a pair the WOF orders opposite to the ranks. **Co-temporal events share a
  rank**, so swapping them is not an inversion. The History section is located by *normalized name*
  (so a "Timeline"-titled section still grades).

## The result contract

`gbf-result.schema.json` + the Pydantic v2 models in `evals/gbf/models.py` are the source of truth
(a mismatch fails at emit time). It reuses the shared envelope from `evals/common` (`Status` /
`Evidence` / `Item` / `SubScore` / `ModelGrader`) and specializes only `summary`:

- **Split sub-scores, no combined score.** `summary` = `{complete, faithfulness, ordering}`, each an
  applicable-only `{pass, partial, fail, na, applicable, score, score_display}`. Faithfulness and
  ordering measure different things and are never averaged.
- **`complete` is run-level and derived** (`skipped == 0`): a degraded run reports `complete=false`.
- **IDs are the join key** — `gbf.faithfulness.<gold-entry-slug>` (one per gold entry, the universe)
  + the single `gbf.ordering`.
- **Item status vocabulary is the shared 5-value enum** (`pass/partial/fail/na/skipped`); the
  per-claim `affirmed/contradicted/omitted` verdicts live only inside a faithfulness item's evidence,
  never as a status.

## The reference files

Both are **gitignored campaign content** and **hard-error-if-missing** (they ARE the reference):

- **Gold:** `output/gol-lore-full.md` — the same reference the FC/rubric golden tests lean on.
- **Ranks:** `output/gbf_event_ranks.yaml` — `{gold-event-slug: rank}`; co-temporal events share a
  rank; rank-less events (e.g. under "Could Not Place") are excluded. A fabricated
  **`example_event_ranks.yaml`** is committed here as the schema reference + offline test fixture.

## The model (adjudicator)

Default judge: **DeepSeek V4 Pro (`deepseek/deepseek-v4-pro-0813`, a PINNED dated slug** — never a
`-latest` redirect, so a longitudinal eval's judge can't drift). It reaches the model over the same
shared OpenAI-compat gateway FC/rubric use (OpenRouter is OpenAI-compatible), so there is **no new
client** — just the model string:

```
LLM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_OPENAI_API_KEY=<your OpenRouter key>
WIKITIZER_GBF_MODEL=deepseek/deepseek-v4-pro-0813   # PINNED; also the built-in default when unset
```

and (all optional, swappable) `WIKITIZER_GBF_TEMPERATURE`, `WIKITIZER_GBF_VOTES` (default **5** —
GBF runs hotter than the other evals; faithfulness is the fuzziest judgment).

**Skipped path:** with `--no-model`, or when the OpenAI-compat credentials are absent, every
criterion is emitted as `skipped` and `complete` is `false` (both sub-scores `0/0`).

## The validation set (the eval-of-the-eval)

`tests/fixtures/gbf/` holds a small **hand-labeled adversarial set** — WOF variants derived from the
gold with a *known* expected grade (`faithful` / `contradictions` / `omissions` / `reordered`) plus
`expected.yaml`. `tests/test_gbf_bakeoff.py` (`@pytest.mark.integration`) runs candidate judges over
it and reports agreement with the labels + vote stability. This is how the judge is chosen and how
the ranks are proven before the eval is trusted.

## Scope (what this is NOT)

GBF takes a WOF path + the gold/ranks in and emits one result out. Cross-eval orchestration and the
final-grade synthesizer are separate layers; the result envelope is shared (`eval` discriminates
fc/rubric/gbf).

## Tests

Committed in the top-level `tests/` (`tests/test_gbf_*.py`), offline (a per-file `FakeModelClient`
feeds canned JSON). `tests/test_gbf_golden.py` runs `--no-model` over `output/gol-lore-full.md`
(skips when the golden or `output/gbf_event_ranks.yaml` is absent). `tests/test_gbf_integration.py`
and `tests/test_gbf_bakeoff.py` (`@pytest.mark.integration`, `skipif LLM_OPENAI_API_KEY` absent) hit
the real judge. The §1 promotions (`SubScore` → `evals/common/models.py`, the record-voting engine →
`BaseAdjudicator`) are guarded by the unchanged FC + rubric suites.
