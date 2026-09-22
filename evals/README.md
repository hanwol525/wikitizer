# `evals/` — dev A/B model-swap harnesses

Standalone developer scripts that compare a **candidate** model against the
**trusted baseline** over your real (gitignored) logs, so you can decide a
per-role model swap on real data. They make **real, paid** API calls and are
**not** part of the pytest suite (their filenames don't match `test_*.py`, so
`pytest` never collects them). The only unit-tested part is the pure metrics
module, `evals/extract_metrics.py` (`tests/test_extract_metrics.py`).

The extractor fan-out dominates the pipeline's spend, so `extract_ab.py` (the
`EXTRACT` role) is the swap that matters most.

## `extract_ab.py` — the six extractors, baseline vs candidate

Runs the six extractor agents **twice over the same filtered messages** — once on
the Sonnet baseline, once on a candidate — and diffs the results. Sonnet is the
reference (there's no golden set); the question it answers is *"does the candidate
drop entities or quotes relative to what I run today?"*

```bash
# baseline claude-sonnet-4-6  vs  candidate DeepSeek V4 Flash over OpenRouter
python -m evals.extract_ab --logs logs/ \
    --candidate openrouter:deepseek/deepseek-v4-flash \
    --report-file evals/last_report.md
```

For an `openrouter:`/`openai:` candidate you need these in `.env` (the OpenAI-compat
endpoint the candidate routes to):

```
LLM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_OPENAI_API_KEY=<your OpenRouter key>
```

The harness sets `WIKITIZER_EXTRACT_MODEL` (and, for an OpenAI-compat candidate,
`WIKITIZER_OPENAI_REASONING=off`) per run itself, then restores your env — you do
**not** flip `.env` to run the eval. The dev cache is on, so each side is paid once
and re-running the eval is free.

**Confirm the candidate slug.** `openrouter:deepseek/deepseek-v4-flash` is a default,
not a promise — check the exact slug on OpenRouter's model list (V4.1 Flash or
others can be passed with `--candidate`). Wikitizer passes the bare slug through
unvalidated; a wrong slug surfaces as API errors / empty responses (which then show
up as low entity counts and json-retries).

### Reading the report

Two tiers, labelled in the output:

- **PRIMARY** (robust; from the returned lore objects) — per-type entity
  `base / cand / matched / missed / extra`, and surviving-quote counts per side.
  Candidate coverage ≈ baseline (low `missed`, `extra` mostly explained by renames)
  ⇒ the candidate isn't dropping entities.
- **DIAGNOSTIC** (best-effort; from WARNING logs; cache-sensitive) — dropped-quote
  count split into **cosmetic** vs **reword**, and json-repair retry counts.
  - **reword** low ⇒ the candidate copies quotes faithfully. **cosmetic** high ⇒
    usually a one-line `agents/base_extractor._normalize_for_match` fold, not a
    model problem — but eyeball the printed samples first: a **case-only** diff also
    lands in "cosmetic", and production rejects case on purpose (a casefold would
    let real rewords past the verbatim net), so that one isn't a free fold.
  - json-retry rate low ⇒ plain JSON is fine, no schema follow-up needed. High ⇒
    that's the trigger for a `response_format`/JSON-schema brief.
  - *Retry counts are accurate only on the first uncached (paid) run — a fully
    cached re-run shows 0. Drop counts survive cache re-runs (the verbatim check
    re-runs on cached JSON).*

## Flipping the real pipeline swap

Once the eval looks good, the actual swap is just `.env` (routing, the
reasoning-off knob, the verbatim-quote net, and the `json_repair` fallback are
already model-agnostic — no pipeline change):

```
LLM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_OPENAI_API_KEY=<your OpenRouter key>
WIKITIZER_EXTRACT_MODEL=openrouter:deepseek/deepseek-v4-flash   # confirm the slug
WIKITIZER_OPENAI_REASONING=off
```

`.env.example` already documents these variables. Model choice is per-role
(`WIKITIZER_<ROLE>_MODEL`); this swaps only `EXTRACT`. The noise-filter and
reconciler roles are separate swaps — evaluate one at a time with their own
`evals/` harness.
