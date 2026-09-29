"""The GBF (Golden-Baseline-File) eval for Wikitizer.

Given one Wiki Output File (WOF) and the gold reference (``output/gol-lore-full.md``), it asks,
for the entries the WOF actually contains: is its prose **faithful in meaning** to the gold, and
are its **History events in the right chronological order**? It is **all-model** -- every verdict
rests on model judgment over ``ParsedWOF``-extracted structure (never raw markdown dumped whole).
Python parses BOTH files, pairs entries (a hybrid pass: exact Python locks + a model residual),
narrows each comparison to one entry-pair, aggregates deterministically, and assembles the result;
the model renders the meaning judgments.

It is NOT presence (that's the rubric) or formatting (that's FC). It emits ONE JSON result
conforming to ``gbf-result.schema.json`` (repo root), reusing the shared envelope from
``evals.common`` and specializing ``summary`` into two independent sub-scores -- ``faithfulness``
and ``ordering`` -- with a run-level ``complete`` and NO combined top score.

Run it::

    python -m evals.gbf <wof_path> [--out result.json] [--votes N] [--no-model]
                        [--gold PATH] [--ranks PATH]

The default judge is DeepSeek (``deepseek/deepseek-v4-pro-0813``, a PINNED dated slug) via the
shared OpenAI-compat gateway; set it with ``WIKITIZER_GBF_MODEL`` (unset falls back to the shared
``DEFAULT_MODEL``). GBF runs hotter than the other evals (votes default 5 -- faithfulness is the
fuzziest judgment).
"""

# Which eval produced the file.
EVAL_NAME = "gbf"

# The GBF judge's vote-count env var (the CLI reads it; the shared default lives in evals.common).
ENV_VOTES = "WIKITIZER_GBF_VOTES"

# The model-client env prefix -> WIKITIZER_GBF_MODEL / WIKITIZER_GBF_TEMPERATURE.
MODEL_PREFIX = "GBF"

# The gold reference WOF (gitignored campaign content; missing is a hard error -- it IS the
# reference). A fabricated schema-reference/fixture is not needed here: the committed FC/rubric
# golden tests already exercise the real gold via skipif.
DEFAULT_GOLD_PATH = "output/gol-lore-full.md"

# The chronological-rank map for the gold's History events (gitignored; missing is a hard error).
# A fabricated ``example_event_ranks.yaml`` (committed next to this package) is the schema
# reference + offline fixture.
DEFAULT_RANKS_PATH = "output/gbf_event_ranks.yaml"

# GBF votes hotter than the shared DEFAULT_VOTES=3 -- meaning-faithfulness is the fuzziest call, so
# a wider self-consistency window is worth the extra calls.
DEFAULT_GBF_VOTES = 5

# The per-entity faithfulness calls are independent and network-bound, so the runner fans them out
# over a thread pool (same basis + cap as the pipeline's PipelineConfig.max_workers=6, which keeps
# 429 risk bounded). Tunable via --max-workers / WIKITIZER_GBF_MAX_WORKERS.
DEFAULT_MAX_WORKERS = 6
ENV_MAX_WORKERS = "WIKITIZER_GBF_MAX_WORKERS"
