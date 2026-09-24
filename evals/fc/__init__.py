"""The FC (Formatting-Checklist) eval for Wikitizer.

Grades one Wiki Output File (WOF) against the structural/formatting conventions the
golden response established -- heading depths, inline anchors, kebab-case slugs,
resolving cross-links, footnote shape, ``[TBD]``/``*`` markers -- and emits ONE JSON
result conforming to ``fc-result.schema.json``.

The design is "LLM-decides / Python-assembles" applied to grading:

  * 24 DETERMINISTIC structural checks run in pure Python (the linter, ``fc_lint.py``).
  * 3 GENUINE-JUDGMENT checks run through a small LLM adjudicator (``adjudicator.py``),
    which only ever renders a per-candidate verdict on a tiny pre-extracted candidate
    set -- it never sees the whole file.

Python owns parsing (``parse.py``), scoring (``scoring.py``), and the output contract
(``models.py`` / ``emit.py``). The criterion **IDs are the join key** across the
fc/rubric/gbf evals; the authoritative id->prose map lives in
``output/patterns-checklist.md`` (the linter never reads it at runtime).

Run it::

    python -m evals.fc <wof_path> [--out result.json] [--votes N] [--no-model]
"""

# --- shared constants (imported across the package so the literals can't drift) --- #

# Envelope shape version (NOT the criteria). Bump only when the JSON structure changes.
SCHEMA_VERSION = "1.0"
# Which eval produced the file (this package only ever emits "fc").
EVAL_NAME = "fc"

# The mechanical grader's identity, echoed into every result's ``graders`` list.
FC_LINT_TOOL = "fc_lint.py"
FC_LINT_VERSION = "0.1.0"

# The default model adjudicator: Qwen3-8B over OpenRouter. Everything here is a
# *default* -- each is overridable via the matching env var so the judge is swappable
# without touching code (see model_client.build_model_client).
DEFAULT_FC_MODEL = "qwen/qwen3-8b"      # env WIKITIZER_FC_MODEL
DEFAULT_FC_PROVIDER = "openrouter"      # serving route, recorded in the ModelGrader
DEFAULT_TEMPERATURE = 0.6              # env WIKITIZER_FC_TEMPERATURE (non-greedy; Qwen3 discourages greedy)
DEFAULT_VOTES = 3                      # env WIKITIZER_FC_VOTES (odd, so a per-candidate majority never ties)

# The env vars the eval reads (reusing the pipeline's existing OpenAI-compat path).
ENV_MODEL = "WIKITIZER_FC_MODEL"
ENV_TEMPERATURE = "WIKITIZER_FC_TEMPERATURE"
ENV_VOTES = "WIKITIZER_FC_VOTES"
ENV_OPENAI_BASE_URL = "LLM_OPENAI_BASE_URL"
ENV_OPENAI_API_KEY = "LLM_OPENAI_API_KEY"
