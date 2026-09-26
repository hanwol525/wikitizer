"""Shared eval infrastructure for Wikitizer's grading evals.

This package holds the eval-agnostic pieces every eval reuses: the result-envelope
models (``Status`` / ``Engine`` / ``Evidence`` / ``Item`` / graders / ``Summary``), the
WOF parser (``parse.ParsedWOF`` / ``parse_wof``), the golden-convention ``slugify``,
``scoring.build_summary``, the OpenAI-compat ``model_client``, and the self-consistency
voting engine (``adjudicator.BaseAdjudicator``).

``evals/fc/`` consumes these (and, coming next, the rubric and GBF evals). Nothing here
imports from a specific eval package -- ``evals/common/`` is the layer root and stays
standalone.
"""

# --- shared, eval-agnostic constants ------------------------------------------ #

# Envelope shape version (NOT the criteria). Bump only when the JSON structure changes.
SCHEMA_VERSION = "1.0"

# The default model adjudicator: Qwen3-8B over OpenRouter. Everything here is a
# *default* -- each is overridable via the matching env var (see
# ``model_client.build_model_client``), so the judge is swappable without touching code.
DEFAULT_MODEL = "qwen/qwen3-8b"      # env WIKITIZER_<PREFIX>_MODEL
DEFAULT_PROVIDER = "openrouter"      # serving route, recorded in the ModelGrader
DEFAULT_TEMPERATURE = 0.6            # env WIKITIZER_<PREFIX>_TEMPERATURE (non-greedy; Qwen3 discourages greedy)
DEFAULT_VOTES = 3                    # odd, so a per-candidate majority never ties
