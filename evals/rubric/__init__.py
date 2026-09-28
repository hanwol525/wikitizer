"""The rubric (presence/sourcing) eval for Wikitizer.

Given one Wiki Output File (WOF) and a campaign rubric, it asks: does the WOF contain the
sections, entities, PC markings, and footnote-support the rubric requires? It is **all-model**
-- every criterion's verdict comes from the judge model reading ``ParsedWOF``-extracted
structure (never raw markdown). Python parses + narrows + numbers candidates and assembles the
result; the model judges every criterion.

It emits ONE JSON result conforming to ``rubric-result.schema.json`` (repo root), reusing the
shared envelope from ``evals.common`` and specializing ``summary`` into two independent
sub-scores (presence, sourcing) with a run-level ``complete`` and NO combined top score.

Run it::

    python -m evals.rubric <wof_path> [--out result.json] [--votes N] [--no-model] [--rubric PATH]
"""

# Which eval produced the file.
EVAL_NAME = "rubric"

# The rubric judge's vote-count env var (the CLI reads it; the default lives in evals.common).
ENV_VOTES = "WIKITIZER_RUBRIC_VOTES"

# The model-client env prefix -> WIKITIZER_RUBRIC_MODEL / WIKITIZER_RUBRIC_TEMPERATURE.
MODEL_PREFIX = "RUBRIC"

# The rubric checklist file (gitignored; missing is a hard error -- it IS the checklist).
DEFAULT_RUBRIC_PATH = "output/rubric.yaml"

# The five entity categories that drive presence + sourcing. Each tuple is:
#   (rubric attribute, id token [hyphens only, so it's id-pattern-safe], WOF section name).
# The section names match the renderer's fixed sections (Locations -> History -> People &
# Cultures -> Organizations -> Characters -> Items); ``events`` -> ``History`` is the one
# non-identity mapping. ``sections`` is its own separate check, not an entity category here.
ENTITY_CATEGORIES = [
    ("locations", "locations", "Locations"),
    ("characters", "characters", "Characters"),
    ("organizations", "organizations", "Organizations"),
    ("people_and_cultures", "people-and-cultures", "People & Cultures"),
    ("events", "events", "History"),
]
