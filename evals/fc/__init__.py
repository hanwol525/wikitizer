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

The shared eval infrastructure (result-envelope models, the WOF parser, ``slugify``,
``build_summary``, the model client, and the ``BaseAdjudicator`` voting engine) lives in
``evals/common/``; this package holds only the FC-specific linter, judge prompts, result
assembly, and CLI. The criterion **IDs are the join key** across the fc/rubric/gbf evals;
the authoritative id->prose map lives in ``output/patterns-checklist.md`` (the linter never
reads it at runtime).

Run it::

    python -m evals.fc <wof_path> [--out result.json] [--votes N] [--no-model]
"""

# --- FC-specific constants (the eval-agnostic ones live in evals.common) --- #

# Which eval produced the file (this package only ever emits "fc").
EVAL_NAME = "fc"

# The mechanical grader's identity, echoed into every result's ``graders`` list.
FC_LINT_TOOL = "fc_lint.py"
FC_LINT_VERSION = "0.1.0"

# The FC judge's vote-count env var (the CLI reads it; the default lives in evals.common).
ENV_VOTES = "WIKITIZER_FC_VOTES"
