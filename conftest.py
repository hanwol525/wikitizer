"""Repo-root conftest.

Its one job today: load ``.env`` into the process environment at collection
time, so the live extractor tests (``tests/test_extractors_integration.py``) can
see ``ANTHROPIC_API_KEY`` *before* their module-level ``skipif`` is evaluated.
conftest is imported before any test module, so the key is already in
``os.environ`` when the skip condition runs. Without this, a key sitting only in
``.env`` would produce a confusing false skip even though it's right there.

A no-op (returns False) when there's no ``.env``, so it's always safe to call.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

# Only load the API key needed for integration-test gating, to avoid `.env`
# side-effects on unit tests (e.g. WIKITIZER_LLM_CACHE).
_env_path = Path(__file__).with_name(".env")
if _env_path.exists() and not os.getenv("ANTHROPIC_API_KEY"):
    api_key = dotenv_values(_env_path).get("ANTHROPIC_API_KEY")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
