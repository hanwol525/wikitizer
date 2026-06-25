"""Repo-root conftest.

Its one job today: load ``.env`` into the process environment at collection
time, so the live extractor tests (``tests/test_extractors_integration.py``) can
see ``ANTHROPIC_API_KEY`` *before* their module-level ``skipif`` is evaluated.
conftest is imported before any test module, so the key is already in
``os.environ`` when the skip condition runs. Without this, a key sitting only in
``.env`` would produce a confusing false skip even though it's right there.

A no-op (returns False) when there's no ``.env``, so it's always safe to call.
"""

from dotenv import load_dotenv

load_dotenv()
