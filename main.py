"""Pipeline entry point.

Right now its one real job is to load configuration from a local ``.env`` into
the process environment *before* anything constructs an Anthropic client. The
SDK reads ``ANTHROPIC_API_KEY`` from the environment, and a ``.env`` file is NOT
auto-loaded -- so loading it belongs here, at the entry point, not in
``agents/base.py`` (which deliberately never reads the key itself). Keep the key
in ``.env`` (gitignored); never hardcode it.

The rest of the pipeline -- parse -> filter reactions -> LLM agents -> wiki
output -- gets wired up here in later phases.
"""

from dotenv import load_dotenv


def main() -> None:
    # Pull .env into os.environ so the Anthropic SDK can find ANTHROPIC_API_KEY.
    # A no-op (returns False) if there's no .env, so it's safe to always call.
    load_dotenv()


if __name__ == "__main__":
    main()
