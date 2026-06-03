# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
Wikitizer turns a D&D group's exported text-chat logs into structured worldbuilding lore (a free WorldAnvil alternative). The pipeline is: raw chat logs → pure-Python parser → LLM agents that extract structured lore → wiki output.

## Commands
- Activate env: `source venv/bin/activate`
- Run all tests: `pytest` (run from repo root)
- Run one file: `pytest tests/test_speaker_map.py`
- Run one test: `pytest tests/test_speaker_map.py::test_load_speaker_map`
- Build the speaker map from real logs (interactive, one-off): `python scripts/build_speaker_map.py`

## Hard constraints
- **Python 3.9.6.** `dict[str, str]` / `list[X]` builtin generics are fine (PEP 585), but `X | None` union syntax is NOT supported — use `Optional[X]` from `typing`. A 3.10 upgrade is deferred.
- **`logs/` is gitignored and must never be committed.** Files there contain real phone numbers. Keep all chat data local.
- Always pass `encoding="utf-8"` when reading/writing files (names may contain accented chars).

## Architecture
Phased build; most of it is still ahead. Current + planned structure:
- `models/` — pydantic v2 data models. `message.py` (`Message`) is parser output; `lore.py` (`Quote`, `Location`, `Character`, `HistoryEvent`, `OtherDetail`) is the LLM-extraction schema. Use `Field(default_factory=list)` for list fields so instances don't share state.
- `speaker_map.py` (top-level) — `load_speaker_map(path) -> dict[str, str]`. Loads `config/speaker_map.json`, a flat map of phone number → name. Phone keys start with `+`; the special `"exporter"` key names whoever exported the chat (their messages have a bare timestamp, no phone prefix). Loader is deliberately minimal — lets `FileNotFoundError`/`JSONDecodeError` bubble up.
- `config/` — data only, no source. Holds `speaker_map.json`.
- `scripts/build_speaker_map.py` — one-off helper (discover phone numbers via regex → prompt for names → save JSON). NOT part of the runtime pipeline. Edit the `files` list inside to point at the real logs.
- `parsers/` — pure Python, no LLM; separate single-responsibility functions. `chat_parser.py` (done, Phase 2.1) turns a log into `list[Message]` via `parse_chat_log(filepath, speaker_map)`: a state machine over two **footer** regexes (the `+phone`/bare-timestamp line *ends* a message and names the sender of the block *above* it — a footer, not a header). `read_clean` reads raw bytes and strips `\r` to dodge the `\r\r\n` double-blank-line trap. Group files attribute the sender by footer phone; the 1-on-1 file has no phones, so it attributes by body left/right alignment. `reaction_filter.py` (done, Phase 2.2) strips export-artifact noise via `filter_reactions(messages) -> list[Message]`: a **pure** function (returns a new list; never mutates the input or its `Message`s — clones changed ones with pydantic v2 `model_copy`) that does NOT parse and does NOT call an LLM. Because `chat_parser` has already merged every multi-line message — including wrapped reactions whose closing quote was stranded on its own line (Pattern 7) — into one `.content`, it checks each content as a **single string** and keys off a fixed opener vocabulary at the *start*, rather than scanning line by line. It drops reaction/tapback pseudo-messages in both English **and** Spanish, across quoted (`Loved “…`), image (`Loved an image`), and emoji (`Reacted 😀 to “…`) forms — matched with **curly** quotes (U+201C), since a straight `"` matches zero reactions (Pattern 10). The opener set was verified exhaustive against all three real logs. Bare `[photo]` tokens are stripped but their captions are kept (captions can be real lore, Pattern 8). URL / link-only filtering is intentionally NOT here — a bare link is human-typed content, not an export artifact, so judging it is deferred to the Phase 3 LLM noise filter.
- `agents/` (planned, Phase 3) — LLM layer. A `BaseAgent` handles the Claude API call (low temperature, retry, strip ``` fences from JSON); specialized agents inherit it.

## Chat log format
Each message ends with a **footer** line — either `+16303463392 MM/DD/YYYY HH:MM:SS` (phone-prefixed sender) or a bare `MM/DD/YYYY HH:MM:SS` (the exporter) — and the phone names the message *above* it (footer, not header). The parser maps the phone through the speaker map; no phone → exporter; an unknown phone falls back to the raw number + a logged warning. The 1-on-1 log has no phone numbers at all, so there the sender is read from body left/right alignment instead. Real exports are truncated mid-message (final footer cut off); the parser recovers that last message rather than dropping it. Full quirk catalogue in `FORMAT_NOTES.md`.

## Testing conventions
Use pytest's `tmp_path` fixture to write throwaway config/log files rather than touching real ones. Test that bad input raises the right error with `pytest.raises(...)`.
