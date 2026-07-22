# wikitizer
Agentic project (Claude Code) to wikify worldbuilding notes. Intended to be a free basic alternative to sites like WorldAnvil.
Requires an Anthropic API key and/or an OpenRouter API key to run locally.
This project is currently undergoing end-to-end testing, debugging, and an API-driven refactor.

## Full Pipeline:

### Finished
iMessage .txt chat logs in `logs/` -> `python scripts/build_speaker_map.py` for name-number pairs in `config/speaker_map.json` -> conversations go through parser/reaction filter -> conversations go through noise extractor agent -> 6 extraction agents extract relevant lore information from .txt chat logs (sonnet, Locations, History, Characters, Organizations, Items, and People & Cultures) -> extracted lore goes to reconciler for dedup/timeline ordering -> footnotes are built for quote attribution/anti-hallucination -> cross-link pass assigns slugs to linkable content (i.e. content that will hyperlink to another wiki entry when clicked) -> content goes through the markdown renderer -> renderer delivers extracted lore wiki-style in a new separate `.md` file.

### In Progress
Content goes through the markdown renderer (after cross-link pass) -> markdown renderer delivers extracted lore wiki-style in a separate `.md` file.

## Input formats
Two chat-log formats are supported, selected with `--input-format {auto,imessage,legacy}` (default `auto`, which sniffs each file):
- **`imessage`** — a structured TXT export from the [imessage-exporter](https://github.com/ReagentX/imessage-exporter) CLI (`imessage-exporter -f txt ...`). **Recommended:** each message carries its sender on its own line, so speaker attribution is exact (no alignment guessing) and reactions/attachments are stripped structurally. The exporter's own messages (`Me`) map to the `"exporter"` key in `config/speaker_map.json`; if you export with `--custom-name`, add that name to the speaker map. Phone-number handles match E.164 speaker-map keys even when prettily formatted.
- **`legacy`** — the older copy-pasted iMessage `.txt` (participant line + dashes + footers). Still fully supported.

Both produce the same internal message stream, so everything downstream is identical.

### Future
- Support for extracted RCS/SMS logs
- Support for extracted Discord logs
- Support for notes in text documents
- ~~Source selection and multi-doc generation capabilities (i.e. "include all lore from the three given sources in the real wiki document, and include lore from Source A and Source B but not Source C in a secondary fake wiki document so I can keep Source C's info secret from my players")~~ COMPLETE
- GUI
