# wikitizer
Agentic project (Claude Code) to wikify worldbuilding notes. Intended to be a free basic alternative to sites like WorldAnvil.
Requires an Anthropic API key and/or an OpenRouter API key to run locally.
This project is currently undergoing end-to-end testing, debugging, and an API-driven refactor.

## Full Pipeline:

### Finished
imessage-exporter TXT chat logs in `logs/` -> `python scripts/build_imessage_speaker_map.py` for handle→name pairs in `config/speaker_map.json` -> conversations go through the parser -> conversations go through noise extractor agent -> 6 extraction agents extract relevant lore information from .txt chat logs (sonnet, Locations, History, Characters, Organizations, Items, and People & Cultures) -> extracted lore goes to reconciler for dedup/timeline ordering -> footnotes are built for quote attribution/anti-hallucination -> cross-link pass assigns slugs to linkable content (i.e. content that will hyperlink to another wiki entry when clicked) -> content goes through the markdown renderer -> renderer delivers extracted lore wiki-style in a new separate `.md` file.

### In Progress
Content goes through the markdown renderer (after cross-link pass) -> markdown renderer delivers extracted lore wiki-style in a separate `.md` file.

## Running it
Two config files must be built **before** the pipeline (they're the ground-truth anchors for
speaker attribution and player↔character identity), so the run is a short, ordered sequence:

**0. Setup.** Create a virtualenv and put your key in `.env`:
```bash
source venv/bin/activate
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env      # (or OPENROUTER_API_KEY, for an OpenRouter backend)
```

**1. Build the speaker map** (contact/handle → real name) from your
[imessage-exporter](https://github.com/ReagentX/imessage-exporter) TXT exports (`imessage-exporter -f txt ...`),
written to the default `config/speaker_map.json`:
```bash
python scripts/build_imessage_speaker_map.py logs/*.txt -o config/speaker_map.json
```

**2. Build the player map** — REQUIRED. Declare each player's character(s): main name (the
heading), an optional last name (paired with the main + alias names for recognition, so
"Kriggy", "Krigius", "Kriggy Krieger", "Krigius Krieger" all fold to one page), optional
aliases, and pronouns (default they/them). A player may have more than one PC; non-players
(the DM, a guest) are skippable. Written to `config/player_map.json`:
```bash
python scripts/build_player_map.py
```

**3. Run the pipeline** (the pipeline refuses to run without a player map — or pass
`--no-player-map` to opt out):
```bash
python main.py --files logs/*.txt                                   # the full wiki
python main.py --files logs/*.txt --exclude-sources dm.txt          # + a restricted copy with dm.txt's secrets hidden
```

### CLI flags (`main.py`)
| Flag | Meaning | Example |
| --- | --- | --- |
| `--files` | **Required.** The chat-log file(s) to ingest (shell globs work). | `--files logs/*.txt` |
| `--output` | Where the full wiki is written (default `output/wiki.md`). | `--output output/gol.md` |
| `--exclude-sources` | Source filename(s) whose secrets are hidden from a SECOND restricted doc (a players' copy). Bare filenames. | `--exclude-sources dm.txt` |
| `--speaker-map` | Path to the speaker map (default `config/speaker_map.json`). | `--speaker-map config/sm.json` |
| `--player-map` | Path to the declared-party map (default `config/player_map.json`). | `--player-map config/party.json` |
| `--current-year` | Present-day in-world year, to resolve relative dates ("200 years ago"). | `--current-year 1424` |
| `--cache` | Reuse cached LLM responses for identical extractor/noise calls (gitignored `.llm_cache/`); cheap re-runs while debugging. | `--cache` |
| `--no-player-map` | Opt out of the REQUIRED player map (party-less/test run; PCs may duplicate or mis-attribute). | `--no-player-map` |
| `--confirm-players` | After the run, interactively assign a player to each discovered PC and save it back to `--player-map`; takes effect next run. Only the player is rewritten — each character's `main_name`/`last_name`/`aliases`/`pronouns` are preserved. (The step-2 builder is still the richer path for building one from scratch.) | `--confirm-players` |

## Input format
The pipeline ingests **imessage-exporter TXT exports** — the structured TXT the [imessage-exporter](https://github.com/ReagentX/imessage-exporter) CLI writes (`imessage-exporter -f txt ...`). Each message carries its sender on its own line, so speaker attribution is exact (no alignment guessing) and reactions/attachments/receipts are stripped structurally. The exporter's own messages (`Me`) map to the `"exporter"` key in `config/speaker_map.json`; if you export with `--custom-name`, add that name to the speaker map. Phone-number handles match E.164 speaker-map keys even when prettily formatted.

### Future
- Support for extracted RCS/SMS logs
- Support for extracted Discord logs
- Support for notes in text documents
- ~~Source selection and multi-doc generation capabilities (i.e. "include all lore from the three given sources in the real wiki document, and include lore from Source A and Source B but not Source C in a secondary fake wiki document so I can keep Source C's info secret from my players")~~ COMPLETE
- GUI
