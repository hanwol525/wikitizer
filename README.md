Besties this is not done. Please be demure and don't use this, queen. I need it public for reasons.
# wikitizer
Agentic project (Claude Code) to wikify worldbuilding notes. Free basic alternative to sites like WorldAnvil.
Requires an Anthropic API key to run locally.

## Full Pipeline:

### Finished
iMessage .txt chat logs in `logs/` -> `python scripts/build_speaker_map.py` for name-number pairs in `config/speaker_map.json` -> conversations go through parser/reaction filter -> conversations go through noise extractor agent (haiku) -> 6 extraction agents extract relevant lore information from .txt chat logs (sonnet, Locations, History, Characters, Organizations, Items, and People & Cultures) -> extracted lore goes to reconciler for dedup/timeline ordering -> footnotes are built for quote attribution/anti-hallucination -> cross-link pass assigns slugs to linkable content (i.e. content that will hyperlink to another wiki entry when clicked)

### In Progress
Content goes through the markdown renderer (after cross-link pass) -> markdown renderer delivers extracted lore wiki-style in a separate `.md` file.

### Future
- Support for extracted RCS/SMS logs
- Support for extracted Discord logs
- Support for notes in text documents
- Source selection and multi-doc generation capabilities (i.e. "include all lore from the three given sources in the real wiki document, and include lore from Source A and Source B but not Source C in a secondary fake wiki document so I can keep Source C's info secret from my players")
- GUI
