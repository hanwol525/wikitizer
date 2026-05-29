# FORMAT_NOTES.md

Notes on the quirks of the exported iMessage chat logs (`dndgroup.txt`,
`royalty.txt`, `dm_convo.txt`), written during Phase 1.4 so future-me doesn't
get blindsided in Phase 2 when building the parser.

The big idea: these files **look** like "one line of text per message" but
they really aren't. There's a pile of formatting weirdness baked in by whatever
exported them. Every item below is something the parser has to survive.

---

## Read this first: the three things most likely to bite you

1. **The line endings are `\r\r\n` (carriage-return, carriage-return,
   line-feed), not normal `\n`.** If you open the file the normal way
   (`open(path).read()`), Python's "universal newlines" turns every `\r\r\n`
   into **two** `\n`, so you get a phantom blank line after *every* real line.
   `dm_convo.txt` is 34 real lines but reads as 68 if you do this wrong. Read
   raw bytes and strip the `\r`s yourself (helper below). Related gotcha: anchor
   any regex with `\s*$`, **not** `$` — the trailing whitespace/carriage returns
   otherwise defeat `$`. The footer regex from the project plan matches **0**
   lines on the raw bytes and 10 once the carriage returns are removed, so
   normalize *before* running any regex.

2. **The phone-number / timestamp line is a FOOTER, not a header.** This is the
   one the project plan has backwards. The `+phone  timestamp` line comes
   *after* a message and names the sender of the message **above** it — not the
   one below it. A message with **no** phone on its footer was sent by the
   **exporter**. (See Pattern 2 for the full shape, including the timestamp
   nuance.)

3. **The 1-on-1 file (`dm_convo.txt`) has NO phone numbers anywhere**, so the
   footer can't tell you who's who there. You have to use left/right text
   **alignment** instead (right = exporter, left = the other person). The naive
   "no phone number = the exporter" shortcut would mark *every* line in this
   file as the exporter, which is wrong (see Pattern 3).

A safe way to read the files (Python 3.9.6):

```python
def read_clean(path: str) -> list[str]:
    # Read as raw bytes, decode ourselves, then nuke ALL carriage returns.
    # This collapses \r\r\n (and the occasional \r\n) down to clean lines.
    raw = open(path, "rb").read().decode("utf-8")
    return raw.replace("\r", "").split("\n")
```

Do **not** use plain `open(path).read()` / `Path.read_text()` here — universal
newline handling will mangle the `\r\r\n` into double blank lines.

---

## 1. The file header block (first two lines)

Every file starts with a participant line, then a line of exactly 100 dashes,
*then* the messages. The participant line differs by file type.

**Group chat** — a leading comma, then the comma-separated phone numbers of the
*other* participants. The leading comma is the exporter's own (empty) slot, so
the exporter is **not** in this list (that's why `speaker_map.json` keeps the
exporter as a separate `"exporter"` key):

```
, +15555550101, +15555550102, +15555550103, +15555550104, +15555550105
----------------------------------------------------------------------------------------------------
```

**1-on-1 chat** — just the single other person's number, no leading comma:

```
+15555550101
----------------------------------------------------------------------------------------------------
```

Parser implication: skip these two lines (or use line 1 to learn who the "other
person" is — important for `dm_convo.txt`, see Pattern 3).

---

## 2. Every message ends with a FOOTER line (two shapes)

A message is one or more **body** lines, followed by a **footer** line. The
footer has one of two shapes, and that's how you both (a) find where a message
ends and (b) learn who sent it.

**(a) Phone footer** — someone other than the exporter. The body comes first,
then a line with the phone number at column 0 and `MM/DD/YYYY HH:MM:SS` after a
wall of spaces. The phone is the sender of the message *above*:

```
If you don’t want to answer yet that’s fine. I’ll be lore dumping
world info tomorrow. You can decide after that if you’d like
+15555550101                                        05/13/2026 18:52:38
```

**(b) Bare footer** — the exporter. Same idea, but the footer is just an
indented timestamp with **no phone number**:

```
                                       I have a dwarf tank build i wanted to try maybe ill play that
                                        05/13/2026 19:54:05
```

The catch the project plan gets wrong: **the phone on a footer line belongs to
the message above it**, so to attribute a message you look at the footer that
comes *after* its body. (The *timestamp* on a footer line actually pairs with
the message *below* it, and the file opens with a lone timestamp for the very
first message — but for sender attribution you only care about the phone, and it
always refers to the block just above.)

Rough regexes (run them on **normalized** text, and prefer `\s*$` over `$`):

```python
import re
PHONE_FOOTER = re.compile(r"^\+\d+\s+\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s*$")
BARE_FOOTER  = re.compile(r"^\s*\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s*$")
```

Note: `BARE_FOOTER` will *also* match a phone footer's timestamp portion if
you're not careful, so check `PHONE_FOOTER` **first**, then fall back to
`BARE_FOOTER`.

---

## 3. `dm_convo.txt` has no phone numbers — use alignment instead (IMPORTANT)

This is the sneakiest one. In `dm_convo.txt` **every** footer is a bare
timestamp — there are zero `+phone` footers in the whole file (I checked:
hundreds in the group files, exactly one `+phone` line in `dm_convo.txt`, and
that one is the participant header at the very top).

So the footer can't tell the two people apart. Instead, the message **body** is
aligned:

- **Right-aligned body** (heavy leading whitespace, text pushed to ~column 100)
  = the **exporter**.
- **Left-aligned body** (starts at column 0) = the **other person** — whoever's
  number is in the file's header line.

Both speakers' *timestamps* are indented the same way, so the timestamp doesn't
help — only the body alignment does.

Exporter (right-aligned):

```
                                    Do any cultures in the world correspond in naming to real world
                                cultures? Or what kinda names do ppl have, Im thinking of names for
                                                                                              my guy
```

Other person (left-aligned):

```
I’m not quite that meticulous, go for whatever name you’d like
```

Parser implication: detect "1-on-1 mode" (single number in the header line + no
`+phone` footers in the body) and, in that mode, assign the sender by measuring
the leading whitespace of the first body line. In the group files you don't have
to rely on this — there the footer phone gives you the exact sender (see
Pattern 4).

---

## 4. Alignment is a SENDER signal, not noise — but trailing space is noise

Careful here, because the two kinds of whitespace mean different things:

- **Leading whitespace (alignment) is meaningful.** Even in the **group** files,
  right-aligned vs left-aligned tracks who's talking: right = the exporter,
  left = everyone else. (In dndgroup, 103 of the 104 exporter messages are
  right-aligned, and **zero** non-exporter messages are right-aligned.) So
  alignment cleanly separates *exporter vs not* — but it does **not** tell the
  non-exporters apart from each other (they're all left-aligned). To know *which*
  non-exporter sent a group message, use the footer phone (Pattern 2). The
  footer phone and the alignment agree, so either works for spotting the
  exporter; the phone is what pins the exact person.
- **Trailing whitespace is noise.** Basically every wrapped line carries a
  trailing space before the break. Strip it.

So: read the alignment (or the footer) to get the sender *first*, then `.strip()`
each body line before storing it. The one caution: stripping also flattens the
blank lines that live *inside* a multi-line message (Pattern 6) — decide on
purpose whether to keep those as paragraph breaks or collapse them.

---

## 5. Reactions / tapbacks (these are noise — filter them out)

By far the most common "weird" line. When someone taps back on a message, the
export writes a whole pseudo-message that quotes the original in **curly**
quotes (`“ ”`, U+201C / U+201D — *not* straight `"`). These carry no lore and
should be dropped by the reaction filter in Phase 2.2.

Counts across all three files: `Loved “` ×65, `Laughed at “` ×43,
`Emphasized “` ×23, `Liked “` ×4, `Disliked “` ×3, plus the rarer ones below.

English text reactions — `Loved`, `Liked`, `Laughed at`, `Emphasized`,
`Disliked`, `Removed a laugh from`, and `Reacted <emoji> to`:

```
Loved “Dungeon Master here. Who wants to play a new campaign
designed by yours truly?”
```

```
Reacted 🤫 to “Is Silvery Barbs banned in this campaign or am I
allowed to be a problem”
```

Image reactions — same idea but on a photo, so there's no quoted text:

```
Loved an image
```

```
Emphasized an image
```

**Watch out — some reactions are localized into Spanish.** Same chat, same
people; the exporter's locale leaked through. Your filter has to catch these too
or they'll sail through as fake content:

| Spanish | English equivalent |
|---|---|
| `Le encantó “…”` | Loved |
| `Le dio risa “…”` | Laughed at |
| `Exclamó por “…”` | Emphasized |
| `Reaccionó con <emoji> a “…”` | Reacted `<emoji>` to |
| `Le encantó una imagen` | Loved an image |

```
Le encantó “All hail emperor Terrasque Krieger”
```

```
Reaccionó con 😂 a “Mundi probably got shape humanoid”
```

Filter strategy: match lines that (after stripping) **start with** any of these
openers. Don't try to match the closing quote on the same line — see Pattern 7,
the quote often wraps onto later lines.

---

## 6. Multi-line messages (one message, many lines, sometimes blank lines)

A single message frequently wraps across many lines. There is **one** footer at
the bottom, and the body runs from the previous footer down to it. Crucially, the
body can contain **blank lines** (the sender pressed Enter for a paragraph
break).

This one message (one body, footer at 18:57:54) spans 7 lines including a blank
line:

```
I will say that I’ve never been more excited about a homebrewed
setting that I’ve crafted than I am with this one. I feel very good
about the flexibility with the plot, with the world history, the
politics, everything.

That being said, I’m more than willing to
change things up to make sure your character and their backstory fit
perfectly
```

The huge Lake Mundi lore dump (one message) is the extreme case — it runs dozens
of lines with several internal blank-line paragraph breaks.

Parser implication: **do not split messages on blank lines.** Message boundaries
come *only* from the footer regexes (Pattern 2). Use a "state machine": keep
appending lines to the current message until you hit a footer line, then close
that message (the footer gives you its sender) and start the next. A blank line
in the middle is just part of the body.

---

## 7. Multi-line reactions — the closing quote lands on its own line

When a reaction quotes a message that was itself long, the quoted text wraps,
and the closing curly quote `”` very often ends up **alone on its own line**:

```
                               Loved “If you don’t want to answer yet that’s fine. I’ll be lore
                               dumping world info tomorrow. You can decide after that if you’d like
                                                                                                  ”
```

This is why the reaction filter should key off the **opening** word
(`Loved “`, `Laughed at “`, etc.) rather than trying to match a full
`Loved “...”` on a single line — the closing `”` is frequently somewhere below.
Practically: once you detect a reaction's opening line, consume body lines until
the next footer, and throw the whole block away.

---

## 8. `[photo]` lines (and the caption underneath is real content)

An attached image shows up as a literal `[photo]` line (14 of them across the
files). Sometimes that's the whole message. But sometimes the **next line is a
caption**, and captions can contain actual lore — so don't blanket-delete the
line after a `[photo]`.

Bare photo:

```
[photo]
```

Photo **with** a caption that matters (this one literally describes the map of
the continent — that's lore):

```
[photo]
Very simple map here of the continent of Gol. Countries and major
natural landmarks are present. Almost every country southeast of the
Cloud Mountains are under the control of the Krieger Imperium
```

Parser implication: strip the `[photo]` token itself (it's noise), but keep
parsing the following lines as normal body — let the noise-filter / extractor
agents decide whether the caption is lore.

---

## 9. URLs — on their own line, and sometimes wrapped mid-URL

Links appear as their own body lines. Most are fine:

```
https://dnd5e.wikidot.com/feat:spear-mastery
https://www.dndbeyond.com/characters/000000000
```

But at least one (a Discord CDN attachment link) is **wrapped across three
lines mid-URL**, with each fragment indented (the real one had an attachment
filename + IDs + a hash; the placeholder below keeps the shape):

```
                              https://cdn.example.com/attachments/000000000000000000/00000000000000
                              0000/file-000000000-0.pdf?ex=000000&is=000000&hm=0000000000000000000000
                                             0000000000000000000000000000000000000000000000000000000&
```

Parser implication: this is really just a normal multi-line message body (one
footer, three lines) — the state machine from Pattern 6 already handles it,
*as long as* you don't choke on a line that's a bare URL fragment. If you ever
need the URL whole, you'd have to rejoin the lines, but for lore extraction
these are noise anyway.

---

## 10. Special characters: curly quotes, em-dashes, ellipses, emoji

The text is full of "smart" punctuation, not ASCII:

- Curly apostrophe `’` (U+2019), e.g. `I’ve`, `don’t` — *not* `'`.
- Curly double quotes `“ ”` (U+201C / U+201D) — these wrap every reaction's
  quoted text, so your filter regexes must use the curly characters (or a
  character class that includes both curly and straight).
- Em-dashes `–`/`—`, ellipsis `…`, and emoji (😒 🤫 😂 🥺 …) show up inside
  both message bodies and reactions.

Always decode as **UTF-8** (the `read_clean` helper above does). Don't assume
ASCII anywhere.

---

## 11. Very short / near-empty messages

Plenty of message bodies are a single word or even a single character. These are
legitimate messages (not footers, not reactions), so boundary detection must not
trip over them. Examples seen: `2014`, `Dope`, `lol`, `Fuck`, and a message that
is literally just `?`:

```
                                                                                                   ?
                                        05/18/2026 13:16:10
```

Parser implication: don't assume a "real" message has some minimum length, and
don't treat a 1-character body as a parsing error.

---

## Quick reference / checklist for the Phase 2 parser

- [ ] Read via raw-bytes + `.replace("\r", "")` — never plain text mode.
- [ ] Skip the first two lines (participant line + 100 dashes).
- [ ] Footers, not headers: a message is body line(s) **then** a footer; the
      footer's phone names the message **above** it. No phone ⇒ exporter
      (Pattern 2).
- [ ] Detect mode: if line 1 has a single number and the body has no `+phone`
      footers → **1-on-1 mode** (assign sender by body alignment: right =
      exporter, left = the other person, Pattern 3). Otherwise → **group mode**
      (footer phone = exact sender; alignment agrees but only separates exporter
      vs others, Pattern 4).
- [ ] Check `PHONE_FOOTER` before `BARE_FOOTER`; anchor with `\s*$`, not `$`.
- [ ] State machine: accumulate body lines until the next footer (Patterns 6, 7,
      9). Never split on blank lines.
- [ ] Read alignment/footer to get the sender, *then* `.strip()` body lines
      (Pattern 4); decide what to do with internal blank lines.
- [ ] Reaction filter keys off the **opening** word, English **and** Spanish,
      including image variants (Patterns 5, 7).
- [ ] Strip the `[photo]` token but keep the following caption lines (Pattern 8).
- [ ] Decode as UTF-8; match curly quotes, not straight ones (Pattern 10).

---

*Files studied: `dndgroup.txt` (1612 raw lines), `royalty.txt` (613),
`dm_convo.txt` (34). Reaction/format counts above were taken from a one-off
scan of all three. Phone numbers, the Discord attachment link, and the character-
sheet ID in the examples have been replaced with placeholders; names are shown
by role.*
