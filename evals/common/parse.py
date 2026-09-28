"""One shared tokenization of a Wiki Output File (WOF) into a ``ParsedWOF`` structure.

Every mechanical check reads from this instead of re-scanning the raw markdown, so the
line-number bookkeeping (used for evidence) and the messy structural judgments (what's a
title vs a section, which headings are entries, where the footnotes live) are decided ONCE
here and the checks stay short.

The two structural calls that matter most:

  * **title vs `#`-section.** The Wikitizer renderer produces either
    ``# Title`` / ``## Section`` / ``### Entry`` (a document title exists) OR
    ``# Section`` / ``## Entry`` (no title). We can't tell those apart from heading levels
    alone, so we use the load-bearing difference: a TITLE is followed by *top matter* (a
    prose subtitle and/or a ``---`` rule) before the first deeper heading, whereas a
    ``#`` SECTION runs straight into its entries. ``opens_with_title`` encodes exactly that,
    independent of how many ``#`` headings exist -- so a stray second ``#`` in a titled doc
    is still caught by ``fc.title.single-h1`` rather than silently reclassifying the doc.
  * **footnotes section (FIX #3).** Located STRUCTURALLY as the section that contains the
    ``[^n]:`` definition lines -- never by matching the title "Footnotes" -- so a WOF that
    names it "Notes"/"References" still grades correctly (the section NAME is substance FC
    deliberately ignores).

Pure: ``re`` + ``dataclasses`` only. No I/O.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- line-level regexes ----------------------------------------------------- #

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# A leading inline anchor on a heading's content: "<a id="slug"></a>Visible Name".
_ANCHOR_LEADS = re.compile(r'^<a id="([^"]*)"></a>\s*(.*)$')
# Any canonical anchor anywhere on a line.
_ANCHOR_ANY = re.compile(r'<a id="([^"]*)"></a>')
# Any <a ...> opening tag (canonical or not) -- used to spot non-canonical anchor syntax.
_ATAG_ANY = re.compile(r"<a\b[^>]*>")
# A kramdown-style trailing anchor "{#slug}" -- a non-canonical form we must flag.
_KRAMDOWN_ANCHOR = re.compile(r"\{#[A-Za-z0-9\-_]+\}")
# A list bullet: "- ...", "* ...", "+ ...".
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
# A bullet that opens with an inline anchor then a bold name (the list-entry shape).
_BULLET_ANCHOR_BOLD = re.compile(
    r'^\s*[-*+]\s+<a id="([^"]*)"></a>\s*\*\*(.+?)\*\*\s*(.*)$'
)
# Intra-document fragment link: "[Display Text](#slug)".
_FRAGMENT_LINK = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")
# A footnote DEFINITION line: "[^12]: ...".
_FN_DEF = re.compile(r"^\[\^(\d+)\]:\s*(.*)$")
# An inline footnote REFERENCE: "[^12]" (not a definition).
_FN_REF = re.compile(r"\[\^(\d+)\]")


def _is_hr(line: str) -> bool:
    """A horizontal rule: a line of three-or-more hyphens (the `---` the golden uses)."""
    s = line.strip()
    return len(s) >= 3 and set(s) == {"-"}


def _is_fence(line: str) -> bool:
    """A fenced-code delimiter line (opens or closes a ``` block)."""
    return line.strip().startswith("```")


# --- parsed pieces ---------------------------------------------------------- #

@dataclass
class Heading:
    level: int
    text: str                       # visible text, with any leading inline anchor stripped
    lineno: int                     # 1-based
    anchor_id: Optional[str]        # from a leading "<a id=...>", else None
    raw: str


@dataclass
class Entry:
    """One entry within a section -- either an ATX-heading entry or a list bullet.

    ``is_grouping`` marks an entry-level ATX heading that carries NO anchor (a sub-grouping
    label like "Could Not Place"); the model check ``fc.entries.grouping-no-anchor`` decides
    whether that's legitimate or an entry that lost its anchor. ``body`` is the entry's prose
    (the paragraph under an ATX entry, or the text after a bullet's bold name)."""

    name: str
    anchor_id: Optional[str]
    lineno: int
    raw: str
    body: str
    is_bullet: bool
    is_grouping: bool = False


@dataclass
class Section:
    title: str
    level: int
    lineno: int
    anchor_id: Optional[str]        # a section header should have NO anchor (checked)
    body: List[Tuple[int, str]] = field(default_factory=list)   # (lineno, line) after the header
    entries: List[Entry] = field(default_factory=list)

    @property
    def has_bullet_entries(self) -> bool:
        return any(e.is_bullet for e in self.entries)


@dataclass
class Anchor:
    id: str
    lineno: int
    context: str                    # "heading" | "bullet" | "inline"


@dataclass
class Link:
    text: str
    slug: str                       # without the leading "#"
    lineno: int


@dataclass
class FnRef:
    n: int
    lineno: int
    start: int                      # column offsets, for the adjacency check
    end: int


@dataclass
class FnDef:
    n: int
    lineno: int
    raw: str                        # the text after "[^n]:"
    line: str                       # the whole definition line


@dataclass
class ParsedWOF:
    lines: List[str]
    code_mask: List[bool]           # parallel to lines: True where the line is inside a fence
    title: Optional[Heading]
    opens_with_title: bool
    h1_count: int
    subtitle: Optional[Tuple[int, str]]
    title_level: int                # 1 if a title exists, else 0
    section_level: int              # S
    entry_level: int                # S + 1
    headings: List[Heading]
    sections: List[Section]
    footnotes_section: Optional[Section]
    anchors: List[Anchor]
    non_canonical_anchors: List[Tuple[int, str]]   # (lineno, offending text)
    links: List[Link]
    fn_refs: List[FnRef]
    fn_defs: List[FnDef]


# --- parsing ---------------------------------------------------------------- #

def _split_heading_anchor(content: str) -> Tuple[Optional[str], str]:
    """From a heading's content, peel a leading "<a id=...>" -> (anchor_id, visible_text)."""
    m = _ANCHOR_LEADS.match(content)
    if m:
        return m.group(1), m.group(2).strip()
    return None, content.strip()


def _bullet_name(text: str) -> Tuple[Optional[str], str, str]:
    """For a bullet's content, return (anchor_id, visible_name, trailing_body).

    The list-entry shape is "<a id=...>**Name** rest"; if it doesn't match we fall back to
    a best-effort name (bold text if any, else the raw text) with anchor None so the
    anchor-bold check can flag it."""
    m = _BULLET_ANCHOR_BOLD.match(text)
    if m:
        return m.group(1), m.group(2).strip(), m.group(3).strip()
    # Fallbacks (malformed bullet): a bold span, else the whole line.
    bold = re.search(r"\*\*(.+?)\*\*", text)
    inner = _BULLET.match(text)
    raw_inner = inner.group(2).strip() if inner else text.strip()
    name = bold.group(1).strip() if bold else raw_inner
    return None, name, raw_inner


def _compute_code_mask(lines: List[str]) -> List[bool]:
    """Mark lines that sit INSIDE a ``` fenced block (the fence lines themselves are False,
    treated as delimiters). Keeps heading/anchor scanning out of code."""
    mask = [False] * len(lines)
    in_code = False
    for i, line in enumerate(lines):
        if _is_fence(line):
            in_code = not in_code
            mask[i] = False           # the fence line itself is a delimiter, not content
            continue
        mask[i] = in_code
    return mask


def _find_headings(lines: List[str], code_mask: List[bool]) -> List[Heading]:
    headings: List[Heading] = []
    for i, line in enumerate(lines):
        if code_mask[i]:
            continue
        m = _ATX_HEADING.match(line)
        if not m:
            continue
        level = len(m.group(1))
        anchor_id, text = _split_heading_anchor(m.group(2))
        headings.append(Heading(level=level, text=text, lineno=i + 1,
                                 anchor_id=anchor_id, raw=line))
    return headings


def _detect_title(lines: List[str], code_mask: List[bool], headings: List[Heading]):
    """Return (opens_with_title, title_or_None).

    A TITLED doc uses three heading tiers (``#`` title / ``##`` section / ``###`` entry); an
    UNTITLED doc uses only two (``#`` section / ``##`` entry). So:
      1. Primary tell -- a leading ``#`` with a level-3+ heading anywhere below it is titled,
         even when the top matter (subtitle/HR) is missing (this catches a titled doc that
         merely forgot its subtitle, so its missing-subtitle violation still surfaces).
      2. Secondary -- a leading ``#`` then a ``##``, with top matter between them: an ``---``
         rule OR a plain PROSE subtitle line. A list bullet or an anchor-bearing entry line is
         NOT top matter (it's the bulleted first section of an untitled doc), so it's excluded.
    See the module docstring for why this beats a heading-count heuristic.
    """
    if not headings:
        return False, None
    first = headings[0]
    if first.level != 1:
        return False, None
    following = [h for h in headings if h.lineno > first.lineno]
    if not following:
        return False, None
    if any(h.level >= 3 for h in following):        # three-tier ladder -> titled
        return True, first
    nxt = following[0]
    if nxt.level != 2:
        return False, None
    for i in range(first.lineno, nxt.lineno - 1):   # 0-based lines strictly between
        line = lines[i]
        if code_mask[i]:
            continue
        if _is_hr(line):
            return True, first
        if (line.strip() and not _ATX_HEADING.match(line)
                and not _BULLET.match(line) and not _ANCHOR_ANY.search(line)):
            return True, first
    return False, None


def _find_subtitle(lines: List[str], code_mask: List[bool],
                   title: Optional[Heading]) -> Optional[Tuple[int, str]]:
    """The first non-blank, non-``---``, non-heading line after the title -- the descriptive
    prose line the title should carry."""
    if title is None:
        return None
    for i in range(title.lineno, len(lines)):       # 0-based, starting just after the title
        line = lines[i]
        if code_mask[i]:
            continue
        if not line.strip():
            continue
        if _is_hr(line) or _ATX_HEADING.match(line):
            return None                             # hit the next structural line first
        return (i + 1, line.strip())
    return None


def _extract_entries(section: Section, entry_level: int,
                     lines: List[str], code_mask: List[bool]) -> None:
    """Populate ``section.entries`` from its body: entry-level ATX headings (and anchorless
    grouping headings) plus anchor-bearing list bullets. ATX-entry bodies run to the next
    entry/heading/HR; bullet bodies are the text after the bold name."""
    body = section.body
    n = len(body)
    idx = 0
    # True while a REAL (anchored) ATX entry is "open": its trailing top-level bullets are detail
    # content of that entry's body, NOT entries of their own. A grouping heading (anchorless,
    # e.g. "Could Not Place") does NOT open one -- bullets beneath it are genuine entries. This
    # keeps a bulleted detail-list inside an entity page from becoming phantom bullet entries,
    # while History (bullets, then a grouping heading, then more bullets) still parses correctly.
    suppress_bullets = False
    while idx < n:
        lineno, line = body[idx]
        zero = lineno - 1
        heading = _ATX_HEADING.match(line) if not code_mask[zero] else None
        bullet_m = _BULLET.match(line) if not code_mask[zero] else None
        # Only a TOP-LEVEL bullet (no indentation) is a candidate entry; an indented sub-bullet
        # is body content.
        bullet_ab = bullet_m if (bullet_m and not bullet_m.group(1)) else None
        if heading and len(heading.group(1)) == entry_level:
            anchor_id, text = _split_heading_anchor(heading.group(2))
            is_grouping = anchor_id is None
            suppress_bullets = not is_grouping
            # Body = following lines until the next heading / HR / bullet.
            j = idx + 1
            body_lines = []
            while j < n:
                _ln, _line = body[j]
                if (_ATX_HEADING.match(_line) or _is_hr(_line) or _BULLET.match(_line)):
                    break
                if _line.strip():
                    body_lines.append(_line.strip())
                j += 1
            section.entries.append(Entry(
                name=text, anchor_id=anchor_id, lineno=lineno, raw=line,
                body=" ".join(body_lines), is_bullet=False, is_grouping=is_grouping,
            ))
            idx = j
            continue
        if bullet_ab:
            if suppress_bullets:                    # a detail bullet inside a real ATX entry's body
                idx += 1
                continue
            anchor_id, name, trailing = _bullet_name(line)
            section.entries.append(Entry(
                name=name, anchor_id=anchor_id, lineno=lineno, raw=line,
                body=trailing, is_bullet=True, is_grouping=False,
            ))
            idx += 1
            continue
        idx += 1


def _build_sections(lines: List[str], code_mask: List[bool], headings: List[Heading],
                    section_level: int, entry_level: int) -> List[Section]:
    sec_heads = [h for h in headings if h.level == section_level]
    sections: List[Section] = []
    for k, h in enumerate(sec_heads):
        start = h.lineno                            # 1-based; body starts on the next line
        end = sec_heads[k + 1].lineno - 1 if k + 1 < len(sec_heads) else len(lines)
        sec = Section(title=h.text, level=h.level, lineno=h.lineno, anchor_id=h.anchor_id)
        for z in range(start, end):                 # 0-based lines after the header, up to next section
            sec.body.append((z + 1, lines[z]))
        _extract_entries(sec, entry_level, lines, code_mask)
        sections.append(sec)
    return sections


def _collect_anchors(lines, code_mask, headings, sections):
    """All canonical anchors (with context) + any non-canonical anchor-like tokens."""
    heading_lines = {h.lineno for h in headings}
    bullet_lines = {e.lineno for s in sections for e in s.entries if e.is_bullet}
    anchors: List[Anchor] = []
    non_canonical: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        if code_mask[i]:
            continue
        lineno = i + 1
        for m in _ANCHOR_ANY.finditer(line):
            if lineno in heading_lines:
                ctx = "heading"
            elif lineno in bullet_lines:
                ctx = "bullet"
            else:
                ctx = "inline"
            anchors.append(Anchor(id=m.group(1), lineno=lineno, context=ctx))
        # Non-canonical anchor forms: any <a ...> tag that isn't the exact canonical anchor,
        # and any kramdown {#slug}. (A canonical "<a id=x></a>" also matches _ATAG_ANY, so
        # subtract those.)
        canonical_spans = [m.span() for m in re.finditer(r'<a id="[^"]*"></a>', line)]
        for m in _ATAG_ANY.finditer(line):
            if not any(cs[0] <= m.start() < cs[1] for cs in canonical_spans):
                non_canonical.append((lineno, m.group(0)))
        for m in _KRAMDOWN_ANCHOR.finditer(line):
            non_canonical.append((lineno, m.group(0)))
    return anchors, non_canonical


def _collect_links(lines, code_mask) -> List[Link]:
    links: List[Link] = []
    for i, line in enumerate(lines):
        if code_mask[i]:
            continue
        for m in _FRAGMENT_LINK.finditer(line):
            links.append(Link(text=m.group(1), slug=m.group(2), lineno=i + 1))
    return links


def _collect_footnotes(lines, code_mask, sections):
    """Return (fn_refs, fn_defs, footnotes_section). The footnotes section is located
    STRUCTURALLY (FIX #3) as the section containing the ``[^n]:`` definition lines."""
    fn_defs: List[FnDef] = []
    def_linenos = set()
    for i, line in enumerate(lines):
        if code_mask[i]:
            continue
        m = _FN_DEF.match(line)
        if m:
            fn_defs.append(FnDef(n=int(m.group(1)), lineno=i + 1, raw=m.group(2), line=line))
            def_linenos.add(i + 1)

    fn_refs: List[FnRef] = []
    for i, line in enumerate(lines):
        if code_mask[i] or (i + 1) in def_linenos:
            continue                                # skip definition lines; their "[^n]:" isn't a ref
        for m in _FN_REF.finditer(line):
            # A "[^n]" immediately followed by ":" would be a (malformed) def start; skip it.
            if m.end() < len(line) and line[m.end()] == ":":
                continue
            fn_refs.append(FnRef(n=int(m.group(1)), lineno=i + 1, start=m.start(), end=m.end()))

    footnotes_section = None
    for sec in sections:
        if any(lineno in def_linenos for lineno, _ in sec.body):
            footnotes_section = sec
            break
    return fn_refs, fn_defs, footnotes_section


def parse_wof(text: str) -> ParsedWOF:
    """Tokenize a WOF's markdown into a ``ParsedWOF`` (all the checks read from this)."""
    lines = text.split("\n")
    code_mask = _compute_code_mask(lines)
    headings = _find_headings(lines, code_mask)
    h1_count = sum(1 for h in headings if h.level == 1)

    opens_with_title, title = _detect_title(lines, code_mask, headings)
    title_level = 1 if opens_with_title else 0
    if opens_with_title:
        section_level = 2
    else:
        section_level = min((h.level for h in headings), default=1)
    entry_level = section_level + 1

    subtitle = _find_subtitle(lines, code_mask, title)
    sections = _build_sections(lines, code_mask, headings, section_level, entry_level)
    anchors, non_canonical = _collect_anchors(lines, code_mask, headings, sections)
    links = _collect_links(lines, code_mask)
    fn_refs, fn_defs, footnotes_section = _collect_footnotes(lines, code_mask, sections)

    return ParsedWOF(
        lines=lines,
        code_mask=code_mask,
        title=title,
        opens_with_title=opens_with_title,
        h1_count=h1_count,
        subtitle=subtitle,
        title_level=title_level,
        section_level=section_level,
        entry_level=entry_level,
        headings=headings,
        sections=sections,
        footnotes_section=footnotes_section,
        anchors=anchors,
        non_canonical_anchors=non_canonical,
        links=links,
        fn_refs=fn_refs,
        fn_defs=fn_defs,
    )
