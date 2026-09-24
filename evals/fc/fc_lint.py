"""The 24 DETERMINISTIC (mechanical) FC checks, plus the candidate extraction that feeds
the 3 model checks.

Each ``check_*`` reads a ``ParsedWOF`` and returns one ``FCItem`` (``engine="mechanical"``).
Evidence is required on every item: terse on a pass (a count / "present"), specific on a
fail/partial (line numbers + the offending value, and ``expected``/``found`` where it reads
naturally). ``run_lint`` runs all 24 in a fixed ID order.

The golden-file conventions several checks encode (slug shape, article/`-N` tolerance,
"ignore leading The" collation, the tolerant footnote-def regex) are only as correct as the
rules pinned here -- so they are validated against ``output/gol-lore-full.md`` in the tests.

The three MODEL checks (``fc.entries.grouping-no-anchor``, ``fc.list-entries.approx-tilde``,
``fc.markers.tbd``) are NOT graded here; this module only pre-extracts their tiny candidate
sets (``grouping_candidates`` / ``figure_candidates`` / ``tbd_candidates``) so the adjudicator
never sees the whole file.
"""

import re
from typing import List

from evals.fc.models import Engine, Evidence, FCItem, Status
from evals.fc.parse import ParsedWOF
from evals.fc.slugify import (
    anchor_matches_name,
    has_non_ascii_letter,
    slugify,
    strip_leading_article,
)

# --- criterion id -> short prose (informational `description`; ids are the join key) --- #

CRITERION_TEXT = {
    "fc.title.single-h1": "one title H1 at top (na if no title)",
    "fc.title.subtitle-present": "prose line under the title (na if no title)",
    "fc.sections.consistent-depth": "section headers all one level below the title",
    "fc.sections.no-anchor": "no anchor on section headers",
    "fc.layout.hr-separators": "--- between top matter and each section",
    "fc.entries.depth": "entries one level below their section",
    "fc.entries.inline-anchor": "anchor placed right before each entry name",
    "fc.entries.alphabetized": "entries A-Z within a section, ignoring leading 'The'",
    "fc.entries.grouping-no-anchor": "group label vs entry-missing-anchor",
    "fc.anchors.html-form": "raw <a id=...></a> anchor form",
    "fc.anchors.unique-ids": "all anchor ids unique",
    "fc.slugs.kebab-case": "slugs lowercase, hyphen-joined",
    "fc.slugs.punctuation": "apostrophes/quotes dropped, parens flattened in slugs",
    "fc.slugs.non-ascii": "non-ASCII letters preserved in slugs",
    "fc.slugs.disambiguation": "-N suffix for cross-section name clashes",
    "fc.links.format": "well-formed [text](#slug)",
    "fc.links.resolve": "every #slug resolves to a real id",
    "fc.list-entries.anchor-bold": "bullet starts with anchor + bold name",
    "fc.list-entries.approx-tilde": "approximate figures marked ~",
    "fc.fnref.format": "[^n] marker format",
    "fc.fnref.adjacent": "stacked refs, no separator",
    "fc.fndef.collected": "all defs in one end section",
    "fc.fndef.numeric-order": "defs in ascending order",
    "fc.fndef.format": "`quote` - Name, file.txt shape",
    "fc.fndef.multi-quote-sep": "multi-quotes joined by ' / '",
    "fc.markers.tbd": "unmarked stubs that should be [TBD]",
    "fc.markers.asterisk-format": "trailing asterisk form/placement uniform",
}

# The 24 mechanical ids in canonical (report) order, and the 3 model ids.
MECHANICAL_IDS = [
    "fc.title.single-h1", "fc.title.subtitle-present", "fc.sections.consistent-depth",
    "fc.sections.no-anchor", "fc.layout.hr-separators", "fc.entries.depth",
    "fc.entries.inline-anchor", "fc.entries.alphabetized", "fc.anchors.html-form",
    "fc.anchors.unique-ids", "fc.slugs.kebab-case", "fc.slugs.punctuation",
    "fc.slugs.non-ascii", "fc.slugs.disambiguation", "fc.links.format", "fc.links.resolve",
    "fc.list-entries.anchor-bold", "fc.fnref.format", "fc.fnref.adjacent",
    "fc.fndef.collected", "fc.fndef.numeric-order", "fc.fndef.format",
    "fc.fndef.multi-quote-sep", "fc.markers.asterisk-format",
]
MODEL_IDS = [
    "fc.entries.grouping-no-anchor", "fc.list-entries.approx-tilde", "fc.markers.tbd",
]

_KEBAB_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*$", re.UNICODE)
_PUNCT_IN_NAME = re.compile(r"['‘’\"“”()]")
_BACKTICK_SPAN = re.compile(r"`[^`]*`")
_FN_REF_INLINE = re.compile(r"\[\^([^\]]*)\]")


# --- shared helpers --------------------------------------------------------- #

def _item(cid: str, status: Status, evidence: Evidence, engine: Engine = Engine.MECHANICAL) -> FCItem:
    return FCItem(id=cid, description=CRITERION_TEXT.get(cid), engine=engine,
                  status=status, evidence=evidence)


def _na(cid: str, why: str) -> FCItem:
    return _item(cid, Status.NA, Evidence(lines=[], detail=why))


def status_from_counts(total: int, violations: int) -> Status:
    """pass with 0 violations; partial for a single isolated straggler (a small minority --
    still scores as a fail but stays visible); fail otherwise. na is decided by the caller."""
    if violations == 0:
        return Status.PASS
    if violations <= max(1, total // 10):
        return Status.PARTIAL
    return Status.FAIL


def _alpha_key(name: str) -> str:
    """Collation key for alphabetization: drop a trailing PC-marker ``*``, drop a leading
    'The', casefold. Ties fall back to raw code-point order (documented, deterministic)."""
    n = name.rstrip("*").strip()
    return strip_leading_article(n).casefold().strip()


def _preceded_by_hr(p: ParsedWOF, lineno: int) -> bool:
    """True if, scanning up from ``lineno`` past blank lines, the first non-blank line is a
    horizontal rule."""
    z = lineno - 2                          # 0-based line just above the header
    while z >= 0 and not p.lines[z].strip():
        z -= 1
    return z >= 0 and set(p.lines[z].strip()) == {"-"} and len(p.lines[z].strip()) >= 3


def _all_entries(p: ParsedWOF):
    for s in p.sections:
        for e in s.entries:
            yield s, e


# --- title & top matter ----------------------------------------------------- #

def check_title_single_h1(p: ParsedWOF) -> FCItem:
    cid = "fc.title.single-h1"
    if not p.opens_with_title:
        return _na(cid, "no document-title H1; omitting a title is allowed")
    extras = [h for h in p.headings if h.level == 1 and h is not p.title]
    if not extras:
        return _item(cid, Status.PASS, Evidence(lines=[p.title.lineno], detail="exactly one H1 title"))
    return _item(cid, Status.FAIL, Evidence(
        lines=[h.lineno for h in extras],
        detail=f"{p.h1_count} level-1 headers; a titled doc must have exactly one",
        expected="1 H1", found=f"{p.h1_count} H1s"))


def check_title_subtitle(p: ParsedWOF) -> FCItem:
    cid = "fc.title.subtitle-present"
    if not p.opens_with_title:
        return _na(cid, "no document-title H1, so no subtitle is expected")
    if p.subtitle is not None:
        return _item(cid, Status.PASS, Evidence(lines=[p.subtitle[0]], detail="subtitle prose present"))
    return _item(cid, Status.FAIL, Evidence(lines=[p.title.lineno],
                                            detail="no prose line directly under the title"))


# --- sections --------------------------------------------------------------- #

def check_sections_consistent_depth(p: ParsedWOF) -> FCItem:
    cid = "fc.sections.consistent-depth"
    if not p.sections:
        return _na(cid, "no sections")
    # Section-divider headings = those preceded by an HR (the `---` between sections). They
    # must all sit at the expected section level S (one below the title, else the top level).
    dividers = [h for h in p.headings if _preceded_by_hr(p, h.lineno)]
    checked = dividers or [p.headings[i] for i, _ in enumerate([s for s in p.sections])]
    offenders = [h for h in checked if h.level != p.section_level]
    if not offenders:
        return _item(cid, Status.PASS, Evidence(
            lines=[], detail=f"{len(p.sections)} section headers, all at level {p.section_level}"))
    return _item(cid, status_from_counts(len(checked), len(offenders)), Evidence(
        lines=[h.lineno for h in offenders],
        detail=f"{len(offenders)} section header(s) not at the consistent level {p.section_level}",
        expected=f"level {p.section_level}", found=",".join(str(h.level) for h in offenders)))


def check_sections_no_anchor(p: ParsedWOF) -> FCItem:
    cid = "fc.sections.no-anchor"
    if not p.sections:
        return _na(cid, "no sections")
    # s.anchor_id is set only when the anchor LEADS the heading content; a non-leading anchor
    # (e.g. "## Locations <a id=...>") parses as anchor_id=None, so re-scan the raw header line
    # too (mirrors FIX #2 on entries) -- a section header must carry NO anchor at all.
    offenders = [s for s in p.sections
                 if s.anchor_id is not None
                 or re.search(r'<a id="[^"]*"></a>', p.lines[s.lineno - 1])]
    st = status_from_counts(len(p.sections), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(p.sections)} section headers, none anchored"))
    return _item(cid, st, Evidence(lines=[s.lineno for s in offenders],
                                   detail=f"{len(offenders)} section header(s) carry an anchor"))


def check_layout_hr(p: ParsedWOF) -> FCItem:
    cid = "fc.layout.hr-separators"
    if not p.sections:
        return _na(cid, "no sections")
    offenders = []
    for i, s in enumerate(p.sections):
        # The first section needs a leading HR only when there is top matter (a title) above it.
        if i == 0 and not p.opens_with_title:
            continue
        if not _preceded_by_hr(p, s.lineno):
            offenders.append(s)
    st = status_from_counts(len(p.sections), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail="--- separates the top matter and every section"))
    return _item(cid, st, Evidence(lines=[s.lineno for s in offenders],
                                   detail=f"{len(offenders)} section(s) not preceded by a '---' rule"))


# --- entries ---------------------------------------------------------------- #

def check_entries_depth(p: ParsedWOF) -> FCItem:
    cid = "fc.entries.depth"
    subs = [h for h in p.headings if h.level > p.section_level]
    if not subs:
        return _na(cid, "no entry-level (sub-section) headings")
    offenders = [h for h in subs if h.level != p.entry_level]
    st = status_from_counts(len(subs), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(subs)} entry headers, all at level {p.entry_level}"))
    return _item(cid, st, Evidence(lines=[h.lineno for h in offenders],
                                   detail=f"{len(offenders)} entry header(s) not at level {p.entry_level}",
                                   expected=f"level {p.entry_level}",
                                   found=",".join(str(h.level) for h in offenders)))


def check_entries_inline_anchor(p: ParsedWOF) -> FCItem:
    # FIX #2: verify the POSITION of anchors that EXIST on ATX entries; do NOT require
    # presence (an anchorless heading is the grouping model check's job). An anchor that is
    # on the line but NOT leading the name parses as anchor_id=None while the raw line still
    # contains "<a id=...>", which is exactly the mis-positioned case we flag here.
    cid = "fc.entries.inline-anchor"
    with_anchor_on_line = [
        (s, e) for s, e in _all_entries(p)
        if not e.is_bullet and re.search(r'<a id="[^"]*"></a>', e.raw)
    ]
    if not with_anchor_on_line:
        return _na(cid, "no ATX entries carry an anchor")
    offenders = [(s, e) for s, e in with_anchor_on_line if e.anchor_id is None]
    st = status_from_counts(len(with_anchor_on_line), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(with_anchor_on_line)} entry anchors, all lead the name"))
    return _item(cid, st, Evidence(lines=[e.lineno for _, e in offenders],
                                   detail=f"{len(offenders)} entry anchor(s) not immediately before the name"))


def check_entries_alphabetized(p: ParsedWOF) -> FCItem:
    cid = "fc.entries.alphabetized"
    considered = 0
    bad_sections = []
    first_bad = None
    for s in p.sections:
        seq = [e for e in s.entries if not e.is_bullet and e.anchor_id]
        if len(seq) < 2:
            continue
        considered += 1
        keys = [_alpha_key(e.name) for e in seq]
        if keys != sorted(keys):
            bad_sections.append(s)
            if first_bad is None:
                for a, b in zip(seq, seq[1:]):
                    if _alpha_key(a.name) > _alpha_key(b.name):
                        first_bad = (s, a, b)
                        break
    if considered == 0:
        return _na(cid, "no section has 2+ anchored heading-style entries to order")
    st = status_from_counts(considered, len(bad_sections))
    if not bad_sections:
        return _item(cid, st, Evidence(lines=[], detail=f"{considered} section(s) alphabetized (ignoring leading 'The')"))
    ev = Evidence(lines=[first_bad[1].lineno] if first_bad else [],
                  detail=f"{len(bad_sections)} section(s) out of order")
    if first_bad:
        ev.found = f"'{first_bad[1].name}' before '{first_bad[2].name}' in {first_bad[0].title}"
    return _item(cid, st, ev)


# --- anchors & slugs -------------------------------------------------------- #

def check_anchors_html_form(p: ParsedWOF) -> FCItem:
    cid = "fc.anchors.html-form"
    total = len(p.anchors) + len(p.non_canonical_anchors)
    if total == 0:
        return _na(cid, "no anchors")
    st = status_from_counts(total, len(p.non_canonical_anchors))
    if not p.non_canonical_anchors:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(p.anchors)} anchors, all raw <a id=...></a>"))
    return _item(cid, st, Evidence(lines=[ln for ln, _ in p.non_canonical_anchors],
                                   detail=f"{len(p.non_canonical_anchors)} non-canonical anchor form(s)",
                                   found="; ".join(t for _, t in p.non_canonical_anchors[:5])))


def check_anchors_unique(p: ParsedWOF) -> FCItem:
    cid = "fc.anchors.unique-ids"
    if not p.anchors:
        return _na(cid, "no anchors")
    seen = {}
    dupes = {}
    for a in p.anchors:
        if a.id in seen:
            dupes.setdefault(a.id, [seen[a.id]]).append(a.lineno)
        else:
            seen[a.id] = a.lineno
    dup_occurrences = sum(len(v) - 1 for v in dupes.values())
    st = status_from_counts(len(p.anchors), dup_occurrences)
    if not dupes:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(p.anchors)} anchor ids, 0 collisions"))
    return _item(cid, st, Evidence(lines=sorted({ln for v in dupes.values() for ln in v}),
                                   detail=f"{len(dupes)} duplicated id(s): " + ", ".join(list(dupes)[:5])))


def check_slugs_kebab(p: ParsedWOF) -> FCItem:
    cid = "fc.slugs.kebab-case"
    if not p.anchors:
        return _na(cid, "no anchors")
    offenders = [a for a in p.anchors if not (a.id == a.id.lower() and _KEBAB_RE.fullmatch(a.id))]
    st = status_from_counts(len(p.anchors), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(p.anchors)} slugs, all lowercase kebab-case"))
    return _item(cid, st, Evidence(lines=[a.lineno for a in offenders],
                                   detail=f"{len(offenders)} slug(s) not lowercase-kebab",
                                   found="; ".join(a.id for a in offenders[:5])))


def check_slugs_punctuation(p: ParsedWOF) -> FCItem:
    cid = "fc.slugs.punctuation"
    if not p.anchors:
        return _na(cid, "no anchors")
    # (a) no anchor id may carry an apostrophe/quote/paren; (b) an entry whose NAME has such
    # punctuation must still slug to a matching anchor (flattened / dropped correctly).
    id_offenders = [a for a in p.anchors if _PUNCT_IN_NAME.search(a.id)]
    name_offenders = [
        (s, e) for s, e in _all_entries(p)
        if e.anchor_id and _PUNCT_IN_NAME.search(e.name) and not anchor_matches_name(e.anchor_id, e.name)
    ]
    total = len(p.anchors)
    violations = len(id_offenders) + len(name_offenders)
    st = status_from_counts(total, violations)
    if violations == 0:
        return _item(cid, st, Evidence(lines=[], detail="apostrophes/quotes dropped and parentheticals flattened"))
    lines = [a.lineno for a in id_offenders] + [e.lineno for _, e in name_offenders]
    return _item(cid, st, Evidence(lines=sorted(lines),
                                   detail=f"{violations} slug(s) mishandle punctuation"))


def check_slugs_non_ascii(p: ParsedWOF) -> FCItem:
    cid = "fc.slugs.non-ascii"
    cases = [(s, e) for s, e in _all_entries(p) if e.anchor_id and has_non_ascii_letter(e.name)]
    if not cases:
        return _na(cid, "no entry names contain non-ASCII letters")
    offenders = [(s, e) for s, e in cases if not has_non_ascii_letter(e.anchor_id)]
    st = status_from_counts(len(cases), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(cases)} non-ASCII name(s) preserved in the slug"))
    return _item(cid, st, Evidence(lines=[e.lineno for _, e in offenders],
                                   detail=f"{len(offenders)} non-ASCII name(s) transliterated in the slug",
                                   found="; ".join(f"{e.name}->{e.anchor_id}" for _, e in offenders[:5])))


def check_slugs_disambiguation(p: ParsedWOF) -> FCItem:
    cid = "fc.slugs.disambiguation"
    # Group entries by their article-stripped slug base; a base used by entries in 2+ distinct
    # sections is a cross-section clash that must be disambiguated with a -N suffix.
    groups = {}
    for s, e in _all_entries(p):
        if not e.anchor_id:
            continue
        base = slugify(strip_leading_article(e.name))
        if not base:
            continue
        groups.setdefault(base, []).append((s.title, e))
    clashes = {b: v for b, v in groups.items()
               if len({sec for sec, _ in v}) >= 2}
    if not clashes:
        return _na(cid, "no name is used for entries in more than one section")
    offenders = []
    for base, members in clashes.items():
        anchors = [e.anchor_id for _, e in members]
        # OK when the anchors are all distinct and follow base / base-N.
        distinct = len(set(anchors)) == len(anchors)
        suffixed_ok = all(a == base or re.fullmatch(re.escape(base) + r"-\d+", a) for a in anchors)
        if not (distinct and suffixed_ok):
            offenders.append((base, members))
    st = status_from_counts(len(clashes), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(clashes)} cross-section name clash(es) disambiguated with -N"))
    return _item(cid, st, Evidence(lines=[e.lineno for _, members in offenders for _, e in members],
                                   detail=f"{len(offenders)} clash(es) not disambiguated",
                                   found="; ".join(b for b, _ in offenders[:5])))


# --- cross-links ------------------------------------------------------------ #

def check_links_format(p: ParsedWOF) -> FCItem:
    cid = "fc.links.format"
    # Scan for link-like fragments loosely so a malformed one is visible, not skipped.
    loose = []
    for i, line in enumerate(p.lines):
        if p.code_mask[i]:
            continue
        for m in re.finditer(r"\[([^\]]*)\](\s*)\((#[^)]*)\)", line):
            loose.append((i + 1, m.group(1), m.group(2), m.group(3)))
    if not loose:
        return _na(cid, "no intra-document fragment links")
    offenders = []
    for lineno, text, gap, target in loose:
        slug = target[1:]
        if gap != "" or not text.strip() or not slug or re.search(r"\s", slug):
            offenders.append(lineno)
    st = status_from_counts(len(loose), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(loose)} fragment links, all well-formed [text](#slug)"))
    return _item(cid, st, Evidence(lines=sorted(set(offenders)),
                                   detail=f"{len(offenders)} malformed fragment link(s)"))


def check_links_resolve(p: ParsedWOF) -> FCItem:
    cid = "fc.links.resolve"
    if not p.links:
        return _na(cid, "no intra-document fragment links")
    ids = {a.id for a in p.anchors}
    broken = [ln for ln in p.links if ln.slug not in ids]
    st = status_from_counts(len(p.links), len(broken))
    if not broken:
        return _item(cid, st, Evidence(lines=[], detail=f"all {len(p.links)} links resolve to a real anchor id"))
    return _item(cid, st, Evidence(lines=[ln.lineno for ln in broken],
                                   detail=f"{len(broken)} link(s) point at a missing anchor",
                                   found="; ".join('#' + ln.slug for ln in broken[:5])))


# --- list-style entries ----------------------------------------------------- #

def check_list_anchor_bold(p: ParsedWOF) -> FCItem:
    cid = "fc.list-entries.anchor-bold"
    bullets = [(s, e) for s, e in _all_entries(p) if e.is_bullet]
    if not bullets:
        return _na(cid, "no bullet-list entry sections")
    # A well-formed bullet entry parsed an anchor id AND a bold name (raw still shows both).
    offenders = [(s, e) for s, e in bullets
                 if e.anchor_id is None or not re.search(r"\*\*.+?\*\*", e.raw)]
    st = status_from_counts(len(bullets), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(bullets)} bullet entries open with anchor + **bold** name"))
    return _item(cid, st, Evidence(lines=[e.lineno for _, e in offenders],
                                   detail=f"{len(offenders)} bullet(s) missing a leading anchor or bold name"))


# --- footnote references ---------------------------------------------------- #

def check_fnref_format(p: ParsedWOF) -> FCItem:
    cid = "fc.fnref.format"
    def_lines = {d.lineno for d in p.fn_defs}
    candidates = []
    offenders = []
    for i, line in enumerate(p.lines):
        if p.code_mask[i] or (i + 1) in def_lines:
            continue
        for m in _FN_REF_INLINE.finditer(line):
            if m.end() < len(line) and line[m.end()] == ":":
                continue                    # a "[^n]:" def start, not an inline ref
            candidates.append((i + 1, m.group(0)))
            if not re.fullmatch(r"\d+", m.group(1)):
                offenders.append((i + 1, m.group(0)))
    if not candidates:
        return _na(cid, "no inline footnote references")
    st = status_from_counts(len(candidates), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(candidates)} inline refs, all [^n] form"))
    return _item(cid, st, Evidence(lines=[ln for ln, _ in offenders],
                                   detail=f"{len(offenders)} malformed footnote ref(s)",
                                   found="; ".join(t for _, t in offenders[:5])))


def check_fnref_adjacent(p: ParsedWOF) -> FCItem:
    cid = "fc.fnref.adjacent"
    by_line = {}
    for r in p.fn_refs:
        by_line.setdefault(r.lineno, []).append(r)
    stacked = 0
    offenders = []
    for lineno, refs in by_line.items():
        refs = sorted(refs, key=lambda r: r.start)
        line = p.lines[lineno - 1]
        for a, b in zip(refs, refs[1:]):
            between = line[a.end:b.start]
            if between == "":
                stacked += 1               # correctly adjacent
            elif re.fullmatch(r"[\s,]+", between):
                stacked += 1               # meant to be stacked but has a separator
                offenders.append(lineno)
    if stacked == 0:
        return _na(cid, "no consecutive footnote references to stack")
    st = status_from_counts(stacked, len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{stacked} stacked ref pair(s), no separators"))
    return _item(cid, st, Evidence(lines=sorted(set(offenders)),
                                   detail=f"{len(offenders)} stacked ref(s) separated by a space/comma"))


# --- footnote definitions --------------------------------------------------- #

def check_fndef_collected(p: ParsedWOF) -> FCItem:
    cid = "fc.fndef.collected"
    if not p.fn_defs:
        return _na(cid, "no footnote definitions")
    if p.footnotes_section is None:
        return _item(cid, Status.FAIL, Evidence(lines=[d.lineno for d in p.fn_defs],
                                                detail="footnote definitions not inside a single section"))
    sec_lines = {ln for ln, _ in p.footnotes_section.body}
    outside = [d for d in p.fn_defs if d.lineno not in sec_lines]
    level_bad = p.footnotes_section.level != p.section_level
    # Criterion: the definitions live in one dedicated section AT THE END of the file.
    not_last = bool(p.sections) and p.footnotes_section is not p.sections[-1]
    violations = len(outside) + (1 if level_bad else 0) + (1 if not_last else 0)
    st = status_from_counts(len(p.fn_defs), violations)
    if violations == 0:
        return _item(cid, st, Evidence(lines=[p.footnotes_section.lineno],
                                       detail=f"all {len(p.fn_defs)} defs in one end section at level {p.section_level}"))
    detail = []
    if outside:
        detail.append(f"{len(outside)} def(s) outside the footnotes section")
    if level_bad:
        detail.append(f"footnotes section at level {p.footnotes_section.level}, not {p.section_level}")
    if not_last:
        detail.append("footnotes section is not the last section (should be at the end of the file)")
    return _item(cid, st, Evidence(lines=[d.lineno for d in outside] or [p.footnotes_section.lineno],
                                   detail="; ".join(detail)))


def check_fndef_numeric_order(p: ParsedWOF) -> FCItem:
    cid = "fc.fndef.numeric-order"
    if len(p.fn_defs) < 2:
        return _na(cid, "fewer than two footnote definitions")
    nums = [d.n for d in p.fn_defs]
    offenders = [p.fn_defs[i + 1] for i in range(len(nums) - 1) if nums[i + 1] <= nums[i]]
    st = status_from_counts(len(nums) - 1, len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(nums)} defs in strict ascending order"))
    return _item(cid, st, Evidence(lines=[d.lineno for d in offenders],
                                   detail=f"{len(offenders)} def(s) out of ascending order"))


def check_fndef_format(p: ParsedWOF) -> FCItem:
    cid = "fc.fndef.format"
    if not p.fn_defs:
        return _na(cid, "no footnote definitions")
    offenders = []
    for d in p.fn_defs:
        raw = d.raw
        has_quote = raw.count("`") >= 2
        has_emdash = "—" in raw
        # Tolerant attribution: an em-dash followed (before the next em-dash) by a "Name, source"
        # comma. Source may lack an extension ("CONVO_2"), so we don't require ".txt".
        has_attr = re.search(r"—[^—]*,", raw) is not None
        if not (has_quote and has_emdash and has_attr):
            offenders.append(d)
    st = status_from_counts(len(p.fn_defs), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"all {len(p.fn_defs)} defs: `quote` — Name, source"))
    return _item(cid, st, Evidence(lines=[d.lineno for d in offenders[:20]],
                                   detail=f"{len(offenders)} def(s) not in `quote` — Name, source shape"))


def check_fndef_multi_quote_sep(p: ParsedWOF) -> FCItem:
    cid = "fc.fndef.multi-quote-sep"
    multi = []
    offenders = []
    for d in p.fn_defs:
        spans = list(_BACKTICK_SPAN.finditer(d.raw))
        if len(spans) < 2:
            continue
        multi.append(d)
        for a, b in zip(spans, spans[1:]):
            gap = d.raw[a.end():b.start()]
            # A legit boundary between excerpts carries ' / ' (the list separator), an em-dash
            # (attribution change), a colon (an "Also:"/"with Matt:" aside), or a connective
            # WORD ("and", "with", "confirmed"). Only a "thin" gap -- empty or bare punctuation
            # like a lone comma/semicolon -- is a genuine run-on with no separator.
            if not (any(tok in gap for tok in ("/", "—", ":")) or re.search(r"[A-Za-z]", gap)):
                offenders.append(d)
                break
    if not multi:
        return _na(cid, "no multi-quote footnote definitions")
    st = status_from_counts(len(multi), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(multi)} multi-quote def(s) use ' / ' separators"))
    return _item(cid, st, Evidence(lines=[d.lineno for d in offenders],
                                   detail=f"{len(offenders)} multi-quote def(s) join excerpts without ' / '"))


# --- other markers ---------------------------------------------------------- #

def check_markers_asterisk(p: ParsedWOF) -> FCItem:
    cid = "fc.markers.asterisk-format"
    marked = [(s, e) for s, e in _all_entries(p)
              if e.name.rstrip().endswith("*") and not e.name.rstrip().endswith("**")]
    if not marked:
        return _na(cid, "no trailing-asterisk markers in use")
    # Uniform = a single trailing '*' with no space before it.
    offenders = [(s, e) for s, e in marked if e.name.rstrip().endswith(" *")]
    st = status_from_counts(len(marked), len(offenders))
    if not offenders:
        return _item(cid, st, Evidence(lines=[], detail=f"{len(marked)} '*' markers, all a uniform trailing '*'"))
    return _item(cid, st, Evidence(lines=[e.lineno for _, e in offenders],
                                   detail=f"{len(offenders)} '*' marker(s) placed inconsistently"))


# --- run all + candidate extraction ----------------------------------------- #

_CHECKS = [
    check_title_single_h1, check_title_subtitle, check_sections_consistent_depth,
    check_sections_no_anchor, check_layout_hr, check_entries_depth,
    check_entries_inline_anchor, check_entries_alphabetized, check_anchors_html_form,
    check_anchors_unique, check_slugs_kebab, check_slugs_punctuation, check_slugs_non_ascii,
    check_slugs_disambiguation, check_links_format, check_links_resolve,
    check_list_anchor_bold, check_fnref_format, check_fnref_adjacent, check_fndef_collected,
    check_fndef_numeric_order, check_fndef_format, check_fndef_multi_quote_sep,
    check_markers_asterisk,
]


def run_lint(p: ParsedWOF) -> List[FCItem]:
    """Run all 24 mechanical checks, in canonical id order."""
    items = [fn(p) for fn in _CHECKS]
    order = {cid: i for i, cid in enumerate(MECHANICAL_IDS)}
    items.sort(key=lambda it: order.get(it.id, 999))
    return items


# --- candidate extraction for the 3 model checks ---------------------------- #

def _clean_prose(text: str) -> str:
    """Strip footnote refs, inline anchors, and the ``(#slug)`` half of fragment links from a
    line, leaving prose (link text kept). FIX #4: this is what keeps ``[^41]`` /
    ``krieger-imperium-2`` slug digits out of ``figure_candidates``."""
    text = re.sub(r"\[\^\d+\]", "", text)                    # footnote refs
    text = re.sub(r'<a id="[^"]*"></a>', "", text)           # inline anchors
    text = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", text)     # links -> their display text
    return text


_FIGURE_RE = re.compile(r"(?<!~)\b\d[\d,]*\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def grouping_candidates(p: ParsedWOF) -> List[dict]:
    """Every entry-level heading that has NO anchor, with the entries beneath it -- the model
    decides group-label (fine) vs an entry that lost its anchor."""
    out = []
    for s in p.sections:
        entries = s.entries
        for i, e in enumerate(entries):
            if not e.is_grouping:
                continue
            beneath = []
            for nxt in entries[i + 1:]:
                if nxt.is_grouping:
                    break
                beneath.append(nxt.name)
            out.append({"label": e.name, "lineno": e.lineno,
                        "section": s.title, "entries_beneath": beneath})
    return out


def figure_candidates(p: ParsedWOF) -> List[dict]:
    """Every numeric prose figure NOT already tilde-marked, with the sentence it sits in.
    Excludes footnote refs and digits inside anchors/links/slugs (FIX #4). Deduped."""
    out = []
    seen = set()
    for i, line in enumerate(p.lines):
        if p.code_mask[i] or _is_def_line(line):
            continue
        cleaned = _clean_prose(line)
        if not _FIGURE_RE.search(cleaned):
            continue
        for sentence in _SENTENCE_SPLIT.split(cleaned):
            for m in _FIGURE_RE.finditer(sentence):
                fig = m.group(0)
                key = (fig, sentence.strip())
                if key in seen:
                    continue
                seen.add(key)
                out.append({"figure": fig, "sentence": sentence.strip(), "lineno": i + 1})
    return out


def tbd_candidates(p: ParsedWOF) -> List[dict]:
    """Each real entry's name + body text -- the model decides whether it's a stub that
    should carry ``[TBD]`` but doesn't (an already-``[TBD]`` body is a legitimate 'ok')."""
    out = []
    for s, e in _all_entries(p):
        if e.is_grouping:
            continue
        out.append({"name": e.name, "body": e.body, "lineno": e.lineno, "section": s.title})
    return out


def _is_def_line(line: str) -> bool:
    return re.match(r"^\[\^\d+\]:", line) is not None
