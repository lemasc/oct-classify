"""Serve the Typst paper (English and Thai) with an optional word-level redline between two Git revisions.

    uv run --group paper python scripts/paper/paper_diff.py [--port 8731]
    http://localhost:8731/main..HEAD       compare two revisions (branch, tag or SHA)
    http://localhost:8731/HEAD~3..         compare against the working tree (uncommitted edits)
    http://localhost:8731/4b9df74          view one revision without any diff
    http://localhost:8731/th/main..HEAD    the same three forms for the Thai paper (docs/paper/th/)

Each revision is extracted with `git archive` and compiled with `typst --features html`; the
built HTML is cached per commit and variant under artifacts/paper-diff/, next to the PDF Typst
lays out (two columns, as submitted): aligning its words with the HTML words tells where each page and
column begins, which the single-column page shows as small markers. The two HTML bodies are
tokenised and diffed with difflib (Thai runs are segmented into words with pythainlp first).
Tables are diffed as whole blocks (old table dimmed above the new one, changed cells shaded green
in the new table, removed cells red in the old), since a per-cell diff breaks when columns move.
Images (and inline SVG drawings, e.g. the fletcher flowchart, hashed as a whole) get the same treatment: each is identified by its Git blob id (the id `git ls-tree` reports
for the figure file, recovered from the data URI Typst embeds), and a changed blob shows old dimmed
above new. Pages that show the working tree stay connected over
Server-Sent Events: while any is open a watcher polls the paper sources, and an edit makes the page
offer a reload (or reload itself, keeping its place in the text). The page chrome (contents with per-section change counts, language switch, copy tools)
lives in viewer/templates (Jinja2) and viewer/static; dependencies are in the `paper` group.
"""

import argparse
import base64
import hashlib
import html
import mimetypes
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from difflib import SequenceMatcher
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, unquote

from jinja2 import Environment, FileSystemLoader, select_autoescape

REPO = Path(__file__).resolve().parents[2]
PAPER = "docs/paper"
DATA_DIRS = ("data/paper", "data/analysis")  # result files the paper loads via json()
CACHE = REPO / "artifacts/paper-diff"
VIEWER = Path(__file__).resolve().parent / "viewer"
WORKTREE = "WORKTREE"
LANGS = ("en", "th")
ENTRY = {"en": "main.typ", "th": "th/main.typ"}  # relative to PAPER
MIN_ANCHOR = 4  # equal runs shorter than this between two edits are folded into the edit
HISTORY = 150  # commits scanned for the revision picker
BUILD_LOCK = threading.Lock()
POLL = 0.5  # seconds between looks at the paper sources while a live page is open
KEEPALIVE = 10  # seconds of silence before a comment line probes the stream (and drops dead clients)
TEMPLATES = Environment(
    loader=FileSystemLoader(VIEWER / "templates"),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

BLOCKS = {"p", "h1", "h2", "h3", "h4", "li", "ul", "ol", "figure", "figcaption", "tr", "div"}
TOKEN = re.compile(r"<[^>]+>|[^<\s]+|\s+")
DATA_URI = re.compile(r'src="data:[^;"]*;base64,([^"]*)"')
THAI = re.compile(r"[\u0e00-\u0e7f]")
ENTITY = re.compile(r"(&#?\w+;)")
FIGURE = re.compile(r'<figure\s+id="((?:fig|tab)-[^"]*)"[^>]*>(.*?)</figure>', re.DOTALL)
CAPTION = re.compile(r"<figcaption[^>]*>(.*?)</figcaption>", re.DOTALL)
HEADING = re.compile(r"<h([23])((?:\s[^>]*)?)>(.*?)</h\1>", re.DOTALL)
TABLE = re.compile(r"<table.*?</table>", re.DOTALL)
DRAWN = re.compile(r"<table.*?</table>|<svg.*?</svg>", re.DOTALL)  # blocks diffed whole, not per token
ROW = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
CELL = re.compile(r"<(t[dh])([^>]*)>(.*?)</\1>", re.DOTALL)
SECTION = re.compile(r"<(thead|tbody)>(.*?)</\1>", re.DOTALL)


class Error(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# --- building one revision ------------------------------------------------------------------


def git(*args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False, **kw)


def resolve(rev: str) -> str:
    if rev == WORKTREE:
        return rev
    done = git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", text=True)
    if done.returncode:
        raise Error(404, f"unknown revision: {rev}")
    return done.stdout.strip()


def typst(workdir: Path, out: Path, lang: str, fmt: str = "html") -> None:
    if not (workdir / ENTRY[lang]).exists():
        raise Error(404, f"no {lang} paper ({PAPER}/{ENTRY[lang]}) in this revision")
    features = ["--features", "html"] if fmt == "html" else []
    done = subprocess.run(
        ["typst", "compile", "--root", "../..", *features, "--format", fmt, ENTRY[lang], str(out)],
        cwd=workdir, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if done.returncode:
        raise Error(500, f"typst failed in {workdir}:\n{done.stderr}")


def compiled(sha: str, lang: str, fmt: str) -> bytes:
    """The `fmt` output of the `lang` paper at `sha` (or the working tree), compiling on a cache miss."""
    if sha == WORKTREE:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / f"main.{fmt}"
            typst(REPO / PAPER, out, lang, fmt)
            return out.read_bytes()
    cached = CACHE / (f"{sha}.{fmt}" if lang == "en" else f"{sha}.{lang}.{fmt}")
    with BUILD_LOCK:
        if not cached.exists():
            with tempfile.TemporaryDirectory() as tmp:
                for path in (PAPER, *DATA_DIRS):
                    archive = git("archive", sha, path)
                    if archive.returncode:
                        if path == PAPER:
                            raise Error(404, f"{PAPER} not found at {sha[:7]}")
                        continue  # older revisions may predate this data folder
                    subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
                CACHE.mkdir(parents=True, exist_ok=True)
                partial = cached.with_suffix(".tmp")
                typst(Path(tmp) / PAPER, partial, lang, fmt)
                partial.replace(cached)
    return cached.read_bytes()


def build(sha: str, lang: str) -> str:
    """HTML of the `lang` paper at `sha`."""
    return compiled(sha, lang, "html").decode()


# --- watching the working tree --------------------------------------------------------------


def fingerprint() -> str:
    """Digest of name, mtime and size of every file the working-tree paper builds from."""
    state = []
    for root in (PAPER, *DATA_DIRS):
        for path in sorted((REPO / root).rglob("*")):
            if path.name == "main.pdf" or not path.is_file():
                continue  # a PDF built by hand is an output, not a source
            stat = path.stat()
            state.append((str(path), stat.st_mtime_ns, stat.st_size))
    return hashlib.md5(repr(state).encode()).hexdigest()[:12]


class Watch:
    """Polls `fingerprint()`, but only while some live page is listening."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.listeners = 0
        self.published = ""
        self.thread: threading.Thread | None = None

    @contextmanager
    def listening(self):
        with self.cond:
            self.listeners += 1
            if self.thread is None:
                self.published = fingerprint()
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()
        try:
            yield
        finally:
            with self.cond:
                self.listeners -= 1

    def run(self) -> None:
        candidate = None
        while True:
            time.sleep(POLL)
            with self.cond:
                if not self.listeners:
                    self.thread = None
                    return
            now = fingerprint()
            if now == self.published:
                candidate = None
            elif now == candidate:  # unchanged for a whole poll: the editor has finished writing
                with self.cond:
                    self.published = now
                    self.cond.notify_all()
            else:
                candidate = now

    def wait(self, seen: str, timeout: float) -> str | None:
        """The new fingerprint once it differs from `seen`, or None after `timeout`."""
        with self.cond:
            self.cond.wait_for(lambda: self.published != seen, timeout)
            return self.published if self.published != seen else None


WATCH = Watch()


# --- diff -----------------------------------------------------------------------------------


def cell_text(cell: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", cell))).strip()


def table_rows(table: str) -> dict:
    """Map (section, first cell, occurrence) -> cell texts, so rows match across edits."""
    rows, seen = {}, {}
    for section in SECTION.finditer(table):
        for row in ROW.finditer(section.group(2)):
            cells = [cell_text(c.group(3)) for c in CELL.finditer(row.group(1))]
            key = (section.group(1), cells[0] if cells else "")
            rows[(*key, seen.get(key, 0))] = cells
            seen[key] = seen.get(key, 0) + 1
    return rows


def mark_table(old: str, new: str, cls: str = "chg") -> str:
    """Shade cells of `new` whose text is not in the matching row of `old`."""
    old_rows, seen = table_rows(old), {}

    def mark_row(row: re.Match) -> str:
        head = new.rfind("<thead>", 0, row.start()) > new.rfind("</thead>", 0, row.start())
        cells = [cell_text(c.group(3)) for c in CELL.finditer(row.group(1))]
        key = ("thead" if head else "tbody", cells[0] if cells else "")
        pool = old_rows.get((*key, seen.get(key, 0)))
        pool = list(pool) if pool is not None else None
        seen[key] = seen.get(key, 0) + 1

        def mark_cell(cell: re.Match) -> str:
            text = cell_text(cell.group(3))
            if pool is not None and text in pool:
                pool.remove(text)
                return cell.group(0)
            tag, attrs, inner = cell.groups()
            return f'<{tag}{attrs} class="{cls}">{inner}</{tag}>'

        return "<tr>" + CELL.sub(mark_cell, row.group(1)) + "</tr>"

    return ROW.sub(mark_row, new)


def body_of(page: str) -> str:
    start = page.index(">", page.index("<body")) + 1
    return page[start : page.rindex("</body>")]


def title_of(page: str) -> str:
    return html.unescape(re.search(r"<title>(.*?)</title>", page).group(1))


def blob_id(img_tag: str) -> str:
    """Git blob id of the file embedded in an <img> tag, i.e. what `git ls-tree` lists for it."""
    m = DATA_URI.search(img_tag)
    if not m:
        return "tag:" + hashlib.md5(img_tag.encode()).hexdigest()  # external src: compare the tag
    data = base64.b64decode(m.group(1))
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@cache
def segment(word: str) -> tuple[str, ...]:
    """Split a whitespace-free run into diffable pieces; Thai has no spaces, so use a dictionary."""
    if not THAI.search(word):
        return (word,)
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:
        raise Error(
            500, "diffing Thai needs pythainlp: run with `uv run --group paper ...`"
        ) from None
    pieces: list[str] = []
    for part in ENTITY.split(word):  # keep &amp; and friends whole
        if ENTITY.fullmatch(part) or not THAI.search(part):
            pieces.append(part)
        else:
            pieces += word_tokenize(part, engine="newmm", keep_whitespace=False)
    return tuple(p for p in pieces if p) or (word,)


def count_words(text: str) -> int:
    return sum(
        any(c.isalnum() for c in piece)
        for word in html.unescape(text).split()
        for piece in segment(word)
    )


def tokenize(body: str) -> list[tuple]:
    """Tokens are (kind, raw, key); keys ignore id/href so renumbered references still match."""
    tokens, space, pos = [], "", 0
    chunks = []
    for m in DRAWN.finditer(body):
        kind = "table" if m.group(0).startswith("<table") else "svg"
        chunks += [("html", body[pos : m.start()]), (kind, m.group(0))]
        pos = m.end()
    chunks.append(("html", body[pos:]))
    for kind, chunk in chunks:
        if kind in ("table", "svg"):
            digest = hashlib.md5(re.sub(r'\s(id|href)="[^"]*"', "", chunk).encode()).hexdigest()
            # an inline drawing (Typst html.frame) is an image: it gets the old-above-new treatment
            tokens.append(("table" if kind == "table" else "img", space + chunk, f"{kind}:{digest}"))
            space = ""
            continue
        for m in TOKEN.finditer(chunk):
            text = m.group(0)
            if text.isspace():
                space = " "
            elif text.startswith("<"):
                name = re.match(r"</?\s*([a-zA-Z0-9]+)", text)
                name = name.group(1).lower() if name else "?"
                if name == "img":
                    tokens.append(("img", space + text, "img:" + blob_id(text)))
                else:
                    tokens.append(("tag", space + text, ("/" if text[1] == "/" else "") + name))
                space = ""
            else:
                for piece in segment(text):
                    tokens.append(("text", space + piece, piece))
                    space = ""
    return tokens


def opcodes(old: list, new: list) -> list[tuple]:
    ops = [list(o) for o in SequenceMatcher(None, [t[2] for t in old], [t[2] for t in new],
                                            autojunk=False).get_opcodes()]  # fmt: skip
    changed = True
    while changed:  # fold short equal runs between two edits into one edit
        changed = False
        for i in range(1, len(ops) - 1):
            tag, a0, a1, _, _ = ops[i]
            if (
                tag == "equal"
                and a1 - a0 < MIN_ANCHOR
                and "equal" not in (ops[i - 1][0], ops[i + 1][0])
            ):
                ops[i - 1 : i + 2] = [
                    ["replace", ops[i - 1][1], ops[i + 1][2], ops[i - 1][3], ops[i + 1][4]]
                ]
                changed = True
                break

    def first_block(tokens, lo, hi, kind=None):
        """Index of the first table or image in the range (of the given kind, if any)."""
        return next(
            (i for i in range(lo, hi) if tokens[i][0] in ("table", "img") and kind in (None, tokens[i][0])),
            None,
        )

    def pair_tables(tag, a0, a1, b0, b1):
        """Split an edit at its first table/image on each side so old and new stay together."""
        ia = first_block(old, a0, a1)
        ib = None if ia is None else first_block(new, b0, b1, old[ia][0])
        if tag == "equal" or ia is None or ib is None:
            return [(tag, a0, a1, b0, b1)]
        return [*pair_tables("replace", a0, ia, b0, ib), ("replace", ia, ia + 1, ib, ib + 1),
                *pair_tables("replace", ia + 1, a1, ib + 1, b1)]  # fmt: skip

    return [piece for op in ops for piece in pair_tables(*op)]


def redline(old_page: str | None, new_page: str, pdf: bytes | None = None) -> str:
    """Body HTML of `new_page`, with the edits since `old_page` marked up (none if it is None).

    `pdf` is the same revision as laid out by Typst; with it, page and column starts are marked."""
    new = tokenize(body_of(new_page))
    old = new if old_page is None else tokenize(body_of(old_page))
    if pdf is not None:
        new = with_marks(new, page_marks(new, read_pdf(pdf).pieces))  # keys are untouched, so the diff is the same
    out: list[str] = []

    def wrap(tokens: list, tag: str, keep_tags: bool) -> None:
        run: list[str] = []

        def flush() -> None:
            if run:
                text = "".join(run)
                out.append(
                    (" " if text[:1] == " " else "")
                    + f'<{tag} class="{tag}">{text.lstrip()}</{tag}>'
                )
                run.clear()

        for kind, raw, _ in tokens:
            if kind == "text":
                run.append(raw)
                continue
            flush()
            if kind == "table":
                out.append(f'<div class="tbl-{tag}"><div class="tbl">{raw}</div></div>')
            elif kind == "img":
                out.append(f'<div class="tbl-{tag}">{raw}</div>')
            elif keep_tags:
                out.append(raw)
            elif re.match(rf"<({'|'.join(BLOCKS)})[\s>]", raw.lstrip()):
                out.append("<br>")  # a deleted block boundary
        flush()

    for tag, a0, a1, b0, b1 in opcodes(old, new):
        if tag == "equal":
            out += [
                f'<div class="tbl">{raw}</div>' if kind == "table" else raw
                for kind, raw, _ in new[b0:b1]
            ]
        elif (
            tag == "replace"
            and old[a0][0] == new[b0][0] in ("table", "img")
            and a1 - a0 == b1 - b0 == 1
        ):
            if old[a0][0] == "img":
                out.append(f'<div class="tbl-del" data-pair>{old[a0][1]}</div>')
                out.append(f'<div class="tbl-ins">{new[b0][1]}</div>')
            else:
                out.append(
                    f'<div class="tbl-del" data-pair><div class="tbl">'
                    f'{mark_table(new[b0][1], old[a0][1], "gone")}</div></div>'
                )
                out.append(
                    f'<div class="tbl-ins"><div class="tbl">{mark_table(old[a0][1], new[b0][1])}</div></div>'
                )
        else:
            wrap(old[a0:a1], "del", keep_tags=False)
            wrap(new[b0:b1], "ins", keep_tags=True)
    return "".join(out)


# --- pages and columns -----------------------------------------------------------------------

# Aligned runs shorter than this are not trusted for where a page or column starts: floats (tables,
# captions) sit elsewhere in the PDF than in the text, and a stray "the" can match by coincidence.
MIN_RUN = 20


# Hyphens are dropped when comparing: a word broken over two lines ("Sec-" "tion", "B-" "scans") must
# equal the same word set on one line, and "B-scans" is not worth telling apart from "Bscans".
BREAKS = "-\u00ad/\u2013\u2014"  # a word may be broken after any of these (DOIs, ranges, hyphens)
HYPHENS = {ord(c): None for c in "-\u00ad\u2010\u2011"}


def fold(word: str) -> str:
    """Comparable form of a word: HTML entities decoded, ligatures and width variants unfolded,
    hyphens dropped."""
    return unicodedata.normalize("NFKC", html.unescape(word)).translate(HYPHENS)


class Piece(NamedTuple):
    """One diffable piece of laid-out text with its box on the page (PDF points, y down)."""

    text: str  # folded, for comparing
    raw: str  # as extracted, for showing
    space: bool  # starts a whitespace-separated word
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    col: int  # 1 or 2; 0 if the word spans both columns
    more: tuple = ()  # boxes (page, x0, y0, x1, y1) of the rest of a piece broken over later lines


class Pdf(NamedTuple):
    title: str
    sizes: list[tuple[float, float]]  # per page: width, height
    pieces: list[Piece]  # in laid-out order
    figures: list[tuple[int, tuple, str]]  # page, box, fingerprint of each picture (see `figures_of`)
    lines: list[tuple]  # page, x0, y0, x1, y1, text of each line the PDF sets
    outline: list[tuple]  # level, title, page, column, y of each heading


def joins(prev: tuple, word: tuple) -> bool:
    """Whether `word` carries on the word `prev` broke off (both are (x0, y0, x1, y1, text, page))."""
    last, first = prev[4][-1], word[4][0]
    if THAI.search(last) and THAI.search(first):  # Thai has no spaces: a line break falls inside a run
        return True
    return last in BREAKS and len(prev[4]) > 1 and first.isalnum()  # "Sec-" "tion", "10.1016/" "j.cell"


def join(run: list[tuple], word: tuple) -> None:
    if run[-1][4].endswith("\u00ad"):  # a hyphen the line breaker added is not part of the word
        run[-1] = (*run[-1][:4], run[-1][4][:-1], run[-1][5])
    run.append(word)


def word_runs(words: list, mid: float, number: int) -> list[list[tuple]]:
    """Group a page's extracted words so that a word wrapped across lines is one run.

    Cut at the break, the two fragments would segment differently (Thai) or compare unequal (Latin)
    from the same text laid out with another line width."""
    runs: list[list[tuple]] = []
    for x0, y0, x1, y1, text, *_ in words:
        word = (x0, y0, x1, y1, text.replace("\u0e4d\u0e32", "\u0e33"), number)  # nikhahit + sara aa
        prev = runs[-1][-1] if runs else None
        wraps = prev and (
            (y0 > prev[1] + 3 and x0 < prev[0] + 1)  # the next line
            or (prev[0] < mid <= x0 and y0 < prev[1] - 3)  # foot of column 1 to head of column 2
        )
        if wraps and joins(prev, word):
            join(runs[-1], word)
        else:
            runs.append([word])
    return runs


def run_pieces(run: list[tuple], mid: float) -> list[Piece]:
    """Cut a run of words into diffable pieces, each with the box of the line it starts on."""
    owner = [(k, j) for k, word in enumerate(run) for j in range(len(word[4]))]
    pieces, at = [], 0
    for part in segment("".join(word[4] for word in run)):
        (k, j), (end, last) = owner[at], owner[at + len(part) - 1]
        x0, y0, x1, y1, text, number = run[k]
        left = x0 + (x1 - x0) * j / len(text)
        right = x0 + (x1 - x0) * (last + 1) / len(text) if end == k else x1
        more = []
        for m in range(k + 1, end + 1):  # the rest of the piece, on the lines (or page) that follow
            mx0, my0, mx1, my1, mtext, mpage = run[m]
            used = last + 1 if m == end else len(mtext)
            more.append((mpage, mx0, my0, mx0 + (mx1 - mx0) * used / len(mtext), my1))
        column = 1 if right <= mid + 1 else 2 if left >= mid - 1 else 0
        pieces.append(
            Piece(fold(part), part, at == 0, number, left, y0, right, y1, column, tuple(more))
        )
        at += len(part)
    return pieces


def figures_of(sheet, digests: dict) -> list[tuple[tuple, str]]:
    """Pictures on a page: embedded images and vector drawings clustered into figures.

    A figure is drawn from many parts (a stack of scans, boxes and arrows), so parts that touch are
    one figure, whose fingerprint covers every part. Positions are taken from the figure's own
    corner, so a figure that merely moved keeps its fingerprint. Thin strokes (arrows, ticks) join a
    figure but cannot make one, so a table of rules is not a figure."""
    import pymupdf

    parts: list[tuple] = []  # (rect, what it is, whether it can make a figure on its own)
    for info in sheet.get_image_info(xrefs=True):
        if info["xref"]:
            if info["xref"] not in digests:
                digests[info["xref"]] = hashlib.md5(sheet.parent.extract_image(info["xref"])["image"]).hexdigest()
            parts.append((pymupdf.Rect(info["bbox"]), ("image", digests[info["xref"]]), True))
    for path in sheet.get_drawings():
        rect = path["rect"]
        kind = ("path", round(rect.width), round(rect.height), str(path.get("fill")),
                str(path.get("color")), len(path["items"]))  # fmt: skip
        parts.append((rect, kind, min(rect.width, rect.height) >= FIGURE_PART))

    def touch(a, b) -> bool:  # `intersects` is false for a rule with no thickness
        return (a.x0 - FIGURE_GAP <= b.x1 and b.x0 - FIGURE_GAP <= a.x1
                and a.y0 - FIGURE_GAP <= b.y1 and b.y0 - FIGURE_GAP <= a.y1)  # fmt: skip

    groups: list[list[int]] = []
    for i, (rect, *_) in enumerate(parts):  # merge every group the part touches
        touching = [g for g in groups if any(touch(rect, parts[j][0]) for j in g)]
        groups = [g for g in groups if g not in touching] + [[i, *(j for g in touching for j in g)]]
    found = []
    for group in groups:
        box = pymupdf.Rect(parts[group[0]][0])
        for j in group:
            box |= parts[j][0]
        if box.width >= FIGURE_MIN and box.height >= FIGURE_MIN and any(parts[j][2] for j in group):
            shape = sorted((kind, round(r.x0 - box.x0), round(r.y0 - box.y0)) for r, kind, _ in (parts[j] for j in group))
            found.append((tuple(box), hashlib.md5(repr(shape).encode()).hexdigest()))
    return found


def read_pdf(data: bytes, images: bool = False) -> Pdf:
    """Text pieces (and, on request, embedded images) of a PDF as Typst laid it out.

    Words are cut into the same pieces as the HTML text (see `segment`), so the two line up."""
    try:
        import pymupdf
    except ImportError:
        raise Error(500, "the paper pages need pymupdf: run with `uv run --group paper ...`") from None
    sizes, pictures, lines, chains = [], [], [], []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        digests: dict[int, str] = {}
        for number, sheet in enumerate(doc, 1):
            sizes.append((sheet.rect.width, sheet.rect.height))
            mid = sheet.rect.width / 2
            found = sheet.get_text("words")
            line: dict[tuple, list] = {}
            for word in found:  # (block, line) is how the PDF groups words into lines
                line.setdefault(word[5:7], []).append(word)
            for group in line.values():
                lines.append((number, min(w[0] for w in group), min(w[1] for w in group),
                              max(w[2] for w in group), max(w[3] for w in group),
                              " ".join(w[4] for w in group)))  # fmt: skip
            body = [w for w in found if w[1] < sheet.rect.height - FOOTER]
            page_runs = word_runs(body, mid, number)
            if page_runs and chains and joins(chains[-1][-1], page_runs[0][0]):  # cut by the page
                first = page_runs.pop(0)
                join(chains[-1], first[0])
                chains[-1] += first[1:]
            chains += page_runs
            if images:
                pictures += [(number, box, sig) for box, sig in figures_of(sheet, digests)]
        pieces = [p for chain in chains for p in run_pieces(chain, sizes[0][0] / 2)]
        outline = [
            (level, title, page, 1 if dest["to"].x < sizes[page - 1][0] / 2 else 2, dest["to"].y)
            for level, title, page, dest in doc.get_toc(simple=False)
            if 1 <= page <= len(sizes) and "to" in dest
        ]
        return Pdf(doc.metadata.get("title") or "", sizes, pieces, pictures, lines, outline)


def page_marks(tokens: list[tuple], laid: list[Piece]) -> dict[int, str]:
    """Token index -> label for the words where the PDF starts a new page or column."""
    starts = [i for i, token in enumerate(tokens) if token[0] == "text"]
    words = [fold(tokens[i][2]) for i in starts]
    blocks = SequenceMatcher(None, words, [w.text for w in laid], autojunk=False).get_matching_blocks()
    marks, page, column = {}, 0, 0
    for block in blocks:
        if block.size < MIN_RUN:
            continue
        for k in range(block.size):
            number, col = laid[block.b + k].page, laid[block.b + k].col
            label = None
            if number != page:
                label = f"page {number}" + (f" · col {col}" if col else "")
                page, column = number, col
            elif col > column:  # within a page the flow only moves from column 1 to 2
                label, column = (f"col {col}" if column else None), col
            if label:
                marks[starts[block.a + k]] = label
    return marks


def with_marks(tokens: list[tuple], marks: dict[int, str]) -> list[tuple]:
    """Tokens with an empty <span class="pb"> in front of each marked word (after its space)."""
    out = list(tokens)
    for i, label in marks.items():
        kind, raw, key = out[i]
        space = raw[: len(raw) - len(raw.lstrip())]
        cls = "pb page" if label.startswith("page") else "pb"
        out[i] = (kind, f'{space}<span class="{cls}" data-label="{label}"></span>{raw.lstrip()}', key)
    return out


# --- the pages as laid out --------------------------------------------------------------------

SHEETS: dict[str, bytes] = {}  # PDFs of working-tree pages being viewed, by key (the newest few)
SHEET_KEY = re.compile(rf"([0-9a-f]{{40}}|{WORKTREE}-[0-9a-f]{{12}})")
FIGURE_PART = 3  # points: a stroke thinner than this cannot make a figure, only join one
FIGURE_GAP = 6  # points between parts that still belong to one figure
FIGURE_MIN = 40  # points: a cluster smaller than this is not a figure
FOOTER = 56  # points from the page bottom below which a word is the page number, not the text
MIN_MOVE = 8  # pieces: a leftover run this long on both sides is text that moved
MOVE_BUDGET = 25_000_000  # old x new leftovers beyond which moves are not looked for (too slow)
GAP = 8  # points between two highlighted words that still count as one run


def sheet_key(sha: str, version: str) -> str:
    return sha if sha != WORKTREE else f"{WORKTREE}-{version}"


def sheet_pdf(key: str, lang: str) -> bytes:
    if key.startswith(WORKTREE):
        if key not in SHEETS:
            raise Error(404, "that working-tree page is out of date")
        return SHEETS[key]
    return compiled(key, lang, "pdf")


def sheet_svg(pdf: bytes, number: int) -> bytes:
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        if not 1 <= number <= len(doc):
            raise Error(404, f"no page {number}")
        return doc[number - 1].get_svg_image(text_as_path=True).encode()


def runs(pieces: list[Piece]) -> list[list]:
    """Merge neighbouring pieces on one line into [page, x0, y0, x1, y1] boxes."""
    boxes: list[list] = []
    for p in pieces:
        for page, x0, y0, x1, y1 in ((p.page, p.x0, p.y0, p.x1, p.y1), *p.more):
            last = boxes[-1] if boxes else None
            if last and last[0] == page and abs(last[2] - y0) < 3 and -1 <= x0 - last[3] < GAP:
                last[3], last[4] = max(last[3], x1), max(last[4], y1)
            else:
                boxes.append([page, x0, y0, x1, y1])
    return boxes


def spoken(pieces: list[Piece]) -> str:
    return "".join((" " if p.space else "") + p.raw for p in pieces).strip()


def worded(pieces: list[Piece]) -> int:
    return sum(any(c.isalnum() for c in p.raw) for p in pieces)


def contents(new: Pdf, events: list[tuple]) -> list[dict]:
    """The PDF outline as a two-level contents; `events` are (page, column, y, kind) of each edit.

    Like the text page, an h2 entry counts everything up to the next h2 and an h3 only its own."""
    heads = [(page, col, y) for _, _, page, col, y in new.outline]
    entries = []
    for n, (level, title, *_) in enumerate(new.outline):
        end = next((heads[m] for m in range(n + 1, len(heads)) if new.outline[m][0] <= level), None)
        mine = [kind for *at, kind in events if heads[n] <= tuple(at) and (end is None or tuple(at) < end)]
        entries.append({
            "level": level, "id": f"toc-{n}", "text": title, "page": heads[n][0], "y": heads[n][2],
            "add": sum(k == "add" for k in mine), "rem": sum(k == "rem" for k in mine),
            "blocks": sum(k == "blocks" for k in mine),
        })  # fmt: skip
    toc: list[dict] = []
    for entry in entries:
        if entry["level"] == 1 or not toc:
            toc.append({**entry, "children": []})
        else:
            toc[-1]["children"].append(entry)
    return toc


def layout(new_pdf: bytes, old_pdf: bytes | None) -> tuple[list[dict], list[dict]]:
    """Per page of `new_pdf`: its size and, against `old_pdf`, the boxes to draw over it.
    Also the contents, with the edits counted per section.

    Added words are boxed, removed words become a tick (with the old text) where they were, and an
    embedded image that is not in the old PDF gets a frame. Boxes are percentages of the page."""
    new = read_pdf(new_pdf, images=True)
    sheets = [{"n": n, "w": w, "h": h, "added": [], "gone": [], "frames": [], "ink": [], "anchors": [], "add": 0, "rem": 0, "blocks": 0}
              for n, (w, h) in enumerate(new.sizes, 1)]  # fmt: skip

    def pct(page: int, x0: float, y0: float, x1: float, y1: float) -> str:
        w, h = new.sizes[page - 1]
        return (f"left:{x0 / w:.3%};top:{y0 / h:.3%};width:{(x1 - x0) / w:.3%};"
                f"height:{(y1 - y0) / h:.3%}")  # fmt: skip

    for page, x0, y0, x1, y1, text in new.lines:  # invisible text over the picture, to select and search
        size = (y1 - y0) * 0.85
        sheets[page - 1]["ink"].append({"x": round(x0, 2), "y": round(y1 - (y1 - y0) * 0.22, 2),
                                        "len": round(x1 - x0, 2), "size": round(size, 2), "text": text})  # fmt: skip
    for n, (_, _, page, _, y) in enumerate(new.outline):  # where each contents entry lands
        sheets[page - 1]["anchors"].append({"id": f"toc-{n}", "top": f"{max(y, 0) / new.sizes[page - 1][1]:.3%}"})
    events: list[tuple] = []
    if old_pdf is None:
        return sheets, contents(new, events)
    old = read_pdf(old_pdf, images=True)

    def keys(pieces: list[Piece]) -> list[tuple]:
        return [("text", "", p.text) for p in pieces]  # the shape `opcodes` takes

    ops = [op for op in opcodes(keys(old.pieces), keys(new.pieces)) if op[0] != "equal"]
    # A float (a wide figure or table sits at the top or bottom of some page) or a paragraph moved by
    # the page break is deleted in one place and inserted in another. Match what is left over once
    # more, and a run that turns up on both sides is a move, not an edit.
    gone = [i for _, a0, a1, _, _ in ops for i in range(a0, a1)]
    came = [j for _, _, _, b0, b1 in ops for j in range(b0, b1)]
    moved_old, moved_new = set(), set()
    while gone and came and len(gone) * len(came) <= MOVE_BUDGET:
        # one pass keeps a single in-order alignment, so moves that cross need another pass
        left = SequenceMatcher(None, [old.pieces[i].text for i in gone],
                               [new.pieces[j].text for j in came], autojunk=False)  # fmt: skip
        found = [m for m in left.get_matching_blocks() if m.size >= MIN_MOVE]
        if not found:
            break
        for m in found:
            moved_old.update(gone[m.a : m.a + m.size])
            moved_new.update(came[m.b : m.b + m.size])
        gone = [i for i in gone if i not in moved_old]
        came = [j for j in came if j not in moved_new]
    for _, a0, a1, b0, b1 in ops:
        added = [new.pieces[j] for j in range(b0, b1) if j not in moved_new]
        removed = [old.pieces[i] for i in range(a0, a1) if i not in moved_old]
        for page, x0, y0, x1, y1 in runs(added):
            sheets[page - 1]["added"].append(pct(page, x0 - 1, y0, x1 + 1, y1))
        for p in added:
            sheets[p.page - 1]["add"] += worded([p])
            events += [(p.page, 2 if p.col == 2 else 1, p.y0, "add")] * worded([p])
        if removed:
            at = new.pieces[min(b0, len(new.pieces) - 1)]
            sheet = sheets[at.page - 1]
            sheet["gone"].append({"style": pct(at.page, at.x0, at.y0, at.x0, at.y1), "text": spoken(removed)})
            sheet["rem"] += worded(removed)
            events += [(at.page, 2 if at.col == 2 else 1, at.y0, "rem")] * worded(removed)
    before = Counter(d for _, _, d in old.figures)
    for page, (x0, y0, x1, y1), digest in new.figures:
        if not before[digest]:
            sheets[page - 1]["frames"].append(pct(page, x0 - 3, y0 - 3, x1 + 3, y1 + 3))
            sheets[page - 1]["blocks"] += 1
            events.append((page, 1 if x0 < new.sizes[page - 1][0] / 2 else 2, y0, "blocks"))
        else:
            before[digest] -= 1
    return sheets, contents(new, events)


# --- contents -------------------------------------------------------------------------------


def plain(fragment: str) -> str:
    """Heading text for the contents: removed words dropped, tags stripped."""
    fragment = re.sub(r"<del\b.*?</del>", "", fragment, flags=re.DOTALL)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def edit_counts(part: str) -> dict:
    """Words added/removed and tables or figures changed within a stretch of redlined HTML."""

    def words(tag: str) -> int:
        runs = re.findall(rf'<{tag} class="{tag}">(.*?)</{tag}>', part, re.DOTALL)
        return sum(count_words(run) for run in runs)

    return {
        "add": words("ins"),
        "rem": words("del"),
        "blocks": part.count('class="tbl-ins"') + part.count('class="tbl-del">'),
    }


def listing(body: str) -> dict[str, list[dict]]:
    """Figures and tables in order, each with its label, caption and edit counts."""
    found: dict[str, list[dict]] = {"fig": [], "tab": []}
    for m in FIGURE.finditer(body):
        ident, part = m.groups()
        caption = CAPTION.search(part)
        text = plain(caption.group(1)) if caption else ""
        label, _, name = text.partition(":")
        if not name or len(label) > 30:
            label, name = ident, text
        found[ident[:3]].append(
            {"id": ident, "label": label.strip(), "text": name.strip()[:120], **edit_counts(part)}
        )
    return found


def outline(body: str) -> tuple[str, list[dict]]:
    """Give every h2/h3 an id and return (body, contents) where each entry counts its edits.

    h3 entries count their own text; an h2 entry counts everything up to the next h2.
    """
    heads = list(HEADING.finditer(body))
    out, entries, last = [], [], 0
    for n, m in enumerate(heads):
        level, attrs, inner = m.groups()
        found = re.search(r'\sid="([^"]*)"', attrs)
        ident = found.group(1) if found else f"s-{n}"
        if not found:
            attrs += f' id="{ident}"'
        end = heads[n + 1].start() if n + 1 < len(heads) else len(body)
        part = body[m.end() : end]
        entries.append(
            {"level": int(level), "id": ident, "text": plain(inner), **edit_counts(part)}
        )
        out += [body[last : m.start()], f"<h{level}{attrs}>{inner}</h{level}>"]
        last = m.end()
    out.append(body[last:])

    toc: list[dict] = []
    for entry in entries:
        if entry["level"] == 2 or not toc:
            toc.append({**entry, "children": []})
        else:
            toc[-1]["children"].append(entry)
    for h2 in toc:  # an h2 summarises its subsection too
        for key in ("add", "rem", "blocks"):
            h2[key] += sum(c[key] for c in h2["children"])
    return "".join(out), toc


# --- revision history -----------------------------------------------------------------------


class Commit(NamedTuple):
    sha: str
    when: str
    subject: str
    variants: frozenset  # which paper variants the commit can change: subset of LANGS
    shared: bool  # touches a file both variants build from (lib.typ, figures, ...)


def variants_of(files: list[str]) -> tuple[frozenset, bool]:
    """Classify a commit by the paper files it touches. Only th/ and i18n/ are Thai-only, only
    main.typ and sections/ are English-only; other source files feed both papers."""
    found, shared = set(), False
    for path in files:
        rel = path.removeprefix(PAPER + "/") if path.startswith(PAPER + "/") else None
        if rel is None:
            found |= set(LANGS)  # a data file the paper loads
            shared = True
        elif rel.startswith(("th/", "i18n/")):
            if not rel.endswith(".pdf"):
                found.add("th")
        elif rel == "main.typ" or rel.startswith("sections/"):
            found.add("en")
        elif rel.endswith((".typ", ".bib")) or rel.startswith("figures/"):
            found |= set(LANGS)
            shared = True
    return frozenset(found), shared


def history() -> list[Commit]:
    """Newest-first commits touching what the paper builds from, preceded by the working tree."""
    log = git(
        "log", f"-{HISTORY}", "--format=%x01%H\t%ad\t%s", "--date=format:%m-%d %H:%M", "--name-only",
        "--", PAPER, *DATA_DIRS, text=True,
    )  # fmt: skip
    rows = [Commit(WORKTREE, "", "working tree (uncommitted edits)", frozenset(LANGS), False)]
    for chunk in log.stdout.split("\x01")[1:]:
        header, *files = chunk.splitlines()
        sha, when, subject = header.split("\t", 2)
        variants, shared = variants_of([f for f in files if f])
        if variants:
            rows.append(Commit(sha, when, subject, variants, shared))
    return rows


def short(sha: str) -> str:
    return sha if sha == WORKTREE else sha[:7]


def prefix(lang: str) -> str:
    return "" if lang == "en" else f"/{lang}"


def picker_rows(old: str, new: str) -> list[dict]:
    """Rows for history.html: every commit, tagged with the variants it touches."""
    commits = []
    for c in history():
        commits.append(
            {
                "sha": c.sha, "ref": short(c.sha), "name": "working tree" if c.sha == WORKTREE else c.sha[:7],
                "when": c.when, "subject": c.subject, "variants": c.variants, "shared": c.shared,
                "never": c.sha == WORKTREE,  # cannot diff *from* the tree
                "from_on": c.sha == old, "to_on": c.sha == new,
            }
        )  # fmt: skip
    return commits


# --- pages ----------------------------------------------------------------------------------


def describe(rev: str, sha: str) -> str:
    if sha == WORKTREE:
        return "<code>working tree</code>"
    return f"<code>{html.escape(rev)}</code> ({sha[:7]})"


def page(lang: str, rest: str, *, label: str, old: str, new: str, body: str, title: str,
         diffing: bool, version: str, sheets: list[dict] | None = None, key: str = "",
         outline_toc: list[dict] | None = None) -> str:  # fmt: skip
    """`sheets` (from `layout`) shows the pages as Typst lays them out, in place of the text `body`."""
    toc, lists = outline_toc or [], {"fig": [], "tab": []}
    if sheets is None:
        body, toc = outline(body)
        lists = listing(body)
    tail = "?layout=pages" if sheets is not None else ""
    return TEMPLATES.get_template("page.html").render(
        lang=lang, lang_prefix=prefix(lang), label=label, title=title, body=body, toc=toc,
        figures=lists["fig"], tables=lists["tab"], diffing=diffing, commits=picker_rows(old, new),
        version=version if new == WORKTREE else "",  # only the working tree can change under us
        sheets=sheets, key=key, layout=tail,
        switch={code: f"{prefix(code)}/{rest}{tail}" for code in LANGS},
        modes={"text": f"{prefix(lang)}/{rest}", "pages": f"{prefix(lang)}/{rest}?layout=pages"},
    )  # fmt: skip


def stage(sha: str, lang: str, version: str) -> tuple[bytes, str]:
    """The PDF of a revision, and the key its page images are served under."""
    pdf, key = compiled(sha, lang, "pdf"), sheet_key(sha, version)
    if sha == WORKTREE:
        SHEETS[key] = pdf
        for stale in list(SHEETS)[:-4]:
            del SHEETS[stale]
    return pdf, key


def diff_page(lang: str, base: str, head: str, pages: bool = False) -> str:
    version = fingerprint()  # before building, so an edit during the build still shows as new
    old_sha, new_sha = resolve(base), resolve(head or WORKTREE)
    label = f"{describe(base, old_sha)} → {describe(head or WORKTREE, new_sha)}"
    rest = f"{base}..{head}"
    new_pdf, key = stage(new_sha, lang, version)
    if pages:
        sheets, toc = layout(new_pdf, compiled(old_sha, lang, "pdf"))
        return page(
            lang, rest, label=label, old=old_sha, new=new_sha, diffing=True, body="", version=version,
            title=read_pdf(new_pdf).title or lang, key=key, sheets=sheets, outline_toc=toc,
        )  # fmt: skip
    new_page = build(new_sha, lang)
    return page(
        lang, rest, label=label, old=old_sha, new=new_sha, diffing=True,
        body=redline(build(old_sha, lang), new_page, new_pdf),
        title=title_of(new_page), version=version,
    )  # fmt: skip


def view_page(lang: str, rev: str, pages: bool = False) -> str:
    """One revision, styled like the diff pages but with nothing marked."""
    version = fingerprint()
    sha = resolve(rev)
    pdf, key = stage(sha, lang, version)
    label = f"viewing {describe(rev, sha)}"
    if pages:
        sheets, toc = layout(pdf, None)
        return page(
            lang, rev, label=label, old="", new=sha, diffing=False, body="", version=version,
            title=read_pdf(pdf).title or lang, key=key, sheets=sheets, outline_toc=toc,
        )  # fmt: skip
    new_page = build(sha, lang)
    return page(
        lang, rev, label=label, old="", new=sha, diffing=False,
        body=redline(None, new_page, pdf), title=title_of(new_page), version=version,
    )  # fmt: skip


def index_page(lang: str) -> str:
    rows = [c for c in history() if lang in c.variants and c.sha != WORKTREE]  # the tree is no default
    latest, previous = rows[0].sha, rows[min(1, len(rows) - 1)].sha
    return TEMPLATES.get_template("index.html").render(
        lang=lang, lang_prefix=prefix(lang), commits=picker_rows(previous, latest),
        switch={code: f"{prefix(code)}/" for code in LANGS},
    )  # fmt: skip


def compare_redirect(lang: str, query: dict[str, list[str]]) -> str | None:
    """Turn the picker's ?from=&to= into /<from>..<to>, swapping them if given newest-first."""
    if "to" not in query:
        return None
    new = query["to"][0]
    tail = "?layout=pages" if query.get("layout") == ["pages"] else ""
    if "from" not in query:
        return f"{prefix(lang)}/{short(new)}{tail}"
    old = query["from"][0]
    order = [c.sha for c in history()]
    if old in order and new in order and order.index(old) < order.index(new):
        old, new = new, old
    return f"{prefix(lang)}/{short(old)}..{'' if new == WORKTREE else short(new)}{tail}"


# --- server ---------------------------------------------------------------------------------


def split_lang(path: str) -> tuple[str, str]:
    """`th/main..HEAD` -> ("th", "main..HEAD"). Unprefixed paths are English, as they always were."""
    head, _, rest = path.partition("/")
    return (head, rest) if head in LANGS else ("en", path)


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, payload: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def stream(self, seen: str) -> None:
        """Server-Sent Events: one `data: <fingerprint>` line per settled edit of the sources."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with WATCH.listening():
                while True:
                    now = WATCH.wait(seen, KEEPALIVE)
                    self.wfile.write(b": keepalive\n\n" if now is None else f"data: {now}\n\n".encode())
                    self.wfile.flush()
                    seen = seen if now is None else now
        except (BrokenPipeError, ConnectionResetError):
            pass  # the tab went away

    def sheet(self, path: str) -> None:
        """`_pages/<key>/<lang>/<n>.svg`: one page of a PDF Typst laid out, drawn as an SVG."""
        try:
            _, key, lang, name = path.split("/")
            if not (SHEET_KEY.fullmatch(key) and lang in LANGS and name.endswith(".svg")):
                raise Error(404, "no such page")
            svg = sheet_svg(sheet_pdf(key, lang), int(name.removesuffix(".svg")))
        except (ValueError, Error) as error:
            status = error.status if isinstance(error, Error) else 404
            return self.respond(status, str(error).encode(), "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(svg)))
        self.send_header("Cache-Control", "max-age=31536000, immutable")  # the key names the content
        self.end_headers()
        self.wfile.write(svg)

    def do_GET(self) -> None:
        raw_path, _, query = self.path.partition("?")
        path = unquote(raw_path).strip("/")
        if path == "_events":
            return self.stream(parse_qs(query).get("v", [""])[0])
        if path.startswith("_pages/"):
            return self.sheet(path)
        if path.startswith("_static/"):
            asset = VIEWER / "static" / Path(path).name  # name only: no traversal
            if not asset.is_file():
                return self.respond(404, b"not found", "text/plain")
            kind = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            return self.respond(200, asset.read_bytes(), f"{kind}; charset=utf-8")
        lang, path = split_lang(path)
        pages = parse_qs(query).get("layout") == ["pages"]
        try:
            if path == "" and (target := compare_redirect(lang, parse_qs(query))):
                self.send_response(303)
                self.send_header("Location", target)
                self.end_headers()
                return
            if path == "":
                body, status = index_page(lang), 200
            elif path == "favicon.ico":
                raise Error(404, "no favicon")
            elif ".." in path:
                base, head = path.split("..", 1)
                body, status = diff_page(lang, base, head, pages), 200
            else:
                body, status = view_page(lang, path, pages), 200
        except Error as error:
            body, status = f"<pre>{html.escape(str(error))}</pre>", error.status
        self.respond(status, body.encode(), "text/html; charset=utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"serving on http://{args.host}:{args.port}/  (try /main..HEAD or /th/main..HEAD)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
