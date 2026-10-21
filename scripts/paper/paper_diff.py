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
        new = with_marks(new, page_marks(new, pdf))  # keys are untouched, so the diff is the same
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


def fold(word: str) -> str:
    """Comparable form of a word: HTML entities decoded, ligatures and width variants unfolded."""
    return unicodedata.normalize("NFKC", html.unescape(word))


def pdf_words(data: bytes) -> list[tuple[str, int, int]]:
    """(piece, page, column) in laid-out order; column 0 means the word spans both columns.

    Words are cut into the same pieces as the HTML text (see `segment`): in Thai a PDF "word" is a
    line fragment, so only dictionary pieces line up."""
    try:
        import pymupdf
    except ImportError:
        raise Error(500, "page markers need pymupdf: run with `uv run --group paper ...`") from None
    found = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for number, sheet in enumerate(doc, 1):
            mid = sheet.rect.width / 2
            for x0, _, x1, _, word, *_ in sheet.get_text("words"):
                column = 1 if x1 <= mid + 1 else 2 if x0 >= mid - 1 else 0
                word = word.replace("\u0e4d\u0e32", "\u0e33")  # extracted as nikhahit + sara aa
                found += [(fold(piece), number, column) for piece in segment(word)]
    return found


def page_marks(tokens: list[tuple], pdf: bytes) -> dict[int, str]:
    """Token index -> label for the words where the PDF starts a new page or column."""
    starts = [i for i, token in enumerate(tokens) if token[0] == "text"]
    words = [fold(tokens[i][2]) for i in starts]
    laid = pdf_words(pdf)
    blocks = SequenceMatcher(None, words, [w for w, _, _ in laid], autojunk=False).get_matching_blocks()
    marks, page, column = {}, 0, 0
    for block in blocks:
        if block.size < MIN_RUN:
            continue
        for k in range(block.size):
            _, number, col = laid[block.b + k]
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
         diffing: bool, version: str) -> str:  # fmt: skip
    body, toc = outline(body)
    lists = listing(body)
    return TEMPLATES.get_template("page.html").render(
        lang=lang, lang_prefix=prefix(lang), label=label, title=title, body=body, toc=toc,
        figures=lists["fig"], tables=lists["tab"], diffing=diffing, commits=picker_rows(old, new),
        version=version if new == WORKTREE else "",  # only the working tree can change under us
        switch={code: f"{prefix(code)}/{rest}" for code in LANGS},
    )  # fmt: skip


def diff_page(lang: str, base: str, head: str) -> str:
    version = fingerprint()  # before building, so an edit during the build still shows as new
    old_sha, new_sha = resolve(base), resolve(head or WORKTREE)
    label = f"{describe(base, old_sha)} → {describe(head or WORKTREE, new_sha)}"
    new_page = build(new_sha, lang)
    return page(
        lang, f"{base}..{head}", label=label, old=old_sha, new=new_sha, diffing=True,
        body=redline(build(old_sha, lang), new_page, compiled(new_sha, lang, "pdf")),
        title=title_of(new_page), version=version,
    )  # fmt: skip


def view_page(lang: str, rev: str) -> str:
    """One revision, styled like the diff pages but with nothing marked."""
    version = fingerprint()
    sha = resolve(rev)
    new_page = build(sha, lang)
    return page(
        lang, rev, label=f"viewing {describe(rev, sha)}", old="", new=sha, diffing=False,
        body=redline(None, new_page, compiled(sha, lang, "pdf")),
        title=title_of(new_page), version=version,
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
    if "from" not in query:
        return f"{prefix(lang)}/{short(new)}"
    old = query["from"][0]
    order = [c.sha for c in history()]
    if old in order and new in order and order.index(old) < order.index(new):
        old, new = new, old
    return f"{prefix(lang)}/{short(old)}..{'' if new == WORKTREE else short(new)}"


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

    def do_GET(self) -> None:
        raw_path, _, query = self.path.partition("?")
        path = unquote(raw_path).strip("/")
        if path == "_events":
            return self.stream(parse_qs(query).get("v", [""])[0])
        if path.startswith("_static/"):
            asset = VIEWER / "static" / Path(path).name  # name only: no traversal
            if not asset.is_file():
                return self.respond(404, b"not found", "text/plain")
            kind = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            return self.respond(200, asset.read_bytes(), f"{kind}; charset=utf-8")
        lang, path = split_lang(path)
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
                body, status = diff_page(lang, base, head), 200
            else:
                body, status = view_page(lang, path), 200
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
