"""Serve a word-level redline of the Typst paper between two Git revisions, like latexdiff.

    uv run python scripts/paper/paper_diff.py [--port 8731]
    http://localhost:8731/main..HEAD       compare two revisions (branch, tag or SHA)
    http://localhost:8731/HEAD~3..         compare against the working tree (uncommitted edits)
    http://localhost:8731/4b9df74          view one revision without any diff

Each revision is extracted with `git archive` and compiled with `typst --features html`; the
built HTML is cached per commit under artifacts/paper-diff/. The two HTML bodies are tokenised
and diffed with difflib. Tables are diffed as whole blocks (old table dimmed above the new one,
changed cells shaded), since a per-cell diff breaks when columns move. Standard library only.
"""

import argparse
import hashlib
import html
import re
import subprocess
import tempfile
import threading
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

REPO = Path(__file__).resolve().parents[2]
PAPER = "docs/paper"
CACHE = REPO / "artifacts/paper-diff"
WORKTREE = "WORKTREE"
MIN_ANCHOR = 4  # equal runs shorter than this between two edits are folded into the edit
BUILD_LOCK = threading.Lock()

CSS = """
:root{color-scheme:light dark;--ins:#d6f5d6;--insfg:#064;--del:#fcd9d9;--delfg:#900;--chg:#b9ecb9}
@media(prefers-color-scheme:dark){:root{--ins:#1d4a28;--insfg:#9fe6b0;--del:#5a2323;--delfg:#f3a0a0;--chg:#2c6b3c}}
body{font:16px/1.6 Georgia,serif;max-width:46rem;margin:0 auto;padding:0 1rem 4rem}
h1{text-align:center;font-size:1.6rem;margin:2rem 0 1rem}
h2{margin-top:2.2rem;border-bottom:1px solid #8884}
figure{margin:1.5rem 0;text-align:center}
figure img{max-width:100%;height:auto}
figcaption{font-size:.9rem;opacity:.85;text-align:left}
.tbl{overflow-x:auto}
table{border-collapse:collapse;margin:0 auto;font-size:.85rem}
td,th{border-bottom:1px solid #8886;padding:.2rem .5rem}
ins.ins{background:var(--ins);color:var(--insfg);text-decoration:none}
del.del{background:var(--del);color:var(--delfg)}
.tbl-ins,.tbl-del{padding:.4rem;margin:.5rem 0;border-radius:4px}
.tbl-ins{outline:2px solid #6c6}
.tbl-del{outline:2px dashed #c66;opacity:.7}
.tbl-ins::before,.tbl-del::before{font:12px sans-serif}
.tbl-ins::before{content:'new / changed table';color:var(--insfg)}
.tbl-del::before{content:'removed table';color:var(--delfg)}
td.chg,th.chg{background:var(--chg)}
.legend{position:sticky;top:0;background:Canvas;font:13px sans-serif;padding:.4rem 0;
 border-bottom:1px solid #8884;z-index:1}
.legend code{font-size:12px}
.legend details{max-height:60vh;overflow:auto}
.history{font:13px sans-serif}
.history table{margin:.5rem 0;width:100%}
.history td{border:0;padding:.1rem .4rem}
.history td:first-child,.history td:nth-child(2){text-align:center;width:2rem}
.history tr:has(:checked){background:#8882}
.history button{font:inherit;padding:.2rem .8rem}
"""

BLOCKS = {"p", "h1", "h2", "h3", "h4", "li", "ul", "ol", "figure", "figcaption", "tr", "div"}
TOKEN = re.compile(r"<[^>]+>|[^<\s]+|\s+")
TABLE = re.compile(r"<table.*?</table>", re.DOTALL)
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


def typst(workdir: Path, out: Path) -> None:
    done = subprocess.run(
        ["typst", "compile", "--root", "../..", "--features", "html", "--format", "html",
         "main.typ", str(out)],
        cwd=workdir, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if done.returncode:
        raise Error(500, f"typst failed in {workdir}:\n{done.stderr}")


def build(sha: str) -> str:
    """Return the HTML of the paper at `sha` (or the working tree), compiling on a cache miss."""
    if sha == WORKTREE:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "main.html"
            typst(REPO / PAPER, out)
            return out.read_text()
    cached = CACHE / f"{sha}.html"
    with BUILD_LOCK:
        if not cached.exists():
            with tempfile.TemporaryDirectory() as tmp:
                archive = git("archive", sha, PAPER, "data/paper")
                if archive.returncode:
                    raise Error(404, f"{PAPER} not found at {sha[:7]}")
                subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
                CACHE.mkdir(parents=True, exist_ok=True)
                partial = cached.with_suffix(".tmp")
                typst(Path(tmp) / PAPER, partial)
                partial.replace(cached)
    return cached.read_text()


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


def mark_table(old: str, new: str) -> str:
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
            return f'<{tag}{attrs} class="chg">{inner}</{tag}>'

        return "<tr>" + CELL.sub(mark_cell, row.group(1)) + "</tr>"

    return ROW.sub(mark_row, new)


def body_of(page: str) -> str:
    start = page.index(">", page.index("<body")) + 1
    return page[start : page.rindex("</body>")]


def title_of(page: str) -> str:
    return html.unescape(re.search(r"<title>(.*?)</title>", page).group(1))


def tokenize(body: str) -> list[tuple]:
    """Tokens are (kind, raw, key); keys ignore id/href so renumbered references still match."""
    tokens, space, pos = [], "", 0
    chunks = []
    for m in TABLE.finditer(body):
        chunks += [("html", body[pos : m.start()]), ("table", m.group(0))]
        pos = m.end()
    chunks.append(("html", body[pos:]))
    for kind, chunk in chunks:
        if kind == "table":
            digest = hashlib.md5(re.sub(r'\s(id|href)="[^"]*"', "", chunk).encode()).hexdigest()
            tokens.append(("table", space + chunk, "table:" + digest))
            space = ""
            continue
        for m in TOKEN.finditer(chunk):
            text = m.group(0)
            if text.isspace():
                space = " "
            elif text.startswith("<"):
                name = re.match(r"</?\s*([a-zA-Z0-9]+)", text)
                name = name.group(1).lower() if name else "?"
                key = "img:" + hashlib.md5(text.encode()).hexdigest() if name == "img" else (
                    ("/" if text.startswith("</") else "") + name)  # fmt: skip
                tokens.append(("tag", space + text, key))
                space = ""
            else:
                tokens.append(("text", space + text, text))
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

    def first_table(tokens, lo, hi):
        return next((i for i in range(lo, hi) if tokens[i][0] == "table"), None)

    def pair_tables(tag, a0, a1, b0, b1):
        """Split an edit at its first table on each side so old and new tables stay together."""
        ia, ib = first_table(old, a0, a1), first_table(new, b0, b1)
        if tag == "equal" or ia is None or ib is None:
            return [(tag, a0, a1, b0, b1)]
        return [*pair_tables("replace", a0, ia, b0, ib), ("replace", ia, ia + 1, ib, ib + 1),
                *pair_tables("replace", ia + 1, a1, ib + 1, b1)]  # fmt: skip

    return [piece for op in ops for piece in pair_tables(*op)]


def redline(old_page: str | None, new_page: str, label: str, picker: str = "") -> str:
    """Diff `old_page` against `new_page`; with no `old_page`, just render `new_page` styled."""
    new = tokenize(body_of(new_page))
    old = new if old_page is None else tokenize(body_of(old_page))
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
            and old[a0][0] == "table"
            and new[b0][0] == "table"
            and a1 - a0 == b1 - b0 == 1
        ):
            out.append(f'<div class="tbl-del"><div class="tbl">{old[a0][1]}</div></div>')
            out.append(
                f'<div class="tbl-ins"><div class="tbl">{mark_table(old[a0][1], new[b0][1])}</div></div>'
            )
        else:
            wrap(old[a0:a1], "del", keep_tags=False)
            wrap(new[b0:b1], "ins", keep_tags=True)

    title = html.escape(title_of(new_page))
    key = ' <ins class="ins">added</ins> <del class="del">removed</del>' if old_page else ""
    legend = (
        f'<div class="legend">{label}{key}'
        f"<details><summary>change range</summary>{picker}</details></div>"
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title} (diff)</title><style>{CSS}</style></head>"
        f"<body>{legend}<h1>{title}</h1>{''.join(out)}</body></html>"
    )


# --- server ---------------------------------------------------------------------------------


def history() -> list[tuple[str, str, str]]:
    """Newest-first (sha, date, subject) of commits touching the paper, preceded by the working tree."""
    log = git(
        "log", "-60", "--format=%H\t%ad\t%s", "--date=format:%m-%d %H:%M", "--", PAPER, text=True
    )
    rows = [tuple(line.split("\t", 2)) for line in log.stdout.splitlines()]
    return [(WORKTREE, "", "working tree (uncommitted edits)"), *rows]


def picker(rows: list[tuple[str, str, str]], old: str, new: str) -> str:
    """Wikipedia-style history form: pick one 'from' and one 'to' radio, then Compare."""
    body = ""
    for sha, when, subject in rows:
        never = " disabled data-never" if sha == WORKTREE else ""  # cannot diff *from* the tree
        from_on = " checked" if sha == old else ""
        to_on = " checked" if sha == new else ""
        name = "working tree" if sha == WORKTREE else sha[:7]
        body += (
            f'<tr><td><input type="radio" name="from" value="{sha}"{from_on}{never}></td>'
            f'<td><input type="radio" name="to" value="{sha}"{to_on}></td>'
            f"<td><code>{name}</code></td><td>{when}</td><td>{html.escape(subject)}</td></tr>"
        )
    return (
        '<form class="history" action="/" method="get">'
        f'<table>{body}</table><button type="submit">Compare selected</button>'
        "<script>"
        "const f=document.currentScript.parentElement,"
        "r=n=>[...f.querySelectorAll(`input[name=${n}]`)],"
        "at=n=>r(n).findIndex(x=>x.checked),"
        "sync=()=>{const a=at('from'),b=at('to');"
        "r('from').forEach((x,i)=>x.disabled=x.hasAttribute('data-never')||(b>=0&&i<=b));"
        "r('to').forEach((x,i)=>x.disabled=a>=0&&i>=a)};"
        "f.addEventListener('change',sync);sync()"
        "</script></form>"
    )


def describe(rev: str, sha: str) -> str:
    if sha == WORKTREE:
        return "<code>working tree</code>"
    return f"<code>{html.escape(rev)}</code> ({sha[:7]})"


def diff_page(base: str, head: str) -> str:
    old_sha, new_sha = resolve(base), resolve(head or WORKTREE)
    label = f"{describe(base, old_sha)} → {describe(head or WORKTREE, new_sha)}"
    return redline(build(old_sha), build(new_sha), label, picker(history(), old_sha, new_sha))


def view_page(rev: str) -> str:
    """One revision, styled like the diff pages but with nothing marked."""
    sha = resolve(rev)
    return redline(None, build(sha), f"viewing {describe(rev, sha)}", picker(history(), "", sha))


def index_page() -> str:
    rows = history()
    latest, previous = rows[1][0], rows[2][0]  # the working tree row is not a default
    return (
        "<!DOCTYPE html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        f"<style>{CSS}</style><body><h1>Paper diff</h1>"
        "<p>Pick a range below, or open <code>/&lt;base&gt;..&lt;head&gt;</code> directly, e.g. "
        "<code>/main..HEAD</code>; an empty head means the working tree. "
        "<code>/&lt;rev&gt;</code> alone views that revision with no diff.</p>"
        f"{picker(rows, previous, latest)}</body>"
    )


def compare_redirect(query: dict[str, list[str]]) -> str | None:
    """Turn the picker's ?from=&to= into /<from>..<to>, swapping them if given newest-first."""
    if "to" not in query:
        return None
    new = query["to"][0]
    if "from" not in query:
        return f"/{'WORKTREE' if new == WORKTREE else new[:7]}"
    old = query["from"][0]
    order = [sha for sha, _, _ in history()]
    if old in order and new in order and order.index(old) < order.index(new):
        old, new = new, old
    return f"/{old[:7]}..{'' if new == WORKTREE else new[:7]}"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        raw_path, _, query = self.path.partition("?")
        path = unquote(raw_path).strip("/")
        try:
            if path == "" and (target := compare_redirect(parse_qs(query))):
                self.send_response(303)
                self.send_header("Location", target)
                self.end_headers()
                return
            if path == "":
                body, status = index_page(), 200
            elif path == "favicon.ico":
                raise Error(404, "no favicon")
            elif ".." in path:
                base, head = path.split("..", 1)
                body, status = diff_page(base, head), 200
            else:
                body, status = view_page(path), 200
        except Error as error:
            body, status = f"<pre>{html.escape(str(error))}</pre>", error.status
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"serving on http://{args.host}:{args.port}/  (try /main..HEAD)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
