"""Count words in Typst sources, segmenting Thai runs with pythainlp (Thai has no spaces).

Usage: uv run --group paper count_thai_words.py FILE...
Prints one `count name` line per file and a total, like `wc -w`.
"""

import re
import sys
from pathlib import Path

from pythainlp.tokenize import word_tokenize

THAI = re.compile(r"[฀-๿]")


def count_words(text: str) -> int:
    n = 0
    for chunk in text.split():
        if THAI.search(chunk):
            pieces = word_tokenize(chunk, engine="newmm", keep_whitespace=False)
        else:
            pieces = [chunk]
        n += sum(any(c.isalnum() for c in p) for p in pieces)
    return n


def main(paths: list[str]) -> None:
    total = 0
    for path in paths:
        n = count_words(Path(path).read_text(encoding="utf-8"))
        total += n
        print(f"{n:7d} {Path(path).name}")
    print(f"{total:7d} total")


if __name__ == "__main__":
    main(sys.argv[1:])
