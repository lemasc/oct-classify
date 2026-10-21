#!/usr/bin/env bash
# Measure the paper against the length budget. Run from anywhere in the repo.
# Usage: preflight.sh [outdir]   (default: a temp dir; PDFs are written there, not into docs/paper)
set -euo pipefail
root=$(git rev-parse --show-toplevel)
out=${1:-$(mktemp -d)}
mkdir -p "$out"
cd "$root/docs/paper"

for lang in en th; do
  [ "$lang" = en ] && m=main.typ || m=th/main.typ
  typst compile --root ../.. "$m" "$out/$lang.pdf" 2>&1 | grep -E '^error' || true
  echo "$lang pages: $(pdfinfo "$out/$lang.pdf" 2>/dev/null | awk '/^Pages/{print $2}')"
done

echo; echo "words per English section (source, includes inline code):"
wc -w sections/*.typ | sed 's|sections/||'
echo; echo "words per Thai section (source, segmented with pythainlp; still not directly comparable to English):"
(cd "$root" && uv run --quiet --group paper .claude/skills/paper-editing/scripts/count_thai_words.py docs/paper/th/sections/*.typ)

echo; echo "run-in bold paragraphs (should be 0; each is a point that needs its own paragraph or heading):"
grep -nE '^\*[^*]+\*' sections/*.typ | cut -c1-100 || true

echo; echo "paragraphs over 150 words (English):"
awk -v RS= -v FS='\n' '{n=gsub(/[^ \n]+/,"&"); if(n>150) printf "%s: paragraph %d, %d words\n", FILENAME, NR, n}' sections/*.typ

echo; echo "full-width floats (wide-figures in template.typ):"
grep -oE '"(fig|tab)-[a-z0-9-]+"' template.typ | sort -u | tr '\n' ' '; echo
echo "figures: $(grep -ohE '<fig-[a-z0-9-]+>' sections/*.typ | sort -u | wc -l), tables: $(grep -ohE '<tab-[a-z0-9-]+>' sections/*.typ | sort -u | wc -l)"
echo "PDFs in $out"
