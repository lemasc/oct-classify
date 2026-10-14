# Paper draft

Typst project. Build with `typst compile --root ../.. main.typ` (or `typst watch --root ../.. main.typ`) from this folder; `--root` lets it read `data/paper/results.json`, which `scripts/paper/export_results.py` writes.

- `sections/`: one file per section, included from `main.typ`.
- `figures/`: generated figures. Do not edit by hand; regenerate with the scripts in `scripts/paper/`.
- Every number in the text comes from tracked outputs under `data/` (`data/final-scoring/`, `data/paper/`).
