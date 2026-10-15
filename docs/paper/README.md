# Paper draft

Typst project. Build with `typst compile --root ../.. main.typ` (or `typst watch --root ../.. main.typ`) from this folder; `--root` lets it read `data/paper/results.json`, which `scripts/paper/export_results.py` writes.

- `sections/`: one file per section, included from `main.typ`.
- `figures/`: generated figures. Do not edit by hand; regenerate with the scripts in `scripts/paper/`.
- Every number in the text comes from tracked outputs under `data/` (`data/final-scoring/`, `data/paper/`).

Figures (run from the repository root; matplotlib is not a project dependency):

```bash
uv run python scripts/paper/export_results.py
for f in examples design size_class gap kermany_size; do uv run --with matplotlib python scripts/paper/fig_$f.py; done
```
