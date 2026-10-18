# Paper draft

Typst project. Build with `typst compile --root ../.. main.typ` (or `typst watch --root ../.. main.typ`) from this folder; `--root` lets it read `data/paper/results.json`, which `scripts/paper/export_results.py` writes.

- `sections/`: one file per section, included from `main.typ` (English).
- `th/`: Thai companion; see [Thai version](#thai-version).
- `i18n/th.typ`: Thai strings (title, source and split names, the `translated` list). Thai section files import it after `lib.typ` so these names shadow the English ones.
- `template.typ`: page and font setup shared by both languages.
- `figures/`: generated figures. Do not edit by hand; regenerate with the scripts in `scripts/paper/`.
- Every number in the text comes from tracked outputs under `data/` (`data/final-scoring/`, `data/paper/`).

## Thai version

`th/main.typ` has the same sections as `main.typ`. For each one it uses the Thai file `th/sections/<name>.typ` if `<name>` is listed in `translated` (`i18n/th.typ`), and otherwise the English `sections/<name>.typ`, marked "ยังไม่ได้แปล" (hidden with `--input final=1`). So the Thai paper always builds, and sections switch over one at a time.

```bash
typst compile --root ../.. th/main.typ   # or `typst watch`; the PDF is th/main.pdf (git-ignored)
```

Translating a section:

1. Copy `sections/<name>.typ` over the stub `th/sections/<name>.typ` and translate it. Keep the two import lines at the top: `../../lib.typ` and, from `../../i18n/th.typ`, `source-name, split-name` (add other names you need; do not import `*`, it would shadow Typst's `title`).
2. Add `"<name>"` to `translated` in `i18n/th.typ`. Do this last: the stubs are placeholders and, if listed, break the build.
3. Compile. Every other section (Thai or English) cites labels from this one, so a missing label is a compile error.

Rules:

- Translate prose, captions and table headers. Leave unchanged: code (`#pt(…)`, `#ci(…)`, `#dp(…)`, `#num(…)`, `#let …`), numbers, `@refs`, `<labels>` and citations (`@srinivasan2014`). Numbers are the same in both languages, so move the call to where the Thai word order puts it rather than retyping the value.
- Names and mathematics that are Latin stay Latin: dataset and model names (Kermany, ResNet-50), metrics (macro-F1, AUROC), LOSO, AdaBN, Grad-CAM, `$…$`. Give a Thai term on first use where one exists.
- Table text is a fixed fraction of the body size (see `template.typ`), so Thai tables get larger text than English ones without any change. Do not hard-code `pt` sizes.
- Each table builds its header inline, so translate it in place. `recall-table`, `rc` and `stage-rc` in `results.typ` have English headers hard-coded; they are copied with the rest of the file, so translate them there.
- Source and split names in tables come from `source-name` and `split-name` in `i18n/th.typ`; edit them there, not in the section.
- Title and abstract title are in `i18n/th.typ`. Set `abstract` there once the English abstract is written; `none` shows a TODO.
- Figures in `figures/` are shared. Their text is English and baked in (the SVGs have outlined glyphs, so it cannot be edited in Typst); the Thai captions describe them. To localise one, add a language option to its `scripts/paper/fig_*.py` and point the Thai section at it.
- The bibliography is shared (`refs.bib`).
- Fonts: TH Sarabun PSK, at 14pt, with Libertinus Serif as the Latin fallback, chosen in `template.typ`. It must be installed on the machine (`typst fonts` lists it); otherwise Typst silently falls back.
- When the English text changes, update the matching Thai section by hand; `scripts/paper/paper_diff.py` shows what changed in the English one (and in the Thai one under `/th`).

Figures (run from the repository root; matplotlib is not a project dependency):

```bash
uv run python scripts/paper/export_results.py
uv run python scripts/paper/export_data_stats.py   # needs artifacts/ (manifests, splits, audits); writes data/paper/data-stats.json
uv run python scripts/paper/export_seeds.py        # needs the exp. 2 seed runs (--extra-root); writes data/paper/seeds.json
for f in examples design padding size_class gap kermany_size; do uv run --with matplotlib python scripts/paper/fig_$f.py; done
```

Review changes between two revisions (redline, like latexdiff; standard library only, needs `typst`):

```bash
uv run --group paper python scripts/paper/paper_diff.py   # then open http://localhost:8731/main..HEAD  (`<base>..` = vs. working tree; `/<rev>` = view without diff; prefix `/th` for the Thai paper)
```
