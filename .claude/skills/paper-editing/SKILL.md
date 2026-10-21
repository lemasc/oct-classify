---
name: paper-editing
description: >-
  Edit the paper text in docs/paper/sections/*.typ (English) and docs/paper/th/sections/*.typ
  (Thai): add, rewrite, cut or trim prose, captions, tables and figures, and keep claims
  calibrated to the evidence. Use for any change to the paper text, including a cleanup pass
  to fit the page budget (de-duplicate, shorten, split run-in-bold paragraphs).
---

# Paper editing

Both languages are written. English is the source of truth; Thai is kept in step with it.
Audience: technical, not medical. The conventions below are language-neutral and apply to every
edit. The cleanup pass (end of file) is the extra workflow for trimming to the page budget.

## Both languages

- Edit English first, then bring the matching Thai section to the same content. Do not change
  content in Thai alone; Thai-only fixes are wording only.
- Whether to port each change in the same pass or after the user approves the English text is the
  user's call per session. Ask once at the start; do not assume.
- `scripts/paper/paper_diff.py` serves a word-level redline between two Git revisions (English,
  and Thai under `/th`). Use it to see what a Thai section is missing.
- Thai conventions and mechanics (what to leave untranslated, `i18n/th.typ`, fonts, shared
  figures) live in `docs/paper/README.md`. Follow them; do not copy them here.

## Working tree (uncommitted, two workflows)

Every edit you make stays uncommitted, in the working tree. Commit only when the user asks (or
they commit themselves). Do not run `git add`, `commit`, `stash`, `checkout --` or `reset` on
paper files; use read-only `git status` and `git diff`. Staging is the user's marker of what is
yours, so touching the index breaks it.

Start of every session, run `git status --short` and `git diff` / `git diff --cached` on
`docs/paper` to see which workflow applies:

1. **User edited first** (unstaged changes, nothing staged): the unstaged diff is the user's
   work. Read it, name the pattern(s) it applies (a cut, a reworded claim, a table change, a
   convention), and check whether the same pattern holds elsewhere: other sections, and the
   other language. Report where it applies and ask before searching further or applying it.
2. **User edits after your changes** (the user stages your changes first): the staged diff is
   yours, the unstaged diff is theirs. Analyze the unstaged changes the same way, against the
   staged state, and ask. Keep making your own edits in the working tree; the user re-stages them.
   If you see unstaged changes with a staged diff you do not recognize, ask whose they are.

In both: a user edit is settled content, never revert or rewrite it to fit a rule; if it
conflicts with a rule here, say so and let the user decide. When a pattern is approved, apply it
mechanically, and bring the other language to the same content (same pass or after, as the user
chose). Report what you changed per file, so the user can review the diff.

## Where content lives (one home per number)

- **Abstract**: headline findings, concise, must read as flowing prose.
- **Introduction**: research questions only. No inline answers, no forward references that spoil.
- **Results**: owns the numbers. Tables carry them; prose gives only the highlights.
- **Discussion**: meaning, not numbers. Cite the table instead of restating gaps.
- **Data / Method**: state each fact once (e.g. PAIMA masked loss lives in Method only).
- **Background**: cite for a claim, not a narrative. Design justifications move to the section
  that uses them. Forward pointers stay neutral ("tests whether", not "shows that").
- Same claim in two places: keep the natural home, replace the other with `@sec-`/`@tab-`.
- New content goes where its number or claim naturally lives, not where it was noticed.

## Paragraphs

- **No run-in bold**: `*Bold lead.* long explanation...` is forbidden. Thai has no way to signal
  a new subject except a paragraph break, so the literal translation is unreadable.
- One paragraph = one point, about 3-4 sentences, first sentence states the point.
  New subject = new paragraph or a `===` subheading.
- Settings and hyperparameters as bullets.
- Splitting adds breaks, so do it *with* cuts, not instead of them.

## Tables and figures

Two columns make these rules hard constraints.
- Small, single-purpose, one point each. Too dense: keep the significant part, or split into
  dedicated tables. Do not merge tables back together.
- A table that needs more than one column's width must be a deliberate exception in `wide-figures`
  (`template.typ`, one list for both languages). Review wide floats first when over budget.
- **Raw before derived**: show a difference only if its raw numbers appear somewhere. If you
  show one set of numbers, show the others for a fair comparison.
- Rows of straight zeros lose to one plain sentence.
- Direct model outputs deserve a table; question later tables that restate them.
- Drop a figure when a table already holds the same quantity and the text explains more than
  the figure shows. Keep Grad-CAM as illustration only; the intervention is the evidence.
- For every table ask: what is its purpose, what does the interval/noise prove, does it weaken
  a later claim? "The numbers are inline so the table is redundant" is not enough by itself;
  improve the table or give a reason.
- Never hard-code `pt` sizes; table and caption text scale with the body.

## Captions

- State at high level what the figure/table does, units and interval type. **At most two lines
  in a column.**
- Column definitions, per-source notes and findings go in the paragraph. If the paragraph
  already has the detail, drop it from the caption; add it to the paragraph only if missing.
- Keep detail with no other home (scale-bar note, "not the same physical scale").
- "Columns as in @tab-x" instead of redefining.

## Claims

- Check for overclaiming; edit or drop the weak ones. Do not name a cause the data did not test.
- Verbs match evidence: Grad-CAM "illustrates", interventions are "evidence", partial causes
  are "partial" with their share.
- State scope with the number (test split vs full source) especially in the abstract.
- Say what is unexplained once, then point to limitations.
- **Verify every number and reference** before adding or keeping it ("is it significant enough to
  include?"). Check references against sources; do not work around fetch restrictions, defer to
  the user.
- Soften, don't delete, when the evidence is mixed.

## Mechanics

- Keep Typst helper calls (`#pt`, `#dl`, `#num`, `#span`, ...) and `<labels>`. When deleting
  prose, drop its helper calls; remove `#let` helpers and imports left unused; check nothing
  still references a removed label (compile error), in either language.
- Every number comes from tracked outputs under `data/`; never retype one.
- Build with `typst compile --root ../.. main.typ` and `th/main.typ` from `docs/paper`. Look at the
  render of any figure or table you changed.
- Run `scripts/preflight.sh` after edits: run-in bold must stay at 0, and both languages must build.
- Never commit, amend, stage or stash on your own; see Working tree. When the user asks for a
  commit, one theme per commit: `docs(paper): <what changed>`, body with bullets per section.

## Cleanup pass (trimming to the page budget)

Use when asked to clean up, trim, shorten or de-duplicate.

**Budget.** The paper has a soft page budget in the two-column layout. Confirm the current target
with the user at the start; do not assume one. Both languages must fit and Thai runs longer, so
measure the ratio with preflight each time and trim English hard enough for Thai to fit. Words are
only a proxy: **floats cost more than prose**. A full-width table or figure takes far more room
than its word count suggests, so cutting or shrinking floats often saves more than trimming text.

1. Run `scripts/preflight.sh` (from anywhere in the repo): pages for both languages, words per
   section, run-in bold, long paragraphs, full-width floats. PDFs go to a temp dir; do not
   overwrite `docs/paper/main.pdf`. For pages per section, read the PDF outline positions. Do not
   copy these numbers into this skill or its notes.
2. **Plan first, change nothing.** Read abstract and all six sections. Report findings in tiers:
   - **A**: pure duplication, safe to drop.
   - **B**: judgement calls, likely worth cutting.
   - **C**: not recommended without the user's call.
   Rank sections by where the pages actually go (measured now), including which floats are
   full-width, and say how much each cut should save.
3. The user picks items ("Drop on 1", "Defer 2 and 4"). Apply **text-only, non-breaking** cuts
   first, then table/figure changes.
   Keep deferred items and list them on request. For a risky change, say so first; the user can
   stage or commit what they want kept.
4. Port each approved cut to the matching Thai section, in the same pass or after, as the user
   chose. A cut is not done until both languages build and the label references still resolve.
5. Re-run preflight and report page deltas per language.

See `reference.md` for the commits these rules came from.
