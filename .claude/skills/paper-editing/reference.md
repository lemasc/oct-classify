# Where these rules came from

Branch `docs/paper-cleanup` (merged as c3492a8) plus earlier trims. Sessions Oct 17-18.

| Rule | Evidence |
|---|---|
| Intro keeps research questions only | 422c9cc |
| One home per number; discussion points to tables | 524f015, 63e8393 |
| Results prose cites the table instead of reciting it | a2680cc, f383cfc (tab-seen: "gives up at most X" instead of four differences) |
| Captions high-level, detail into text | f23ae6b; user reverted fig 1 (scale bar) and fig 2 (physical scale) captions |
| Raw before derived; AUROC own table | e6610e7 (user reverted a version that collapsed AUROC into differences) |
| Dense table: split into one-point tables | b340043 (tab-square -> result + mechanism) |
| Drop a figure the table supersedes | f94598f (fig-gap) |
| Limit controls to what the argument needs | 453f9df (zero-DME only on Kermany) |
| Grad-CAM illustrates, intervention is evidence | f383cfc (cites Arun, Hooker; user verified refs first) |
| Don't name untested causes | 5d54001 (gap sits in different classes per source) |
| Footnotes only when needed | ab7fb98 |
| Trim related work; move design citations to use | 91913dc, 158b3ae, e911f32 |
| Drop imports left behind | 422c9cc |

Those sessions were English-only, before Thai existed. All six sections are now translated, so
the skill treats Thai as a port of the English text. The earlier Thai preferences document
(removed in bc74040) is outdated and deliberately not carried over; `docs/paper/README.md` is the
current source.

Not recoverable from history: rejected ideas that never reached a commit; whether the remaining
Tier B items from the first plan (dry-AMD thread, Background OCT-physics detail and fig-scan,
diabetic-retinopathy statistics in the intro, per-eye scoring footnote in Data) are still wanted.
Ask the user.
