# Post-Training Analysis Guide

This guide explains how to produce the post-training analysis outputs, how to browse them in
`notebooks/analysis_results.py`, and how to draw conclusions from each section. Everything is
computed from artifacts a finished run already has (prediction CSVs, `checkpoint-best.pt`), so no
retraining is needed. For the conclusions drawn from the current ResNet-50 runs, see
[reports/20260927-resnet50-analysis.md](reports/20260927-resnet50-analysis.md).

## 1. Produce the outputs

Run the three commands in order. The notebook only reads their outputs.

```bash
# 1. Per run: CIs, calibration, curves, slices, and gallery indexes (CPU only, seconds per run).
uv run oct-classify analyze run artifacts/runs/fused/resnet50/loso-paima-local-20260918-082742-fold{1..5}

# 2. Per campaign: aggregate the same prediction set across checkpoints.
uv run oct-classify analyze campaign --name loso-paima \
    artifacts/runs/fused/resnet50/loso-paima-local-20260918-082742-fold{1..5}

# 3. Per run: Grad-CAM for the gallery images (GPU by default; needs step 1).
uv run oct-classify analyze gradcam artifacts/runs/fused/resnet50/loso-paima-local-20260918-082742-fold1
```

In zsh, pass run lists as an array or glob. A single quoted string of paths is treated as one
argument.

| Command | Writes | Main options |
|---|---|---|
| `analyze run` | `<run>/analysis/<set>/` | `--n-boot 2000`, `--n-boot-auroc 500`, `--gallery-size 24`, `--max-per-group 3`, `--sets`, `--output` |
| `analyze campaign` | `data/analysis/<name>/` | `--name`, `--sets`, `--n-boot` |
| `analyze gradcam` | `<run>/analysis/<set>/gradcam/` | `--device`, `--layers layer3 layer4`, `--randomization-sample 32`, `--checkpoint` |

A **set** is one prediction file inside a run. Its name says where it came from:

- `test-<source>` comes from `predictions-test.csv` or `predictions-test/<source>.csv`.
- `eval-<name>` comes from `evaluations/<name>/predictions.csv`, e.g. `eval-kermany` (a cross-source
  evaluation) or `eval-paima-loso` (a held-out source evaluated on its full manifest).

`analyze run` prints one line per set. If a line ends in `WARNING: differs from saved metrics`, the
recomputed accuracy, balanced accuracy, macro-F1, or macro-AUROC disagrees with the run's
`metrics-test.json` / `metrics.json`. Stop and investigate before trusting anything else for that
set.

For the retained ResNet-50 runs, use these campaign names:

| Campaign | Runs |
|---|---|
| `baseline-duke` | `duke/resnet50/local-20260916-duke-fold{1..5}` |
| `fused-cv` | `fused/resnet50/local-20260917-184617-fold{1..5}` |
| `loso-duke` | `fused/resnet50/loso-duke-local-20260918-082742` |
| `loso-{kermany,octdl,paima}` | `fused/resnet50/loso-<source>-local-20260918-082742-fold{1..5}` |

## 2. Open the notebook

```bash
uv run --group notebook marimo edit notebooks/analysis_results.py   # interactive
uv run --group notebook marimo export html notebooks/analysis_results.py -o analysis.html  # static
```

The notebook discovers analyzed sets under `artifacts/runs/*/*/*/analysis/` and campaign tables
under `data/analysis/*/table.csv`. The `notebook` dependency group adds `altair` for charts. The
"falling back to CSV: No module named 'pyarrow'" warnings are harmless.

It reads top to bottom:

1. **Campaign table**: pick a campaign and a metric.
2. **Run / prediction set**: a searchable dropdown. Every section below it follows this choice.
3. **Metric and calibration tables**, then **disease operating points**.
4. **Charts**: reliability, confidence histogram, ROC, PR.
5. **Slices** and **groups by error count**.
6. **Grad-CAM attention** summary and the randomization check.
7. **Gallery**: pick a true→predicted cell, CAM layer, CAM target, and image count.

## 3. Reading each section

### 3.1 Campaign table

Each row is one set × level (`image` or `eye`) × metric, in percent.

| Column | Meaning |
|---|---|
| `mode` | `shared`: every checkpoint scored the same rows, e.g. five CV checkpoints on the fixed Kermany test set or a LOSO held-out source. `pooled`: checkpoints scored disjoint rows, e.g. the Duke CV outer test folds, which are concatenated. |
| `runs` | Number of checkpoints aggregated. |
| `rows`, `units` | Images, and the resampling units (patients, volumes, or eyes) behind the interval. |
| `point` | `shared`: mean of per-checkpoint point metrics. `pooled`: metric of the concatenated out-of-fold predictions. |
| `ci_low`, `ci_high` | 95% percentile cluster-bootstrap interval. |
| `run_sd` | Standard deviation across checkpoints (checkpoint-to-checkpoint variability). |

How to read it:

- **The interval and `run_sd` answer different questions.** The interval is uncertainty from the
  finite set of test *patients*. `run_sd` is training variability across checkpoints. In `shared`
  mode each bootstrap replicate resamples patients once for all checkpoints, so repeated test images
  are not counted as independent evidence.
- **Look at `units` before trusting precision.** Duke has 9 test eyes per fold and 45 pooled, so its
  intervals are wide by construction. A result of `100.00 [100.00, 100.00]` over 45 eyes means no
  eye was misclassified in this sample. It does not mean the error rate is zero.
- **When comparing two models, check whether the intervals overlap heavily.** If they do, the
  difference is not established. Use a paired comparison on the same rows before claiming one is
  better.
- **`pooled` Duke is the headline Duke CV estimate.** Each eye appears exactly once. Per-fold Duke
  numbers are too small to compare individually.

### 3.2 Metric table (selected set)

The header shows the source, image count, and group count. If the recomputed metrics differ from the
saved ones, it says so in bold.

- **`image (patient bootstrap)`** rows score every image, but resample `source:group_id` groups
  (patients; for Duke, volumes/eyes).
- **`eye`** rows (Duke only) first average B-scan probabilities within each eye, the same way as the
  training CLI's `eye_level` metrics, and then resample eyes. Use these for clinical-unit claims about
  Duke.
- **`recall_<class>`** is per-class sensitivity. If a class has an undefined interval, that class
  was absent from some resamples; check `units`.
- **Unit differences between sources:**
  - PAIMA is labelled per B-scan, so its metrics are image-level only. Do not read them as eye
    diagnoses.
  - Kermany and OCTDL have no eye IDs; their groups are patients.

### 3.3 Calibration table

| Field | Meaning | Reading |
|---|---|---|
| `ECE` | Top-label expected calibration error over 15 equal-width confidence bins, in %. | Roughly: "confidence is off by this many points on average". Below ~3 is good; above ~5 means the probabilities should not be read literally. |
| `MCE` | The worst bin's gap. | High MCE with low ECE usually means one sparse bin; check the histogram. |
| `Brier`, `NLL` | Proper scoring rules (lower is better). | Use them to compare models on the *same* set. They are not comparable across sources. |
| mean confidence (correct / wrong) | Average top-label probability. | Wrong predictions near 0.5 mean the model is unsure at the boundary, which thresholding or calibration can help. Wrong predictions near 0.9+ mean it is confidently wrong: suspect shift, a shortcut, or label noise. |

### 3.4 Disease-versus-normal operating points

The prediction is collapsed to *disease* (any non-normal class) versus *normal*, using the score
`1 - p(normal)`.

| Row | What it is |
|---|---|
| `argmax` | The decision the model actually makes. |
| `sensitivity_90`, `sensitivity_95` | The strictest threshold that still reaches that sensitivity, with its specificity. |
| `best_f1_test_tuned` | The F1-maximizing threshold, **chosen on these same labels**. |

Use `best_f1_test_tuned` only to measure threshold shift, never as a result. If it recovers much
better sensitivity/specificity than `argmax`, the model ranks cases well but its decision boundary
has moved for this source. That problem is fixable by calibration or threshold selection on held-out
data from the target source. If it barely improves on `argmax`, the source is genuinely hard to
separate.

Also check the direction of the shift. A tuned threshold far above 0.5 means the model pushes
normals towards disease (specificity suffers). One far below 0.5 means it misses disease
(sensitivity suffers). See [reports/20260927-resnet50-analysis.md](reports/20260927-resnet50-analysis.md) §2 for both cases in the current runs.

### 3.5 Charts

- **Reliability diagram.** Points below the dashed diagonal are overconfident: accuracy is lower
  than confidence. Points above it are underconfident. Only bins with data are drawn.
- **Confidence histogram (symlog counts),** split into correct and wrong. A tall "wrong" bar near
  1.0 is the clearest sign of confident failures; open the gallery for that cell.
- **ROC (one-vs-rest).** High AUROC next to a mediocre macro-F1 means ranking is fine but the
  threshold or calibration is off (see 3.4).
- **Precision-recall.** More informative than ROC for minority classes (e.g. DME), because it
  reflects prevalence. Compare the curve against the class `prevalence` in `curves.json`.

A class with no positives in the set has no curve.

### 3.6 Slices

Each row is a subset. Its type is one of:

- `all`
- `cohort` (PAIMA CSV class)
- `raw_label` (the source-native label before merging, e.g. PAIMA `DRUSEN` vs `CNV` inside `amd`)
- `in_duplicate_cluster` (membership in an exact or pHash cluster from `artifacts/audits/`)

`true_<class>` counts the true labels in the slice. `predicted_<class>` is the fraction of the slice
sent to each class.

- **Raw labels expose merged classes.** If `raw_label=DRUSEN` has much lower accuracy than
  `raw_label=CNV`, the AMD errors come from early/dry disease. That is a different problem from
  missing neovascular AMD.
- **The duplicate slice is a leakage check.** Noticeably higher accuracy on `in_duplicate_cluster=True`
  suggests near-duplicates are inflating the score.
- For a single-class slice, `macro_f1` is not meaningful; read `accuracy` (which equals that class's
  recall) and the `predicted_*` distribution.

### 3.7 Groups by error count

This shows one row per patient/volume, sorted by number of errors. The header sentence reports how
many groups have errors and the share of errors held by the worst 10% of groups.

| Share held by worst 10% | Reading |
|---|---|
| High (e.g. above 0.8) | A few patients cause almost all errors. Inspect them before changing the model: look for label problems, unusual scanners, or atypical pathology. |
| Low (~10–30%) | Errors are spread out, which points to a systematic, source-wide problem (shift, calibration). |

`targets` lists the true-label mix of a group. A patient with images of several classes is expected
for image-labelled sources such as PAIMA.

### 3.8 Grad-CAM attention

This requires `analyze gradcam`. Rows are `group` × `layer` × `target_role`:

- `group` is `error`, `control`, or `kind:true->predicted` for a single confusion cell.
- `target_role=predicted` explains the class the model chose. `true` explains the class it should
  have chosen.

| Metric | Meaning |
|---|---|
| `outside_retina_mass_mean` / `_median` | Share of CAM mass outside the estimated retina band (from the ILM to about 10 px below the RPE). Images whose mask covers under 5% or over 60% of the content are excluded; `_n` shows how many remain. |
| `padding_mass_mean` / `_median` | Share of CAM mass on letterbox padding or detected fill wedges (e.g. Kermany's white rotation corners). |

How to read it:

- **Compare, don't read absolutes.**
  - The `layer4` map is 7×7 and upsampled to 224², so even well-placed attention spills outside a
    thin retina band. Absolute values of 40–60% are therefore normal.
  - Compare `error` against `control`, one source against another, or one model against another on
    the same images.
  - `layer3` (14×14) is sharper; check that the conclusion holds there too.
- **Padding mass is the shortcut detector, but compare it with the padding area.** Padding carries
  no clinical information. However, letterboxing makes padding a large part of the canvas (about
  56–63% for Kermany, OCTDL and PAIMA, 27% for Duke). A model spreading attention evenly would put
  roughly that share there. Treat padding mass as a shortcut signal when it is:
  - above the source's padding area, or
  - well above what other models put on the same images.

  A model focused on the retina scores far *below* the area share. [reports/20260927-resnet50-analysis.md](reports/20260927-resnet50-analysis.md) §4 has the
  current Duke-baseline example.
- **Errors attending more outside the retina than controls do** suggests failures come from off-retina cues, such as the choroid or vitreous, rather than from
  misreading the retina.
- **Randomization check.** This is the Spearman correlation between CAMs from the trained model and
  from a copy with `layer4` and `fc` re-initialized.
  - Low values (roughly below 0.3) mean the maps depend on the learned weights and are worth
    interpreting.
  - Values near 1 mean the maps mostly reflect image structure (edges, bright layers). In that case,
    do not interpret them.

### 3.9 Gallery

Choose a `true → predicted` cell. Off-diagonal cells are errors, selected as the most confidently
wrong images. Diagonal cells are controls: a random sample of correct predictions. At most three
images come from any one patient.

Each card shows:

- the preprocessed image as the model saw it
- the CAM overlay in red, with the retina-mask outline in green
- the file path, group, and class probabilities

Check for:

- attention on padding, wedges, image corners, or text/scale bars
- attention on the retina but missing the lesion (wrong region inside the retina)
- images that look mislabelled, or whose label unit does not match the image (PAIMA per-B-scan labels
  inside an AMD cohort can legitimately look normal)
- whether one patient or scanner style dominates a cell

Switch `CAM target` to `true` for errors to see where evidence *for the correct class* is, and how
weak it is.

## 4. From observation to next step

| Observation | Likely cause | Next step |
|---|---|---|
| Wide interval, few `units` | Small test population (Duke) | Report the pooled CV and eye-level result; avoid per-fold claims. |
| High AUROC, low F1; tuned threshold much better than `argmax` | Threshold/prior shift | Temperature scaling or threshold selection on target-source validation data (needs saved validation predictions). |
| High ECE, confidently wrong errors | Distribution shift or shortcut | Check padding mass and the gallery; consider augmentation or preprocessing changes. |
| High padding mass | Geometry/fill shortcut | Change letterboxing (crop or randomize the padding), augment padding, then re-check the padding mass. |
| Errors concentrated in a few groups | Patient-specific issues or label noise | Review those patients' images and labels; cross-check the duplicate audit. |
| One raw label drives the errors (e.g. `DRUSEN`) | Merged-class heterogeneity | Report per-raw-label recall; consider class-specific data or loss weighting. |
| Better accuracy on duplicate-cluster slice | Leakage via near-duplicates | Revisit the quarantine rules in `docs/data-contract.md`. |

## 5. Output file reference

Per set, in `<run>/analysis/<set>/`:

| File | Contents |
|---|---|
| `summary.json` | Point metrics, `reference_check`, `bootstrap`, `eye_level` (Duke), calibration scalars, `per_class` one-vs-rest rates, `disease_operating_points`, `error_concentration`. |
| `calibration.json`, `reliability.csv` | Reliability bins and confidence histograms. |
| `curves.json` | Per-class ROC/PR points (≤200), AUROC, AP, prevalence. |
| `slices.csv`, `groups.csv` | Slice metrics; per-group error table. |
| `errors.csv`, `controls.csv` | Gallery indexes, with `row_id` keys used by the Grad-CAM files. |
| `gradcam/summary.json`, `gradcam/gradcam.csv` | Attention summaries and the randomization check; per-image, per-layer, per-target masses. |
| `gradcam/cams.npz`, `gradcam/masks.npz` | Native-resolution CAMs (`<row_id>__<layer>__<role>`) and packed masks (`<row_id>__content`, `<row_id>__retina`). |

Per campaign, in `data/analysis/<name>/`:

| File | Contents |
|---|---|
| `summary.json` | Per-set aggregation: mode, per-run metrics, bootstrap, and eye-level blocks. |
| `table.csv` | Flat table used by the notebook. |

Defaults:

- **Bootstrap:** seed `20260927`, 2000 replicates for count metrics, 500 for AUROC. A replicate
  missing a class is skipped for AUROC; `n_valid` records how many were used.
- **Implementation:** `src/oct_classify/analysis/`.
