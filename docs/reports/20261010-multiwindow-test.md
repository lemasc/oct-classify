# Multi-Window Square Crops at Test Time

A no-training preview of experiment 3a's square-crop arm ([cross-device plan](../cross-device-plan.md)). The reference
fill-aware LOSO models for Kermany and PAIMA are rescored with each image cut into overlapping square windows, and
the window predictions are pooled. The models were trained on letterboxed images, so this measures what windows do
to models that have never seen them, not what a model trained on crops would do.

## Summary

- **PAIMA's ranking gap closes at test time.** Averaging three windows raises held-out PAIMA AUROC from 92.8 to
  **95.6**, the seen PAIMA value. Macro-F1 does not move (86.4 → 85.7) because the cutoff shifts: fewer normal images
  are called AMD (5.0% → 1.2%) and more AMD images are called normal (22.3% → 27.1%).
- **Kermany gains overall (+4.4 macro-F1) but only where the size cue hurt.** 768×496 gains 14.7 points (normal → DME
  36.6% → 7.4%). 1024×496 loses 8.0 and 1536×496 loses 11.3; on the test split, which is mostly 1024×496, macro-F1
  falls from 82.6 to 74.0.
- **Pooling hardly matters.** Mean, center window and per-class max are within about 1.5 macro-F1 points everywhere.
  If wide scans lost accuracy because lesions fell outside the center window, max pooling ("diseased if any window
  is") would recover it, and it does not. At test time the loss comes from showing these scans at an unfamiliar
  scale (a window draws the tissue 2-3× larger than the letterbox did), not from missing lesions.
- **For experiment 3a:** a model trained on square crops would see this scale for every source, so the 1024/1536 losses
  here say little about the trained arm. Whether a training crop that misses the lesion hurts (label noise) is still
  untested. Use mean pooling for the trained arm; the choice is not critical.

## Setup

- **Models:** `loso-{kermany,paima}-local-20261008-174938-fold{1..5}/checkpoint-best.pt`, each with its own
  preprocessing (fill-aware) and normalization, fp32. PAIMA uses the softmax over normal and AMD only, as in its
  evaluation.
- **Windows:** side = image height (496 px; 512 for Kermany 512×512), at half-window steps across the width:
  1 window for 512-wide images, 3 for 768, 4 for 1024, 6 for 1536. A square window gets no letterbox bands.
- **Pooling:** mean of window probabilities; the window nearest the image center; per-class max over windows,
  renormalized.
- **Baseline:** the stored letterbox predictions for the same images.
- **Statistics:** paired bootstrap over patients (Kermany) or volumes (PAIMA), 2,000 draws (500 for AUROC), seed
  20261010. Metrics are computed per model and averaged over the 5 models.
- Scripts: `.scratchpad/e4e2145_multiwindow/{rescore.py,compare.py}`. Summary:
  `data/analysis/multiwindow-test/multiwindow.json`. Pooled predictions: `artifacts/analysis/multiwindow-test/`.

## 1. PAIMA

| Pooling | AUROC | Macro-F1 | Normal recall | AMD recall |
|---|---|---|---:|---:|
| Letterbox (stored) | 92.8 | 86.4 | 95.0 | 77.7 |
| Center window | 95.0 (+2.1 [+1.7, +2.6]) | 86.2 (−0.2 [−0.8, +0.5]) | | |
| **Mean of 3 windows** | **95.6** (+2.8 [+2.3, +3.3]) | 85.7 (−0.7 [−1.3, +0.0]) | 98.8 | 72.9 |
| Max | 95.5 (+2.7 [+2.2, +3.2]) | 85.1 (−1.3 [−2.0, −0.6]) | | |

Seen PAIMA (fused training) scores 95.6 AUROC, so at test time the windows recover the whole ranking drop. The
operating point moves toward normal, which leaves macro-F1 flat. A recalibrated cutoff would be needed to turn the
ranking gain into accuracy; the logit-bias experiment on the other branch showed such a shift is cheap to fit but
needs target labels.

## 2. Kermany

Macro-F1 with paired 95% CI; recall for mean pooling.

| Images | Letterbox | Mean of windows | Change | Recall N / A / D, letterbox → windows |
|---|---:|---:|---|---|
| All (101,217) | 73.8 | 78.2 | +4.4 [+4.1, +4.8] | 74.9 / 88.6 / 71.6 → 82.8 / 90.0 / 69.5 |
| Test split (1,000) | 82.6 | 74.0 | −8.6 [−10.5, −6.8] | 63.0 / 98.4 / 83.4 → 62.2 / 93.2 / 65.1 |
| 512×512 | 80.9 | 81.0 | 0.0 | DME 69.1 → 69.1 |
| 512×496 | 71.2 | 71.5 | +0.3 [+0.1, +0.5] | 80.5 / 91.4 / 77.8 → 81.5 / 91.4 / 75.6 |
| 768×496 | 58.7 | 73.5 | **+14.7** [+13.6, +15.8] | 58.4 / 80.6 / 89.0 → 87.2 / 90.4 / 68.8 |
| 1024×496 | 84.1 | 76.1 | **−8.0** [−10.1, −6.0] | 68.5 / 98.7 / 84.4 → 68.9 / 93.7 / 65.1 |
| 1536×496 | 77.6 | 66.3 | **−11.3** [−13.4, −9.2] | 87.7 / 95.1 / 55.9 → 79.1 / 82.3 / 57.9 |

- **512-wide images** get one window that is almost the letterboxed image, so nothing changes.
- **768×496:** the DME pull from the size cue disappears (normal → DME 36.6% → 7.4%, AMD → DME 19.1% → 7.3%). DME recall
  falls from 89.0 to 68.8, as in the single crop of experiment 1b: part of the old DME recall came from the cue.
- **1024×496:** the AMD pull falls (normal → AMD 22.5% → 13.5%), but DME → AMD doubles (14.8% → 30.2%) and normal → DME
  rises (9.1% → 17.7%).
- **1536×496:** normal → DME rises from 3.4% to 14.0% and AMD recall falls from 95.1 to 82.3.

## 3. Pooling Choice

| Images | Center | Mean | Max |
|---|---:|---:|---:|
| Kermany all | 78.0 | 78.2 | 77.9 |
| Kermany 768×496 | 72.9 | 73.5 | 73.2 |
| Kermany 1024×496 | 74.6 | 76.1 | 75.5 |
| Kermany 1536×496 | 66.9 | 66.3 | 64.9 |
| PAIMA AUROC | 95.0 | 95.6 | 95.5 |

Mean is best or close to best everywhere. Max does not beat center on the widest scans, where a lesion outside the
center window is most likely, so missing lesions are not what costs accuracy here.

## 4. What This Changes

- **PAIMA ranking drop:** at test time, geometry accounts for all of it (exp. 1c showed most; windows show all). Whether
  the cause is the shape cue or tissue scale is still open: windows change both.
- **3a-crop stays in.** Its test-time preview helps where the cue hurts and costs accuracy on wide Kermany scans, but
  a trained model would see windows from every source. Train it with mean pooling at test.
- **The lesion concern is narrowed, not settled.** At test time, averaging over windows covers all the tissue and
  missing lesions are not the main loss. In training, each crop carries the image's label whether or not it shows
  the lesion. Only the trained arm measures that.
- **Test-split comparison for 3a:** the Kermany test split is mostly 1024×496, the size where windows hurt untrained
  models most. Report 3a on the full source and by size as well, not on the test split alone.

## Caveats

- Test-time only: the models never saw square windows in training. The results bound what windows do to
  letterbox-trained models, not what the trained square-crop arm would do.
- Windows change both the image's shape (no bands) and the tissue's on-screen size. Kermany's wider scans may also
  have a different lateral pixel spacing, which this analysis cannot check.
- One training seed per fold; CIs cover the sampling of held-out patients or volumes only.
