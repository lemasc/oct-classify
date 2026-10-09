# fp16 validation NaN in OCTDL hold-out fold 3

**Status: investigating.** Evaluation now runs in fp32 as a workaround; root cause not yet traced.

## What happened

The LOSO campaign `local-20261008-174938` (fill-aware padding) stopped during OCTDL hold-out fold 3, epoch 0:

```
ValueError: Out of range float values are not JSON compliant: nan
```

Duke, kermany and OCTDL folds 1–2 had finished. OCTDL folds 3–5 and paima were not run.

## Findings

- The NaN is `validation_by_source/paima/loss`. Training loss (0.256) and the other sources were finite.
- Replaying epoch 0 of fold 3 (same config, split and seed) reproduces it.
- A per-image scan of paima validation with the epoch-0 weights found 29 of 2493 images with NaN logits under fp16 autocast. All 29 are from one volume, `CNV/159` (images 009–030).
- Inputs are finite and in a normal range. The same images give finite fp32 logits (about ±50), so fp16 overflows somewhere inside the network.
- Folds 1–2, and every earlier campaign (fp16 throughout), were not affected.

## Changes

- Autocast is enabled only while training (`run_epoch`). Validation and `train evaluate` run in fp32. Training numerics are unchanged, so earlier runs stay comparable. Validation metrics of the earlier campaigns were computed in fp16 and may differ slightly.
- A non-finite batch loss now raises `NonFiniteLossError` before backward and writes `checkpoint-failed.pt` and `failure.json` (phase, source, batch, per-image logits) into the run directory.

## Open

- Which layer overflows (forward hook on the epoch-0 checkpoint) and whether it persists in later epochs.
- If NaNs recur in training or in fp32 evaluation, reconsider full fp32 or bf16 training.
- Epoch time rose about 22–56% in this campaign versus the median-pad campaigns. The 6 workers and fill-aware padding changed together, so the cause is not separated.
