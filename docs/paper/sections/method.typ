#import "../lib.typ": *

= Method <sec-method>

== Model and training

All models are ResNet-50 networks initialised from ImageNet weights, with a three-way output (normal, AMD, DME).
Each B-scan is scaled to fit a 224×224 canvas with its aspect ratio kept (letterboxing), and the empty bands are
filled with the median intensity of the scan. The bands grow with the width of the scan (@fig-padding): they cover
about 3% of the canvas for a 512×496 scan, 35% for 768×496 and 52% for 1024×496. Intensities are rescaled to the 1st–99th percentile of the tissue.
Some sources rotate or crop their scans and fill the exposed corners with white; we exclude this fill from the
percentiles and redraw it from nearby background (_fill-aware padding_), so that it neither compresses the
tissue contrast nor marks a source. Training augmentations are horizontal flip, rotation up to 10°, small
brightness and contrast jitter, and light Gaussian blur.

#figure(
  kind: image,
  fig("padding"),
  caption: [How a B-scan becomes the model input, for Duke's three image sizes: letterbox (the default) and
    square crop.],
) <fig-padding>

We train with AdamW (learning rate $10^(-4)$, weight decay $10^(-4)$, batch size 16, or 12 for the LOSO models
below) for up to 100 epochs of 1,000 batches, with early stopping after 10 epochs without improvement. Batches are
sampled so that every source and every class within it is equally likely. The loss is a class-weighted
cross-entropy _masked_ to the classes a source labels: for PAIMA the DME output is excluded from the softmax, so
PAIMA images never act as negative DME examples. The checkpoint with the best validation macro-F1, averaged over
the training sources weighted by their validation size, is kept; a LOSO model never selects on its held-out
source. Under this model selection, plain training on pooled data (empirical risk minimization) matches dedicated
domain-generalization methods on standard benchmarks @gulrajani2021domainbed, so we study that baseline rather
than a new method. Per-dataset models use the same recipe on one source. All runs use one seed (20260913); @sec-limits
discusses seed variation.

== Three training variants

#figure(
  kind: image,
  fig("design"),
  caption: [The three training variants and the sources each is tested on.],
) <fig-design>

We train three variants (@fig-design), all scored on each source's test split:

- *Per-dataset*: one model per source, trained and tested on that source. This is the usual single-dataset
  setting and our reference for "own source" accuracy.
- *Fused*: one model trained on all four sources together and tested on each source's test split. It measures
  whether a single model can serve every source it has seen.
- *Leave-one-source-out (LOSO)*: for each source, a model trained on the other three and tested on the held-out
  source. It measures accuracy on an unseen source.

Duke's five folds are used throughout: the per-dataset Duke model and every model that trains on Duke (fused,
and LOSO with Kermany, OCTDL or PAIMA held out) are trained five times, once per Duke fold, so Duke is always
scored on eyes the model did not train on. Kermany, OCTDL and PAIMA have fixed test splits; their fused and LOSO
scores are the mean over the five models. The LOSO model with Duke held out is a single model.

*The gap.* The fused model and the LOSO model are scored on the same test images of a source; the difference is
its _unseen-source gap_. A LOSO model can also be scored on _all_ images of its held-out source, since none were
used in training; we call this the _full-source_ score.

== Square crop: removing the image-size cue <sec-square>

To test whether models use image size as a label cue (@sec-shortcut), we add one arm that hides it. _Square crop_
trains on a random square window (height × height) of each scan and tests on the centred window. Every input is
then a square of tissue with no letterbox bands (@fig-padding, right), so the scan's original width cannot be read. We train square-crop
versions of the fused and LOSO models with otherwise identical settings, and call the default arm _letterbox_ to
contrast the two.

== Evaluation

All metrics are computed per B-scan, for every source. We report macro-F1 (the mean of the per-class F1 scores) as the main metric, because the classes are imbalanced
and differ in balance between sources, together with macro-AUROC (one-vs-rest, averaged over classes), which
measures ranking and does not depend on the decision cutoff. PAIMA is always scored on normal vs AMD, with the
softmax restricted to those two outputs. We also report a _Normal/AMD view_ for every source, which drops both
the DME images and the DME output and renormalises over normal and AMD. It separates errors that involve DME (DME
missed, or other images called DME) from the rest.

Confidence intervals are 95% patient-level cluster bootstrap intervals (2,000 resamples; 500 for AUROC):
patients, or eyes for Duke, are resampled with all their images. When two arms are compared on the same images,
one resample is shared by both, so the difference has a _paired_ interval. We call a difference beyond noise when
its paired interval excludes zero.

*Attribution and intervention.* To illustrate where held-out models find evidence for a class, we compute Grad-CAM
@selvaraju2017gradcam maps at the last convolutional stage for the predicted class, and check them against a
model whose top stage is re-initialised @adebayo2018sanity. Because attribution maps show where gradients point
rather than what causes a prediction, we also change selected pixels of the input (@sec-shortcut) and record how
predictions move.
