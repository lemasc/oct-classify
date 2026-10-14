#import "../lib.typ": *

= Results <sec-results>

#let srcs = ("kermany", "octdl", "paima", "duke")
#let k(kind, src) = kind + "|" + src + "|" + view3(src)

== Seen devices <sec-seen>

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header[Source][Per-dataset][Fused][Fused, square crop],
    ..srcs.map(s => (
      [#source-name.at(s)],
      ci(k("seen", s), "per-dataset"),
      ci(k("seen", s), "fused-fill"),
      ci(k("seen", s), "fused-sq"),
    )).flatten(),
  ),
  caption: [Macro-F1 (%) on each source's test split, with 95% patient-level bootstrap intervals. PAIMA is scored
    on normal vs AMD. Scored per B-scan. Duke is pooled over its five folds (all 3,231 B-scans of its 45 eyes, each
    scored by the model that did not train on it).],
) <tab-seen>

On the device it was trained on, a ResNet-50 reaches #pt(k("seen", "paima"), "per-dataset")–#pt(k("seen", "kermany"),
"per-dataset") macro-F1 (@tab-seen). Training one model on all four sources costs little: against the
per-dataset models the fused model is within noise on OCTDL (#dl(k("seen", "octdl"), "fused-fill - per-dataset")),
PAIMA (#dl(k("seen", "paima"), "fused-fill - per-dataset")) and Duke (#dl(k("seen", "duke"), "fused-fill - per-dataset")), and is slightly but measurably lower on Kermany
(#dl(k("seen", "kermany"), "fused-fill - per-dataset")). Square crop, which hides the image size (@sec-method), also costs nothing on seen devices: no source loses
beyond noise against the fused model, and PAIMA gains #dl(k("seen", "paima"), "fused-sq - fused-fill").

PAIMA is the hardest source even when seen (#pt(k("seen", "paima"), "fused-fill") fused), mostly through AMD recall
(#pt(k("seen", "paima"), "fused-fill", name: "recall_amd")%).

== Unseen devices <sec-unseen>

#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    table.header(
      table.cell(rowspan: 2)[Held out],
      table.cell(rowspan: 2)[Seen (fused)],
      table.cell(colspan: 2)[Unseen (LOSO), test split],
      table.cell(colspan: 2)[Unseen, full source],
      [Macro-F1], [Gap], [Macro-F1], [AUROC],
    ),
    ..srcs.map(s => (
      [#source-name.at(s)],
      pt(k("gap_test", s), "fused-fill"),
      pt(k("gap_test", s), "loso-fill"),
      dl(k("gap_test", s), "loso-fill - fused-fill"),
      ci(k("loso_full", s), "loso-fill"),
      pt(k("loso_full", s), "loso-fill", name: "macro_auroc"),
    )).flatten(),
  ),
  caption: [Seen vs unseen devices (fill-aware models). _Gap_ is the paired difference between the LOSO model and
    the fused model on the same test images, with its 95% interval. _Full source_ scores the LOSO model on every
    image of the held-out source. Duke held out is one model on 45 eyes.],
) <tab-unseen>

#figure(
  kind: image,
  fig("gap"),
  caption: [Macro-F1 on each source's test split when the device is seen (fused, hollow) and unseen (LOSO, filled),
    for fill-aware and square-crop models. Lines are 95% intervals; paired gaps are in @tab-unseen.],
) <fig-gap>

Every source loses accuracy when its device is unseen, beyond noise (@tab-unseen, @fig-gap). For OCTDL, PAIMA and
Duke the gap is 4–6 macro-F1 points; for Kermany it is #dl(k("gap_test", "kermany"), "loso-fill - fused-fill").
AUROC drops much less than macro-F1 for the three-class sources, so the models still rank the right class
highly for most images but the decision falls on the wrong side of the cutoff more often. PAIMA is the exception: its
AUROC falls by #dl(k("gap_test", "paima"), "loso-fill - fused-fill", name: "macro_auroc") on the test split, a
change in ranking, not only in the cutoff.

Kermany's test split is not representative of the source: #todo[942] of its 1,000 images are 1024×496, a size that
makes up about 1% of Kermany overall. On all #num(R.at("loso_full|kermany|all").n_images) images the held-out Kermany
score is #pt("loso_full|kermany|all", "loso-fill"), well below the #pt("gap_test|kermany|all", "loso-fill") on the
test split. We use the full-source score for claims about Kermany as an unseen device. The next section shows why
image size matters so much.

== The image-size shortcut <sec-shortcut>

In Duke, image size identifies the label for a third of the images: every 768×496 scan is DME and every 1024×496
scan is AMD (@fig-size-class). Models that train on Duke can learn "768 wide means DME" as a rule, which is
correct within Duke and wrong elsewhere. Two tests show that they do.

*Changing only the size of test images.* We take the held-out Kermany models and rescore Kermany images after
changing only their width, with no retraining (@fig-size-test). Padding normal 512×496 images to 768 wide (Duke's
DME-only width) moves the share called DME from 12.4% to 59.8%. Cropping 768×496 normal images to 512 wide does
the reverse, from 36.6% to 8.3%. The effect is not limited to Kermany: padding Duke's own 512×496 images by
16 rows to 512×512, a size that is DME-only in Kermany, sends 99% of Duke's normal images to DME in the model that
held Duke out. #todo[move these numbers to a tracked file and load them; now from `data/analysis/kermany-size-test`
and `data/analysis/heldout-size-test`]

#figure(
  kind: image,
  placeholder(4cm)[F6 or table: test-time
    size changes and the share of normal images predicted DME / AMD],
  caption: [Changing only the width of test images moves predictions toward the label Duke associates with that
    width. Held-out Kermany models, no retraining.],
) <fig-size-test>

*Training without the cue.* Square-crop models never see the image width. With Kermany held out, they score
#pt("loso_full|kermany|all", "loso-sq") on the full source against #pt("loso_full|kermany|all", "loso-fill")
(#dl("loso_full|kermany|all", "loso-sq - loso-fill")), and close about 40% of the test-split gap
(#dl("gap_test|kermany|all", "loso-fill - fused-fill") becomes #dl("gap_test|kermany|all", "loso-sq - fused-sq")).
The gain is largest exactly where the cue was (@fig-kermany-size): for 768×496 normal images, the share called DME
falls from #share("size_loso:768x496|kermany|all", "fill", "normal", "dme")% to
#share("size_loso:768x496|kermany|all", "sq", "normal", "dme")%.

#figure(
  kind: image,
  fig("kermany-size"),
  caption: [Held-out Kermany (all images) by image size and true class: recall of fill-aware and square-crop
    LOSO models, with 95% intervals. Square crop helps most for normal and AMD images at 768×496, the width that is
    DME only in Duke, and costs DME recall there.],
) <fig-kermany-size>

*But removing the cue does not help everywhere.* On the other held-out sources square crop is no better and
sometimes worse (@tab-square): OCTDL changes by #dl("loso_full|octdl|all", "loso-sq - loso-fill") on the full source,
almost all through normal images called DME, and Duke by #dl("loso_full|duke|all", "loso-sq - loso-fill"),
through DME recall. PAIMA loses ranking quality (AUROC #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill",
name: "macro_auroc")), and this loss varies strongly with the seed (@sec-limits). In the Normal/AMD view the OCTDL
and Duke losses disappear (#dl("loso_full|octdl|normal_amd", "loso-sq - loso-fill") and
#dl("loso_full|duke|normal_amd", "loso-sq - loso-fill")), so square crop's cost on those sources is in DME.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header[Held out (full source)][Fill-aware][Square crop][Paired difference],
    ..srcs.map(s => (
      [#source-name.at(s)],
      pt(k("loso_full", s), "loso-fill"),
      pt(k("loso_full", s), "loso-sq"),
      dl(k("loso_full", s), "loso-sq - loso-fill"),
    )).flatten(),
  ),
  caption: [LOSO macro-F1 (%) on the full held-out source, with and without the image-size cue.],
) <tab-square>

We therefore keep fill-aware models as the main result and treat square crop as an ablation. It confirms that
the size shortcut is real and explains a large part of Kermany's gap, but it is not a general fix. #todo[state that
this reading rule was fixed before the OCTDL/Duke square-crop results; true per the plan — keep?]

== What remains of the gap <sec-remaining>

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header[Held out][3-class gap][DME recall change][Normal/AMD gap],
    ..("kermany", "octdl", "duke").map(s => (
      [#source-name.at(s)],
      dl("gap_test|" + s + "|all", "loso-fill - fused-fill"),
      dl("gap_test|" + s + "|all", "loso-fill - fused-fill", name: "recall_dme"),
      dl("gap_test|" + s + "|normal_amd", "loso-fill - fused-fill"),
    )).flatten(),
    [PAIMA], [--], [--], dl("gap_test|paima|normal_amd", "loso-fill - fused-fill"),
  ),
  caption: [Where the unseen-device gap comes from (fill-aware, test split, LOSO minus fused, percentage points).
    Dropping DME images removes most of the gap for OCTDL and Duke.],
) <tab-remaining>

Most of what remains is DME (@tab-remaining). On the test split, held-out DME recall falls by 13–16 points for
Kermany, OCTDL and Duke. With DME images removed, the OCTDL and Duke gaps are within noise of zero
(#dl("gap_test|octdl|normal_amd", "loso-fill - fused-fill") and #dl("gap_test|duke|normal_amd", "loso-fill - fused-fill")). Kermany keeps a large Normal/AMD gap, through normal images called AMD (normal recall
#dl("gap_test|kermany|normal_amd", "loso-fill - fused-fill", name: "recall_normal")), consistent with the size cue
above: most of its test split is 1024×496, Duke's AMD-only size. PAIMA, which has no DME, keeps a #dl("gap_test|paima|normal_amd", "loso-fill - fused-fill") gap,
through AMD recall.

DME is hard to transfer because each LOSO model sees DME from fewer devices: PAIMA has none, so with Kermany held
out the model learns DME from OCTDL and Duke alone (122 patients). To separate "the device is unseen" from "the
device's DME is unseen", we train the fused model with every Kermany DME image removed from training, keeping
Kermany's normal and AMD images. #pending[zero-DME rerun on fill-aware models (item F); legacy median-pad result:
Kermany macro-F1 −5.6 [−6.7, −4.5]].

*Causes we ruled out.* Two simpler explanations do not account for the gap. First, the decision cutoff: we fit a
per-class shift of the output log-probabilities on the held-out source's own labels, fitting on one half of its
patients and scoring the other. This raises held-out Kermany from #lb-pt("kermany|all", "none") to
#lb-pt("kermany|all", "bias-crossfit") (#lb-dl("kermany|all", "bias-crossfit")), far short of the
#pt("seen|kermany|all", "fused-fill") it scores when seen, and changes OCTDL (#lb-dl("octdl|all", "bias-crossfit"))
and Duke (#lb-dl("duke|all", "bias-crossfit")) within noise. Even the best shift found by searching on all of
Kermany's labels reaches only #fmt(LB.at("kermany|all").grid.best). Since this uses labels a deployed model would
not have, it is an upper bound; the cutoff explains a few points of Kermany's gap, not most of it.
#pending[AdaBN (item H) on fill-aware models; legacy numbers below.] Re-estimating the BatchNorm
statistics on the unseen device (AdaBN), the standard first remedy for a change in image statistics, helps
Kermany by about one point and costs OCTDL about twenty. Finally, the gap is not a matter of training-set size: a
fused model whose Kermany DME was cut to the 122 patients a LOSO model sees still scores 97.1 on Kermany (earlier
preprocessing variant, not rerun).
