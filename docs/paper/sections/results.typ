#import "../lib.typ": *

= Results <sec-results>

#let srcs = ("kermany", "octdl", "paima", "duke")
#let k(kind, src) = kind + "|" + src + "|" + view3(src)

== Seen sources <sec-seen>

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

On the source it was trained on, a ResNet-50 reaches #pt(k("seen", "paima"), "per-dataset")–#pt(k("seen", "kermany"),
"per-dataset") macro-F1 (@tab-seen). Training one model on all four sources costs little: against the
per-dataset models the fused model is within noise on OCTDL (#dl(k("seen", "octdl"), "fused-fill - per-dataset")),
PAIMA (#dl(k("seen", "paima"), "fused-fill - per-dataset")) and Duke (#dl(k("seen", "duke"), "fused-fill - per-dataset")), and is slightly but measurably lower on Kermany
(#dl(k("seen", "kermany"), "fused-fill - per-dataset")). Square crop, which hides the image size (@sec-square), also costs nothing on seen sources: no source loses
beyond noise against the fused model, and PAIMA gains #dl(k("seen", "paima"), "fused-sq - fused-fill").

PAIMA is the hardest source even when seen (#pt(k("seen", "paima"), "fused-fill") fused), mostly through AMD recall
(#pt(k("seen", "paima"), "fused-fill", name: "recall_amd")%), and within AMD through its dry stage: the fused model
recognises #as-rec("test|paima|normal_amd", "dry", "fused-fill")% of PAIMA's drusen scans but
#as-rec("test|paima|normal_amd", "wet", "fused-fill")% of its wet ones, and calls the missed drusen scans normal. The
per-dataset PAIMA model shows the same (#as-rec("test|paima|normal_amd", "dry", "per-dataset")% vs
#as-rec("test|paima|normal_amd", "wet", "per-dataset")%), so this comes from PAIMA's drusen scans, not from fusing.
Kermany's drusen scans, in contrast, are recognised in at least #as-min("test|kermany|all", "dry")% of cases by
every model in this paper, including the held-out models of the next section.

== Unseen sources <sec-unseen>

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
  caption: [Seen vs unseen sources (letterbox models). _Gap_ is the paired difference between the LOSO model and
    the fused model on the same test images, with its 95% interval. _Full source_ scores the LOSO model on every
    image of the held-out source. Duke held out is one model on 45 eyes.],
) <tab-unseen>

#figure(
  kind: image,
  fig("gap"),
  caption: [Macro-F1 on each source's test split when the source is seen (fused, hollow) and unseen (LOSO, filled),
    for letterbox and square-crop models. Lines are 95% intervals of the unseen score; paired gaps are in
    @tab-unseen. Both scores use the same test images, which for Kermany are mostly 1024×496 and score higher than
    Kermany as a whole (@tab-unseen, full source).],
) <fig-gap>

Every source loses accuracy when it is unseen, beyond noise (@tab-unseen, @fig-gap). For OCTDL, PAIMA and
Duke the gap is 4–6 macro-F1 points; for Kermany it is #dl(k("gap_test", "kermany"), "loso-fill - fused-fill").
AUROC drops much less than macro-F1 for the three-class sources, so the models still rank the right class
highly for most images but the decision falls on the wrong side of the cutoff more often. PAIMA is the exception: its
AUROC changes by #dl(k("gap_test", "paima"), "loso-fill - fused-fill", name: "macro_auroc") on the test split, a
change in ranking, not only in the cutoff.

Kermany's test split is not representative of the source: #todo[942] of its 1,000 images are 1024×496, a size that
makes up about 1% of Kermany overall. On all #num(R.at("loso_full|kermany|all").n_images) images the held-out Kermany
score is #pt("loso_full|kermany|all", "loso-fill"), well below the #pt("gap_test|kermany|all", "loso-fill") on the
test split. We use the full-source score for claims about Kermany as an unseen source. The next section shows why
image size matters so much.

== The image-size shortcut <sec-shortcut>

In Duke, image size identifies the label for a third of the eyes: every 768×496 scan is DME and every 1024×496
scan is AMD (@fig-size-class). Models that train on Duke can learn "768 wide means DME" as a rule, which is
correct within Duke and wrong elsewhere. Three tests show that they do.

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

*Where the cue enters: the letterbox bands.* Width reaches the model through the letterbox. A 768×496 scan
fitted to the square canvas leaves flat bands above and below the tissue, covering about a third of the canvas, and
Duke's DME-only 768×496 images have the same bands. Changing only the band pixels, to black or to speckle drawn
from the darker half of the scan itself, removes most of the shortcut errors. Across the five held-out Kermany
models, of the 768×496 normal images a model called DME, #range0(gc-dme("768 normal → DME", "speckle"))% are
still called DME with speckle bands (#range0(gc-dme("768 normal → DME", "black"))% with black bands); for AMD images
called DME, #range0(gc-dme("768 AMD → DME", "speckle"))% remain. #{
  let rows = GC.intervention_by_fold.filter(r => r.group in ("768 normal → normal", "768 AMD → AMD") and r.at("band fill") != "trained")
  let total = rows.map(r => r.n).sum()
  let changed = rows.map(r => calc.round(r.n * (1 - r.at("called true class")))).sum()
  let to-dme = rows.map(r => calc.round(r.n * r.at("called DME"))).sum()
  [Correct normal and AMD calls barely move: of #total such cases (two band changes, five models), #to-dme become
  DME, and the predicted class changes for #changed.]
} In
two of the five models about half of the AMD errors survive, so the bands are the main but not the only route.
(#gc-group-n("768 normal → DME") images per group and model, one per patient.)

Grad-CAM illustrates the same mechanism (@fig-gradcam), with a caveat. For these errors two of the models put
#range0(gc-band("768 normal → DME").slice(0, 2))% of the map on the bands, against
#range0(gc-band("768 normal → normal"))% for correct calls, but two others put only
#range0(gc-band("768 normal → DME").slice(3, 5))% there although the band change removes most of their errors (the
fifth puts #pct0(gc-band("768 normal → DME").at(2))%). The
maps reflect what the model learned, not image structure: after the top of the network is re-initialised they no
longer correlate with the trained maps (median Spearman
#calc.round(GC.randomization_fold1_768_normal_to_dme.spearman_median, digits: 2)). We therefore use the band
intervention as the evidence and Grad-CAM as an illustration.

#figure(
  image("../figures/gradcam.png", width: 100%),
  caption: [Grad-CAM (last convolutional stage, predicted class) on held-out Kermany 768×496 normal images, from one
    letterbox and one square-crop LOSO-Kermany model. Each column is one image as the model sees it: letterbox
    with its bands (top), square crop without (bottom). In the first four columns the letterbox model calls
    the image DME (bold) and its map sits on the top band; the square-crop model calls it normal and looks at the
    retina. The last two columns are images both models call normal. How much of the map falls on the bands varies
    between models (text).],
) <fig-gradcam>

*Training without the cue.* Square-crop models never see the image width. With Kermany held out, they score
#pt("loso_full|kermany|all", "loso-sq") on the full source against #pt("loso_full|kermany|all", "loso-fill")
(#dl("loso_full|kermany|all", "loso-sq - loso-fill")). On the test split, each arm measured against the fused
model of the same arm, the gap narrows from #dl("gap_test|kermany|all", "loso-fill - fused-fill") to
#dl("gap_test|kermany|all", "loso-sq - fused-sq"), about 40%. The gain is largest exactly where the cue was
(@fig-kermany-size): for 768×496 normal images, recall rises from
#pt("size_loso:768x496|kermany|all", "fill", name: "recall_normal") to
#pt("size_loso:768x496|kermany|all", "sq", name: "recall_normal"), and the share called DME falls from
#share("size_loso:768x496|kermany|all", "fill", "normal", "dme")% to
#share("size_loso:768x496|kermany|all", "sq", "normal", "dme")%.

#figure(
  kind: image,
  fig("kermany-size"),
  caption: [Held-out Kermany (all images) by image size and true class: recall of letterbox and square-crop
    LOSO models, with 95% intervals. Square crop helps most for normal and AMD images at 768×496, the width that is
    DME only in Duke, and costs DME recall there.],
) <fig-kermany-size>

*But removing the cue does not help everywhere.* On the other held-out sources square crop is no better and
sometimes worse (@tab-square): OCTDL changes by #dl("loso_full|octdl|all", "loso-sq - loso-fill") on the full source,
almost all through normal images called DME, and Duke by #dl("loso_full|duke|all", "loso-sq - loso-fill"),
through DME recall. PAIMA loses ranking quality (AUROC #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill",
name: "macro_auroc")), and this loss varies strongly with the seed (@sec-limits). In the Normal/AMD view, which
removes the DME class, the OCTDL and Duke losses disappear (#dl("loso_full|octdl|normal_amd", "loso-sq - loso-fill")
and #dl("loso_full|duke|normal_amd", "loso-sq - loso-fill")), so square crop's cost on those sources involves DME:
too many DME calls on OCTDL, too few on Duke.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header[Held out (full source)][Letterbox][Square crop][Paired difference],
    ..srcs.map(s => (
      [#source-name.at(s)],
      pt(k("loso_full", s), "loso-fill"),
      pt(k("loso_full", s), "loso-sq"),
      dl(k("loso_full", s), "loso-sq - loso-fill"),
    )).flatten(),
  ),
  caption: [LOSO macro-F1 (%) on the full held-out source, with and without the image-size cue.],
) <tab-square>

We therefore keep letterbox models as the main result and treat square crop as an ablation. It confirms that
the size shortcut is real and explains a large part of Kermany's gap, but it is not a general fix. We fixed this
reading before the OCTDL and Duke square-crop results were in: square crop would become the main setup only if
neither source lost beyond noise.

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
  caption: [Where the unseen-source gap comes from (letterbox, test split, LOSO minus fused, percentage points).
    Removing the DME class removes most of the gap for OCTDL and Duke.],
) <tab-remaining>

Most of what remains is DME (@tab-remaining). On the test split, held-out DME recall falls by 13–16 points for
Kermany, OCTDL and Duke. In the Normal/AMD view, the OCTDL and Duke gaps are within noise of zero
(#dl("gap_test|octdl|normal_amd", "loso-fill - fused-fill") and #dl("gap_test|duke|normal_amd", "loso-fill - fused-fill")). Kermany keeps a large Normal/AMD gap, through normal images called AMD (normal recall change
#dl("gap_test|kermany|normal_amd", "loso-fill - fused-fill", name: "recall_normal")), consistent with the size cue
above: most of its test split is 1024×496, Duke's AMD-only size. PAIMA, which has no DME, keeps a gap of
#dl("gap_test|paima|normal_amd", "loso-fill - fused-fill"), through AMD recall, and again mostly dry AMD: held-out
recall changes by #as-dl("test|paima|normal_amd", "dry", "loso-fill - fused-fill") for drusen scans and by
#as-dl("test|paima|normal_amd", "wet", "loso-fill - fused-fill") for wet ones.

A likely reason is that each LOSO model sees DME from fewer sources: PAIMA has none, so with Kermany held out the
model learns DME from OCTDL and Duke alone. We tested this for Kermany with a fused model, trained
on all four sources, from which we removed every Kermany DME image (in training and validation; the other classes
and sources are unchanged; five Duke folds, same recipe as the fused model). Kermany is then a seen source
without its DME, so the comparison with the fused model isolates the missing DME, and the comparison with the LOSO model isolates the source.

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header[Source][Without Kermany DME][Fused][Difference][DME recall change],
    ..(("kermany", "Kermany"), ("octdl", "OCTDL"), ("duke", "Duke")).map(((k, n)) => (
      [#n],
      zd-pt(k + "|all", "zero-dme"),
      zd-pt(k + "|all", "fused-fill"),
      zd-dl(k + "|all", "zero-dme - fused-fill"),
      zd-dl(k + "|all", "zero-dme - fused-fill", name: "recall_dme"),
    )).flatten(),
    [PAIMA], zd-pt("paima|normal_amd", "zero-dme"), zd-pt("paima|normal_amd", "fused-fill"),
    zd-dl("paima|normal_amd", "zero-dme - fused-fill"), [--],
  ),
  caption: [Fused model with and without Kermany's DME images (macro-F1 %, test split, percentage-point differences
    with patient-level 95% intervals; PAIMA on Normal/AMD).],
) <tab-zero-dme>

Removing Kermany's DME changes Kermany by #zd-dl("kermany|all", "zero-dme - fused-fill"), all of it through DME
recall (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_dme")); its normal and AMD recall stay within
noise (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_normal"),
#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_amd")). On the same test images this is about
#zd-share()% the size of the held-out gap (LOSO #zd-pt("kermany|all", "loso-fill"), without Kermany DME
#zd-pt("kermany|all", "zero-dme"), fused #zd-pt("kermany|all", "fused-fill")), but the two models fail
differently, so the gap does not split into these shares. Missing in-device DME alone already costs more DME recall
than the LOSO model loses (DME recall, LOSO minus no-DME #zd-dl("kermany|all", "loso-fill - zero-dme", name: "recall_dme")), so
it can account for the LOSO model's DME errors. The LOSO model's other loss is normal recall
(#zd-dl("kermany|all", "loso-fill - zero-dme", name: "recall_normal") against the model without Kermany DME), which
comes from the source being unseen and matches the size cue. Missing DME is therefore a real but partial cause. The
effect is not confined to the source that lost its DME: Duke's DME recall also changes
(#zd-dl("duke|all", "zero-dme - fused-fill", name: "recall_dme")), which suggests that Kermany's DME helps the model
recognise Duke's; this change rests on 45 eyes, so we take it as supporting evidence. OCTDL and PAIMA are within
noise.

*Causes we ruled out.* Two simpler explanations do not account for the gap. First, the decision cutoff: we fit a
per-class shift of the output log-probabilities on the held-out source's own labels, fitting on one half of its
patients and scoring the other. This changes held-out Kermany by #lb-dl("kermany|all", "bias-crossfit") on the
full source (#lb-pt("kermany|all", "none") to #lb-pt("kermany|all", "bias-crossfit")), against
#dl("loso_full|kermany|all", "loso-sq - loso-fill") for removing the size cue on the same images, and leaves OCTDL
(#lb-dl("octdl|all", "bias-crossfit")) and Duke (#lb-dl("duke|all", "bias-crossfit")) within noise. Even the best
shift found by searching on all of Kermany's labels reaches only #fmt(LB.at("kermany|all").grid.best). Since this
uses labels a deployed model would not have, it is an upper bound; the cutoff explains a few points of Kermany's
gap, not most of it.
Second, BatchNorm statistics. AdaBN @li2016adabn, a common first remedy for such a shift, re-estimates the
BatchNorm statistics on the unseen source's own unlabelled images; we again estimate on one half of the patients
and predict the other. It changes held-out Kermany by only #ab-dl("kermany|all") (#ab-pt("kermany|all", "none") to
#ab-pt("kermany|all", "adabn-all")), leaves Duke within noise (#ab-dl("duke|all")), and changes OCTDL by
#ab-dl("octdl|all"), because it shifts OCTDL's images toward DME: DME recall rises from
#ab-pt("octdl|all", "none", name: "recall_dme") to #ab-pt("octdl|all", "adabn-all", name: "recall_dme") while AMD
recall falls from #ab-pt("octdl|all", "none", name: "recall_amd") to #ab-pt("octdl|all", "adabn-all", name: "recall_amd").
