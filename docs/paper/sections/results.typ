#import "../lib.typ": *

= Results <sec-results>


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
    on normal vs AMD. Scored per B-scan. Duke is pooled over its five folds (all #num(DS.duke.n_images) B-scans of its #DS.duke.n_groups eyes, each
    scored by the model that did not train on it).],
) <tab-seen>

On the source it was trained on, a ResNet-50 reaches #pt(k("seen", "paima"), "per-dataset")–#pt(k("seen", "kermany"),
"per-dataset") macro-F1 (@tab-seen). Training one model on all four sources costs little: against the
per-dataset models the fused model is within noise on OCTDL (#dl(k("seen", "octdl"), "fused-fill - per-dataset")),
PAIMA (#dl(k("seen", "paima"), "fused-fill - per-dataset")) and Duke (#dl(k("seen", "duke"), "fused-fill - per-dataset")), and is slightly but measurably lower on Kermany
(#dl(k("seen", "kermany"), "fused-fill - per-dataset")). Square crop, which hides the image size (@sec-square), also costs nothing on seen sources: no source loses
beyond noise against the fused model, and PAIMA gains #dl(k("seen", "paima"), "fused-sq - fused-fill").

#let rc(m, c) = if ("recall_" + c) in m { fmt(m.at("recall_" + c).point) } else [--]
#let stage-rc(src, arm, st, m) = {
  let key = "test|" + src + "|" + view3(src)
  if src == "duke" { if st == "dry" { rc(m, "amd") } else [--] } // Duke's AMD is all dry
  else if arm == none [--]
  else { fmt(AS.at(key).stages.at(st).arms.at(arm).recall) }
}
#let recall-row(src, label, m, arm) = (
  [#label], rc(m, "normal"), rc(m, "amd"), stage-rc(src, arm, "dry", m), stage-rc(src, arm, "wet", m), rc(m, "dme"),
)

// Per-class recall table; `rows-of(src)` gives that source's rows (from `recall-row`).
#let recall-table(rows-of) = {
  set text(size: 9.5pt)
  table(
    columns: 7,
    align: (left, left, right, right, right, right, right),
    stroke: none,
    table.header(
      table.cell(rowspan: 2)[Source], table.cell(rowspan: 2)[Model], table.cell(rowspan: 2)[Normal],
      table.cell(rowspan: 2)[AMD], table.cell(colspan: 2, align: center)[AMD stage], table.cell(rowspan: 2)[DME],
      [dry], [wet],
    ),
    table.hline(stroke: 0.5pt),
    ..srcs.map(s => {
      let rows = rows-of(s)
      (table.cell(rowspan: rows.len())[#source-name.at(s)], ..rows.flatten(), table.hline(stroke: 0.3pt + gray))
    }).flatten(),
  )
}
#let seen-row(s, label, arm) = recall-row(s, label, R.at("seen|" + s + "|" + view3(s)).arms.at(arm).metrics, arm)

#figure(
  recall-table(s => (seen-row(s, "Per-dataset", "per-dataset"), seen-row(s, "Fused", "fused-fill"))),
  caption: [Recall (%) per class on each source's test split, seen letterbox models, mean over models. _Dry_ and
    _wet_ are recall within AMD; Duke's AMD is all dry, and OCTDL's intermediate stage is not shown. PAIMA has no
    DME.],
) <tab-recall>

Per-class recall (@tab-recall) shows where the errors are. PAIMA is the hardest source even when seen
(#pt(k("seen", "paima"), "fused-fill") fused), through its AMD class (recall
#pt(k("seen", "paima"), "fused-fill", name: "recall_amd")), and within AMD through its dry stage: the fused model
recognises #as-rec("test|paima|normal_amd", "dry", "fused-fill")% of PAIMA's drusen scans but
#as-rec("test|paima|normal_amd", "wet", "fused-fill")% of its wet ones, and calls the missed drusen scans normal.
The per-dataset PAIMA model shows the same (#as-rec("test|paima|normal_amd", "dry", "per-dataset")% vs
#as-rec("test|paima|normal_amd", "wet", "per-dataset")%), so this comes from PAIMA's drusen scans, not from fusing.
OCTDL's drusen scans are also recognised less often than its wet ones
(#as-rec("test|octdl|all", "dry", "fused-fill")% vs #as-rec("test|octdl|all", "wet", "fused-fill")% fused), but its
test split has only #AS.at("test|octdl|all").stages.dry.n_images of them, from
#AS.at("test|octdl|all").stages.dry.n_groups patients. Kermany's drusen scans are recognised in at least
#fmt(calc.min(..("per-dataset", "fused-fill").map(a => AS.at("test|kermany|all").stages.dry.arms.at(a).recall)))%
of cases by both models.

== Unseen sources <sec-unseen>

#figure(
  {
    set text(size: 9.5pt)
    table(
      columns: 7,
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held out],
        table.cell(colspan: 3)[Macro-F1, test split],
        table.cell(colspan: 2)[AUROC, test split],
        table.cell(rowspan: 2)[Macro-F1, \ full source],
        [Seen], [Unseen], [Gap], [Seen], [Unseen],
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        pt(k("gap_test", s), "fused-fill"),
        pt(k("gap_test", s), "loso-fill"),
        dl(k("gap_test", s), "loso-fill - fused-fill"),
        pt(k("gap_test", s), "fused-fill", name: "macro_auroc"),
        pt(k("gap_test", s), "loso-fill", name: "macro_auroc"),
        ci(k("loso_full", s), "loso-fill"),
      )).flatten(),
    )
  },
  caption: [Seen (fused) vs unseen (LOSO) sources, letterbox models. _Gap_ is the paired difference between the
    LOSO model and the fused model on the same test images, with its 95% interval. _Full source_ scores the LOSO
    model on every image of the held-out source. Duke held out is one model on 45 eyes.],
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
Duke the gap is #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) macro-F1 points; for Kermany it is #dl(k("gap_test", "kermany"), "loso-fill - fused-fill").
For the three-class sources the AUROC gap is much smaller than the macro-F1 gap
(#dl(k("gap_test", "kermany"), "loso-fill - fused-fill", name: "macro_auroc") for Kermany,
#dl(k("gap_test", "octdl"), "loso-fill - fused-fill", name: "macro_auroc") for OCTDL and
#dl(k("gap_test", "duke"), "loso-fill - fused-fill", name: "macro_auroc") for Duke), so the models still rank the
right class highly for most images but the decision falls on the wrong side of the cutoff more often. PAIMA is the
exception: its AUROC gap is #dl(k("gap_test", "paima"), "loso-fill - fused-fill", name: "macro_auroc") against
#dp(k("gap_test", "paima"), "loso-fill - fused-fill") in macro-F1, a change in ranking, not only in the cutoff.

Kermany's test split is not representative of the source: #DS.kermany.test_sizes.at("1024x496") of its #num(DS.kermany.n_test) images are 1024×496, a size that
makes up #pct1(size-share("kermany", "1024x496") * 100)% of Kermany overall. On all #num(R.at("loso_full|kermany|all").n_images) images the held-out Kermany
score is #pt("loso_full|kermany|all", "loso-fill"), well below the #pt("gap_test|kermany|all", "loso-fill") on the
test split. We use the full-source score for claims about Kermany as an unseen source. The next section shows why
image size matters so much.

== The image-size shortcut <sec-shortcut>

In Duke, image size identifies the label for #calc.round((int(duke-eyes("amd", size: "1024x496")) + int(duke-eyes("dme", size: "768x496"))) / DS.duke.n_groups * 100)% of the eyes: every 768×496 scan is DME and every 1024×496
scan is AMD (@fig-size-class). Models that train on Duke can learn "768 wide means DME" as a rule, which is
correct within Duke and wrong elsewhere. Three tests show that they do.

*Changing only the size of test images.* We take the held-out Kermany models and rescore Kermany images after
changing only their width, with no retraining (@fig-size-test). Padding normal 512×496 images to 768 wide (Duke's
DME-only width) moves the share called DME from #pct1(KS.pad.at("512x496").at("normal->dme").at(0))% to #pct1(KS.pad.at("512x496").at("normal->dme").at(1))%. Cropping 768×496 normal images to 512 wide does
the reverse, from #pct1(KS.crop.at("768x496").at("normal->dme").at(0))% to #pct1(KS.crop.at("768x496").at("normal->dme").at(1))%. The effect is not limited to Kermany: padding Duke's own 512×496 images by
16 rows to 512×512, a size that is DME-only in Kermany, sends #calc.round(HS.at("duke/pad512x512").all.at("normal->dme").at(1))% of Duke's normal images to DME in the model that
held Duke out.

#figure(
  kind: image,
  placeholder(4cm)[F6 or table: test-time
    size changes and the share of normal images predicted DME / AMD],
  caption: [Changing only the width of test images moves predictions toward the label Duke associates with that
    width. Held-out Kermany models, no retraining.],
) <fig-size-test>

*Where the cue enters: the letterbox bands.* Width reaches the model through the letterbox. A 768×496 scan
fitted to the square canvas leaves flat bands above and below the tissue, covering about a third of the canvas, and
Duke's DME-only 768×496 images have the same bands. For each of the five held-out Kermany models we take
#gc-group-n("768 normal → DME") images of size 768×496 per group, one per patient: normal and AMD images the model wrongly
calls DME, and normal and AMD images it gets right. We then change only the band pixels, to black or to speckle drawn
from the darker half of the scan itself, and rescore (@tab-bands). In models 1, 2 and 4 at most #{
  let xs = (1, 2, 4).map(f => ("768 normal → DME", "768 AMD → DME").map(g => ("black", "speckle").map(b => gc-at(f, g, b)))).flatten()
  pct0(calc.max(..xs))
}% of the errors survive either band change. In models 3 and 5, #pct0(gc-at(3, "768 AMD → DME", "speckle"))% and
#pct0(gc-at(5, "768 AMD → DME", "speckle"))% of the AMD errors and #pct0(gc-at(3, "768 normal → DME", "speckle"))%
and #pct0(gc-at(5, "768 normal → DME", "speckle"))% of the normal errors survive speckle bands, so the bands are the
main but not the only route. #{
  let rows = GC.intervention_by_fold.filter(r => r.group in ("768 normal → normal", "768 AMD → AMD") and r.at("band fill") != "trained")
  let total = rows.map(r => r.n).sum()
  let changed = rows.map(r => calc.round(r.n * (1 - r.at("called true class")))).sum()
  let to-dme = rows.map(r => calc.round(r.n * r.at("called DME"))).sum()
  [Correct calls barely move: of #total such cases (two band changes, five models), #to-dme become DME, and the
  predicted class changes for #changed.]
}

#figure(
  {
    set text(size: 9.5pt)
    table(
      columns: 7,
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Model], table.cell(colspan: 2)[Normal called DME],
        table.cell(colspan: 2)[AMD called DME], table.cell(colspan: 2)[Grad-CAM band share],
        [black], [speckle], [black], [speckle], [errors], [correct],
      ),
      ..range(1, 6).map(f => (
        [#f],
        pct0(gc-at(f, "768 normal → DME", "black")), pct0(gc-at(f, "768 normal → DME", "speckle")),
        pct0(gc-at(f, "768 AMD → DME", "black")), pct0(gc-at(f, "768 AMD → DME", "speckle")),
        pct0(gc-band("768 normal → DME", f)), pct0(gc-band("768 normal → normal", f)),
      )).flatten(),
    )
  },
  caption: [Changing only the letterbox bands of held-out Kermany 768×496 images that a LOSO-Kermany model wrongly
    calls DME (all of them are called DME with the original bands). _Normal called DME_ and _AMD called DME_: share
    (%) still called DME after the bands are replaced by black or by speckle. _Grad-CAM band share_: share (%) of the
    map's mass on the bands, for normal images called DME (_errors_) and called normal (_correct_).],
) <tab-bands>

Grad-CAM illustrates the same mechanism (@fig-gradcam), with a caveat. The share of the map on the bands is
#pct0(calc.min(..range(1, 6).map(f => gc-band("768 normal → normal", f))))–#pct0(calc.max(..range(1, 6).map(f => gc-band("768 normal → normal", f))))%
for correct calls in every model, but for errors it ranges from
#pct0(calc.min(..range(1, 6).map(f => gc-band("768 normal → DME", f))))% to
#pct0(calc.max(..range(1, 6).map(f => gc-band("768 normal → DME", f))))% and does not follow the intervention:
model 4 puts only #pct0(gc-band("768 normal → DME", 4))% of the map on the bands, yet only
#pct0(gc-at(4, "768 normal → DME", "speckle"))% of its normal errors survive speckle bands (@tab-bands). The maps reflect what the model learned, not image structure: after the top of the network is
re-initialised they no longer correlate with the trained maps (median Spearman
#calc.round(GC.randomization_fold1_768_normal_to_dme.spearman_median, digits: 2), model 1). We therefore use the band
intervention as the evidence and Grad-CAM as an illustration.

#figure(
  image("../figures/gradcam.png", width: 100%),
  caption: [Grad-CAM (last convolutional stage, predicted class) on held-out Kermany 768×496 normal images, from one
    letterbox and one square-crop LOSO-Kermany model. Each column is one image as the model sees it: letterbox
    with its bands (top), square crop without (bottom). In the first four columns the letterbox model calls
    the image DME (bold) and its map sits on the top band; the square-crop model calls it normal and looks at the
    retina. The last two columns are images both models call normal. How much of the map falls on the bands varies
    between models (@tab-bands).],
) <fig-gradcam>

*Training without the cue.* Square-crop models never see the image width. With Kermany held out, they score
#pt("loso_full|kermany|all", "loso-sq") on the full source against #pt("loso_full|kermany|all", "loso-fill")
(#dl("loso_full|kermany|all", "loso-sq - loso-fill")). On the test split, each arm measured against the fused
model of the same arm, the gap narrows from #dl("gap_test|kermany|all", "loso-fill - fused-fill") to
#dl("gap_test|kermany|all", "loso-sq - fused-sq"), about #calc.round((1 - delta("gap_test|kermany|all", "loso-sq - fused-sq").point / delta("gap_test|kermany|all", "loso-fill - fused-fill").point) * 100)%. The gain is largest exactly where the cue was
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
sometimes worse (@tab-square). On OCTDL, normal recall changes by
#dl("loso_full|octdl|all", "loso-sq - loso-fill", name: "recall_normal") as the share of normal images called DME
goes from #share("loso_full|octdl|all", "loso-fill", "normal", "dme")% to
#share("loso_full|octdl|all", "loso-sq", "normal", "dme")%; on Duke, DME recall changes by
#dl("loso_full|duke|all", "loso-sq - loso-fill", name: "recall_dme"). In the Normal/AMD view, which removes the DME
class, both losses disappear (#dl("loso_full|octdl|normal_amd", "loso-sq - loso-fill") and
#dl("loso_full|duke|normal_amd", "loso-sq - loso-fill")), so square crop's cost on these sources involves DME: too
many DME calls on OCTDL, too few on Duke. PAIMA changes by #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill")
in macro-F1 and #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill", name: "macro_auroc") in AUROC, a loss in
ranking that varies strongly with the seed (@sec-limits).

#figure(
  {
    set text(size: 9.5pt)
    table(
      columns: 8,
      align: (left, right, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held out], table.cell(colspan: 3)[Macro-F1], table.cell(colspan: 2)[AUROC],
        table.cell(colspan: 2)[Normal/AMD macro-F1],
        [Letterbox], [Square], [Difference], [Letterbox], [Square], [Letterbox], [Square],
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        pt(k("loso_full", s), "loso-fill"),
        pt(k("loso_full", s), "loso-sq"),
        dl(k("loso_full", s), "loso-sq - loso-fill"),
        pt(k("loso_full", s), "loso-fill", name: "macro_auroc"),
        pt(k("loso_full", s), "loso-sq", name: "macro_auroc"),
        ..if s == "paima" { ([--], [--]) } else {
          (pt("loso_full|" + s + "|normal_amd", "loso-fill"), pt("loso_full|" + s + "|normal_amd", "loso-sq"))
        },
      )).flatten(),
    )
  },
  caption: [LOSO models on the full held-out source, with (letterbox) and without (square crop) the image-size cue
    (%). _Difference_ is paired, with its 95% interval. PAIMA is scored on normal vs AMD only, so its Normal/AMD
    view is its main score.],
) <tab-square>

We therefore keep letterbox models as the main result and treat square crop as an ablation. It confirms that
the size shortcut is real and explains a large part of Kermany's gap, but it is not a general fix. We fixed this
reading before the OCTDL and Duke square-crop results were in: square crop would become the main setup only if
neither source lost beyond noise.

== What remains of the gap <sec-remaining>

#figure(
  {
    set text(size: 9.5pt)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held out], table.cell(rowspan: 2)[3-class gap],
        table.cell(rowspan: 2)[DME recall change], table.cell(colspan: 3)[Normal/AMD macro-F1],
        [Seen], [Unseen], [Gap],
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        ..if s == "paima" { ([--], [--]) } else {
          (dl("gap_test|" + s + "|all", "loso-fill - fused-fill"),
           dl("gap_test|" + s + "|all", "loso-fill - fused-fill", name: "recall_dme"))
        },
        pt("gap_test|" + s + "|normal_amd", "fused-fill"),
        pt("gap_test|" + s + "|normal_amd", "loso-fill"),
        dl("gap_test|" + s + "|normal_amd", "loso-fill - fused-fill"),
      )).flatten(),
    )
  },
  caption: [Where the unseen-source gap comes from (letterbox, test split, LOSO minus fused, percentage points, with
    paired 95% intervals). The 3-class scores are in @tab-unseen and the DME recalls in @tab-recall-unseen. Removing the
    DME class removes most of the gap for OCTDL and Duke; PAIMA has no DME.],
) <tab-remaining>

Most of what remains is DME (@tab-remaining). On the test split, held-out DME recall falls by #span(("kermany", "octdl", "duke").map(s => delta("gap_test|" + s + "|all", "loso-fill - fused-fill", name: "recall_dme").point)) points for
Kermany, OCTDL and Duke. In the Normal/AMD view, the OCTDL and Duke gaps are within noise of zero
(#dl("gap_test|octdl|normal_amd", "loso-fill - fused-fill") and #dl("gap_test|duke|normal_amd", "loso-fill - fused-fill")). Kermany keeps a large Normal/AMD gap, through normal images called AMD: held out, its normal recall falls from
#pt("gap_test|kermany|all", "fused-fill", name: "recall_normal") to
#pt("gap_test|kermany|all", "loso-fill", name: "recall_normal") (@tab-recall-unseen), and most of the missed normal images
are called AMD. This is consistent with the size cue above: most of its test split is 1024×496, Duke's AMD-only
size. PAIMA, which has no DME, keeps a gap of
#dl("gap_test|paima|normal_amd", "loso-fill - fused-fill"), through AMD recall, and again mostly dry AMD: held-out
recall changes by #as-dl("test|paima|normal_amd", "dry", "loso-fill - fused-fill") for drusen scans and by
#as-dl("test|paima|normal_amd", "wet", "loso-fill - fused-fill") for wet ones.

#figure(
  recall-table(s => {
    let rows = (seen-row(s, "Fused (seen)", "fused-fill"),)
    if s == "kermany" {
      rows.push(recall-row(s, "Fused, no Kermany DME", ZD.at("kermany|all").arms.at("zero-dme").metrics, none))
    }
    rows.push(recall-row(s, "LOSO (held out)", R.at(k("gap_test", s)).arms.at("loso-fill").metrics, "loso-fill"))
    rows
  }),
  caption: [Recall (%) per class on each source's test split when the source is seen (fused) and held out (LOSO),
    letterbox models; columns as in @tab-recall. For Kermany, also the fused model trained without Kermany's DME
    (@tab-zero-dme). Differences cited in the text carry paired 95% intervals.],
) <tab-recall-unseen>

A likely reason is that each LOSO model sees DME from fewer sources: PAIMA has none, so with Kermany held out the
model learns DME from OCTDL and Duke alone. We tested this for Kermany with a fused model, trained
on all four sources, from which we removed every Kermany DME image (in training and validation; the other classes
and sources are unchanged; five Duke folds, same recipe as the fused model). Kermany is then a seen source
without its DME, so the comparison with the fused model isolates the missing DME, and the comparison with the LOSO model isolates the source.

#figure(
  {
    set text(size: 9.5pt)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Source], table.cell(colspan: 3)[Macro-F1], table.cell(colspan: 2)[DME recall],
        [Without], [Fused], [Difference], [Without], [Fused],
      ),
      ..srcs.map(s => {
        let key = s + "|" + view3(s)
        (
          [#source-name.at(s)],
          zd-pt(key, "zero-dme"),
          zd-pt(key, "fused-fill"),
          zd-dl(key, "zero-dme - fused-fill"),
          ..if s == "paima" { ([--], [--]) } else {
            (zd-pt(key, "zero-dme", name: "recall_dme"), zd-pt(key, "fused-fill", name: "recall_dme"))
          },
        )
      }).flatten(),
    )
  },
  caption: [Fused model trained without Kermany's DME images (_Without_) and with them (_Fused_), on each source's
    test split (%; PAIMA on Normal/AMD, without DME). _Difference_ is paired, with its patient-level 95% interval.],
) <tab-zero-dme>

Per-class recall for the three Kermany models is in @tab-recall-unseen. Removing Kermany's DME changes Kermany by
#zd-dl("kermany|all", "zero-dme - fused-fill"), all of it through DME recall (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_dme")); its normal and AMD recall stay within
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

*Causes we ruled out.* Two simpler explanations do not account for the gap (@tab-posthoc). First, the decision
cutoff: we fit a per-class shift of the output log-probabilities on the held-out source's own labels, fitting on
one half of its patients and scoring the other. This changes Kermany by #lb-dl("kermany|all", "bias-crossfit") and
PAIMA by #lb-dl("paima|normal_amd", "bias-crossfit"), and leaves OCTDL (#lb-dl("octdl|all", "bias-crossfit")) and
Duke (#lb-dl("duke|all", "bias-crossfit")) within noise. Even the best shift found by a grid search on all of a
source's labels, which a deployed model could not do, reaches only #fmt(LB.at("kermany|all").grid.best) on
Kermany, against #pt("loso_full|kermany|all", "loso-sq") for retraining without the size cue. The cutoff explains a
few points of Kermany's gap, not most of it.

#let lbk(s) = s + "|" + view3(s)
#figure(
  {
    set text(size: 9.5pt)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held out], table.cell(rowspan: 2)[LOSO],
        table.cell(colspan: 2)[Cutoff shift], table.cell(rowspan: 2)[AdaBN],
        table.cell(rowspan: 2)[Square crop (retrained)],
        [Cross-fit], [Ceiling],
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        lb-pt(lbk(s), "none"),
        lb-dl(lbk(s), "bias-crossfit"),
        fmt(LB.at(lbk(s)).grid.best),
        ab-dl(lbk(s)),
        dl(k("loso_full", s), "loso-sq - loso-fill"),
      )).flatten(),
    )
  },
  caption: [Fixes for the unseen-source gap, on the full held-out source (macro-F1 %, letterbox LOSO models; PAIMA on
    normal vs AMD). _LOSO_ is the model as trained. Cross-fit shift, AdaBN and square crop are paired differences
    from it with 95% intervals; the shift and AdaBN are fitted on one half of the patients and scored on the other.
    _Ceiling_ is the macro-F1 of the best shift found by grid search on all labels
    (#srcs.filter(s => LB.at(lbk(s)).grid.subsampled).map(s => source-name.at(s)).join(", ") on a random
    subsample).],
) <tab-posthoc>

Second, BatchNorm statistics. AdaBN @li2016adabn, a common first remedy for such a shift, re-estimates the
BatchNorm statistics on the unseen source's own unlabelled images; we again estimate on one half of the patients
and predict the other. It changes Kermany by #ab-dl("kermany|all"), less than the cutoff shift, leaves Duke
within noise (#ab-dl("duke|all")), and changes OCTDL by #ab-dl("octdl|all") and PAIMA by
#ab-dl("paima|normal_amd"). On OCTDL it shifts images toward DME: on the full source, DME recall rises from
#ab-pt("octdl|all", "none", name: "recall_dme") to #ab-pt("octdl|all", "adabn-all", name: "recall_dme") while AMD
recall falls from #ab-pt("octdl|all", "none", name: "recall_amd") to #ab-pt("octdl|all", "adabn-all", name: "recall_amd").
