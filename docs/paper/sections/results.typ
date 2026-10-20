#import "../lib.typ": *

= Results <sec-results>


== Seen sources <sec-seen>

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header([Source], [Per-dataset], [Fused], [Fused, square crop], table.hline()),
    ..srcs.map(s => (
      [#source-name.at(s)],
      ci(k("seen", s), "per-dataset"),
      ci(k("seen", s), "fused-fill"),
      ci(k("seen", s), "fused-sq"),
    )).flatten(),
    table.hline(),
  ),
  caption: [Macro-F1 (%) on each source's test split, with 95% patient-level bootstrap intervals.],
) <tab-seen>

On the source it was trained on, a ResNet-50 reaches #pt(k("seen", "paima"), "per-dataset")–#pt(k("seen", "kermany"),
"per-dataset") macro-F1 (@tab-seen; Duke is pooled over its five folds, each eye scored by the model that did not train on it). Training one model on all four sources costs little: on any source the fused model gives up at most
#whole(calc.min(..srcs.map(s => delta(k("seen", s), "fused-fill - per-dataset").lo))) points against the
per-dataset model (lower end of the paired 95% interval), and the loss is measurable only on Kermany
(#dl(k("seen", "kermany"), "fused-fill - per-dataset")). Square crop, which hides the image size (@sec-square), also
costs nothing on seen sources: no source loses beyond noise against the fused model, and PAIMA gains
#dl(k("seen", "paima"), "fused-sq - fused-fill").

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
  set text(size: (9.5 / 11) * 1em)
  table(
    columns: 7,
    align: (left, left, right, right, right, right, right),
    table.header(
      table.cell(rowspan: 2)[Source], table.cell(rowspan: 2)[Model], table.cell(rowspan: 2)[Normal],
      table.cell(rowspan: 2)[AMD], table.cell(colspan: 2, align: center)[AMD stage], table.cell(rowspan: 2)[DME],
      [dry], [wet],
      table.hline(),
    ),
    ..srcs.map(s => {
      let rows = rows-of(s)
      let rule = if s == srcs.last() { table.hline() } else { table.hline(stroke: 0.3pt + gray) }
      (table.cell(rowspan: rows.len())[#source-name.at(s)], ..rows.flatten(), rule)
    }).flatten(),
  )
}
#let seen-row(s, label, arm) = recall-row(s, label, R.at("seen|" + s + "|" + view3(s)).arms.at(arm).metrics, arm)

#figure(
  recall-table(s => (seen-row(s, "Per-dataset", "per-dataset"), seen-row(s, "Fused", "fused-fill"))),
  caption: [Recall (%) per class on each source's test split, seen letterbox models, mean over models.],
) <tab-recall>

Per-class recall (@tab-recall) shows where the errors are. _Dry_ and _wet_ are recall within AMD; Duke's AMD is all
dry, and OCTDL's intermediate stage is not shown. PAIMA is the hardest source even when seen
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
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held out],
        table.cell(colspan: 3, align: center)[Macro-F1, test split],
        table.cell(rowspan: 2)[Macro-F1, \ full source],
        [Seen], [Unseen], [Gap],
        table.hline(),
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        pt(k("gap_test", s), "fused-fill"),
        pt(k("gap_test", s), "loso-fill"),
        dl(k("gap_test", s), "loso-fill - fused-fill"),
        ci(k("loso_full", s), "loso-fill"),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [Macro-F1 (%) of seen (fused) vs unseen (LOSO) sources, letterbox. _Gap_ is the paired difference, LOSO minus fused, on the same test
    images, with its 95% interval.],
) <tab-unseen>

Every source loses accuracy when it is unseen, beyond noise (@tab-unseen). The full-source column scores the LOSO model on
every image of the held-out source, and Duke held out is one model on #DS.duke.n_groups eyes. For OCTDL, PAIMA and Duke the gap is #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) macro-F1 points; for Kermany it is #dl(k("gap_test", "kermany"), "loso-fill - fused-fill").

For the three-class sources the AUROC gap is much smaller than the macro-F1 gap (@tab-auroc; for Kermany
#dl(k("gap_test", "kermany"), "loso-fill - fused-fill", name: "macro_auroc")), so the models still rank the right
class highly for most images but the decision falls on the wrong side of the cutoff more often. PAIMA is the
exception: its AUROC gap is #dl(k("gap_test", "paima"), "loso-fill - fused-fill", name: "macro_auroc") against
#dp(k("gap_test", "paima"), "loso-fill - fused-fill") in macro-F1, a change in ranking, not only in the cutoff.

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 4,
      align: (left, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held out], table.cell(colspan: 3, align: center)[AUROC, test split],
        [Seen], [Unseen], [Gap],
        table.hline(),
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        pt(k("gap_test", s), "fused-fill", name: "macro_auroc"),
        pt(k("gap_test", s), "loso-fill", name: "macro_auroc"),
        dl(k("gap_test", s), "loso-fill - fused-fill", name: "macro_auroc"),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [Macro-AUROC (%) of seen (fused) vs unseen (LOSO) sources, letterbox models, test split; gaps are paired, with 95% intervals.],
) <tab-auroc>


Kermany's test split is not representative of the source: #DS.kermany.test_sizes.at("1024x496") of its #num(DS.kermany.n_test) images are 1024×496, a size that
makes up #pct1(size-share("kermany", "1024x496") * 100)% of Kermany overall. On all #num(R.at("loso_full|kermany|all").n_images) images the held-out Kermany
score is #pt("loso_full|kermany|all", "loso-fill"), well below the #pt("gap_test|kermany|all", "loso-fill") on the
test split. We use the full-source score for claims about Kermany as an unseen source. The next section shows why
image size matters so much.

== The image-size shortcut <sec-shortcut>

In Duke, image size identifies the label for #calc.round((int(duke-eyes("amd", size: "1024x496")) + int(duke-eyes("dme", size: "768x496"))) / DS.duke.n_groups * 100)% of the eyes (@sec-data). Models that train on Duke can learn "768 wide means DME" as a rule, which is
correct within Duke and wrong elsewhere. Three tests show that they do.

*Changing only the size of test images.* We take the held-out Kermany models and rescore Kermany images after
changing only their width, with no retraining (@tab-size-test). Padding normal 512×496 images to 768 wide (Duke's
DME-only width) moves the share called DME from #pct1(KS.pad.at("512x496").at("normal->dme").at(0))% to #pct1(KS.pad.at("512x496").at("normal->dme").at(1))%, and cropping 768×496 normal
images to 512 wide does the reverse. The effect is not limited to Kermany: padding Duke's own 512×496 images by
16 rows to 512×512, a size that is DME-only in Kermany, has the same effect in the model that held Duke out.

// One test-time size change: [before, after, change, lo, hi] in % (KS / HS in lib.typ).
#let size-row(label, e) = (label, pct1(e.at(0)), pct1(e.at(1)),
  [#fmt(e.at(2) / 100, signed: true) [#fmt(e.at(3) / 100, signed: true), #fmt(e.at(4) / 100, signed: true)]])
#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 4,
      align: (left, right, right, right),
      table.header([Test images, size change], [Before], [After], [Change], table.hline()),
      ..size-row([Kermany, pad 512×496 → 768×496], KS.pad.at("512x496").at("normal->dme")),
      ..size-row([Kermany, crop 768×496 → 512×496], KS.crop.at("768x496").at("normal->dme")),
      ..size-row([Duke, pad 512×496 → 512×512], HS.at("duke/pad512x512").all.at("normal->dme")),
      table.hline(),
    )
  },
  caption: [Share (%) of normal images called DME before and after changing only the test image size, with the paired change and its 95% interval.],
) <tab-size-test>

*Where the cue enters: the letterbox bands.* Width reaches the model through the letterbox. A 768×496 scan
fitted to the square canvas leaves flat bands above and below the tissue, covering about a third of the canvas
(@fig-padding), and
Duke's DME-only 768×496 images have the same bands. For each of the five held-out Kermany models we take
#gc-group-n("768 normal → DME") images of size 768×496 per group, one per patient: normal and AMD images the model wrongly
calls DME, and normal and AMD images it gets right. We then change only the band pixels, to black or to speckle drawn
from the darker half of the scan itself, and rescore (@tab-bands). In models 1, 2 and 4 at most #{
  let xs = (1, 2, 4).map(f => ("768 normal → DME", "768 AMD → DME").map(g => ("black", "speckle").map(b => gc-at(f, g, b)))).flatten()
  pct0(calc.max(..xs))
}% of the errors survive either band change; in models 3 and 5 up to #{
  let xs = (3, 5).map(f => ("768 normal → DME", "768 AMD → DME").map(g => gc-at(f, g, "speckle"))).flatten()
  pct0(calc.max(..xs))
}% survive speckle bands, so the bands are the main but not the only route. #{
  let rows = GC.intervention_by_fold.filter(r => r.group in ("768 normal → normal", "768 AMD → AMD") and r.at("band fill") != "trained")
  let total = rows.map(r => r.n).sum()
  let to-dme = rows.map(r => calc.round(r.n * r.at("called DME"))).sum()
  [Correct calls barely move: of #total such cases (two band changes, five models), #to-dme become DME.]
}

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Model], table.cell(colspan: 2, align: center)[Normal called DME],
        table.cell(colspan: 2, align: center)[AMD called DME],
        [black], [speckle], [black], [speckle],
        table.hline(),
      ),
      ..range(1, 6).map(f => (
        [#f],
        pct0(gc-at(f, "768 normal → DME", "black")), pct0(gc-at(f, "768 normal → DME", "speckle")),
        pct0(gc-at(f, "768 AMD → DME", "black")), pct0(gc-at(f, "768 AMD → DME", "speckle")),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [Errors that survive replacing the letterbox bands with black or speckle: share (%) of 768×496 images still wrongly
    called DME, per held-out Kermany model.],
) <tab-bands>

Grad-CAM maps (last convolutional stage, for the predicted class) illustrate the same mechanism (@fig-gradcam). After the top of the network is re-initialised they no
longer correlate with the trained maps (median Spearman
#calc.round(GC.randomization_fold1_768_normal_to_dme.spearman_median, digits: 2), model 1), so they reflect what the
model learned. Attribution maps can still disagree with what an intervention changes @arun2021saliency
@hooker2019roar, so we use the band intervention as the evidence and Grad-CAM as an illustration.

#figure(
  image("../figures/gradcam.png", width: 100%),
  caption: [Grad-CAM on held-out Kermany 768×496 normal images for the letterbox (top) and square-crop (bottom) LOSO
    model; the last two columns are controls that both models call normal.],
) <fig-gradcam>

*Training without the cue.* Square-crop models never see the image width. With Kermany held out, they score
#pt("loso_full|kermany|all", "loso-sq") on the full source against #pt("loso_full|kermany|all", "loso-fill")
(#dl("loso_full|kermany|all", "loso-sq - loso-fill")). On the test split, measured against the fused model of the
same arm, the gap narrows by about #calc.round((1 - delta("gap_test|kermany|all", "loso-sq - fused-sq").point / delta("gap_test|kermany|all", "loso-fill - fused-fill").point) * 100)%,
from #dl("gap_test|kermany|all", "loso-fill - fused-fill"). The gain is largest exactly where the cue was
(@fig-kermany-size): at 768×496, the width that is DME only in Duke, square crop helps most for normal and AMD images
(normal recall rises from #pt("size_loso:768x496|kermany|all", "fill", name: "recall_normal") to
#pt("size_loso:768x496|kermany|all", "sq", name: "recall_normal")) and costs DME recall.

#figure(
  kind: image,
  fig("kermany-size"),
  caption: [Recall of the held-out Kermany source (all images) by image size and true class, for letterbox and
    square-crop LOSO models, with 95% intervals.],
) <fig-kermany-size>

*But removing the cue does not help everywhere.* On the other held-out sources square crop loses accuracy beyond
noise (@tab-square): #span(("octdl", "duke", "paima").map(s => delta(k("loso_full", s), "loso-sq - loso-fill").point))
macro-F1 points for OCTDL, Duke and PAIMA. PAIMA also loses
#dl("loso_full|paima|normal_amd", "loso-sq - loso-fill", name: "macro_auroc") in AUROC, a loss in ranking that varies
strongly with the seed (@sec-limits).

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held out], table.cell(colspan: 3, align: center)[Macro-F1],
        table.cell(colspan: 2, align: center)[AUROC],
        [Letterbox], [Square], [Difference], [Letterbox], [Square],
        table.hline(),
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        pt(k("loso_full", s), "loso-fill"),
        pt(k("loso_full", s), "loso-sq"),
        dl(k("loso_full", s), "loso-sq - loso-fill"),
        pt(k("loso_full", s), "loso-fill", name: "macro_auroc"),
        pt(k("loso_full", s), "loso-sq", name: "macro_auroc"),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [LOSO models on the full held-out source, with (letterbox) and without (square crop) the image-size cue
    (%); the difference is paired, with its 95% interval.],
) <tab-square>

For OCTDL and Duke the cost involves DME (@tab-square-dme). On OCTDL, normal recall changes by
#dl("loso_full|octdl|all", "loso-sq - loso-fill", name: "recall_normal") as the share of normal images called DME
goes from #share("loso_full|octdl|all", "loso-fill", "normal", "dme")% to
#share("loso_full|octdl|all", "loso-sq", "normal", "dme")%; on Duke, DME recall changes by
#dl("loso_full|duke|all", "loso-sq - loso-fill", name: "recall_dme"). In the Normal/AMD view, which drops the DME
images and the DME output, both losses disappear: too many DME calls on OCTDL, too few on Duke.

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, left, right, right, right),
      table.header([Held out], [Measure], [Letterbox], [Square], [Difference], table.hline()),
      ..("octdl", "duke").map(s => {
        let key = "loso_full|" + s + "|"
        (
          table.cell(rowspan: 3)[#source-name.at(s)],
          [Normal/AMD macro-F1],
          pt(key + "normal_amd", "loso-fill"), pt(key + "normal_amd", "loso-sq"),
          dl(key + "normal_amd", "loso-sq - loso-fill"),
          [Normal recall],
          pt(key + "all", "loso-fill", name: "recall_normal"), pt(key + "all", "loso-sq", name: "recall_normal"),
          dl(key + "all", "loso-sq - loso-fill", name: "recall_normal"),
          [DME recall],
          pt(key + "all", "loso-fill", name: "recall_dme"), pt(key + "all", "loso-sq", name: "recall_dme"),
          dl(key + "all", "loso-sq - loso-fill", name: "recall_dme"),
          table.hline(stroke: 0.3pt + gray),
        )
      }).flatten(),
      table.hline(),
    )
  },
  caption: [OCTDL and Duke held out, full source: letterbox vs square crop (%), including the Normal/AMD view without DME.
    Differences are paired, with 95% intervals.],
) <tab-square-dme>

We therefore keep letterbox models as the main result and treat square crop as an ablation. It confirms that
the size shortcut is real and explains a large part of Kermany's gap, but it is not a general fix. We fixed this
reading before the OCTDL and Duke square-crop results were in: square crop would become the main setup only if
neither source lost beyond noise.

== Where the gap sits <sec-remaining>

#figure(
  kind: image,
  fig("recall-change"),
  caption: [Change in per-class recall (percentage points) when the source is held out (LOSO minus fused, letterbox,
    test split), with paired 95% intervals. Fused levels are in @tab-recall.],
) <fig-recall-change>

The classes that carry the gap differ by source (@fig-recall-change): each source loses most in one class, normal
for Kermany, DME for OCTDL and Duke, and dry AMD for PAIMA. DME recall does not count normal or AMD images wrongly
called DME, so we also score the Normal/AMD view, which drops the DME images and the DME output (@sec-eval). There
the OCTDL and Duke gaps are #dl("gap_test|octdl|normal_amd", "loso-fill - fused-fill") and
#dl("gap_test|duke|normal_amd", "loso-fill - fused-fill"), not distinguishable from zero, which points to DME for
these two sources, but the intervals are too wide to show that DME accounts for all of it. Kermany keeps a
Normal/AMD gap of #dl("gap_test|kermany|normal_amd", "loso-fill - fused-fill"): most of its missed normal images
are called AMD, consistent with the size cue (@fig-kermany-size), since most of its test split is 1024×496, Duke's
AMD-only size. PAIMA, which has no DME, loses more dry than wet AMD recall, as it does when seen (@tab-recall).

A likely reason for the DME losses is that each LOSO model sees DME from fewer sources: PAIMA has none, so with Kermany held out the
model learns DME from OCTDL and Duke alone. We tested this for Kermany with a fused model, trained
on all four sources, from which we removed every Kermany DME image (in training and validation; the other classes
and sources are unchanged; five Duke folds, same recipe as the fused model). Kermany is then a seen source
without its DME, so the comparison with the fused model isolates the missing DME, and the comparison with the LOSO model isolates the source.

// Rows of the zero-DME control: per-arm levels, and paired differences between arms (italic).
#let zd-level(label, key, arm) = (
  [#label], zd-pt(key, arm), zd-pt(key, arm, name: "recall_normal"), zd-pt(key, arm, name: "recall_amd"),
  if "recall_dme" in ZD.at(key).arms.at(arm).metrics { zd-pt(key, arm, name: "recall_dme") } else [--],
)
#let zd-diff(label, key, pair) = (
  [_#label _], zd-dl(key, pair), zd-dl(key, pair, name: "recall_normal"), zd-dl(key, pair, name: "recall_amd"),
  if "recall_dme" in ZD.at(key).deltas.at(pair) { zd-dl(key, pair, name: "recall_dme") } else [--],
)
#let zd-key = "kermany|all"
#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      table.header(
        [Model], [Macro-F1], [Normal], [AMD], [DME],
        table.hline(),
      ),
      ..zd-level("Fused", zd-key, "fused-fill"),
      ..zd-level("No Kermany DME", zd-key, "zero-dme"),
      ..zd-level("LOSO (held out)", zd-key, "loso-fill"),
      table.hline(stroke: 0.3pt + gray),
      ..zd-diff("no DME − fused", zd-key, "zero-dme - fused-fill"),
      ..zd-diff("LOSO − no DME", zd-key, "loso-fill - zero-dme"),
      table.hline(),
    )
  },
  caption: [Kermany's test split under the fused, no-Kermany-DME and LOSO models: macro-F1 and per-class recall (%).
    Italic rows are paired differences with 95% intervals.],
) <tab-zero-dme>

@tab-zero-dme puts the three Kermany models side by side. Removing Kermany's DME changes Kermany by
#zd-dl("kermany|all", "zero-dme - fused-fill"), all of it through DME recall (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_dme")); its normal and AMD recall stay within
noise. On the same test images this is about #zd-share()% of the held-out gap, but the two models fail
differently, so the gap does not split into these shares. Missing in-device DME alone already costs more DME recall
than the LOSO model loses, so it can account for the LOSO model's DME errors. The LOSO model's other loss is normal recall
(#zd-dl("kermany|all", "loso-fill - zero-dme", name: "recall_normal") against the model without Kermany DME), which
comes from the source being unseen and matches the size cue. Missing DME is therefore a real but partial cause.

== Post-hoc fixes <sec-posthoc>

Two standard post-hoc fixes leave most of the gap in place (@tab-posthoc). Both are fitted on one half of the
held-out source's patients and scored on the other. The first tests whether the gap is a misplaced decision cutoff
(@sec-unseen): a per-class shift of the output log-probabilities, fitted on the source's own labels. It gains
Kermany #lb-dl("kermany|all", "bias-crossfit") and changes the other sources by little.
Even the best shift found on all of a source's labels (the Ceiling column; #srcs.filter(s => LB.at(s + "|" + view3(s)).grid.subsampled).map(s => source-name.at(s)).join(", ") on a random subsample), which a deployed model could not do, reaches only
#fmt(LB.at("kermany|all").grid.best), against #pt("loso_full|kermany|all", "loso-sq") for retraining without the
size cue; a shift shared by all images cannot undo errors that depend on image size.

#let lbk(s) = s + "|" + view3(s)
#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held out], table.cell(rowspan: 2)[LOSO],
        table.cell(colspan: 2, align: center)[Cutoff shift], table.cell(rowspan: 2)[AdaBN],
        table.cell(rowspan: 2)[Square crop (retrained)],
        [Cross-fit], [Ceiling],
        table.hline(),
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        lb-pt(lbk(s), "none"),
        lb-dl(lbk(s), "bias-crossfit"),
        fmt(LB.at(lbk(s)).grid.best),
        ab-dl(lbk(s)),
        dl(k("loso_full", s), "loso-sq - loso-fill"),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [Post-hoc fixes on the full held-out source (macro-F1 %, letterbox LOSO models); fix columns are paired
    differences from LOSO, with 95% intervals.],
) <tab-posthoc>

The second fix, AdaBN @li2016adabn, re-estimates the BatchNorm statistics on the unseen source's unlabelled
images. It gains Kermany #ab-dl("kermany|all"), less than the cutoff shift, but lowers OCTDL
(#ab-dl("octdl|all")), where it pushes images toward DME. Neither fix is reliable across sources, and on Kermany neither comes close to
retraining without the size cue.
