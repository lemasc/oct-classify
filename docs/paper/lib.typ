// Draft markers, visible in the PDF until resolved.
#let todo(body) = text(fill: red)[[TODO: #body]]
#let pending(body) = text(fill: orange)[[pending: #body]]

// Results. Paths are relative to the repository root, so compile with `--root ../..`.
#let R = json("/data/paper/results.json")

// Fraction -> percent with one decimal, always showing the decimal ("98.0", not "98").
#let fmt(x, signed: false) = {
  let v = calc.round(x * 100, digits: 1)
  let s = str(calc.abs(v))
  if not s.contains(".") { s += ".0" }
  if v < 0 { "−" + s } else if signed and v > 0 { "+" + s } else { s }
}

#let metric(key, arm, name: "macro_f1") = R.at(key).arms.at(arm).metrics.at(name)
#let delta(key, pair, name: "macro_f1") = R.at(key).deltas.at(pair).at(name)

// Point estimate, point with 95% CI, and paired delta with CI.
#let pt(key, arm, name: "macro_f1") = fmt(metric(key, arm, name: name).point)
#let ci(key, arm, name: "macro_f1") = {
  let m = metric(key, arm, name: name)
  [#fmt(m.point) [#fmt(m.lo), #fmt(m.hi)]]
}
#let dl(key, pair, name: "macro_f1") = {
  let m = delta(key, pair, name: name)
  [#fmt(m.point, signed: true) [#fmt(m.lo, signed: true), #fmt(m.hi, signed: true)]]
}
// Signed point of a paired delta, without its interval.
#let dp(key, pair, name: "macro_f1") = fmt(delta(key, pair, name: name).point, signed: true)
// Mean ± SD over folds, for arms with five models.
#let sd(key, arm, name: "macro_f1") = {
  let m = metric(key, arm, name: name)
  if "fold_sd" in m and m.fold_sd != none [#fmt(m.point) ± #fmt(m.fold_sd)] else [#fmt(m.point)]
}

#let source-name = (kermany: "Kermany", octdl: "OCTDL", paima: "PAIMA", duke: "Duke")
// PAIMA has no DME, so its only view is Normal/AMD.
#let view3(src) = if src == "paima" { "normal_amd" } else { "all" }

// Integer with thousands separators: 101217 -> "101,217".
#let num(n) = {
  let s = str(n)
  let out = ""
  for (i, c) in s.clusters().enumerate() {
    if i > 0 and calc.rem(s.len() - i, 3) == 0 { out += "," }
    out += c
  }
  out
}

// Grey box standing in for a figure that is not generated yet.
#let placeholder(height, body) = rect(width: 100%, height: height, stroke: 0.5pt + gray)[
  #align(center + horizon)[#pending(body)]
]

// Logit-bias diagnostic (data/paper/logit-bias). Arms: "none", "bias-crossfit", "bias-oracle".
#let LB = json("/data/paper/logit-bias/results.json")
#let lb-pt(key, arm) = fmt(LB.at(key).arms.at(arm).metrics.macro_f1.point)
#let lb-dl(key, arm) = {
  let m = LB.at(key).arms.at(arm).delta.macro_f1
  [#fmt(m.point, signed: true) [#fmt(m.lo, signed: true), #fmt(m.hi, signed: true)]]
}

// Duke eye-level scores (scripts/paper/duke_eye_level.py), for the footnote only.
#let EYE = json("/data/paper/duke-eye-level.json")
#let eye-wrong(arm) = str(EYE.arms.at(arm).misclassified_eyes)
#let eye-f1(arm) = fmt(EYE.arms.at(arm).macro_f1.point)

// Share (%) of true class `t` predicted as class `p`, mean over models, from a result's confusion matrix.
#let share(key, arm, t, p) = {
  let order = ("normal", "amd", "dme")
  let row = R.at(key).arms.at(arm).confusion.at(order.position(c => c == t))
  fmt(row.at(order.position(c => c == p)) / row.sum())
}
#let fig(name, width: 100%) = image("figures/" + name + ".svg", width: width)

// Grad-CAM shortcut analysis (notebooks/gradcam_shortcut.py -> data/paper/gradcam/summary.json).
#let GC = json("/data/paper/gradcam/summary.json")
#let pct0(x) = str(calc.round(x * 100)) // whole percent
// Share (0-1) of a group's images called DME with the given band fill, for one model (fold).
#let gc-at(fold, group, fill) = GC.intervention_by_fold.find(r => (
  r.fold == fold and r.group == group and r.at("band fill") == fill
)).at("called DME")
// Mean share (0-1) of the Grad-CAM map on the bands, for one group and model (fold).
#let gc-band(group, fold) = GC.mean_band_share_by_fold.at(group).at(str(fold))
#let gc-group-n(group) = GC.intervention_by_fold.find(r => r.group == group).n

// AdaBN diagnostic (data/paper/adabn). Arms: "none", "adabn-all".
#let AB = json("/data/paper/adabn/results.json")
#let ab-pt(key, arm, name: "macro_f1") = fmt(AB.at(key).arms.at(arm).metrics.at(name).point)
#let ab-dl(key, name: "macro_f1") = {
  let m = AB.at(key).arms.at("adabn-all").delta.at(name)
  [#fmt(m.point, signed: true) [#fmt(m.lo, signed: true), #fmt(m.hi, signed: true)]]
}

// Fused model without Kermany DME (data/paper/zero-dme). Arms: "zero-dme", "fused-fill", "loso-fill" (Kermany, Duke).
#let ZD = json("/data/paper/zero-dme/results.json")
#let zd-pt(key, arm, name: "macro_f1") = fmt(ZD.at(key).arms.at(arm).metrics.at(name).point)
#let zd-dl(key, pair, name: "macro_f1") = {
  let m = ZD.at(key).deltas.at(pair).at(name)
  [#fmt(m.point, signed: true) [#fmt(m.lo, signed: true), #fmt(m.hi, signed: true)]]
}
// Share of Kermany's LOSO gap (fused minus LOSO, same images) that the zero-DME model already shows.
#let zd-share() = {
  let f(a) = ZD.at("kermany|all").arms.at(a).metrics.macro_f1.point
  str(calc.round((f("fused-fill") - f("zero-dme")) / (f("fused-fill") - f("loso-fill")) * 100))
}

// AMD recall by stage (data/paper/amd-stage). Keys "test|<src>|<view>" and "full|<src>|<view>"; stages dry, wet,
// intermediate (OCTDL), normal, dme. Arms: per-dataset, fused-fill, fused-sq, loso-fill, loso-sq (full: LOSO only).
#let AS = json("/data/paper/amd-stage/results.json")
#let as-rec(key, st, arm) = fmt(AS.at(key).stages.at(st).arms.at(arm).recall)
#let as-dl(key, st, pair) = {
  let m = AS.at(key).deltas.at(st).at(pair)
  [#fmt(m.point, signed: true) [#fmt(m.lo, signed: true), #fmt(m.hi, signed: true)]]
}

// Dataset statistics (scripts/paper/export_data_stats.py): counts, group units, image sizes, deduplication.
#let DS = json("/data/paper/data-stats.json").sources
// AMD stage share of a source's AMD images (data/paper/amd-stage, full source). Duke's AMD is all dry.
#let stage-share(src, st) = {
  if src == "duke" { return if st == "dry" { "100" } else { none } }
  let stages = AS.at("full|" + src + "|" + view3(src)).stages
  let total = ("dry", "wet", "intermediate").map(s => if s in stages { stages.at(s).n_images } else { 0 }).sum()
  str(calc.round(stages.at(st).n_images / total * 100))
}
// Share (0-1) of a source's images with a given size ("WxH").
#let size-share(src, size) = DS.at(src).sizes.at(size) / DS.at(src).n_images

// Test-time image-size changes (data/analysis/{kermany,heldout}-size-test): entries are [before, after, delta, lo, hi] in %.
#let KS = json("/data/analysis/kermany-size-test/size-test.json")
#let HS = json("/data/analysis/heldout-size-test/size-test.json")
#let pct1(x) = str(calc.round(x, digits: 1))
// Range "a–b" of fractions as whole percentage points, from the rounded values.
#let span(xs) = {
  let v = xs.map(x => calc.round(calc.abs(x) * 100))
  [#int(calc.min(..v))–#int(calc.max(..v))]
}

// Duke eyes of a class (at one image size, or in total).
#let duke-eyes(cls, size: none) = {
  let by = DS.duke.eyes_by_size_class
  str(by.pairs().filter(((sz, _)) => size == none or sz == size).map(((_, c)) => c.at(cls, default: 0)).sum())
}
// Kermany images outside the five sizes shown in the size figure.
#let kermany-rare = num(DS.kermany.n_images - ("512x496", "768x496", "1024x496", "1536x496", "512x512").map(z => DS.kermany.sizes.at(z)).sum())
// Whole percentage points of a fraction (absolute value).
#let whole(x) = str(calc.round(calc.abs(x) * 100))

#let srcs = ("kermany", "octdl", "paima", "duke")
#let k(kind, src) = kind + "|" + src + "|" + view3(src)

// Seed variation, LOSO full source (scripts/paper/export_seeds.py): per source, arm ("loso-fill", "loso-sq") and
// seed, the mean over the 5 models.
#let SEED = json("/data/paper/seeds.json")
#let seed-vals(src, arm, name: "macro_f1") = SEED.at(src).at(arm).values().map(m => m.at(name))
// Per seed, square crop minus letterbox macro-F1 (fractions).
#let seed-gain(src) = SEED.at(src).at("loso-sq").keys().map(s => (
  SEED.at(src).at("loso-sq").at(s).macro_f1 - SEED.at(src).at("loso-fill").at(s).macro_f1
))
// Range "a–b" of fractions as percent with one decimal, absolute values.
#let span1(xs) = {
  let v = xs.map(calc.abs)
  [#fmt(calc.min(..v))–#fmt(calc.max(..v))]
}

// Multi-window pooling on square-crop LOSO models (data/analysis/3a-multiwindow, exp. 3a section 4): entries are
// [single centre window, pooled, delta lo, delta hi] in %.
#let MW = json("/data/analysis/3a-multiwindow/compare.json")
