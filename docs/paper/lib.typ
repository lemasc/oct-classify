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
// Share (0-1) of a group's images called DME after changing the band pixels, for each fold.
#let gc-dme(group, fill) = GC.intervention_by_fold.filter(r => r.group == group and r.at("band fill") == fill).map(r => r.at("called DME"))
#let gc-band(group) = GC.mean_band_share_by_fold.at(group).values()
#let pct0(x) = str(calc.round(x * 100)) // whole percent
#let range0(xs) = pct0(calc.min(..xs)) + "–" + pct0(calc.max(..xs))
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
// Lowest recall of a stage over all arms of a key.
#let as-min(key, st) = fmt(calc.min(..AS.at(key).stages.at(st).arms.values().map(a => a.recall)))
// PAIMA held out, full source: AUROC normal vs dry / wet, fill-aware vs square crop, per seed.
#let PA = json("/data/paper/amd-stage/paima-auroc.json")
#let pa-dl(seed, st) = {
  let r = PA.at(str(seed) + "|normal_vs_" + st)
  let d = fmt(r.sq - r.fill, signed: true)
  if "delta_ci" in r [#d [#fmt(r.delta_ci.at(0), signed: true), #fmt(r.delta_ci.at(1), signed: true)]] else [#d]
}
