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
  if v < 0 { "−" + s } else if signed and v > 0 { "+" + s } else if signed { "±" + s } else { s }
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
