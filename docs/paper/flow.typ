// Study overview for the top of Method: the pipeline, and the loop from diagnosis back to preprocessing that added
// the square-crop arm. Labels are content, so each language passes its own (English defaults below; Thai in
// i18n/th.typ). Sizes are in em of the body, so the diagram scales with the language like captions and tables.
#import "@preview/fletcher:0.5.8": diagram, edge, node

// Arm colours from scripts/paper/style.py, so they match the generated figures.
#let ink-2 = rgb("#52514e")
#let ink-3 = rgb("#8a8984")
#let letterbox-color = rgb("#6250d6")
#let square-color = rgb("#008300")
#let letterbox(body) = text(fill: letterbox-color, body)
#let square(body) = text(fill: square-color, body)

#let flow-en = (
  stages: (
    ([Data], [4 sources, 3 classes; deduplicate; patient-level split]),
    ([Preprocess], [#letterbox[letterbox] (default) or #square[square crop] (@fig-padding)]),
    ([Train], [ResNet-50 (ImageNet); per-dataset, fused, LOSO (@fig-design)]),
    ([Score], [per B-scan macro-F1, paired bootstrap; gap = LOSO − fused]),
    ([Diagnose], [resize test images; replace the bands; Grad-CAM]),
  ),
  loop: [size cue found:\ add the\ square-crop arm],
)

#let method-flow(labels: flow-en) = {
  set text(size: (9.5 / 11) * 1em)
  // Stacked (name above the description) so the boxes fit one column. The return arrow and its label hang outside
  // the boxes, and fletcher does not count them against the given width, so the boxes are sized from a measured
  // draw (see below) to make the whole diagram exactly as wide as the column.
  let stage((name, body)) = align(left, {
    text(weight: "bold", name)
    linebreak()
    text(fill: ink-2, size: (8.5 / 9.5) * 1em, body)
  })
  let n = labels.stages.len()
  let draw(width) = diagram(
    spacing: (1.5em, 1.2em),
    node-stroke: 0.6pt + ink-3,
    node-corner-radius: 3pt,
    node-inset: 0.6em,
    edge-stroke: 0.7pt + ink-2,
    ..labels.stages.enumerate().map(((i, s)) => (
      node((0, i), stage(s), width: width, name: label("flow-" + str(i))),
      if i + 1 < n { edge("-|>") },
    )).flatten().filter(x => x != none),
    // diagnosis -> preprocess: the square-crop arm came from what the diagnosis found
    edge(label("flow-" + str(n - 1)), "r," + "u" * (n - 2) + ",l", "-|>", stroke: 0.7pt + square-color,
      corner-radius: 4pt, label: square(text(size: (8.5 / 9.5) * 1em, labels.loop)), label-side: right),
  )
  // HTML export has no layout engine for the diagram, so embed it as an inline SVG there.
  context if target() == "html" { html.frame(draw(30em)) } else { layout(size => {
    let overshoot = measure(draw(size.width)).width - size.width
    draw(size.width - overshoot)
  }) }
}
