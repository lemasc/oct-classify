// Labels of the figures and tables that span both columns; every other one stays in its column.
// One list for all languages, so the English and Thai papers lay out alike. Add a label here to opt a float out of the column.
#let wide-figures = ("fig-examples", "tab-data", "tab-seen", "tab-recall", "tab-unseen", "fig-gap", "tab-size-test", "fig-gradcam", "fig-kermany-size", "tab-square", "tab-remaining", "tab-recall-unseen", "tab-zero-dme", "tab-posthoc",)

// Page and text setup shared by every language. `lang` picks the hyphenation, the figure/table supplements and the font.
// Thai uses TH Sarabun PSK (installed on the system), which also covers Latin
// and runs small, so it is set at 14pt. Typst falls back per glyph to the next font.
#let paper(title: none, lang: "en", body) = {
  set document(title: title)
  set page(paper: "a4", margin: (x: 1.8cm, y: 2.2cm), columns: 2, numbering: "1")
  set columns(gutter: 0.7cm)
  set text(lang: lang, ..if lang == "th" {
    (size: 14pt, font: ("TH SarabunPSK", "Libertinus Serif"))
  } else {
    (size: 11pt, font: "Libertinus Serif")
  })
  // Sarabun runs small, so the Thai body is 14pt; the math font does not, and would look large next to it.
  // Scale equations back to the size they have in the English body (11pt).
  show math.equation: set text(size: (11 / 14) * 1em) if lang == "th"
  // Figures and tables float to the top or bottom of a column. Those listed in `wide-figures` span both columns.
  set figure(placement: auto)
  show selector.or(..wide-figures.map(label)): set figure(scope: "parent", placement: top)
  set par(justify: true)
  set heading(numbering: "1.1")
  // Captions and table text (in the sections) are 9.5/11 of the body, so they scale with the language's body size.
  show figure.caption: set text(size: (9.5 / 11) * 1em)
  // Tables use rules only: one on top (here), one under the header and one at the bottom (`table.hline()` in each table).
  set table(stroke: (_, y) => if y == 0 { (top: 0.5pt) })
  set table.hline(stroke: 0.5pt)
  body
}

// Title block and abstract span both columns and sit at the top of page 1; the body flows into two columns below.
#let front(body) = place(top + center, scope: "parent", float: true, clearance: 1em, body)
