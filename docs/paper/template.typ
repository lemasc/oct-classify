// Page and text setup shared by every language. `lang` picks the hyphenation, the figure/table supplements and the font.
// Thai uses TH Sarabun PSK (installed on the system), which also covers Latin
// and runs small, so it is set at 14pt. Typst falls back per glyph to the next font.
#let paper(title: none, lang: "en", body) = {
  set document(title: title)
  set page(paper: "a4", margin: 2.5cm, numbering: "1")
  set text(lang: lang, ..if lang == "th" {
    (size: 14pt, font: ("TH SarabunPSK", "Libertinus Serif"))
  } else {
    (size: 11pt, font: "Libertinus Serif")
  })
  // Sarabun runs small, so the Thai body is 14pt; the math font does not, and would look large next to it.
  // Scale equations back to the size they have in the English body (11pt).
  show math.equation: set text(size: (11 / 14) * 1em) if lang == "th"
  // Thai has no spaces between words, and the optimized (Knuth-Plass) breaker mishandles the Thai/Latin boundary:
  // it breaks before the space, so the next line starts with a stray space. First-fit breaking does not.
  set par(justify: true, ..if lang == "th" { (linebreaks: "simple", first-line-indent: (amount: 1.5em, all: true)) })
  set heading(numbering: "1.1", ..if lang == "th" { (supplement: [หัวข้อที่]) })
  // Captions and table text (in the sections) are 9.5/11 of the body, so they scale with the language's body size.
  show figure.caption: set text(size: (9.5 / 11) * 1em)
  // Tables use rules only: one on top (here), one under the header and one at the bottom (`table.hline()` in each table).
  set table(stroke: (_, y) => if y == 0 { (top: 0.5pt) })
  set table.hline(stroke: 0.5pt)
  body
}
