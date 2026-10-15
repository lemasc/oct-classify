// Thai companion. Build (from docs/paper): typst compile --root ../.. th/main.typ
// A section is taken from th/sections/ when listed in `translated` (i18n/th.typ), else from the English ../sections/.
// `--input final=1` hides the "not translated" markers.
#import "../template.typ": paper
#import "../lib.typ": *
#import "../i18n/th.typ"

#show: paper.with(title: th.title, lang: "th")

#let final = sys.inputs.at("final", default: "0") == "1"

#let section(name) = {
  if name in th.translated {
    include "sections/" + name + ".typ"
  } else {
    [
      #set text(lang: "en")
      #if not final { pending[ยังไม่ได้แปล: #name] }
      #include "../sections/" + name + ".typ"
    ]
  }
}

#align(center)[
  #text(size: 1.45em, weight: "bold")[#th.title]
  #v(0.5em)
  #todo[authors, affiliation]
]

#heading(outlined: false, numbering: none)[#th.abstract-title]
#if th.abstract == none [#todo[write last]] else [#th.abstract]

#for name in ("introduction", "data", "method", "results", "discussion") { section(name) }

#bibliography("../refs.bib")
