// Presubmission draft (English). Build: typst compile --root ../.. main.typ
// Thai companion: th/main.typ (same sections, English wherever a Thai section is not written yet).
#import "template.typ": paper
#import "lib.typ": *

#show: paper.with(title: [Cross-Dataset Generalization of CNN Retinal OCT Classifiers])

#align(center)[
  #text(size: 1.45em, weight: "bold")[Cross-Dataset Generalization of CNN Retinal OCT Classifiers]
  #v(0.5em)
  #todo[authors, affiliation]
]

#heading(outlined: false, numbering: none)[Abstract]
#todo[write last]

#include "sections/introduction.typ"
#include "sections/background.typ"
#include "sections/data.typ"
#include "sections/method.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"

#bibliography("refs.bib")
