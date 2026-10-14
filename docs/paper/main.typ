// Presubmission draft. Build: typst compile main.typ
#set document(title: [Cross-Device Generalization of CNN Retinal OCT Classifiers])
#set page(paper: "a4", margin: 2.5cm, numbering: "1")
#set text(size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show figure.caption: set text(size: 9.5pt)

#import "lib.typ": *

#align(center)[
  #text(size: 16pt, weight: "bold")[Cross-Device Generalization of CNN Retinal OCT Classifiers]
  #v(0.5em)
  #todo[authors, affiliation]
]

#heading(outlined: false, numbering: none)[Abstract]
#todo[write last]

#include "sections/introduction.typ"
#include "sections/data.typ"
#include "sections/method.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"

// #bibliography("refs.bib")  // enable once refs.bib has entries
