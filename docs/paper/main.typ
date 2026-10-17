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
Retinal OCT classifiers are usually tested on the dataset they were trained on. We measure how much accuracy they
keep on a dataset they have not seen, and why they lose it. Using four public datasets (Kermany, OCTDL, PAIMA and
Duke) and an ImageNet-pretrained ResNet-50 that labels B-scans (two-dimensional cross-sections of the retina) as normal, AMD or DME, we compare models scored on
sources they trained on with leave-one-source-out models, using patient-level splits and paired bootstrap
intervals. Seen sources score #span(("kermany", "octdl", "paima", "duke").map(s => metric(k("seen", s), "per-dataset").point)) macro-F1. Every source loses accuracy beyond noise when unseen:
#span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) points for OCTDL, PAIMA and Duke and about #whole(delta(k("gap_test", "kermany"), "loso-fill - fused-fill").point) for Kermany, although three of the four
sources share one manufacturer. Much of Kermany's gap is a shortcut imported from another dataset: in Duke, image
size reveals the label, and models trained with Duke apply this rule to Kermany through the padding bands added
when wide scans are fitted to a square input. Training without the size cue raises held-out Kermany from
#pt("loso_full|kermany|all", "loso-fill") to #pt("loso_full|kermany|all", "loso-sq") macro-F1 but lowers the other unseen sources. Most of the remaining gap is DME recall,
and shifting the decision cutoff or using AdaBN leaves most of it in place. Comparing scan settings against labels
before pooling datasets would have exposed the shortcut.

#include "sections/introduction.typ"
#include "sections/background.typ"
#include "sections/data.typ"
#include "sections/method.typ"
#include "sections/results.typ"
#include "sections/discussion.typ"

#bibliography("refs.bib")
