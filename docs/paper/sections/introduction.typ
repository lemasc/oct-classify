#import "../lib.typ": *

= Introduction <sec-intro>

Optical coherence tomography (OCT) images the retina in cross-section and has become a standard of care for
diagnosing and treating retinal disease @kermany2018. Two of the diseases it is used to find are common and growing: age-related macular degeneration (AMD)
was projected to affect 196 million people in 2020 and 288 million by 2040 @wong2014amd, and diabetic macular edema
(DME) affects about 5.5% of people with diabetes @im2022dme, a population in which diabetic retinopathy is expected
to rise from 103 million to 161 million cases by 2045 @teo2021dr. In 2020, AMD and diabetic retinopathy together
caused about 2.7 million cases of blindness among people aged 50 and over @gbd2021blindness. Reading scans at this
volume is repetitive work, which makes automatic classification an attractive aid. @sec-background explains how
OCT images are formed and how the two diseases appear on them.

Convolutional networks already classify OCT B-scans, two-dimensional cross-sections of the retina, with high
accuracy; Kermany et al. report 96.6% on four classes
@kermany2018 (@sec-prior-classification). Nearly all such numbers come from a test split of the dataset the model
was trained on. A deployed model instead meets scans from a clinic it has never seen: another site, population,
acquisition protocol and labelling practice, and sometimes another device. How much accuracy survives that move,
and why it is lost, is rarely measured for B-scan classifiers trained on public datasets (@sec-prior-shift).

A natural remedy is to train on several datasets at once. Pooling gives the model more variety, but it also mixes
datasets with different quirks: a property that happens to predict the label in one dataset, such as how its scans
were acquired, can become a rule the model applies to all of them. Such shortcuts @geirhos2020shortcut stay hidden
as long as the test data share the quirk (@sec-prior-shortcut).

We study this with four public OCT datasets (Kermany, OCTDL, PAIMA and Duke), three classes (normal, AMD and DME)
and the common baseline, an ImageNet-pretrained ResNet-50. Each model is scored on sources it trained on and, by
leaving one source out, on a source it never saw; splits are made by patient and differences carry paired bootstrap
intervals. We ask three questions:

/ RQ1: _How much accuracy does a classifier lose on a source it has not seen, and does the loss require a change
  of device?_ Every source loses accuracy beyond noise: #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) macro-F1 points for OCTDL, PAIMA and Duke and
  about #whole(delta(k("gap_test", "kermany"), "loso-fill - fused-fill").point) for Kermany, although three of the four sources come from one manufacturer (@sec-unseen).
/ RQ2: _Where does the loss come from?_ Partly from a shortcut imported from another dataset. In Duke, image size
  gives the label away; models trained with Duke learn it through the padding bands added when wide scans are
  fitted to a square input, and apply it to Kermany. Changing only the size or the bands of test images moves
  predictions (@sec-shortcut). Most of what remains is DME recall: removing Kermany's DME from an otherwise
  identical fused model costs about #zd-share()% as much as holding Kermany out (@sec-remaining).
/ RQ3: _Can the gap be closed without new labelled data?_ Not with one fix for every source. Training without the
  size cue raises held-out Kermany from #pt("loso_full|kermany|all", "loso-fill") to #pt("loso_full|kermany|all", "loso-sq") macro-F1 but costs the other unseen sources
  (@sec-shortcut), and two standard post-hoc fixes, shifting the decision cutoff and re-estimating BatchNorm
  statistics (AdaBN), leave most of the gap in place (@sec-posthoc).
