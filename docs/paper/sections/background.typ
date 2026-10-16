#import "../lib.typ": *

= Background and related work <sec-background>

== Optical coherence tomography <sec-oct>

Optical coherence tomography (OCT) images the retina in cross-section, without contact and without dye
@huang1991. It works like ultrasound with light instead of sound: a near-infrared beam enters the eye, and
low-coherence interferometry measures how much light comes back from each depth @huang1991. One measurement gives
an _A-scan_, a profile of brightness against depth at one point of the retina (@fig-scan a) @fujimoto2016. Sweeping
the beam sideways and placing the A-scans side by side gives a _B-scan_, a two-dimensional cross-section (@fig-scan
b) @fujimoto2016. A B-scan is the image a classifier sees. A macular _volume_ is a stack of parallel B-scans across
the centre of the retina (@fig-scan c). On a B-scan the retina is a band of bright and dark layers above the
choroid, with a dip at the _fovea_, the centre of sharp vision.

#figure(
  kind: image,
  fig("scan"),
  caption: [How an OCT volume is built, from one Duke normal eye @srinivasan2014. (a) One A-scan. (b) A B-scan
    through the fovea, 512 A-scans side by side; depth is stretched about 3× (see the scale bar). (c) The volume,
    a stack of 97 B-scans.],
) <fig-scan>

The number of A-scans in a B-scan, the number of B-scans in a volume and the depth of each A-scan are scan
settings, so they vary between devices and between protocols on one device. In the RETOUCH benchmark, which covers
the same 6×6 mm area with devices from three vendors, a B-scan is 512 A-scans wide on all of them but 1,024 pixels
deep on Cirrus, 496 on Spectralis and 885 or 650 on Topcon, and a volume holds 128, 49 or 128 B-scans
@bogunovic2019retouch. On one device,
Duke's Spectralis B-scans range from 512 to 1,024 A-scans (@sec-data). Image size is therefore a property of the
acquisition, not of the eye.

== AMD and DME on OCT <sec-diseases>

The hallmark of *age-related macular degeneration (AMD)* is _drusen_, deposits of extracellular material between
the retinal pigment epithelium (RPE) and Bruch's membrane @sotoudehpaima2022, which appear on a B-scan as bumps under
the retina (@fig-examples). The clinical classification grades AMD by drusen size and late lesions: medium drusen
mark early AMD, large drusen, or pigment changes with medium drusen, intermediate AMD, and neovascular AMD or geographic
atrophy late AMD @ferris2013. In neovascular ("wet") AMD, new vessels grow from the choroid, a process called
choroidal neovascularization (CNV), and leak fluid into and under the retina @kermany2018 @sotoudehpaima2022. Our
sources call the drusen stage "dry AMD" (Duke) or "drusen", and we follow them. The clinical classification instead
reserves "dry" for geographic atrophy @ferris2013, so in this paper dry AMD means early and intermediate AMD with
drusen, not atrophy.

*Diabetic macular edema (DME)* is a complication of diabetes in which fluid builds up in the central retina
@kermany2018. On a B-scan the retina is thickened and holds dark, fluid-filled spaces (@fig-examples)
@kermany2018. Fluid also appears in wet AMD @kermany2018, so the presence of fluid alone does not separate the two
diseases.

== Classifying OCT scans <sec-prior-classification>

Early classifiers used hand-crafted features: Srinivasan et al. built Duke's dataset and classified its 45 eyes
with histograms of oriented gradients and a support vector machine @srinivasan2014. Convolutional networks
pretrained on ImageNet then took over @lee2017 @karri2017. Kermany et al. reached 96.6% accuracy on four classes
(normal, drusen, CNV, DME) after training on 108,312 B-scans @kermany2018, later work added multi-scale models
@rasti2018 @sotoudehpaima2022 and lesion attention @fang2019, and the OCTDL release reports VGG16 and ResNet-50
baselines @kulyabin2024octdl. Reviews of this work @muchuchuti2023 @abdi2025 name dataset bias and model
generalization among the main open problems @abdi2025. Nearly all of these models are tested on the dataset they
were trained on; Rasti et al. and Fang et al. each use two datasets but train and test within each one @rasti2018
@fang2019.#footnote[Both use a 148-eye dataset from Noor Eye Hospital, Tehran (NEH), as do Zhang et al. @zhang2025
  and MIRAGE @morano2025mirage. It is not PAIMA, which comes from the same hospital @sotoudehpaima2022 and which
  Chutani and Yalavarthy also call NEH @chutani2026.]

== Generalization across datasets and devices <sec-prior-shift>

Most OCT models "have been trained and tested on OCT images obtained from a single commercial device in a single
center, with a lack of external data sets to test generalizability" @tang2021. External tests that do exist mostly
come from clinical groups working with whole volumes. Tang et al. trained a DME classifier on Cirrus, Spectralis and
Triton scans from one centre and kept an AUROC above 0.906 on seven external datasets @tang2021. Zhang et al.
trained on private Spectralis volumes and tested on OCTDL and NEH, and every method they compared lost accuracy on
OCTDL @zhang2025. For B-scan classifiers trained on the public datasets, tests on another dataset are rare. Chutani
and Yalavarthy trained on Kermany and tested on PAIMA, two Spectralis datasets that differ in protocol, site and
labelling, and a ResNet-18 reached 77.8% accuracy on normal, drusen and CNV @chutani2026. An independent benchmark
found ImageNet-pretrained compact models as accurate as RETFound for OCT classification, but within single
datasets, and names external validation as missing @isztl2026.

Methods that adapt a model or its inputs to a new device, by image translation @romobucheli2020 @chen2023, noise
transfer @koch2022svdna or test-time adaptation @hucke2026, have mostly been developed for segmentation.
Self-supervised pretraining on unlabelled OCT is another route @jannat2025selfnet @zhou2023retfound
@morano2025mirage. We do neither: we study the common baseline, an ImageNet-pretrained ResNet-50, and ask where its
gap on an unseen source comes from.

== Shortcuts <sec-prior-shortcut>

Geirhos et al. define _shortcuts_ as "decision rules that perform well on standard benchmarks but fail to transfer
to more challenging testing conditions" @geirhos2020shortcut. In medical imaging the shortcut is often the source
itself: a network can tell which hospital a chest radiograph comes from, and where disease prevalence differs
between hospitals, the hospital predicts the label @zech2018. In OCT, image size has been treated as a nuisance to
remove: Rasti et al. resize every B-scan to 496×512 to give their two datasets "a unique field of view"
@rasti2018. Whether OCT classifiers use image size as a label cue has not been tested; @sec-shortcut tests it.
