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
@bogunovic2019retouch. Vendors also differ in image quality: Spectralis averages several B-scans of the same
location to reduce speckle noise, at the cost of fewer B-scans per volume @bogunovic2019retouch. On one device,
Duke's Spectralis B-scans range from 512 to 1,024 A-scans (@sec-data). Image size is therefore a property of the
acquisition, not of the eye.

== AMD and DME on OCT <sec-diseases>

*Age-related macular degeneration (AMD)* was projected to affect 196 million people in 2020 and 288 million by 2040
@wong2014amd. Its hallmark is _drusen_, deposits of extracellular material between the
retinal pigment epithelium (RPE) and Bruch's membrane @sotoudehpaima2022, which appear on a B-scan as bumps under
the retina (@fig-examples). The clinical classification grades AMD by drusen size and late lesions: medium drusen
(63–125 µm) mark early AMD, large drusen, or pigment changes with medium drusen, intermediate AMD, and neovascular AMD or geographic
atrophy late AMD @ferris2013. In neovascular ("wet") AMD, new vessels grow from the choroid, a process called
choroidal neovascularization (CNV), and leak fluid into and under the retina @kermany2018 @sotoudehpaima2022. Our
sources call the drusen stage "dry AMD" (Duke) or "drusen", and we follow them. The clinical classification instead
reserves "dry" for geographic atrophy @ferris2013, so in this paper dry AMD means early and intermediate AMD with
drusen, not atrophy.

*Diabetic macular edema (DME)* is a complication of diabetes in which fluid builds up in the central retina
@kermany2018. On a B-scan the retina is thickened and holds dark, fluid-filled spaces (@fig-examples)
@kermany2018. A meta-analysis of OCT-based studies puts its prevalence at 5.47% of people with diabetes
@im2022dme. Fluid also appears in wet AMD @kermany2018, so the presence of fluid alone does not separate the two
diseases. AMD and diabetic retinopathy, of which DME is a form, are leading causes of blindness in older and
working-age Americans respectively @srinivasan2014.

== Classifying OCT scans <sec-prior-classification>

Early classifiers used hand-crafted features. Srinivasan et al. built Duke's dataset and classified its 45 eyes
with histograms of oriented gradients and a support vector machine, recognising every AMD and DME eye and 13 of 15
normal eyes @srinivasan2014. Convolutional networks pretrained on ImageNet then took over: Lee et al. separated
normal from AMD on about 100,000 Heidelberg B-scans with VGG16 (AUROC 92.8% per image, 97.5% per patient)
@lee2017, Karri et al. fine-tuned GoogLeNet on Duke @karri2017, and Kermany et al. reached 96.6% accuracy on four
classes (normal, drusen, CNV, DME) after training on 108,312 B-scans @kermany2018. Later work added multi-scale
ensembles @rasti2018, lesion attention @fang2019 and multi-scale feature pyramids @sotoudehpaima2022, and the OCTDL
release reports VGG16 and ResNet-50 baselines @kulyabin2024octdl. Reviews survey this work for OCT and other
retinal imaging @muchuchuti2023 @abdi2025, and name dataset bias and model generalization among the main open
problems @abdi2025.

Two features of this literature matter here. First, models are nearly always scored on test data from the
dataset they were trained on. Rasti et al. and Fang et al. each use two datasets but train and test within each
one @rasti2018 @fang2019.#footnote[Both use a 148-eye dataset from Noor Eye Hospital, Tehran, which they call NEH.
  It is not PAIMA, which comes from the same hospital @sotoudehpaima2022.] Second, how a dataset is split matters
as much as the model. Tampu et al. showed that splitting OCT data by image instead of by patient inflates test
accuracy by 5–30% across three datasets, and that the train and test sets of Kermany version 2 share
patients and images @tampu2022. For version 3, which we use, the supplied split scored only 0.04 MCC above a split by patient @tampu2022. We split
every source by patient and remove near-duplicates (@sec-data).

== Generalization across datasets and devices <sec-prior-shift>

Few OCT studies test on a source that was not used in training. De Fauw et al. trained a segmentation network and
a separate classifier, and showed that the tissue segmentations act as a device-independent representation:
referral accuracy is kept when the segmentations come from a different type of device @defauw2018. The RETOUCH
challenge was built to measure how fluid segmentation varies across vendors @bogunovic2019retouch. For
classification across datasets, the closest work is OCT-SelfNet @jannat2025selfnet and its multi-class successor
@jannat2026multiselfnet, which pretrain a transformer without labels on images from several datasets. In their
experiments a supervised ResNet-50 trained on Duke reaches only 0.59 AUROC on a test set from another dataset,
against 0.90 for their pretrained model @jannat2026multiselfnet. Their pretraining includes images from the target
datasets and their Kermany split is by image @jannat2025selfnet, so their numbers are not comparable with ours. We
do not pretrain: we study their baseline, an ImageNet-pretrained ResNet-50, and ask where its gap comes from.
Foundation models such as RETFound, pretrained by masked autoencoding on 904,170 fundus photographs (one model) and
736,442 OCT B-scans (another) @zhou2023retfound, are a further option. On one OCT task, RETFound beat an ImageNet
ResNet-50 clearly only with 100 or fewer labelled scans; with 500 or more the two were similar @kuo2025retfound.

Our design follows domain generalization practice: each source is held out in turn, and checkpoints are chosen on
validation data from the training sources only. Under this model selection, plain training on the pooled data
(empirical risk minimization) matches the best dedicated domain-generalization methods on standard benchmarks
@gulrajani2021domainbed, so we study that baseline rather than a new method.

== Shortcuts and dataset bias <sec-prior-shortcut>

Datasets carry signatures of how they were made. Torralba and Efros showed that a classifier can tell which of 12
datasets an image comes from (39% accuracy against 8% by chance), and that models trained on one dataset lose
accuracy on another @torralba2011bias. Geirhos et al. call the resulting decision rules _shortcuts_: "decision
rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions"
@geirhos2020shortcut. In medical imaging the shortcut is often the source itself. A network can identify which
hospital system a chest radiograph comes from with 95.6–99.98% accuracy, and since pneumonia prevalence differed
between the hospitals (34.2% against about 1%), the hospital predicts the label @zech2018. Radiographs reveal the
scanner model (AUROC 1.00), and a hip fracture model whose AUROC is 0.78 falls to 0.52 on a test set balanced for
such patient and hospital variables @badgeley2019. COVID-19 models relied on laterality markers and patient position
rather than lung pathology, and testing on an external hospital alone did not reveal it @degrave2021.

In OCT, image size has been treated as a nuisance to remove rather than a possible shortcut: Rasti et al. resize
every B-scan to 496×512 to give their two datasets "a unique field of view" @rasti2018. We know of no study that
tests whether OCT classifiers use image size as a label cue. @sec-shortcut shows that they do when one source ties
a scan setting to the label, and that pooling sources carries the cue to sources where it is wrong.
