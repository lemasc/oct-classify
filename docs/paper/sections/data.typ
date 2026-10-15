#import "../lib.typ": *

= Data <sec-data>

We use four public retinal OCT datasets and treat each as one _source_. The sources differ in clinical site,
population, acquisition protocol and labelling, but mostly not in device. Kermany @kermany2018 (five centres in the
United States and China), Duke @srinivasan2014 (three centres in the United States) and PAIMA
@sotoudehpaima2022 (one hospital in Iran) were all acquired with Heidelberg Engineering spectral-domain OCT (Spectralis for Kermany and Duke; the PAIMA
paper names only the manufacturer); only
OCTDL @kulyabin2024octdl comes from another vendor (Optovue Avanti RTVue XR). An unseen source is therefore an
unseen dataset and, only for OCTDL, also an unseen device. @fig-examples shows one scan per source and class.

#figure(
  image("../figures/examples.png", width: 100%),
  caption: [One B-scan per source and class, drawn at the same pixel scale (not the same physical scale) and
    labelled with its size in pixels. Dry and wet AMD are merged into one class for training.],
) <fig-examples>

#figure(
  {
    set text(size: 9.5pt)
    table(
      columns: 9,
      align: (left, right, right, right, right, right, right, right, left),
      stroke: (x, y) => if y <= 1 { (bottom: 0.5pt) },
      table.header(
        table.cell(rowspan: 2)[Source], table.cell(rowspan: 2)[Images], table.cell(rowspan: 2)[Groups],
        table.cell(rowspan: 2)[Normal], table.cell(rowspan: 2)[AMD], table.cell(colspan: 2, align: center)[AMD stage],
        table.cell(rowspan: 2)[DME], table.cell(rowspan: 2)[Split],
        [dry], [wet],
      ),
      [Kermany], [101,217], [5,419 patients], [50,263], [39,857], [20%], [80%], [11,097], [supplied test],
      [OCTDL], [1,709], [637 patients], [332], [1,230], [22%], [58%#super[a]], [147], [70/15/15],
      [PAIMA], [16,601], [437 patients], [8,521], [8,080], [60%], [40%], [--], [70/15/15],
      [Duke], [3,231], [45 eyes], [1,407], [723], [*100%*], [--], [1,101], [5-fold CV],
    )
  },
  caption: [The four sources after deduplication. Groups are the unit kept within one split and resampled by the
    bootstrap; Duke labels whole eyes, the others single B-scans. _AMD stage_ is the share of each source's AMD
    images that are dry (drusen) or wet (CNV/MNV); both are one class in training. #super[a]The other 20% of OCTDL's
    AMD images are intermediate stage with suspected MNV. #todo[generate from manifests rather than typing]],
) <tab-data>

*Taxonomy.* We merge the source labels into three classes: _normal_; _AMD_ (age-related macular degeneration);
and _DME_ (diabetic macular edema). AMD merges two stages that look different on a B-scan (@fig-examples): dry AMD,
with drusen under the retina, and wet AMD, with choroidal neovascularization (CNV, called MNV in OCTDL). The
sources mix the two stages very differently (@tab-data): dry AMD is a fifth of Kermany's and OCTDL's AMD images,
60% of PAIMA's and all of Duke's, whose AMD eyes are dry only @srinivasan2014. PAIMA contains only normal and AMD
scans. Its lack of DME is structural, not a negative label: PAIMA images never contribute to the DME output during
training (masked loss, @sec-method), and PAIMA is scored on normal vs AMD only.

*Label unit.* Duke labels a whole eye (volume) and propagates the diagnosis to every B-scan; the other sources label
individual B-scans. For comparability we score every source per B-scan, but Duke's 45 eyes are the unit of
splitting and of bootstrap resampling, so its intervals reflect 45 independent eyes, not 3,231 images.#footnote[Duke can also be scored per eye, by averaging each eye's B-scan probabilities. At that level the
  per-dataset, fused, fused square-crop, LOSO and LOSO square-crop models misclassify #eye-wrong("per-dataset"),
  #eye-wrong("fused-fill"), #eye-wrong("fused-sq"), #eye-wrong("loso-fill") and #eye-wrong("loso-sq") of the
  #EYE.n_eyes eyes (macro-F1 #eye-f1("per-dataset"), #eye-f1("fused-fill"), #eye-f1("fused-sq"), #eye-f1("loso-fill")
  and #eye-f1("loso-sq")). We keep B-scans as the main unit so that all four sources are scored alike.]

*Deduplication and splits.* Exact duplicates and perceptual-hash (distance zero) near-duplicates across patients are
removed or linked into one split unit, which removes 7,771 same-label copies from Kermany alone. Splits are made at
the patient (or eye/volume) level so that no patient appears in more than one split. Kermany keeps its supplied test
split and divides the rest 85/15 into train and validation; OCTDL and PAIMA are split 70/15/15 into train, validation and test; Duke, with only 45 eyes, uses five-fold
cross-validation (9 test eyes per fold, 3 per class).

*Image geometry.* The sources differ in image size (@fig-size-class). PAIMA is all 768×496; Kermany has five main
sizes (512×496, 768×496, 1024×496, 1536×496 and 512×512, the last DME only); OCTDL images are cropped to varying
sizes, mostly around 880×320. Duke has three sizes, and two of them carry a single label: every 768×496 Duke image
is DME and every 1024×496 image is AMD. The link comes from scan settings on the same device, not from the
disease: in Duke's acquisition protocol @srinivasan2014, all 15 normal volumes have 512 A-scans per B-scan, 6 of the
15 AMD volumes have 1,024 and 9 of the 15 DME volumes have 768. A model trained on Duke can therefore learn image
size as a label cue; @sec-shortcut shows that it does.

#figure(
  kind: image,
  fig("size-class"),
  caption: [Class mix at each image size. In Duke, every 768×496 image is DME and every 1024×496 image is AMD;
    in Kermany, every 512×512 image is DME. OCTDL images are cropped to many different sizes and are shown as one
    row; 23 Kermany images of rare sizes are omitted.],
) <fig-size-class>
