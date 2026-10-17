#import "../lib.typ": *

#let split-name = (kermany: "supplied test", octdl: "70/15/15", paima: "70/15/15", duke: "5-fold CV")
= Data <sec-data>

We use four public retinal OCT datasets and treat each as one _source_. The sources differ in clinical site,
population, acquisition protocol and labelling, but mostly not in device. Kermany @kermany2018 (five centres in
the United States and China), Duke @srinivasan2014 (three centres in the United States) and PAIMA
@sotoudehpaima2022 (one hospital in Iran) were all acquired with Heidelberg Engineering spectral-domain OCT (Spectralis for Kermany and Duke; the PAIMA
paper names only the manufacturer); only
OCTDL @kulyabin2024octdl comes from another vendor (Optovue Avanti RTVue XR). An unseen source is therefore an
unseen dataset and, only for OCTDL, also an unseen device. @fig-examples shows one scan per source and class.

#figure(
  image("../figures/examples.png", width: 100%),
  caption: [One B-scan per source and class, drawn at the same pixel scale (not the same physical scale)
    and labelled with its size in pixels.],
) <fig-examples>

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 9,
      align: (left, right, right, right, right, right, right, right, left),
      table.header(
        table.cell(rowspan: 2)[Source], table.cell(rowspan: 2)[Images], table.cell(rowspan: 2)[Groups],
        table.cell(rowspan: 2)[Normal], table.cell(rowspan: 2)[AMD], table.cell(colspan: 2, align: center)[AMD stage],
        table.cell(rowspan: 2)[DME], table.cell(rowspan: 2)[Split],
        [dry], [wet],
        table.hline(),
      ),
      ..srcs
        .map(s => {
          let d = DS.at(s)
          let stage(st) = if stage-share(s, st) == none [--] else if s == "duke" [*100%*] else [#stage-share(s, st)%]
          (
            [#source-name.at(s)],
            [#num(d.n_images)],
            [#num(d.n_groups) #if d.label_unit == "eye" [eyes] else [patients]],
            [#num(d.classes.normal)],
            [#num(d.classes.amd)],
            stage("dry"),
            if s == "octdl" [#stage-share(s, "wet")%#super[a]] else { stage("wet") },
            if d.classes.dme == 0 [--] else [#num(d.classes.dme)],
            [#split-name.at(s)],
          )
        })
        .flatten(),
      table.hline(),
    )
  },
  caption: [The four sources after deduplication, with class counts, AMD stage mix and split.
    #super[a]The other #stage-share("octdl", "intermediate")% of OCTDL's AMD images are intermediate stage with suspected MNV.],
) <tab-data>

*Taxonomy.* We merge the source labels into three classes: _normal_; _AMD_ (age-related macular degeneration);
and _DME_ (diabetic macular edema). AMD merges two stages that look different on a B-scan (@sec-diseases, @fig-examples):
dry AMD, with drusen under the retina, and wet AMD, with choroidal neovascularization (CNV, called MNV in OCTDL). The
sources mix the two stages very differently (@tab-data): dry AMD is a fifth of Kermany's and OCTDL's AMD images,
#stage-share("paima", "dry")% of PAIMA's and all of Duke's, whose AMD eyes are dry only @srinivasan2014. PAIMA contains only normal and AMD
scans; its lack of DME is structural, not a negative label (masked loss, @sec-method), and it is scored on normal
vs AMD only.

*Label unit.* Duke labels a whole eye (volume) and propagates the diagnosis to every B-scan; the other sources label
individual B-scans. For comparability we score every source per B-scan, but Duke's #DS.duke.n_groups eyes are the unit of
splitting and of bootstrap resampling, so its intervals reflect #DS.duke.n_groups independent eyes, not #num(DS.duke.n_images) images.#footnote[Duke can also be scored per eye, by averaging each eye's B-scan probabilities. At that level the
  per-dataset, fused, fused square-crop, LOSO and LOSO square-crop models misclassify #eye-wrong("per-dataset"),
  #eye-wrong("fused-fill"), #eye-wrong("fused-sq"), #eye-wrong("loso-fill") and #eye-wrong("loso-sq") of the
  #EYE.n_eyes eyes (macro-F1 #eye-f1("per-dataset"), #eye-f1("fused-fill"), #eye-f1("fused-sq"), #eye-f1("loso-fill")
  and #eye-f1("loso-sq")). We keep B-scans as the main unit so that all four sources are scored alike.]

*Deduplication and splits.* Exact duplicates and perceptual-hash (distance zero) near-duplicates across patients are
removed or linked into one split unit, which removes #num(DS.kermany.dedup.same_label_copies_removed) same-label copies from Kermany alone. Splits are made at
the patient (or eye/volume) level so that no patient appears in more than one split; splitting OCT data by image
instead inflates test accuracy by 5–30% @tampu2022. Kermany keeps its supplied test split and divides the rest 85/15 into train and validation; OCTDL and PAIMA are split 70/15/15 into train, validation and test; Duke, with only #DS.duke.n_groups eyes, uses five-fold
cross-validation (#DS.duke.test_eyes_per_fold.first() test eyes per fold, #DS.duke.test_eyes_per_class_per_fold.first() per class).

*Image geometry.* The sources differ in image size (@fig-size-class). PAIMA is all 768×496; Kermany has five main
sizes (512×496, 768×496, 1024×496, 1536×496 and 512×512, the last DME only); OCTDL images are cropped to varying
sizes, mostly around 880×320. Duke has three sizes, and two of them carry a single label: every 768×496 Duke image
is DME and every 1024×496 image is AMD. The link comes from scan settings on the same device, not from the
disease: in Duke's acquisition protocol @srinivasan2014, all #duke-eyes("normal", size: "512x496") normal volumes have 512 A-scans per B-scan, #duke-eyes("amd", size: "1024x496") of the
#duke-eyes("amd") AMD volumes have 1,024 and #duke-eyes("dme", size: "768x496") of the #duke-eyes("dme") DME volumes have 768. A model trained on Duke can therefore learn image
size as a label cue; @sec-shortcut tests whether it does.

#figure(
  kind: image,
  fig("size-class"),
  caption: [Class mix at each image size. OCTDL images are cropped to many different sizes and are shown as one
    row; #kermany-rare Kermany images of rare sizes are omitted.],
) <fig-size-class>
