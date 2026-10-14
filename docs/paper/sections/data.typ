#import "../lib.typ": *

= Data <sec-data>

We use four public retinal OCT datasets, each acquired with different devices and protocols. We treat each
dataset as one _source_ and, because each comes from a distinct acquisition setup, as a proxy for one device.
#todo[confirm device/vendor per source from the dataset papers; Kermany, Duke and PAIMA are Spectralis-like 496-row
scans, OCTDL is not — check before claiming "different devices" for every pair]

#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, left),
    table.header[Source][Images][Groups][Normal / AMD / DME][Label unit][Split],
    [Kermany], [101,217], [5,419 patients], [50,263 / 39,857 / 11,097], [image], [supplied test; train 85/15],
    [OCTDL], [1,709], [637 patients], [332 / 1,230 / 147], [image], [70/15/15 by patient],
    [PAIMA], [16,601], [437 patients], [8,521 / 8,080 / --], [image], [70/15/15 by patient],
    [Duke], [3,231], [45 eyes], [1,407 / 723 / 1,101], [eye], [5-fold CV by eye],
  ),
  caption: [The four sources after deduplication. Groups are the unit kept within one split and resampled by the
    bootstrap. PAIMA has no DME images. #todo[generate from manifests rather than typing]],
) <tab-data>

*Taxonomy.* We merge the source labels into three classes: _normal_; _AMD_ (age-related macular degeneration,
including its CNV and drusen subtypes); and _DME_ (diabetic macular edema). PAIMA contains only normal and AMD
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
split; OCTDL and PAIMA are split 70/15/15 into train, validation and test; Duke, with only 45 eyes, uses five-fold
cross-validation (9 test eyes per fold, 3 per class).

*Image geometry.* The sources differ in image size (@fig-size-class). PAIMA is all 768×496; Kermany has five main
sizes (512×496, 768×496, 1024×496, 1536×496 and 512×512, the last DME only); OCTDL images are cropped to varying
sizes, mostly around 880×320. Duke has three sizes, and two of them carry a single label: every 768×496 Duke image
is DME and every 1024×496 image is AMD. A model
trained on Duke can therefore learn image size as a label cue; @sec-shortcut shows that it does.

#figure(
  kind: image,
  fig("size-class"),
  caption: [Class mix at each image size. In Duke, every 768×496 image is DME and every 1024×496 image is AMD;
    in Kermany, every 512×512 image is DME. OCTDL images are cropped to many different sizes and are shown as one
    row; 23 Kermany images of rare sizes are omitted.],
) <fig-size-class>
