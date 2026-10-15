#import "../lib.typ": *

= Discussion <sec-discussion>

*Accuracy on a seen source overstates accuracy on a new one.* All four sources score 90–99 macro-F1 when the model
has seen them, and every one of them loses accuracy beyond noise when it has not: 4–6 points for OCTDL,
PAIMA and Duke, and about 15 for Kermany on its test split. A single-dataset benchmark, the most common way OCT
classifiers are evaluated, cannot show this. A
different device is not needed for the drop: three of the four sources come from the same manufacturer
(Heidelberg), and OCTDL, the only other vendor, has the smallest gap. Site, population, protocol and labelling are enough.

*Pooling datasets can import their shortcuts.* The largest gap, Kermany's, is partly caused by a property of
_another_ dataset: in Duke, image size gives the label away, and models trained with Duke apply that rule to
Kermany. The shortcut is invisible on Duke itself, where it is always right, and the fused model scores well
on every source. It only shows when the model meets a source where the rule is wrong. The rule is an artifact of
acquisition settings on one device (@sec-data), so it would not be caught by matching devices across datasets. A
simple check would have
caught it before training: tabulate image size (and other acquisition properties) against the label in each
source, as in @fig-size-class. The route matters too: here the cue reaches the model through the letterbox bands that
fitting a wide scan into a square canvas creates, so any preprocessing that leaves shape-dependent padding can
carry it.

*Removing a shortcut is not free.* Square crop removes the size cue and recovers much of Kermany's gap, but it
costs the other unseen sources in different ways: OCTDL loses normal recall (normal images called DME), Duke loses
DME recall, and PAIMA loses ranking quality. One possible cause for OCTDL and Duke is lost tissue: the square window keeps nearly the whole width of a
512×496 scan, but only about two thirds of a 768×496 scan and about a third of a typical 880×320 OCTDL scan, and
almost every OCTDL scan is that elongated. About half of Kermany is 512×496, where little tissue is lost. We have not
tested this for OCTDL or Duke. It does not explain PAIMA: averaging predictions over several square windows that
together cover the whole PAIMA scan leaves its AUROC unchanged. We found no single preprocessing that is best
on every unseen source.

*DME is the weak point, but not the whole story.* Once the size cue is accounted for, most of the remaining gap is DME
recall. DME is also the class with the fewest training examples from the fewest sources: PAIMA has none, so a model
with Kermany held out learns DME from 122 patients or eyes in two sources, one per vendor. A fused model trained
without Kermany's DME separates "unseen source" from "unseen DME". On Kermany's test split, missing in-device DME
alone costs about #zd-share()% as much as the held-out gap, all through DME recall, and at least as much DME recall
as the held-out model loses. The held-out model's other loss, normal recall, comes from the source being unseen, as
the size cue predicts. Whether DME from more sources would close the DME part is still untested.

*Dry AMD is a weak point for one source.* AMD merges a dry and a wet stage, and the sources mix them very
differently (@tab-data). PAIMA, where most AMD is dry, misses its drusen scans far more often than its wet ones,
both when seen and when held out, while Kermany's drusen scans are recognised almost perfectly by every model
(@sec-seen). The dry stage is therefore not hard in general; something about PAIMA's drusen scans is, and our data
cannot tell what. A single AMD score hides this difference, which is why we report the two stages separately.

== Limitations <sec-limits>

- *One seed.* Every table reports seed 20260913. Two extra seeds exist for the LOSO Kermany and PAIMA models:
  held-out Kermany scores 73.8–76.4 (letterbox) and 84.8–86.6 (square crop) across the three seeds, so its gain
  holds at every seed; held-out PAIMA with square crop ranges 75.6–85.2 against 86.4–87.7 for letterbox, so its
  loss is real but its size is unstable. #todo[load from a tracked file]
- *Small Duke.* Duke has 45 eyes, and Duke held out is a single LOSO model. Its intervals are wide, and we use it
  only to support directions seen on the other sources.
- *Unexplained PAIMA loss under square crop.* Shrinking the test tissue toward the scale seen in training lowers
  PAIMA's AUROC instead of recovering it, so tissue scale does not explain the loss (enlarging was not tested).
  Nor does AMD stage: with seed 20260913 the AUROC loss is in separating normal from dry AMD
  (#pa-dl(20260913, "dry")) rather than from wet AMD (#pa-dl(20260913, "wet")), but with the other two seeds both
  stages lose about equally (dry #pa-dl(20260914, "dry") and wet #pa-dl(20260914, "wet") with seed 20260914, dry
  #pa-dl(20260915, "dry") and wet #pa-dl(20260915, "wet") with seed 20260915).
- *Hardware and batch size.* The fused letterbox models were trained on a different GPU from every other model,
  and no seed noise floor exists for the fused models. Every comparison with them therefore also crosses hardware:
  the unseen-source gap, fused vs per-dataset, and the two fused arms. The LOSO models also train with batch size
  12 instead of 16, so the gap compares models that differ in batch size as well as in training data. Kermany's
  gap is far larger than such effects; differences of a few points against the fused models are less certain.
- *Zero-DME control.* The fused model without Kermany DME is one seed trained on a different GPU from the fused
  reference, and removing Kermany's DME also changes Kermany's share of each training batch. Its Duke result rests on
  45 eyes and mostly two folds.
- *Source is not device.* Three of the four sources share one manufacturer, so our gaps measure a shift between datasets
  (site, population, acquisition protocol, labelling and label unit; Duke labels whole eyes, the others single
  B-scans), not a shift between devices. Only OCTDL adds a vendor change, and a single source cannot separate it
  from the rest.
- *One architecture.* All results are for ImageNet-pretrained ResNet-50.

== Future work

Two directions follow from the remaining gap. Augmentation that mimics other sources' appearance (speckle,
contrast, resolution) would target the part of the gap that size and DME do not explain. OCT-specific pretraining,
for example self-supervised pretraining on unlabelled scans from all sources including the unseen one, would change
the features themselves, which our diagnostics point to as the limiting factor.

= Conclusion

#todo[3–4 sentences once results are final]
