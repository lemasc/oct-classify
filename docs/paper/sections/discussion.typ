#import "../lib.typ": *

= Discussion <sec-discussion>

*Accuracy on a seen source overstates accuracy on a new one.* Every source loses accuracy beyond noise when the
model has not seen it (@tab-unseen). A single-dataset benchmark, the most common way OCT classifiers are evaluated,
cannot show this. A
different device is not needed for the drop: three of the four sources come from the same manufacturer
(Heidelberg), and OCTDL, the only other vendor, does not have a larger gap than they do. Site, population, protocol and labelling are enough.

*Pooling datasets can import their shortcuts.* The largest gap, Kermany's, is partly caused by a property of
_another_ dataset: in Duke, image size gives the label away, and models trained with Duke apply that rule to
Kermany. The shortcut is invisible on Duke itself, where it is always right, and the fused model scores well
on every source. It only shows when the model meets a source where the rule is wrong. Chest radiographs show the same pattern (@sec-prior-shortcut): testing COVID-19 models on an external hospital
alone did not reveal that they read markers and patient position rather than the lungs @degrave2021. Here, too, the cause showed only
when we changed the inputs (@sec-shortcut). The rule is an artifact of
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
almost every OCTDL scan is that elongated. About #whole(size-share("kermany", "512x496"))% of Kermany is 512×496, where little tissue is lost. We have not
tested this for OCTDL or Duke. It does not explain PAIMA, whose loss stays unexplained (@sec-limits). We found no single preprocessing that is best
on every unseen source.

*DME is the weak point for some sources, not all.* For OCTDL and Duke, DME recall is the largest per-class loss when the
source is held out (@fig-recall-change), though with wide intervals. Kermany's largest loss is normal recall, and PAIMA, which has
no DME, loses AMD recall. DME is also the class with the fewest training examples from the fewest sources: PAIMA has none, so a model
with Kermany held out learns DME from only two sources, one per vendor. Removing Kermany's DME from a fused
model (@tab-zero-dme) reproduces about #zd-share()% of Kermany's held-out gap, so missing DME is a real but partial
cause; the rest, a loss in normal recall, comes from the source being unseen, as the size cue predicts. Whether DME
from more sources would close the DME part is still untested.

*Dry AMD is hard on some sources, not all.* AMD merges a dry and a wet stage, and the sources mix them very
differently (@tab-data). PAIMA, where most AMD is dry, misses its drusen scans far more often than its wet ones,
both when seen and when held out, and OCTDL shows the same on a few dozen scans, while Kermany's drusen scans are
recognised almost perfectly when seen and lose only about #whole(AS.at("test|kermany|all").deltas.dry.at("loso-fill - fused-fill").point) point when held out (@tab-recall, @fig-recall-change). The dry stage is therefore not hard in general; something
about PAIMA's drusen scans is, and our data cannot tell what. A single AMD score hides this difference, which is why we report the two stages separately.

== Limitations <sec-limits>

- *One seed.* Every table reports seed 20260913. Two extra seeds exist for the LOSO Kermany and PAIMA models, in
  both arms. On the full source, held-out Kermany scores #span1(seed-vals("kermany", "loso-fill")) macro-F1
  across the three seeds, and square crop's gain holds at every seed (#fmt(calc.min(..seed-gain("kermany")), signed: true) to
  #fmt(calc.max(..seed-gain("kermany")), signed: true)). Held-out PAIMA
  with square crop loses against letterbox at every seed, but by #span1(seed-gain("paima")) points (AUROC
  #seed-vals("paima", "loso-sq", name: "macro_auroc").map(x => fmt(x)).join(", ") against
  #span1(seed-vals("paima", "loso-fill", name: "macro_auroc"))), so the loss is real but its size is unstable.
- *Small Duke.* Duke has 45 eyes, and Duke held out is a single LOSO model. Its intervals are wide, and we use it
  only to support directions seen on the other sources.
- *Unexplained PAIMA loss under square crop.* Tissue scale does not explain it: shrinking the test tissue toward the scale seen in training lowers PAIMA's
  AUROC instead of recovering it (enlarging was not tested), and averaging predictions over several square windows
  that together cover the whole scan moves its AUROC only from #fmt(MW.paima.all.mean.auroc.at(0) / 100) to
  #fmt(MW.paima.all.mean.auroc.at(1) / 100).
  AMD stage does not explain it either: with seed 20260913 the AUROC loss is mostly in separating normal from dry AMD, but with
  the other two seeds both stages lose about equally.
- *Hardware and batch size.* The fused letterbox models were trained on a different GPU from every other model,
  and no seed noise floor exists for the fused models. Every comparison with them therefore also crosses hardware:
  the unseen-source gap, fused vs per-dataset, and the two fused arms. The LOSO models also train with batch size
  12 instead of 16, so the gap compares models that differ in batch size as well as in training data. Kermany's
  gap is far larger than such effects; differences of a few points against the fused models are less certain.
- *Zero-DME control.* The fused model without Kermany DME is one seed trained on a different GPU from the fused
  reference, and removing Kermany's DME also changes Kermany's share of each training batch.
- *Source is not device.* Three of the four sources share one manufacturer, so our gaps measure a shift between datasets
  (site, population, acquisition protocol, labelling and label unit; Duke labels whole eyes, the others single
  B-scans), not a shift between devices. Only OCTDL adds a vendor change, and a single source cannot separate it
  from the rest.
- *One architecture.* All results are for ImageNet-pretrained ResNet-50.

== Future work

Two directions follow from the remaining gap (both surveyed in @sec-prior-shift). Augmentation that mimics other
sources' appearance (speckle, contrast, resolution), as built-in speckle noise does @chutani2026, would target the part
of the gap that size and DME do not explain. OCT-specific pretraining, for example self-supervised pretraining on
unlabelled scans from all sources including the unseen one, would change the features themselves, which our
diagnostics point to as the limiting factor.

= Conclusion

A ResNet-50 that scores #span(("kermany", "octdl", "paima", "duke").map(s => metric(k("seen", s), "per-dataset").point)) macro-F1 on OCT datasets it has seen loses
#span(("octdl", "paima", "duke", "kermany").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) points on one it has not, although three of the four come from the same manufacturer.
Part of that loss comes from the training data rather than the new dataset: pooling let a label-predicting image
size in Duke become a rule that fails on Kermany. Removing the cue recovers much of Kermany's gap but is not a
general fix, and what remains differs by source: DME recall for OCTDL and Duke, normal recall for Kermany and AMD recall
for PAIMA. Before pooling OCT datasets,
compare scan settings against labels, and test on a held-out source, not only a held-out split.
