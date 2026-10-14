#import "../lib.typ": *

= Discussion <sec-discussion>

*Accuracy on a seen device overstates accuracy on a new one.* All four sources score 90–99 macro-F1 when the model
has seen their device, and every one of them loses accuracy beyond noise when it has not: 4–6 points for OCTDL,
PAIMA and Duke, and about 15 for Kermany on its test split. A single-dataset benchmark, the most common way OCT
classifiers are evaluated, cannot show this.

*Pooling datasets can import their shortcuts.* The largest gap, Kermany's, is partly caused by a property of
_another_ dataset: in Duke, image size gives the label away, and models trained with Duke apply that rule to
Kermany. The shortcut is invisible on Duke itself, where it is always right, and the fused model scores well
on every source. It only shows when the model meets a device where the rule is wrong. A simple check would have
caught it before training: tabulate image size (and other acquisition properties) against the label in each
source, as in @fig-size-class. The route matters too: here the cue reaches the model through the letterbox bands that
fitting a wide scan into a square canvas creates, so any preprocessing that leaves shape-dependent padding can
carry it.

*Removing a shortcut is not free.* Square crop removes the size cue and recovers much of Kermany's gap, but it
costs the other unseen devices in different ways: OCTDL loses normal recall (normal images called DME), Duke loses
DME recall, and PAIMA loses ranking quality. One possible cause for OCTDL and Duke is lost tissue: the square window keeps nearly the whole width of a
512×496 scan, but only about two thirds of a 768×496 scan and about a third of a typical 880×320 OCTDL scan, and
almost every OCTDL scan is that elongated. About half of Kermany is 512×496, where little tissue is lost. We have not
tested this for OCTDL or Duke. It does not explain PAIMA: averaging predictions over several square windows that
together cover the whole PAIMA scan leaves its AUROC unchanged. We found no single preprocessing that is best
on every unseen device.

*DME is the weak point.* Once the size cue is accounted for, most of the remaining gap is DME recall. DME is also
the class with the fewest training examples from the fewest devices: PAIMA has none, so a model with Kermany held
out learns DME from 122 patients on two devices. Whether more devices' DME would close the gap is untested; a fused
model trained without one device's DME would separate "unseen device" from "unseen DME".

== Limitations <sec-limits>

- *One seed.* Every table reports seed 20260913. Two extra seeds exist for the LOSO Kermany and PAIMA models:
  held-out Kermany scores 73.8–76.4 (fill-aware) and 84.8–86.6 (square crop) across the three seeds, so its gain
  holds at every seed; held-out PAIMA with square crop ranges 75.6–85.2 against 86.4–87.7 for fill-aware, so its
  loss is real but its size is unstable. #todo[load from a tracked file]
- *Small Duke.* Duke has 45 eyes, and Duke held out is a single LOSO model. Its intervals are wide, and we use it
  only to support directions seen on the other sources.
- *Unexplained PAIMA loss under square crop.* Shrinking the test tissue toward the scale seen in training lowers
  PAIMA's AUROC instead of recovering it, so tissue scale does not explain the loss (enlarging was not tested).
- *Hardware.* The fused fill-aware models were trained on a different GPU from the fused square-crop models, and no
  seed noise floor exists for the fused models. Differences of a few points between the fused arms are therefore
  not confirmed.
- *Dataset is not device.* Each source differs from the others in device, but also in population, labelling
  protocol and label unit (Duke labels whole eyes, the others single B-scans). The gaps we measure combine all of
  these.
- *One architecture.* All results are for ImageNet-pretrained ResNet-50.

== Future work

Two directions follow from the remaining gap. Augmentation that mimics other devices' appearance (speckle,
contrast, resolution) would target the part of the gap that size and DME do not explain. OCT-specific pretraining,
for example self-supervised pretraining on unlabelled scans from all devices including the unseen one, would change
the features themselves, which our diagnostics point to as the limiting factor.

= Conclusion

#todo[3–4 sentences once results are final]
