#import "../../lib.typ": *
#import "../../i18n/th.typ": source-name, split-name

= ผลลัพธ์ <sec-results>

== แหล่งข้อมูลที่เคยเห็น <sec-seen>

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header([Source], [Per-dataset], [Fused], [Fused, square crop], table.hline()),
    ..srcs.map(s => (
      [#source-name.at(s)],
      ci(k("seen", s), "per-dataset"),
      ci(k("seen", s), "fused-fill"),
      ci(k("seen", s), "fused-sq"),
    )).flatten(),
    table.hline(),
  ),
  caption: [ค่า Macro-F1 บนชุดข้อมูลทดสอบของแต่ละแหล่งข้อมูล พร้อมช่วงความเชื่อมั่นร้อยละ 95 จากการสุ่มตัวอย่างซ้ำด้วยวิธีบูตสแตรป ในระดับผู้ป่วย #super[a]สำหรับชุดข้อมูล PAIMA ได้รับการประเมินผลเฉพาะการจำแนกระหว่างภาพปกติ (normal) กับโรค AMD ส่วนชุดข้อมูล Duke เป็นการรวมผลลัพธ์จากทั้ง 5 ส่วน (5 folds) (รวมภาพ B-scan ทั้งหมด #num(DS.duke.n_images) ภาพ จากดวงตา #DS.duke.n_groups ดวง โดยดวงตาแต่ละดวงได้รับการประเมินด้วยโมเดลที่ไม่เคยใช้ดวงตาดังกล่าวในการฝึกสอน)],
) <tab-seen>

เมื่อทำการประเมินผลบนแหล่งข้อมูลที่ใช้ในการฝึกสอน โมเดล ResNet-50 สามารถทำค่า Macro-F1 ได้ระหว่างร้อยละ #pt(k("seen", "paima"), "per-dataset") ถึง #pt(k("seen", "kermany"), "per-dataset") (@tab-seen) การฝึกสอนโมเดลเดี่ยวบนทั้ง 4 แหล่งข้อมูลร่วมกัน (โมเดลแบบ fused) แทบไม่สูญเสียประสิทธิภาพ โดยเมื่อเปรียบเทียบกับโมเดลแบบรายชุดข้อมูล (per-dataset) ผลต่างประสิทธิภาพของโมเดลแบบ fused อยู่ในระดับสัญญาณรบกวน (noise level) บนชุดข้อมูล OCTDL (#dl(k("seen", "octdl"), "fused-fill - per-dataset")), PAIMA (#dl(k("seen", "paima"), "fused-fill - per-dataset")) และ Duke (#dl(k("seen", "duke"), "fused-fill - per-dataset")) มีเพียงชุดข้อมูล Kermany เท่านั้นที่มีคะแนนลดลงเล็กน้อยอย่างมีนัยสำคัญ (#dl(k("seen", "kermany"), "fused-fill - per-dataset")) ในทำนองเดียวกัน วิธีการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัส (square crop) ซึ่งขจัดสิ่งชี้แนะจากขนาดภาพออกไป (@sec-square) ไม่ส่งผลกระทบเชิงลบต่อแหล่งข้อมูลที่เคยเห็นเช่นกัน โดยไม่มีแหล่งข้อมูลใดที่มีคะแนนลดลงเกินระดับสัญญาณรบกวนเมื่อเปรียบเทียบกับโมเดลแบบ fused และสำหรับชุดข้อมูล PAIMA ยังคงมีคะแนนเพิ่มขึ้นอีก #dl(k("seen", "paima"), "fused-sq - fused-fill")

#let rc(m, c) = if ("recall_" + c) in m { fmt(m.at("recall_" + c).point) } else [--]
#let stage-rc(src, arm, st, m) = {
  let key = "test|" + src + "|" + view3(src)
  if src == "duke" { if st == "dry" { rc(m, "amd") } else [--] } // Duke's AMD is all dry
  else if arm == none [--]
  else { fmt(AS.at(key).stages.at(st).arms.at(arm).recall) }
}
#let recall-row(src, label, m, arm) = (
  [#label], rc(m, "normal"), rc(m, "amd"), stage-rc(src, arm, "dry", m), stage-rc(src, arm, "wet", m), rc(m, "dme"),
)

// Per-class recall table; `rows-of(src)` gives that source's rows (from `recall-row`).
#let recall-table(rows-of) = {
  set text(size: (9.5 / 11) * 1em)
  table(
    columns: 7,
    align: (left, left, right, right, right, right, right),
    table.header(
      table.cell(rowspan: 2)[Source], table.cell(rowspan: 2)[Model], table.cell(rowspan: 2)[Normal],
      table.cell(rowspan: 2)[AMD], table.cell(colspan: 2, align: center)[AMD Stage], table.cell(rowspan: 2)[DME],
      [dry], [wet],
      table.hline(),
    ),
    ..srcs.map(s => {
      let rows = rows-of(s)
      let rule = if s == srcs.last() { table.hline() } else { table.hline(stroke: 0.3pt + gray) }
      (table.cell(rowspan: rows.len())[#source-name.at(s)], ..rows.flatten(), rule)
    }).flatten(),
  )
}
#let seen-row(s, label, arm) = recall-row(s, label, R.at("seen|" + s + "|" + view3(s)).arms.at(arm).metrics, arm)

#figure(
  recall-table(s => (seen-row(s, "Per-dataset", "per-dataset"), seen-row(s, "Fused", "fused-fill"))),
  caption: [ค่า recall แยกตามคลาสบนชุดข้อมูลทดสอบของแต่ละแหล่งข้อมูล สำหรับโมเดลแบบ letterbox ที่เคยเห็นข้อมูล โดยแสดงค่าเฉลี่ยข้ามโมเดล],
) <tab-recall>

ค่า recall แยกตามคลาส (@tab-recall) เผยให้เห็นลักษณะของข้อผิดพลาดที่เกิดขึ้น โดยชุดข้อมูล PAIMA เป็นแหล่งข้อมูลที่ท้าทายที่สุดแม้จะเป็นแหล่งข้อมูลที่เคยเห็นก็ตาม (ได้ค่า Macro-F1 เท่ากับร้อยละ #pt(k("seen", "paima"), "fused-fill") สำหรับโมเดลแบบ fused) ซึ่งข้อผิดพลาดหลักเกิดขึ้นในคลาส AMD (มีค่า recall เพียงร้อยละ #pt(k("seen", "paima"), "fused-fill", name: "recall_amd")) และเมื่อพิจารณาเจาะลึกในกลุ่ม AMD พบว่าข้อผิดพลาดเกิดจากโรค AMD ชนิดแห้ง (dry AMD) เป็นหลัก กล่าวคือ โมเดลแบบ fused สามารถจำแนกภาพสแกนที่มีดรูเซน (drusen) ในชุดข้อมูล PAIMA ได้ถูกต้องเพียงร้อยละ #as-rec("test|paima|normal_amd", "dry", "fused-fill") เมื่อเทียบกับภาพถ่ายชนิดเปียก (wet AMD) ที่ทำได้ถึงร้อยละ #as-rec("test|paima|normal_amd", "wet", "fused-fill") โดยภาพถ่ายชนิดมีดรูเซนที่ทำนายผิดเกือบทั้งหมดถูกจำแนกว่าเป็นภาพปกติ (normal) โมเดลแบบรายชุดข้อมูล (per-dataset) ของชุดข้อมูล PAIMA ก็แสดงผลลัพธ์ในลักษณะเดียวกัน (ร้อยละ #as-rec("test|paima|normal_amd", "dry", "per-dataset") เทียบกับร้อยละ #as-rec("test|paima|normal_amd", "wet", "per-dataset")) ข้อจำกัดนี้จึงเกิดจากลักษณะเฉพาะของภาพสแกนชนิดดรูเซนในชุดข้อมูล PAIMA เอง มิได้เกิดจากกระบวนการรวมข้อมูล (fusing) ภาพถ่ายชนิดมีดรูเซนในชุดข้อมูล OCTDL ได้รับการจำแนกถูกต้องน้อยกว่าชนิดเปียกเช่นกัน (ร้อยละ #as-rec("test|octdl|all", "dry", "fused-fill") เทียบกับร้อยละ #as-rec("test|octdl|all", "wet", "fused-fill") สำหรับโมเดลแบบ fused) ทว่าในชุดข้อมูลทดสอบ (test split) ของ OCTDL มีภาพถ่ายชนิดมีดรูเซนเพียง #AS.at("test|octdl|all").stages.dry.n_images ภาพจากผู้ป่วย #AS.at("test|octdl|all").stages.dry.n_groups รายเท่านั้น สำหรับภาพถ่ายชนิดมีดรูเซนในชุดข้อมูล Kermany นั้น ทั้งสองโมเดลสามารถจำแนกได้อย่างถูกต้องอย่างน้อยร้อยละ #fmt(calc.min(..("per-dataset", "fused-fill").map(a => AS.at("test|kermany|all").stages.dry.arms.at(a).recall))) ของกรณีทั้งหมด

== แหล่งข้อมูลที่ไม่เคยเห็น (Unseen sources) <sec-unseen>

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 7,
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูลที่กันไว้],
        table.cell(colspan: 3, align: center)[Macro-F1 (test split)],
        table.cell(colspan: 2, align: center)[AUROC (test split)],
        table.cell(rowspan: 2)[Macro-F1 \ (full source)],
        [Seen], [Unseen], [Gap], [Seen], [Unseen],
        table.hline(),
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        pt(k("gap_test", s), "fused-fill"),
        pt(k("gap_test", s), "loso-fill"),
        dl(k("gap_test", s), "loso-fill - fused-fill"),
        pt(k("gap_test", s), "fused-fill", name: "macro_auroc"),
        pt(k("gap_test", s), "loso-fill", name: "macro_auroc"),
        ci(k("loso_full", s), "loso-fill"),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [การเปรียบเทียบแหล่งข้อมูลที่เคยเห็น (fused) กับไม่เคยเห็น (LOSO) สำหรับโมเดล letterbox โดย _Gap_ คือผลต่างแบบจับคู่ระหว่างโมเดล LOSO กับโมเดล fused บนภาพทดสอบชุดเดียวกัน พร้อมช่วงความเชื่อมั่น 95% ส่วน _Full source_ คือการวัดผลโมเดล LOSO บนทุกภาพของแหล่งข้อมูลที่กันไว้ สำหรับ Duke กรณี held out เป็นโมเดลเดี่ยวที่ทดสอบบน 45 ดวงตา],
) <tab-unseen>

#figure(
  kind: image,
  fig("gap"),
  caption: [Macro-F1 บน test split ของแต่ละแหล่งข้อมูล เมื่อแหล่งข้อมูลนั้นเป็นแบบเคยเห็น (fused, จุดโปร่ง) และไม่เคยเห็น (LOSO, จุดทึบ) สำหรับโมเดล letterbox โดยเส้นแทนช่วงความเชื่อมั่น 95% ของคะแนนแบบ unseen ส่วนผลต่างแบบจับคู่ (paired gaps) แสดงใน @tab-unseen ทั้งสองคะแนนประเมินบนชุดภาพทดสอบเดียวกัน ซึ่งสำหรับ Kermany ภาพส่วนใหญ่มีขนาด 1024×496 และได้คะแนนสูงกว่า Kermany ในภาพรวมทั้งชุดข้อมูล (@tab-unseen, full source)],
) <fig-gap>

ทุกแหล่งข้อมูลมีความแม่นยำลดลงอย่างมีนัยสำคัญเหนือสัญญาณรบกวนเมื่อกลายเป็นแหล่งข้อมูลที่ไม่เคยเห็น (unseen source) (@tab-unseen, @fig-gap) สำหรับ OCTDL, PAIMA และ Duke ส่วนต่างของ macro-F1 อยู่ที่ #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) จุดเปอร์เซ็นต์ ส่วน Kermany มีส่วนต่างอยู่ที่ #dl(k("gap_test", "kermany"), "loso-fill - fused-fill")

สำหรับแหล่งข้อมูลแบบ 3 คลาส ส่วนต่างของ AUROC มีค่าน้อยกว่าส่วนต่างของ macro-F1 อย่างชัดเจน (#dl(k("gap_test", "kermany"), "loso-fill - fused-fill", name: "macro_auroc") สำหรับ Kermany, #dl(k("gap_test", "octdl"), "loso-fill - fused-fill", name: "macro_auroc") สำหรับ OCTDL และ #dl(k("gap_test", "duke"), "loso-fill - fused-fill", name: "macro_auroc") สำหรับ Duke)

ชี้ให้เห็นว่าโมเดลยังคงจัดอันดับคลาสที่ถูกต้องไว้ในลำดับต้น ๆ ได้ดีในภาพส่วนใหญ่ แต่คะแนนทำนายมักตกไปอยู่อีกฝั่งของจุดตัดการตัดสินใจ (cutoff threshold)

PAIMA เป็นข้อยกเว้นเพียงหนึ่งเดียว โดยมีส่วนต่าง AUROC อยู่ที่ #dl(k("gap_test", "paima"), "loso-fill - fused-fill", name: "macro_auroc") เมื่อเทียบกับส่วนต่าง macro-F1 ที่ #dp(k("gap_test", "paima"), "loso-fill - fused-fill") ซึ่งสะท้อนถึงการเปลี่ยนแปลงในความสามารถของการจัดอันดับ (ranking) ไม่ใช่เพียงแค่การคลาดเคลื่อนของจุดตัดเท่านั้น

ชุด test ของ Kermany ไม่ได้เป็นตัวแทนที่ดีของชุดข้อมูลโดยรวม: ภาพถึง #DS.kermany.test_sizes.at("1024x496") ภาพจากทั้งหมด #num(DS.kermany.n_test) ภาพ มีขนาด 1024×496 ซึ่งเป็นขนาดที่มีสัดส่วนเพียง #pct1(size-share("kermany", "1024x496") * 100)% ใน Kermany ทั้งหมด

เมื่อประเมินบนภาพทั้งหมด #num(R.at("loso_full|kermany|all").n_images) ภาพ คะแนนของ Kermany แบบ held-out ได้เพียง #pt("loso_full|kermany|all", "loso-fill") ซึ่งต่ำกว่าคะแนน #pt("gap_test|kermany|all", "loso-fill") บน test split อย่างมาก เราจึงใช้คะแนนแบบ full-source เป็นเกณฑ์หลักในการอ้างอิงข้อสรุปเกี่ยวกับ Kermany ในฐานะ unseen source โดยในหัวข้อถัดไปจะแสดงให้เห็นว่าเหตุใดขนาดของภาพจึงส่งผลอย่างยิ่งยวด

== ทางลัดจากขนาดภาพ (The image-size shortcut) <sec-shortcut>

ใน Duke ขนาดของภาพสามารถระบุคลาสของข้อมูลได้ถึง #calc.round((int(duke-eyes("amd", size: "1024x496")) + int(duke-eyes("dme", size: "768x496"))) / DS.duke.n_groups * 100)% ของดวงตาทั้งหมด: สแกนขนาด 768×496 ทุกภาพคือ DME และสแกนขนาด 1024×496 ทุกภาพคือ AMD (@fig-size-class)

โมเดลที่เทรนด้วย Duke จึงสามารถเรียนรู้กฎทางลัดว่า "ความกว้าง 768 พิกเซลหมายถึง DME" ซึ่งเป็นกฎที่ถูกต้องภายใน Duke แต่ผิดพลาดเมื่อนำไปใช้กับชุดข้อมูลอื่น การทดสอบ 3 ประการต่อไปนี้ยืนยันว่าโมเดลได้เรียนรู้กฎดังกล่าวจริง

*การเปลี่ยนเฉพาะขนาดของภาพทดสอบ:* เรานำโมเดล LOSO ที่กัน Kermany ไว้มาประเมินภาพของ Kermany ซ้ำ โดยเปลี่ยนเฉพาะความกว้างของภาพโดยไม่มีการเทรนใหม่ (@tab-size-test) การเติมขอบ (padding) ภาพ normal ขนาด 512×496 ให้มีความกว้าง 768 พิกเซล (ซึ่งเป็นความกว้างเฉพาะของ DME ใน Duke) ทำให้สัดส่วนภาพที่ถูกทำนายเป็น DME พุ่งสูงขึ้นจาก #pct1(KS.pad.at("512x496").at("normal->dme").at(0))% เป็น #pct1(KS.pad.at("512x496").at("normal->dme").at(1))%

ในทางตรงกันข้าม การครอบตัด (crop) ภาพ normal ขนาด 768×496 ให้เหลือความกว้าง 512 พิกเซล ให้ผลในทิศทางกลับกัน โดยสัดส่วนที่ถูกทำนายเป็น DME ลดลงจาก #pct1(KS.crop.at("768x496").at("normal->dme").at(0))% เหลือ #pct1(KS.crop.at("768x496").at("normal->dme").at(1))%

ปรากฏการณ์นี้ไม่ได้จำกัดอยู่เพียง Kermany เท่านั้น: การเติมขอบภาพขนาด 512×496 ของ Duke ขึ้นไปอีก 16 แถวกลายเป็นขนาด 512×512 (ซึ่งเป็นขนาดที่พบเฉพาะ DME ใน Kermany) ส่งผลให้ภาพ normal ของ Duke ถึง #calc.round(HS.at("duke/pad512x512").all.at("normal->dme").at(1))% ถูกทำนายผิดเป็น DME ในโมเดลที่กัน Duke ไว้

#let size-row(label, e) = (label, pct1(e.at(0)), pct1(e.at(1)),
  [#fmt(e.at(2) / 100, signed: true) [#fmt(e.at(3) / 100, signed: true), #fmt(e.at(4) / 100, signed: true)]])
#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 4,
      align: (left, right, right, right),
      table.header([ภาพทดสอบ และการเปลี่ยนขนาด], [ก่อนหน้า], [ภายหลัง], [ผลต่าง], table.hline()),
      ..size-row([Kermany, pad 512×496 → 768×496], KS.pad.at("512x496").at("normal->dme")),
      ..size-row([Kermany, crop 768×496 → 512×496], KS.crop.at("768x496").at("normal->dme")),
      ..size-row([Duke, pad 512×496 → 512×512], HS.at("duke/pad512x512").all.at("normal->dme")),
      table.hline(),
    )
  },
  caption: [สัดส่วน (%) ของภาพ normal ที่ถูกทำนายเป็น DME ก่อนและหลังการเปลี่ยนเฉพาะขนาดของภาพทดสอบ โดยใช้โมเดล LOSO บนแหล่งข้อมูลที่กันไว้โดยไม่มีการเทรนใหม่ ทั้งนี้ _ผลต่าง_ เป็นค่าแบบจับคู่พร้อมช่วงความเชื่อมั่น 95%],
) <tab-size-test>

*จุดที่สัญญาณขนาดภาพส่งผล: แถบขอบ letterbox:* ความกว้างของภาพส่งผ่านไปถึงโมเดลผ่านแถบขอบ letterbox: เมื่อสแกนขนาด 768×496 ถูกปรับให้พอดีกับกรอบสี่เหลี่ยมจัตุรัส จะเกิดแถบขอบเรียบทั้งด้านบนและด้านล่างของเนื้อเยื่อ กินพื้นที่ประมาณหนึ่งในสามของภาพทั้งหมด ซึ่งภาพขนาด 768×496 ที่เป็น DME ทั้งหมดของ Duke ก็มีแถบขอบในลักษณะเดียวกันนี้

สำหรับโมเดลที่กัน Kermany ไว้ทั้ง 5 โมเดล เราสุ่มภาพขนาด 768×496 กลุ่มละ #gc-group-n("768 normal → DME") ภาพ (ผู้ป่วยละ 1 ภาพ) ได้แก่ ภาพ normal และ AMD ที่โมเดลทำนายผิดเป็น DME และภาพ normal และ AMD ที่โมเดลทำนายได้ถูกต้อง จากนั้นเราทำการแทรกแซงโดยแก้ไขเฉพาะพิกเซลบริเวณแถบขอบ ให้เป็นสีดำหรือเป็นสัญญาณรบกวน speckle ที่สุ่มมาจากครึ่งที่มีความเข้มมืดของภาพสแกน แล้วทำการประเมินคะแนนใหม่ (@tab-bands)

ในโมเดลที่ 1, 2 และ 4 มีข้อผิดพลาดหลงเหลืออยู่ไม่เกิน #{
  let xs = (1, 2, 4).map(f => ("768 normal → DME", "768 AMD → DME").map(g => ("black", "speckle").map(b => gc-at(f, g, b)))).flatten()
  pct0(calc.max(..xs))
}% หลังจากการเปลี่ยนแถบขอบทั้งสองแบบ ส่วนในโมเดลที่ 3 และ 5 พบว่ามีข้อผิดพลาดของ AMD หลงเหลืออยู่ #pct0(gc-at(3, "768 AMD → DME", "speckle"))% และ #pct0(gc-at(5, "768 AMD → DME", "speckle"))% และข้อผิดพลาดของ normal หลงเหลืออยู่ #pct0(gc-at(3, "768 normal → DME", "speckle"))% และ #pct0(gc-at(5, "768 normal → DME", "speckle"))% เมื่อใช้แถบขอบแบบ speckle แสดงว่าแถบขอบเป็นช่องทางหลักแต่ไม่ใช่ช่องทางเดียวที่สัญญาณนี้ส่งผ่าน

#{
  let rows = GC.intervention_by_fold.filter(r => r.group in ("768 normal → normal", "768 AMD → AMD") and r.at("band fill") != "trained")
  let total = rows.map(r => r.n).sum()
  let changed = rows.map(r => calc.round(r.n * (1 - r.at("called true class")))).sum()
  let to-dme = rows.map(r => calc.round(r.n * r.at("called DME"))).sum()
  [ในทางกลับกัน ภาพที่เดิมทำนายถูกต้องแทบไม่ได้รับผลกระทบ: จากทั้งหมด #total กรณีดังกล่าว (การเปลี่ยนแถบขอบ 2 รูปแบบ ใน 5 โมเดล) มีเพียง #to-dme ภาพที่เปลี่ยนเป็น DME และมีภาพที่ผลทำนายเปลี่ยนไปจากคลาสเดิมเพียง #changed ภาพ]
}

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 7,
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[โมเดล], table.cell(colspan: 2, align: center)[Normal ทำนายเป็น DME],
        table.cell(colspan: 2, align: center)[AMD ทำนายเป็น DME],
        table.cell(colspan: 2, align: center)[สัดส่วนบนแถบขอบ Grad-CAM],
        [black], [speckle], [black], [speckle], [ทำนายผิด], [ทำนายถูก],
        table.hline(),
      ),
      ..range(1, 6).map(f => (
        [#f],
        pct0(gc-at(f, "768 normal → DME", "black")), pct0(gc-at(f, "768 normal → DME", "speckle")),
        pct0(gc-at(f, "768 AMD → DME", "black")), pct0(gc-at(f, "768 AMD → DME", "speckle")),
        pct0(gc-band("768 normal → DME", f)), pct0(gc-band("768 normal → normal", f)),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [ภาพขนาด 768×496 จาก Kermany ที่กันไว้ ซึ่งโมเดล LOSO-Kermany ทำนายผิดเป็น DME: สัดส่วน (%) ที่ยังคงถูกทำนายเป็น DME หลังจากแทนที่แถบขอบ letterbox ด้วยสีดำหรือสัญญาณรบกวน speckle ส่วน _สัดส่วนบนแถบขอบ Grad-CAM_ แสดงสัดส่วน (%) ของแผนภาพที่ตกบนแถบขอบ สำหรับภาพ normal ที่ทำนายเป็น DME (_ทำนายผิด_) และที่ทำนายเป็น normal (_ทำนายถูก_)],
) <tab-bands>

แผนภาพ Grad-CAM ช่วยแสดงให้เห็นกลไกนี้เช่นเดียวกัน (@fig-gradcam) แต่มีข้อควรระวัง: สำหรับภาพที่ทำนายถูกต้อง ทุกโมเดลมีสัดส่วนของแผนภาพที่ตกบนแถบขอบอยู่ที่ #pct0(calc.min(..range(1, 6).map(f => gc-band("768 normal → normal", f))))–#pct0(calc.max(..range(1, 6).map(f => gc-band("768 normal → normal", f))))% ทว่าสำหรับภาพที่ทำนายผิด สัดส่วนดังกล่าวมีช่วงกว้างตั้งแต่ #pct0(calc.min(..range(1, 6).map(f => gc-band("768 normal → DME", f))))% ถึง #pct0(calc.max(..range(1, 6).map(f => gc-band("768 normal → DME", f))))% และไม่ได้สอดคล้องกับผลการแทรกแซงโดยตรง

ตัวอย่างเช่น โมเดลที่ 4 มีแผนภาพตกบนแถบขอบเพียง #pct0(gc-band("768 normal → DME", 4))% แต่กลับมีข้อผิดพลาดของ normal หลงเหลือเพียง #pct0(gc-at(4, "768 normal → DME", "speckle"))% เมื่อใส่แถบขอบ speckle (@tab-bands)

แผนภาพเหล่านี้สะท้อนสิ่งที่โมเดลเรียนรู้ ไม่ใช่โครงสร้างของภาพ: เมื่อทำการสุ่มค่าน้ำหนักส่วนบนของโครงข่ายใหม่ (re-initialisation) แผนภาพที่ได้ไม่สัมพันธ์กับแผนภาพของโมเดลที่เทรนแล้วอีกต่อไป (ค่ามัธยฐานสหสัมพันธ์แบบ Spearman เท่ากับ #calc.round(GC.randomization_fold1_768_normal_to_dme.spearman_median, digits: 2) ในโมเดลที่ 1) ดังนั้นเราจึงใช้การทดลองแทรกแซงแถบขอบเป็นหลักฐานเชิงประจักษ์หลัก และใช้ Grad-CAM เป็นเพียงภาพประกอบ

#figure(
  image("../../figures/gradcam.png", width: 100%),
  caption: [แผนภาพ Grad-CAM (เลเยอร์คอนโวลูชันสุดท้าย สำหรับคลาสที่ทำนาย) บนภาพ normal ขนาด 768×496 จาก Kermany ที่กันไว้ ตามมุมมองของโมเดล LOSO-Kermany แบบ letterbox (แถวบน) และ square-crop (แถวล่าง) โดยแสดงคลาสที่ทำนายไว้เหนือแต่ละช่อง สองคอลัมน์สุดท้ายเป็นกลุ่มควบคุมที่ทั้งสองโมเดลทำนายเป็น normal ถูกต้อง],
) <fig-gradcam>

*การเทรนโดยไม่มีสัญญาณขนาดภาพ:* โมเดลแบบ square crop จะไม่เห็นความกว้างของภาพสแกนเลย เมื่อกัน Kermany ไว้ โมเดลนี้ทำคะแนนบน full source ได้ #pt("loso_full|kermany|all", "loso-sq") เทียบกับ #pt("loso_full|kermany|all", "loso-fill") ของโมเดล letterbox (#dl("loso_full|kermany|all", "loso-sq - loso-fill"))

และเมื่อวัดบน test split เทียบกับโมเดล fused ในกลุ่มการทดลองเดียวกัน ส่วนต่างความแม่นยำลดลงจาก #dl("gap_test|kermany|all", "loso-fill - fused-fill") เหลือ #dl("gap_test|kermany|all", "loso-sq - fused-sq") หรือแคบลงราว #calc.round((1 - delta("gap_test|kermany|all", "loso-sq - fused-sq").point / delta("gap_test|kermany|all", "loso-fill - fused-fill").point) * 100)%

ความแม่นยำที่เพิ่มขึ้นเกิดขึ้นตรงจุดที่มีสัญญาณทางลัดพอดี (@fig-kermany-size): สำหรับภาพ normal ขนาด 768×496 ค่า recall เพิ่มขึ้นจาก #pt("size_loso:768x496|kermany|all", "fill", name: "recall_normal") เป็น #pt("size_loso:768x496|kermany|all", "sq", name: "recall_normal") และสัดส่วนที่ถูกทำนายผิดเป็น DME ลดลงจาก #share("size_loso:768x496|kermany|all", "fill", "normal", "dme")% เหลือ #share("size_loso:768x496|kermany|all", "sq", "normal", "dme")%

#figure(
  kind: image,
  fig("kermany-size"),
  caption: [การประเมินผลบน Kermany ที่กันไว้ (ภาพทั้งหมด) จำแนกตามขนาดภาพและคลาสจริง: Recall ของโมเดล LOSO แบบ letterbox และ square-crop พร้อมช่วงความเชื่อมั่น 95% การตัดภาพแบบ square crop ช่วยเพิ่มความแม่นยำได้มากที่สุดสำหรับภาพ normal และ AMD ขนาด 768×496 ซึ่งเป็นความกว้างที่พบเฉพาะคลาส DME ใน Duke และทำให้ recall ของ DME ลดลงในขนาดดังกล่าว],
) <fig-kermany-size>

*แต่การตัดสัญญาณขนาดภาพไม่ได้ให้ผลดีในทุกกรณี:* ทว่าสำหรับแหล่งข้อมูลที่กันไว้อื่น ๆ วิธี square crop ไม่ได้ให้ผลดีขึ้น และบางกรณีกลับแย่ลง (@tab-square)

บน OCTDL ค่า recall ของ normal เปลี่ยนแปลงไป #dl("loso_full|octdl|all", "loso-sq - loso-fill", name: "recall_normal") โดยสัดส่วนภาพ normal ที่ทำนายเป็น DME เปลี่ยนจาก #share("loso_full|octdl|all", "loso-fill", "normal", "dme")% เป็น #share("loso_full|octdl|all", "loso-sq", "normal", "dme")% ส่วนบน Duke ค่า recall ของ DME เปลี่ยนแปลงไป #dl("loso_full|duke|all", "loso-sq - loso-fill", name: "recall_dme")

ในมุมมอง Normal/AMD ซึ่งตัดคลาส DME ออก ความสูญเสียทั้งสองจุดนี้หายไป (#dl("loso_full|octdl|normal_amd", "loso-sq - loso-fill") และ #dl("loso_full|duke|normal_amd", "loso-sq - loso-fill")) แสดงให้เห็นว่าผลเสียของ square crop ในแหล่งข้อมูลเหล่านี้เกี่ยวข้องกับ DME โดยตรง: นั่นคือ ทำนายเป็น DME มากเกินไปใน OCTDL และทำนายเป็น DME น้อยเกินไปใน Duke

สำหรับ PAIMA คะแนน macro-F1 เปลี่ยนแปลงไป #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill") และ AUROC เปลี่ยนแปลงไป #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill", name: "macro_auroc") ซึ่งเป็นการสูญเสียความสามารถในการจัดอันดับที่แปรผันตาม seed อย่างมาก (@sec-limits)

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 8,
      align: (left, right, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูลที่กันไว้], table.cell(colspan: 3, align: center)[Macro-F1],
        table.cell(colspan: 2, align: center)[AUROC], table.cell(colspan: 2, align: center)[Macro-F1 Normal/AMD],
        [Letterbox], [Square], [ผลต่าง], [Letterbox], [Square], [Letterbox], [Square],
        table.hline(),
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        pt(k("loso_full", s), "loso-fill"),
        pt(k("loso_full", s), "loso-sq"),
        dl(k("loso_full", s), "loso-sq - loso-fill"),
        pt(k("loso_full", s), "loso-fill", name: "macro_auroc"),
        pt(k("loso_full", s), "loso-sq", name: "macro_auroc"),
        ..if s == "paima" { ([--], [--]) } else {
          (pt("loso_full|" + s + "|normal_amd", "loso-fill"), pt("loso_full|" + s + "|normal_amd", "loso-sq"))
        },
      )).flatten(),
      table.hline(),
    )
  },
  caption: [โมเดล LOSO บนแหล่งข้อมูลที่กันไว้แบบเต็มชุดข้อมูล (full source) เปรียบเทียบระหว่างแบบมีสัญญาณขนาดภาพ (letterbox) และไม่มีสัญญาณขนาดภาพ (square crop) (%) โดย _ผลต่าง_ เป็นค่าแบบจับคู่พร้อมช่วงความเชื่อมั่น 95% สำหรับ PAIMA ประเมินผลเฉพาะ normal กับ AMD ดังนั้นมุมมอง Normal/AMD จึงเป็นคะแนนหลักของชุดข้อมูลนี้],
) <tab-square>

ด้วยเหตุนี้ เราจึงกำหนดให้โมเดลแบบ letterbox เป็นผลการทดลองหลัก และจัดให้ square crop เป็นการทดลองตัดทอน (ablation) ซึ่งช่วยยืนยันว่าทางลัดจากขนาดภาพเกิดขึ้นจริงและอธิบายส่วนต่างความแม่นยำของ Kermany ได้เป็นส่วนใหญ่ แต่ไม่ใช่วิธีแก้ปัญหาที่ใช้ได้ทั่วไป

ทั้งนี้เราได้กำหนดแนวทางการตีความนี้ไว้ล่วงหน้าก่อนทราบผล square crop ของ OCTDL และ Duke ว่า square crop จะถูกใช้เป็นการตั้งค่าหลักก็ต่อเมื่อไม่มีแหล่งข้อมูลใดสูญเสียความแม่นยำเกินระดับสัญญาณรบกวนเท่านั้น

== ส่วนต่างความแม่นยำที่ยังคงหลงเหลือ (What remains of the gap) <sec-remaining>

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูลที่กันไว้], table.cell(rowspan: 2)[ส่วนต่าง 3 คลาส],
        table.cell(rowspan: 2)[DME recall \ เปลี่ยนแปลง], table.cell(colspan: 3, align: center)[Macro-F1 Normal/AMD],
        [Seen], [Unseen], [Gap],
        table.hline(),
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        ..if s == "paima" { ([--], [--]) } else {
          (dl("gap_test|" + s + "|all", "loso-fill - fused-fill"),
           dl("gap_test|" + s + "|all", "loso-fill - fused-fill", name: "recall_dme"))
        },
        pt("gap_test|" + s + "|normal_amd", "fused-fill"),
        pt("gap_test|" + s + "|normal_amd", "loso-fill"),
        dl("gap_test|" + s + "|normal_amd", "loso-fill - fused-fill"),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [ที่มาของ unseen-source gap (โมเดล letterbox บน test split คำนวณจาก LOSO ลบด้วย fused หน่วยเป็น percentage points พร้อมช่วงความเชื่อมั่นแบบจับคู่ 95%) คะแนนแบบ 3 คลาสแสดงใน @tab-unseen และค่า DME recall แสดงใน @tab-recall-unseen การตัดคลาส DME ออกช่วยลดส่วนต่างเกือบทั้งหมดสำหรับ OCTDL และ Duke ส่วน PAIMA ไม่มีคลาส DME],
) <tab-remaining>

ส่วนต่างความแม่นยำที่ยังหลงเหลืออยู่เกือบทั้งหมดเกี่ยวข้องกับ DME (@tab-remaining) บน test split ค่า recall ของ DME เมื่อเป็นแหล่งข้อมูลที่กันไว้ ลดลงถึง #span(("kermany", "octdl", "duke").map(s => delta("gap_test|" + s + "|all", "loso-fill - fused-fill", name: "recall_dme").point)) จุดเปอร์เซ็นต์สำหรับ Kermany, OCTDL และ Duke

ในมุมมอง Normal/AMD ส่วนต่างของ OCTDL และ Duke อยู่ในระดับสัญญาณรบกวนใกล้ศูนย์ (#dl("gap_test|octdl|normal_amd", "loso-fill - fused-fill") และ #dl("gap_test|duke|normal_amd", "loso-fill - fused-fill"))

ในขณะที่ Kermany ยังคงมีส่วนต่าง Normal/AMD ขนาดใหญ่ จากการที่ภาพ normal ถูกทำนายเป็น AMD: เมื่อกัน Kermany ไว้ ค่า recall ของ normal ลดลงจาก #pt("gap_test|kermany|all", "fused-fill", name: "recall_normal") เหลือ #pt("gap_test|kermany|all", "loso-fill", name: "recall_normal") (@tab-recall-unseen) และภาพ normal ที่ทำนายผิดส่วนใหญ่ถูกระบุว่าเป็น AMD

ซึ่งสอดคล้องกับสัญญาณขนาดภาพข้างต้น (@fig-kermany-size) เนื่องจาก test split ส่วนใหญ่ของ Kermany มีขนาด 1024×496 ซึ่งเป็นขนาดเฉพาะของ AMD ใน Duke

สำหรับ PAIMA ซึ่งไม่มีคลาส DME ยังคงมีส่วนต่างอยู่ที่ #dl("gap_test|paima|normal_amd", "loso-fill - fused-fill") โดยเกิดจาก recall ของ AMD และเป็นระยะ dry AMD เป็นหลักเช่นเดิม: เมื่อกัน PAIMA ไว้ ค่า recall เปลี่ยนแปลงไป #as-dl("test|paima|normal_amd", "dry", "loso-fill - fused-fill") สำหรับภาพสแกน drusen และ #as-dl("test|paima|normal_amd", "wet", "loso-fill - fused-fill") สำหรับภาพระยะ wet

#figure(
  recall-table(s => {
    let rows = (seen-row(s, "Fused (เคยเห็น)", "fused-fill"),)
    if s == "kermany" {
      rows.push(recall-row(s, "Fused (ไม่รวม Kermany DME)", ZD.at("kermany|all").arms.at("zero-dme").metrics, none))
    }
    rows.push(recall-row(s, "LOSO (กันไว้)", R.at(k("gap_test", s)).arms.at("loso-fill").metrics, "loso-fill"))
    rows
  }),
  caption: [Recall (%) แยกตามคลาสบน test split ของแต่ละแหล่งข้อมูล เมื่อแหล่งข้อมูลนั้นเคยเห็น (fused) และถูกกันไว้ (LOSO) สำหรับโมเดล letterbox โดยคอลัมน์เหมือนใน @tab-recall สำหรับ Kermany แสดงผลของโมเดล fused ที่เทรนโดยไม่รวม DME ของ Kermany ร่วมด้วย (@tab-zero-dme) ทั้งนี้ผลต่างที่อ้างในเนื้อหามีช่วงความเชื่อมั่นแบบจับคู่ 95%],
) <tab-recall-unseen>

สาเหตุที่เป็นไปได้คือโมเดล LOSO แต่ละตัวเห็นข้อมูล DME จากแหล่งข้อมูลจำนวนน้อยลง: PAIMA ไม่มีข้อมูล DME เลย ดังนั้นเมื่อกัน Kermany ไว้ โมเดลจึงได้เรียนรู้ลักษณะของ DME จาก OCTDL และ Duke เท่านั้น

เราได้ทดสอบสมมติฐานนี้สำหรับ Kermany ด้วยโมเดล fused ที่เทรนบนทั้ง 4 แหล่งข้อมูล แต่ตัดภาพ DME ของ Kermany ออกทั้งหมด (ทั้งในชุด train และ validation โดยคลาสอื่นและแหล่งข้อมูลอื่นยังคงเดิม เทรน 5 fold ตาม Duke ภายใต้ขั้นตอนเดียวกับโมเดล fused ปกติ)

สภาวะนี้ทำให้ Kermany เป็นแหล่งข้อมูลที่เคยเห็นแต่ไม่มีข้อมูล DME ของตนเอง การเปรียบเทียบกับโมเดล fused ปกติจึงช่วยแยกผลกระทบของข้อมูล DME ที่หายไปออกมาโดยเฉพาะ และการเปรียบเทียบกับโมเดล LOSO ช่วยแยกผลกระทบของความเป็นแหล่งข้อมูลที่ไม่เคยเห็น

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูล], table.cell(colspan: 3, align: center)[Macro-F1],
        table.cell(colspan: 2, align: center)[DME recall],
        [ไม่รวม], [Fused], [ผลต่าง], [ไม่รวม], [Fused],
        table.hline(),
      ),
      ..srcs.map(s => {
        let key = s + "|" + view3(s)
        (
          [#source-name.at(s)],
          zd-pt(key, "zero-dme"),
          zd-pt(key, "fused-fill"),
          zd-dl(key, "zero-dme - fused-fill"),
          ..if s == "paima" { ([--], [--]) } else {
            (zd-pt(key, "zero-dme", name: "recall_dme"), zd-pt(key, "fused-fill", name: "recall_dme"))
          },
        )
      }).flatten(),
      table.hline(),
    )
  },
  caption: [โมเดล fused ที่เทรนโดยตัดภาพ DME ของ Kermany ออก (_ไม่รวม_) และแบบปกติ (_Fused_) บน test split ของแต่ละแหล่งข้อมูล (%; สำหรับ PAIMA ประเมินแบบ Normal/AMD โดยไม่มี DME) ทั้งนี้ _ผลต่าง_ เป็นค่าแบบจับคู่พร้อมช่วงความเชื่อมั่น 95% ในระดับผู้ป่วย],
) <tab-zero-dme>

ค่า recall แยกตามคลาสของโมเดล Kermany ทั้งสามรูปแบบแสดงใน @tab-recall-unseen การตัดข้อมูล DME ของ Kermany ออกทำให้คะแนนของ Kermany เปลี่ยนแปลงไป #zd-dl("kermany|all", "zero-dme - fused-fill") ซึ่งทั้งหมดเกิดจาก recall ของ DME ที่ลดลง (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_dme")) ขณะที่ recall ของ normal และ AMD ยังคงอยู่ในระดับสัญญาณรบกวน (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_normal"), #zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_amd"))

บนชุดภาพทดสอบเดียวกัน ผลกระทบนี้คิดเป็นสัดส่วนราว #zd-share()% ของขนาดส่วนต่างเมื่อกัน Kermany ไว้ทั้งหมด (LOSO #zd-pt("kermany|all", "loso-fill"), ไม่รวม Kermany DME #zd-pt("kermany|all", "zero-dme"), fused #zd-pt("kermany|all", "fused-fill")) อย่างไรก็ตาม ทั้งสองโมเดลล้มเหลวด้วยรูปแบบที่ต่างกัน ส่วนต่างความแม่นยำจึงไม่ได้แบ่งแยกตามสัดส่วนนี้อย่างตรงไปตรงมา

การขาดข้อมูล DME ภายในอุปกรณ์สแกนเดียวกัน (in-device) เพียงอย่างเดียว ทำให้สูญเสีย recall ของ DME มากกว่าที่โมเดล LOSO สูญเสียเสียอีก (ผลต่างของ DME recall ระหว่าง LOSO ลบด้วยแบบไม่มี DME คือ #zd-dl("kermany|all", "loso-fill - zero-dme", name: "recall_dme")) จึงสามารถอธิบายข้อผิดพลาดด้าน DME ของโมเดล LOSO ได้อย่างเพียงพอ

ความสูญเสียอีกส่วนของโมเดล LOSO คือ recall ของ normal (#zd-dl("kermany|all", "loso-fill - zero-dme", name: "recall_normal") เมื่อเทียบกับโมเดลที่ไม่รวม Kermany DME) ซึ่งเกิดจากการเป็นแหล่งข้อมูลที่ไม่เคยเห็นและสอดคล้องกับสัญญาณขนาดภาพ การขาดข้อมูล DME จึงเป็นสาเหตุที่มีอยู่จริงแต่อธิบายได้เพียงบางส่วนเท่านั้น

นอกจากนี้ ผลกระทบยังไม่ได้จำกัดอยู่เพียงแหล่งข้อมูลที่ถูกตัดข้อมูล DME ไปเท่านั้น: ค่า DME recall ของ Duke ก็เปลี่ยนแปลงไปด้วยเช่นกัน (#zd-dl("duke|all", "zero-dme - fused-fill", name: "recall_dme")) ชี้ให้เห็นว่าข้อมูล DME ของ Kermany ช่วยให้โมเดลจำแนก DME ของ Duke ได้ดีขึ้น ทว่าการเปลี่ยนแปลงนี้อิงจากข้อมูลเพียง 45 ดวงตา เราจึงถือเป็นเพียงหลักฐานสนับสนุน ส่วน OCTDL และ PAIMA อยู่ในระดับสัญญาณรบกวน

== วิธีการแก้ไขหลังเทรน (Post-hoc fixes) <sec-posthoc>

วิธีการแก้ไขหลังการเทรน (post-hoc fixes) แบบมาตรฐาน 2 วิธี ยังคงเหลือส่วนต่างความแม่นยำไว้เกือบทั้งหมด (@tab-posthoc) โดยทั้งสองวิธีปรับจูนพารามิเตอร์บนผู้ป่วยครึ่งหนึ่งของแหล่งข้อมูลที่กันไว้ และนำไปประเมินผลบนผู้ป่วยอีกครึ่งหนึ่งที่เหลือ

วิธีแรกทดสอบว่าส่วนต่างความแม่นยำเกิดจากจุดตัดการตัดสินใจที่คลาดเคลื่อนหรือไม่ (@sec-unseen): โดยการเลื่อนค่า bias ของ log-probability เอาต์พุตแยกตามแต่ละคลาส (cutoff shift) ซึ่งปรับจูนด้วยป้ายกำกับของแหล่งข้อมูลนั้นเอง

วิธีนี้ทำให้คะแนนของ Kermany เปลี่ยนแปลงไป #lb-dl("kermany|all", "bias-crossfit") และ PAIMA เปลี่ยนไป #lb-dl("paima|normal_amd", "bias-crossfit") ขณะที่ OCTDL (#lb-dl("octdl|all", "bias-crossfit")) และ Duke (#lb-dl("duke|all", "bias-crossfit")) ยังคงอยู่ในระดับสัญญาณรบกวน

แม้กระทั่งการเลื่อนจุดตัดที่ดีที่สุดที่หาได้จากป้ายกำกับทั้งหมดของ Kermany (ซึ่งเป็นไปไม่ได้ในโมเดลที่นำไปใช้งานจริง) ก็ทำคะแนนได้เพียง #fmt(LB.at("kermany|all").grid.best) เทียบกับ #pt("loso_full|kermany|all", "loso-sq") จากการเทรนใหม่โดยไม่มีสัญญาณขนาดภาพ แสดงให้เห็นว่าการปรับจุดตัดแบบคงที่ร่วมกันในทุกภาพ ไม่สามารถแก้ไขข้อผิดพลาดที่ขึ้นกับขนาดภาพได้

#let lbk(s) = s + "|" + view3(s)
#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูลที่กันไว้], table.cell(rowspan: 2)[LOSO],
        table.cell(colspan: 2, align: center)[Cutoff shift], table.cell(rowspan: 2)[AdaBN],
        table.cell(rowspan: 2)[Square crop \ (เทรนใหม่)],
        [Cross-fit], [Ceiling],
        table.hline(),
      ),
      ..srcs.map(s => (
        [#source-name.at(s)],
        lb-pt(lbk(s), "none"),
        lb-dl(lbk(s), "bias-crossfit"),
        fmt(LB.at(lbk(s)).grid.best),
        ab-dl(lbk(s)),
        dl(k("loso_full", s), "loso-sq - loso-fill"),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [การแก้ไขหลังเทรน (post-hoc fixes) บนแหล่งข้อมูลที่กันไว้แบบเต็มชุดข้อมูล (macro-F1 %, โมเดล LOSO แบบ letterbox; PAIMA วัดผลเฉพาะ normal กับ AMD) โดยการปรับจุดตัดแบบ _Cross-fit_, AdaBN และ square crop เป็นผลต่างแบบจับคู่เทียบกับ _LOSO_ พร้อมช่วงความเชื่อมั่น 95% ส่วน _Ceiling_ คือการปรับจุดตัดที่ดีที่สุดบนข้อมูลป้ายกำกับทั้งหมด (#srcs.filter(s => LB.at(lbk(s)).grid.subsampled).map(s => source-name.at(s)).join(", ") คำนวณจากกลุ่มตัวอย่างย่อยแบบสุ่ม)],
) <tab-posthoc>

วิธีที่สองคือ AdaBN @li2016adabn ซึ่งทำการประมาณค่าสถิติของ BatchNorm ใหม่จากภาพที่ไม่มีป้ายกำกับของแหล่งข้อมูลที่ไม่เคยเห็น

วิธีนี้ทำให้คะแนนของ Kermany เปลี่ยนแปลงไป #ab-dl("kermany|all") ซึ่งน้อยกว่าการเลื่อนจุดตัดการตัดสินใจ ส่วน Duke ยังคงอยู่ในระดับสัญญาณรบกวน (#ab-dl("duke|all")) และทำให้ OCTDL เปลี่ยนแปลงไป #ab-dl("octdl|all") โดยดันให้ภาพถูกทำนายเป็น DME มากขึ้น และ PAIMA เปลี่ยนแปลงไป #ab-dl("paima|normal_amd")

ทั้งสองวิธีจึงไม่มีความน่าเชื่อถือข้ามแหล่งข้อมูล และสำหรับ Kermany ไม่มีวิธีใดที่ได้ผลใกล้เคียงกับการเทรนใหม่โดยไม่มีสัญญาณขนาดภาพเลย
