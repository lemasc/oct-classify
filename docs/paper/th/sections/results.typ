#import "../../lib.typ": *
#import "../../i18n/th.typ": source-name, split-name

= ผลลัพธ์ <sec-results>

== แหล่งข้อมูลที่เคยเห็น <sec-seen>

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header([Source], [Per-dataset], [Fused], [Fused, square crop], table.hline()),
    ..srcs
      .map(s => (
        [#source-name.at(s)],
        ci(k("seen", s), "per-dataset"),
        ci(k("seen", s), "fused-fill"),
        ci(k("seen", s), "fused-sq"),
      ))
      .flatten(),
    table.hline(),
  ),
  caption: [Macro-F1 (%) บนชุดข้อมูลทดสอบของแต่ละแหล่งข้อมูล พร้อมช่วงความเชื่อมั่นระดับผู้ป่วยร้อยละ 95 จากการบูตสแตรป],
) <tab-seen>

บนแหล่งข้อมูลที่ใช้ฝึกสอน โมเดล ResNet-50 ได้ค่า Macro-F1 ระหว่าง #pt(k("seen", "paima"), "per-dataset")–#pt(k("seen", "kermany"), "per-dataset") (@tab-seen; Duke รวมผลจากทั้ง 5 ส่วน โดยแต่ละดวงตาประเมินด้วยโมเดลที่ไม่ได้ฝึกสอนด้วยดวงตานั้น) โมเดลเดี่ยวที่ฝึกสอนบนทั้ง 4 แหล่งข้อมูลร่วมกัน (โมเดลแบบ fused) แทบไม่สูญเสียประสิทธิภาพ โดยในทุกแหล่งข้อมูลสูญเสียคะแนนไม่เกิน #whole(calc.min(..srcs.map(s => delta(k("seen", s), "fused-fill - per-dataset").lo))) จุดเมื่อเทียบกับโมเดลแบบรายชุดข้อมูล (ขอบเขตล่างของช่วงความเชื่อมั่นแบบจับคู่ร้อยละ 95) และการลดลงมีนัยสำคัญเฉพาะใน Kermany (#dl(k("seen", "kermany"), "fused-fill - per-dataset")) ในทำนองเดียวกัน การครอบตัดเป็นสี่เหลี่ยมจัตุรัส (square crop) ซึ่งขจัดสิ่งชี้แนะจากขนาดภาพ (@sec-square) ไม่ส่งผลเสียต่อแหล่งข้อมูลที่เคยเห็น โดยไม่มีแหล่งข้อมูลใดลดลงเกินระดับสัญญาณรบกวนเมื่อเทียบกับโมเดลแบบ fused และ PAIMA ได้คะแนนเพิ่มขึ้น #dl(k("seen", "paima"), "fused-sq - fused-fill")

#let rc(m, c) = if ("recall_" + c) in m { fmt(m.at("recall_" + c).point) } else [--]
#let stage-rc(src, arm, st, m) = {
  let key = "test|" + src + "|" + view3(src)
  if src == "duke" { if st == "dry" { rc(m, "amd") } else [--] } // Duke's AMD is all dry
  else if arm == none [--] else { fmt(AS.at(key).stages.at(st).arms.at(arm).recall) }
}
#let recall-row(src, label, m, arm) = (
  [#label],
  rc(m, "normal"),
  rc(m, "amd"),
  stage-rc(src, arm, "dry", m),
  stage-rc(src, arm, "wet", m),
  rc(m, "dme"),
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
    ..srcs
      .map(s => {
        let rows = rows-of(s)
        let rule = if s == srcs.last() { table.hline() } else { table.hline(stroke: 0.3pt + gray) }
        (table.cell(rowspan: rows.len())[#source-name.at(s)], ..rows.flatten(), rule)
      })
      .flatten(),
  )
}
#let seen-row(s, label, arm) = recall-row(s, label, R.at("seen|" + s + "|" + view3(s)).arms.at(arm).metrics, arm)

#figure(
  recall-table(s => (seen-row(s, "Per-dataset", "per-dataset"), seen-row(s, "Fused", "fused-fill"))),
  caption: [Recall (%) รายคลาสบนชุดข้อมูลทดสอบของแต่ละแหล่งข้อมูล โมเดลแบบ letterbox ที่เคยเห็นข้อมูล ค่าเฉลี่ยข้ามโมเดล],
) <tab-recall>

ค่า recall รายคลาส (@tab-recall) เผยลักษณะของข้อผิดพลาด โดย _dry_ และ _wet_ คือค่า recall ภายในคลาส AMD AMD ใน Duke เป็นชนิดแห้งทั้งหมด ตารางไม่แสดงระยะปานกลางของ OCTDL และ PAIMA ไม่มีภาวะ DME PAIMA เป็นแหล่งข้อมูลที่ท้าทายที่สุดแม้จะเคยเห็น (Macro-F1 ร้อยละ #pt(k("seen", "paima"), "fused-fill") สำหรับโมเดลแบบ fused) โดยข้อผิดพลาดหลักอยู่ในคลาส AMD (recall เพียงร้อยละ #pt(k("seen", "paima"), "fused-fill", name: "recall_amd")) ภายใน AMD ข้อผิดพลาดเกิดจาก AMD ชนิดแห้ง (dry AMD) เป็นหลัก โดยโมเดลแบบ fused จำแนกภาพที่มีดรูเซน (drusen) ใน PAIMA ได้ถูกต้องเพียงร้อยละ #as-rec("test|paima|normal_amd", "dry", "fused-fill") เทียบกับชนิดเปียก (wet AMD) ที่ถูกต้องร้อยละ #as-rec("test|paima|normal_amd", "wet", "fused-fill") ภาพดรูเซนที่ทำนายผิดเกือบทั้งหมดถูกจำแนกเป็นปกติ (normal) โมเดลแบบ per-dataset ของ PAIMA ให้ผลลักษณะเดียวกัน (ร้อยละ #as-rec("test|paima|normal_amd", "dry", "per-dataset") เทียบกับร้อยละ #as-rec("test|paima|normal_amd", "wet", "per-dataset")) ข้อจำกัดนี้จึงเกิดจากลักษณะของภาพดรูเซนใน PAIMA เอง ไม่ได้เกิดจากการรวมข้อมูล (fusing) ภาพดรูเซนใน OCTDL ถูกจำแนกถูกต้องน้อยกว่าชนิดเปียกเช่นกัน (ร้อยละ #as-rec("test|octdl|all", "dry", "fused-fill") เทียบกับร้อยละ #as-rec("test|octdl|all", "wet", "fused-fill") สำหรับโมเดลแบบ fused) ทว่าชุดทดสอบ (test split) ของ OCTDL มีภาพดรูเซนเพียง #AS.at("test|octdl|all").stages.dry.n_images ภาพจากผู้ป่วย #AS.at("test|octdl|all").stages.dry.n_groups รายเท่านั้น ส่วนภาพดรูเซนใน Kermany ทั้งสองโมเดลจำแนกถูกต้องอย่างน้อยร้อยละ #fmt(calc.min(..("per-dataset", "fused-fill").map(a => AS.at("test|kermany|all").stages.dry.arms.at(a).recall))) ของทั้งหมด

== แหล่งข้อมูลที่ไม่เคยเห็น <sec-unseen>

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูลที่กันไว้],
        table.cell(colspan: 3, align: center)[Macro-F1 (ชุดทดสอบ)],
        table.cell(rowspan: 2)[Macro-F1 \ (ทั้งแหล่งข้อมูล)],
        [เคยเห็น], [ไม่เคยเห็น], [ส่วนต่าง],
        table.hline(),
      ),
      ..srcs
        .map(s => (
          [#source-name.at(s)],
          pt(k("gap_test", s), "fused-fill"),
          pt(k("gap_test", s), "loso-fill"),
          dl(k("gap_test", s), "loso-fill - fused-fill"),
          ci(k("loso_full", s), "loso-fill"),
        ))
        .flatten(),
      table.hline(),
    )
  },
  caption: [Macro-F1 (%) ของแหล่งข้อมูลที่เคยเห็น (fused) เทียบกับไม่เคยเห็น (LOSO) รูปแบบ letterbox โดย _ส่วนต่าง_ คือผลต่างแบบจับคู่ (LOSO ลบด้วย fused) บนชุดภาพทดสอบเดียวกัน พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-unseen>

ทุกแหล่งข้อมูลลดลงเกินระดับสัญญาณรบกวนเมื่อไม่เคยเห็น (@tab-unseen) คอลัมน์ทั้งแหล่งข้อมูลประเมินโมเดล LOSO บนภาพทั้งหมดของแหล่งข้อมูลที่แยกไว้ โดย Duke ประเมินด้วยโมเดลเดี่ยวบนดวงตาทั้ง #DS.duke.n_groups ดวง ส่วนต่างของ Macro-F1 ของ OCTDL, PAIMA และ Duke อยู่ที่ #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) จุด ขณะที่ของ Kermany อยู่ที่ #dl(k("gap_test", "kermany"), "loso-fill - fused-fill")

สำหรับแหล่งข้อมูล 3 คลาส ส่วนต่างของ AUROC น้อยกว่าของ Macro-F1 อย่างมาก (@tab-auroc; สำหรับ Kermany #dl(k("gap_test", "kermany"), "loso-fill - fused-fill", name: "macro_auroc")) ซึ่งชี้ว่าโมเดลยังจัดอันดับคลาสที่ถูกต้องไว้ในลำดับต้น ๆ ได้ในภาพส่วนใหญ่ แต่การตัดสินใจมักตกไปอยู่อีกฝั่งของจุดตัดการตัดสินใจ PAIMA เป็นข้อยกเว้นเดียว โดยมีส่วนต่างของ AUROC ที่ #dl(k("gap_test", "paima"), "loso-fill - fused-fill", name: "macro_auroc") เทียบกับ Macro-F1 ที่ #dp(k("gap_test", "paima"), "loso-fill - fused-fill") ซึ่งสะท้อนการเปลี่ยนแปลงของความสามารถด้านการจัดอันดับ ไม่ใช่เพียงจุดตัดที่คลาดเคลื่อน

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 4,
      align: (left, right, right, right),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูลที่กันไว้], table.cell(colspan: 3, align: center)[AUROC (ชุดทดสอบ)],
        [เคยเห็น], [ไม่เคยเห็น], [ส่วนต่าง],
        table.hline(),
      ),
      ..srcs
        .map(s => (
          [#source-name.at(s)],
          pt(k("gap_test", s), "fused-fill", name: "macro_auroc"),
          pt(k("gap_test", s), "loso-fill", name: "macro_auroc"),
          dl(k("gap_test", s), "loso-fill - fused-fill", name: "macro_auroc"),
        ))
        .flatten(),
      table.hline(),
    )
  },
  caption: [Macro-AUROC ของแหล่งข้อมูลที่เคยเห็นเทียบกับไม่เคยเห็น (ชุดทดสอบ, letterbox) โดย _ส่วนต่าง_ คือผลต่างแบบจับคู่ พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-auroc>

ชุดทดสอบ (test set) ของ Kermany ไม่เป็นตัวแทนที่ดีของชุดข้อมูลโดยรวม โดยมีภาพถึง #DS.kermany.test_sizes.at("1024x496") ภาพจากทั้งหมด #num(DS.kermany.n_test) ภาพที่มีขนาด 1024×496 พิกเซล ซึ่งมีสัดส่วนเพียงร้อยละ #pct1(size-share("kermany", "1024x496") * 100) ในชุดข้อมูล Kermany ทั้งหมด

เมื่อประเมินบนภาพทั้งหมด #num(R.at("loso_full|kermany|all").n_images) ภาพ Kermany ที่ถูกแยกไว้ (held-out) ได้เพียง #pt("loso_full|kermany|all", "loso-fill") ซึ่งต่ำกว่า #pt("gap_test|kermany|all", "loso-fill") บนชุดทดสอบ (test split) อย่างมาก จึงใช้คะแนนทั้งแหล่งข้อมูล (full-source score) เป็นเกณฑ์หลักสำหรับข้อสรุปเกี่ยวกับ Kermany ที่ไม่เคยเห็น หัวข้อถัดไปแสดงเหตุที่ขนาดภาพส่งผลมาก

== สิ่งชี้แนะทางลัดจากขนาดภาพ <sec-shortcut>

ใน Duke ขนาดภาพระบุคลาสได้ถึงร้อยละ #calc.round((int(duke-eyes("amd", size: "1024x496")) + int(duke-eyes("dme", size: "768x496"))) / DS.duke.n_groups * 100) ของดวงตาทั้งหมด (@sec-data) โมเดลที่ฝึกสอนด้วย Duke จึงเรียนรู้กฎทางลัดได้ว่า "ความกว้าง 768 พิกเซลหมายถึงภาวะ DME" ซึ่งถูกต้องใน Duke แต่ผิดเมื่อใช้กับชุดข้อมูลอื่น การทดสอบ 3 ข้อต่อไปนี้ยืนยันว่าโมเดลเรียนรู้กฎนี้จริง

=== การปรับเปลี่ยนเฉพาะขนาดของภาพทดสอบ

งานวิจัยนี้ประเมินโมเดล LOSO ที่แยก Kermany ไว้บนภาพ Kermany ซ้ำ โดยปรับเฉพาะความกว้างของภาพและไม่ฝึกสอนใหม่ (@tab-size-test) การเติมขอบ (padding) ภาพปกติขนาด 512×496 พิกเซลให้กว้าง 768 พิกเซล (ความกว้างเฉพาะของ DME ใน Duke) ทำให้สัดส่วนภาพที่ถูกทำนายเป็น DME เพิ่มจากร้อยละ #pct1(KS.pad.at("512x496").at("normal->dme").at(0)) เป็นร้อยละ #pct1(KS.pad.at("512x496").at("normal->dme").at(1)) ส่วนการครอบตัด (crop) ภาพปกติขนาด 768×496 พิกเซลให้เหลือกว้าง 512 พิกเซลให้ผลตรงข้าม ปรากฏการณ์นี้ไม่จำกัดเฉพาะ Kermany: การเติมขอบภาพ Duke ขนาด 512×496 พิกเซลอีก 16 แถวเป็น 512×512 พิกเซล (ขนาดที่พบเฉพาะ DME ใน Kermany) ให้ผลลักษณะเดียวกันในโมเดลที่แยก Duke ไว้

#let size-row(label, e) = (
  label,
  pct1(e.at(0)),
  pct1(e.at(1)),
  [#fmt(e.at(2) / 100, signed: true) [#fmt(e.at(3) / 100, signed: true), #fmt(e.at(4) / 100, signed: true)]],
)
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
  caption: [สัดส่วน (%) ของภาพปกติที่ถูกทำนายเป็น DME ก่อนและหลังการปรับเปลี่ยนเฉพาะขนาดของภาพทดสอบ พร้อมผลต่างแบบจับคู่และช่วงความเชื่อมั่นร้อยละ 95],
) <tab-size-test>

=== ตำแหน่งที่สัญญาณขนาดภาพส่งผล

ความกว้างของภาพถูกส่งถึงโมเดลผ่านแถบขอบแบบ letterbox เมื่อภาพสแกนขนาด 768×496 พิกเซลถูกปรับให้พอดีกรอบสี่เหลี่ยมจัตุรัส จะเกิดแถบขอบเรียบที่ด้านบนและล่างของเนื้อเยื่อ คิดเป็นประมาณหนึ่งในสามของภาพ (@fig-padding) ภาพ DME ขนาด 768×496 พิกเซลทั้งหมดใน Duke มีแถบขอบเช่นเดียวกัน สำหรับโมเดล LOSO ที่แยก Kermany ไว้ทั้ง 5 โมเดล งานวิจัยนี้สุ่มภาพขนาด 768×496 พิกเซลกลุ่มละ #gc-group-n("768 normal → DME") ภาพ (ผู้ป่วยละ 1 ภาพ) ได้แก่ ภาพปกติ (normal) และ AMD ที่โมเดลทำนายผิดเป็น DME รวมทั้งภาพปกติและ AMD ที่ทำนายถูก แล้วแทรกแซงข้อมูล (intervention) โดยแก้พิกเซลบริเวณแถบขอบเป็นสีดำ หรือแทนที่ด้วยสัญญาณรบกวนแบบเกรน (speckle noise) ที่สุ่มจากครึ่งมืดของภาพสแกน จากนั้นประเมินคะแนนใหม่ (@tab-bands)

โมเดลที่ 1, 2 และ 4 มีข้อผิดพลาดเหลือไม่เกินร้อยละ #{
  let xs = (1, 2, 4).map(f => ("768 normal → DME", "768 AMD → DME").map(g => ("black", "speckle").map(b => gc-at(f, g, b)))).flatten()
  pct0(calc.max(..xs))
}% หลังเปลี่ยนแถบขอบทั้งสองรูปแบบ ขณะที่โมเดลที่ 3 และ 5 เหลือไม่เกินร้อยละ #{
  let xs = (3, 5).map(f => ("768 normal → DME", "768 AMD → DME").map(g => gc-at(f, g, "speckle"))).flatten()
  pct0(calc.max(..xs))
}% เมื่อใช้ speckle ซึ่งแสดงว่าแถบขอบเป็นช่องทางหลักแต่ไม่ใช่ช่องทางเดียวที่สัญญาณนี้ส่งถึงโมเดล ในทางตรงกันข้าม #{
  let rows = GC.intervention_by_fold.filter(r => r.group in ("768 normal → normal", "768 AMD → AMD") and r.at("band fill") != "trained")
  let total = rows.map(r => r.n).sum()
  let to-dme = rows.map(r => calc.round(r.n * r.at("called DME"))).sum()
  [ภาพที่ทำนายถูกต้องแทบไม่เปลี่ยนแปลง: จากทั้งหมด #total กรณี (การเปลี่ยนแถบขอบ 2 รูปแบบ, 5 โมเดล) มีเพียง #to-dme ภาพที่กลายเป็น DME]
}

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[โมเดล], table.cell(colspan: 2, align: center)[ภาพปกติที่ทำนายเป็น DME],
        table.cell(colspan: 2, align: center)[ภาพ AMD ที่ทำนายเป็น DME],
        [ดำ], [speckle], [ดำ], [speckle],
        table.hline(),
      ),
      ..range(1, 6)
        .map(f => (
          [#f],
          pct0(gc-at(f, "768 normal → DME", "black")),
          pct0(gc-at(f, "768 normal → DME", "speckle")),
          pct0(gc-at(f, "768 AMD → DME", "black")),
          pct0(gc-at(f, "768 AMD → DME", "speckle")),
        ))
        .flatten(),
      table.hline(),
    )
  },
  caption: [ข้อผิดพลาดที่ยังคงอยู่หลังแทนที่แถบขอบภาพ letterbox ด้วยสีดำหรือ speckle: สัดส่วน (%) ของภาพขนาด 768×496 ที่ยังคงทำนายผิดเป็น DME ในแต่ละโมเดลที่แยก Kermany ไว้],
) <tab-bands>

Grad-CAM (เลเยอร์คอนโวลูชันสุดท้าย สำหรับคลาสที่ทำนาย) แสดงกลไกเดียวกันนี้เป็นภาพตัวอย่าง (@fig-gradcam) เมื่อสุ่มค่าน้ำหนักชั้นบนสุดของโครงข่ายใหม่ แผนภาพจะไม่สัมพันธ์กับแผนภาพของโมเดลที่ฝึกสอนแล้ว (ค่ามัธยฐานสหสัมพันธ์สเปียร์แมน #calc.round(GC.randomization_fold1_768_normal_to_dme.spearman_median, digits: 2) ในโมเดลที่ 1) จึงสะท้อนสิ่งที่โมเดลเรียนรู้ อย่างไรก็ดี แผนภาพความสำคัญอาจขัดแย้งกับผลของการแทรกแซงจริง @arun2021saliency @hooker2019roar จึงใช้การแทรกแซงบริเวณแถบขอบเป็นหลักฐาน และใช้ Grad-CAM เป็นเพียงภาพประกอบ

#figure(
  image("../../figures/gradcam.png", width: 100%),
  caption: [Grad-CAM บนภาพปกติขนาด 768×496 จากชุดข้อมูล Kermany ที่ถูกแยกไว้ สำหรับโมเดล LOSO แบบ letterbox (บน) และ square-crop (ล่าง) โดยสองคอลัมน์สุดท้ายเป็นกลุ่มควบคุมที่ทั้งสองโมเดลทำนายเป็นปกติ],
) <fig-gradcam>

=== การฝึกสอนโดยไม่มีสิ่งชี้แนะจากขนาดภาพ

โมเดลแบบ square crop ไม่เคยเห็นความกว้างของภาพสแกน เมื่อแยก Kermany ไว้ โมเดลนี้ได้คะแนนบนแหล่งข้อมูลทั้งหมด #pt("loso_full|kermany|all", "loso-sq") เมื่อเทียบกับ #pt("loso_full|kermany|all", "loso-fill") (#dl("loso_full|kermany|all", "loso-sq - loso-fill")) และบนชุดทดสอบเมื่อเทียบกับโมเดล fused ของกลุ่มการทดลองเดียวกัน ส่วนต่างแคบลงประมาณร้อยละ #calc.round((1 - delta("gap_test|kermany|all", "loso-sq - fused-sq").point / delta("gap_test|kermany|all", "loso-fill - fused-fill").point) * 100)% จากเดิมที่ #dl("gap_test|kermany|all", "loso-fill - fused-fill") ประสิทธิภาพที่เพิ่มขึ้นตรงกับจุดที่มีสิ่งชี้แนะพอดี (@fig-kermany-size): ที่ขนาด 768×496 ซึ่งพบเฉพาะ DME ใน Duke square crop ช่วยภาพปกติและ AMD มากที่สุด (ค่า recall ของภาพปกติเพิ่มจาก #pt("size_loso:768x496|kermany|all", "fill", name: "recall_normal") เป็น #pt("size_loso:768x496|kermany|all", "sq", name: "recall_normal")) และทำให้ค่า recall ของ DME ลดลง

#figure(
  kind: image,
  fig("kermany-size"),
  caption: [Recall ของชุดข้อมูล Kermany ที่ถูกแยกไว้ (ภาพทั้งหมด) จำแนกตามขนาดภาพและคลาสจริง สำหรับโมเดล LOSO แบบ letterbox และ square-crop พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <fig-kermany-size>

ทว่าการตัดสิ่งชี้แนะออกไม่ได้ช่วยทุกแหล่งข้อมูล บนแหล่งข้อมูลอื่นที่แยกไว้ square crop สูญเสียความแม่นยำเกินระดับสัญญาณรบกวน (@tab-square): ลดลง #span(("octdl", "duke", "paima").map(s => delta(k("loso_full", s), "loso-sq - loso-fill").point)) จุด Macro-F1 สำหรับ OCTDL, Duke และ PAIMA โดย PAIMA สูญเสีย AUROC อีก #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill", name: "macro_auroc") ซึ่งเป็นการสูญเสียความสามารถด้านการจัดอันดับที่ผันแปรตามค่าซีดอย่างมาก (@sec-limits)

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูลที่กันไว้], table.cell(colspan: 3, align: center)[Macro-F1],
        table.cell(colspan: 2, align: center)[AUROC],
        [Letterbox], [Square], [ผลต่าง], [Letterbox], [Square],
        table.hline(),
      ),
      ..srcs
        .map(s => (
          [#source-name.at(s)],
          pt(k("loso_full", s), "loso-fill"),
          pt(k("loso_full", s), "loso-sq"),
          dl(k("loso_full", s), "loso-sq - loso-fill"),
          pt(k("loso_full", s), "loso-fill", name: "macro_auroc"),
          pt(k("loso_full", s), "loso-sq", name: "macro_auroc"),
        ))
        .flatten(),
      table.hline(),
    )
  },
  caption: [โมเดลแบบ LOSO บนแหล่งข้อมูลที่ถูกแยกไว้ทั้งหมด แบบมี (letterbox) และไม่มี (square crop) สิ่งชี้แนะจากขนาดภาพ (%) โดยผลต่างเป็นผลต่างแบบจับคู่ พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-square>

ผลเสียใน OCTDL และ Duke เกี่ยวข้องกับ DME (@tab-square-dme) ใน OCTDL ค่า recall ของภาพปกติเปลี่ยนไป #dl("loso_full|octdl|all", "loso-sq - loso-fill", name: "recall_normal") เนื่องจากสัดส่วนภาพปกติที่ถูกทำนายเป็น DME เปลี่ยนจากร้อยละ #share("loso_full|octdl|all", "loso-fill", "normal", "dme")% เป็นร้อยละ #share("loso_full|octdl|all", "loso-sq", "normal", "dme")% ขณะที่ใน Duke ค่า recall ของ DME เปลี่ยนไป #dl("loso_full|duke|all", "loso-sq - loso-fill", name: "recall_dme") แต่ในมุมมอง Normal/AMD ซึ่งตัดภาพและผลลัพธ์ DME ออก ผลเสียทั้งสองหายไป กล่าวคือ ทำนายเป็น DME มากเกินไปใน OCTDL และน้อยเกินไปใน Duke

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, left, right, right, right),
      table.header([แหล่งข้อมูลที่กันไว้], [ตัวชี้วัด], [Letterbox], [Square], [ผลต่าง], table.hline()),
      ..("octdl", "duke")
        .map(s => {
          let key = "loso_full|" + s + "|"
          (
            table.cell(rowspan: 3)[#source-name.at(s)],
            [Normal/AMD macro-F1],
            pt(key + "normal_amd", "loso-fill"),
            pt(key + "normal_amd", "loso-sq"),
            dl(key + "normal_amd", "loso-sq - loso-fill"),
            [Normal recall],
            pt(key + "all", "loso-fill", name: "recall_normal"),
            pt(key + "all", "loso-sq", name: "recall_normal"),
            dl(key + "all", "loso-sq - loso-fill", name: "recall_normal"),
            [DME recall],
            pt(key + "all", "loso-fill", name: "recall_dme"),
            pt(key + "all", "loso-sq", name: "recall_dme"),
            dl(key + "all", "loso-sq - loso-fill", name: "recall_dme"),
            table.hline(stroke: 0.3pt + gray),
          )
        })
        .flatten(),
      table.hline(),
    )
  },
  caption: [OCTDL และ Duke ที่ถูกแยกไว้ บนแหล่งข้อมูลทั้งหมด: letterbox เทียบกับ square crop (%) รวมถึงมุมมอง Normal/AMD ที่ตัด DME ออก ผลต่างเป็นผลต่างแบบจับคู่ พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-square-dme>

ด้วยเหตุนี้ จึงกำหนดให้โมเดลแบบ letterbox เป็นการทดลองหลัก และ square crop เป็นการทดลองตัดทอนองค์ประกอบ (ablation study) ซึ่งยืนยันว่าสิ่งชี้แนะทางลัดจากขนาดภาพมีจริง และอธิบายส่วนต่างประสิทธิภาพของ Kermany ได้เป็นส่วนใหญ่ ทว่าไม่ใช่วิธีแก้ปัญหาที่ใช้ได้ทั่วไป

เกณฑ์นี้กำหนดไว้ล่วงหน้าก่อนทราบผล square crop ของ OCTDL และ Duke คือ จะใช้ square crop เป็นรูปแบบหลักก็ต่อเมื่อไม่มีแหล่งข้อมูลใดสูญเสียประสิทธิภาพเกินระดับสัญญาณรบกวนเท่านั้น

== จุดที่เกิดส่วนต่างประสิทธิภาพ <sec-remaining>

#figure(
  kind: image,
  fig("recall-change"),
  caption: [การเปลี่ยนแปลงของค่า recall รายคลาส (จุดเปอร์เซ็นต์) เมื่อแหล่งข้อมูลถูกแยกไว้ (LOSO ลบด้วย fused, letterbox, ชุดทดสอบ) พร้อมช่วงความเชื่อมั่นแบบจับคู่ร้อยละ 95 ระดับของโมเดล fused อยู่ใน @tab-recall],
) <fig-recall-change>

คลาสที่เป็นสาเหตุของส่วนต่างแตกต่างกันไปตามแหล่งข้อมูล (@fig-recall-change): แต่ละแหล่งข้อมูลสูญเสียมากที่สุดในคลาสเดียว ได้แก่ ภาพปกติสำหรับ Kermany, ภาวะ DME สำหรับ OCTDL และ Duke, และ AMD ชนิดแห้งสำหรับ PAIMA

ค่า recall ของ DME ไม่นับภาพปกติหรือภาพ AMD ที่ถูกทำนายผิดเป็น DME เราจึงประเมินในมุมมอง Normal/AMD เพิ่มเติม ซึ่งตัดทั้งภาพ DME และ output ของ DME ออก (@sec-eval) ในมุมมองนี้ ส่วนต่างของ OCTDL และ Duke เท่ากับ #dl("gap_test|octdl|normal_amd", "loso-fill - fused-fill") และ #dl("gap_test|duke|normal_amd", "loso-fill - fused-fill") ซึ่งแยกไม่ออกจากศูนย์ ชี้ว่าภาวะ DME เป็นสาเหตุหลักสำหรับสองแหล่งข้อมูลนี้ แต่ช่วงความเชื่อมั่นยังกว้างเกินกว่าจะระบุว่า DME อธิบายส่วนต่างทั้งหมดได้

Kermany ยังคงมีส่วนต่างในมุมมอง Normal/AMD เท่ากับ #dl("gap_test|kermany|normal_amd", "loso-fill - fused-fill"): ภาพปกติส่วนใหญ่ที่ทำนายผิดถูกระบุว่าเป็น AMD ซึ่งสอดคล้องกับสิ่งชี้แนะจากขนาดภาพ (@fig-kermany-size) เนื่องจากชุดทดสอบส่วนใหญ่เป็นขนาด 1024×496 ซึ่งเป็นขนาดที่มีเฉพาะ AMD ใน Duke สำหรับ PAIMA ซึ่งไม่มีภาวะ DME สูญเสีย recall ของ AMD ชนิดแห้งมากกว่าชนิดเปียก เช่นเดียวกับเมื่อเป็นแหล่งข้อมูลที่เคยเห็น (@tab-recall)

สาเหตุที่เป็นไปได้ของการสูญเสีย recall ของ DME คือ โมเดล LOSO ได้เรียนรู้ DME จากแหล่งข้อมูลที่น้อยลง เนื่องจาก PAIMA ไม่มีข้อมูล DME เลย เมื่อแยก Kermany ไว้ โมเดลจึงเรียนรู้ลักษณะของ DME จาก OCTDL และ Duke เท่านั้น

งานวิจัยนี้ทดสอบสมมติฐานนี้สำหรับ Kermany ด้วยโมเดลแบบ fused ที่ฝึกสอนบนทั้ง 4 แหล่งข้อมูล แต่ตัดภาพ DME ของ Kermany ออกทั้งหมด (ทั้งชุดฝึกสอน (training set) และชุดตรวจสอบความถูกต้อง (validation set)) โดยคลาสและแหล่งข้อมูลอื่นคงเดิม และฝึกสอนแบบ 5 ส่วน (5 folds) ตาม Duke ภายใต้ขั้นตอนเดียวกับโมเดลแบบ fused ปกติ

เงื่อนไขนี้ทำให้ Kermany เป็นแหล่งข้อมูลที่เคยเห็นแต่ไม่มีข้อมูล DME ของตนเอง การเปรียบเทียบกับโมเดลแบบ fused ปกติจึงแยกผลจากการขาด DME ได้ ส่วนการเปรียบเทียบกับโมเดลแบบ LOSO แยกผลจากการเป็นแหล่งข้อมูลที่ไม่เคยเห็น (unseen source)

// Rows of the zero-DME control: per-arm levels, and paired differences between arms (italic).
#let zd-level(label, key, arm) = (
  [#label],
  zd-pt(key, arm),
  zd-pt(key, arm, name: "recall_normal"),
  zd-pt(key, arm, name: "recall_amd"),
  if "recall_dme" in ZD.at(key).arms.at(arm).metrics { zd-pt(key, arm, name: "recall_dme") } else [--],
)
#let zd-diff(label, key, pair) = (
  [_#label _],
  zd-dl(key, pair),
  zd-dl(key, pair, name: "recall_normal"),
  zd-dl(key, pair, name: "recall_amd"),
  if "recall_dme" in ZD.at(key).deltas.at(pair) { zd-dl(key, pair, name: "recall_dme") } else [--],
)
#let zd-key = "kermany|all"
#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      table.header(
        [โมเดล], [Macro-F1], [ปกติ], [AMD], [DME],
        table.hline(),
      ),
      ..zd-level("โมเดลแบบ fused", zd-key, "fused-fill"),
      ..zd-level("ไม่รวม DME ของ Kermany", zd-key, "zero-dme"),
      ..zd-level("LOSO (แยกไว้)", zd-key, "loso-fill"),
      table.hline(stroke: 0.3pt + gray),
      ..zd-diff("ไม่รวม DME − fused", zd-key, "zero-dme - fused-fill"),
      ..zd-diff("LOSO − ไม่รวม DME", zd-key, "loso-fill - zero-dme"),
      table.hline(),
    )
  },
  caption: [ชุดข้อมูลทดสอบของ Kermany ภายใต้โมเดลแบบ fused, โมเดลที่ไม่รวม DME ของ Kermany และโมเดลแบบ LOSO: ค่า Macro-F1 และค่า recall รายคลาส (%) แถวตัวเอียงเป็นผลต่างแบบจับคู่พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-zero-dme>

@tab-zero-dme เปรียบเทียบโมเดลของ Kermany ทั้งสามแบบ การตัดข้อมูล DME ของ Kermany ออกทำให้ Kermany เปลี่ยนไป #zd-dl("kermany|all", "zero-dme - fused-fill") ซึ่งทั้งหมดเกิดจากค่า recall ของภาวะ DME (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_dme")) ขณะที่ค่า recall ของปกติและ AMD ยังคงอยู่ในระดับสัญญาณรบกวน บนชุดภาพทดสอบเดียวกัน ผลกระทบนี้คิดเป็นประมาณร้อยละ #zd-share() ของส่วนต่างเมื่อแยกแหล่งข้อมูลไว้ แต่ทั้งสองโมเดลล้มเหลวในลักษณะที่ต่างกัน ส่วนต่างจึงไม่แบ่งตามสัดส่วนนี้โดยตรง การขาดหายไปของ DME ในอุปกรณ์เดียวกันเพียงอย่างเดียวทำให้สูญเสียค่า recall ของ DME มากกว่าที่โมเดล LOSO สูญเสีย จึงสามารถอธิบายข้อผิดพลาดของ DME ในโมเดล LOSO ได้ ส่วนการสูญเสียอีกส่วนของโมเดล LOSO คือค่า recall ของภาพปกติ (#zd-dl("kermany|all", "loso-fill - zero-dme", name: "recall_normal") เมื่อเทียบกับโมเดลที่ไม่รวม DME ของ Kermany) ซึ่งมาจากการเป็นแหล่งข้อมูลที่ไม่เคยเห็นและตรงกับสิ่งชี้แนะจากขนาดภาพ การขาดหายไปของ DME จึงเป็นสาเหตุที่มีจริงแต่เป็นเพียงสาเหตุบางส่วน

== วิธีการปรับแก้หลังการฝึกสอน (Post-hoc fixes) <sec-posthoc>

วิธีปรับแก้หลังการฝึกสอน (post-hoc fixes) มาตรฐาน 2 วิธียังเหลือส่วนต่างประสิทธิภาพเกือบทั้งหมด (@tab-posthoc) โดยทั้งสองวิธีปรับจูนพารามิเตอร์บนผู้ป่วยครึ่งหนึ่งของแหล่งข้อมูลที่แยกไว้ และประเมินบนผู้ป่วยอีกครึ่งที่เหลือ

วิธีแรกทดสอบว่าส่วนต่างเกิดจากจุดตัดการตัดสินใจที่คลาดเคลื่อนหรือไม่ (@sec-unseen) โดยปรับเลื่อนค่าเอนเอียง (bias) ของค่าความน่าจะเป็นเชิงลอการิทึมที่เอาต์พุตแยกตามคลาส ปรับจูนด้วยป้ายข้อมูลของแหล่งข้อมูลนั้นเอง วิธีนี้เพิ่มคะแนนให้ Kermany #lb-dl("kermany|all", "bias-crossfit") และเปลี่ยนแหล่งข้อมูลอื่นเพียงเล็กน้อย แม้แต่การปรับเลื่อนจุดตัดที่ดีที่สุดซึ่งคำนวณจากป้ายข้อมูลทั้งหมดของแหล่งข้อมูล (คอลัมน์ Ceiling; สำหรับ #srcs.filter(s => LB.at(s + "|" + view3(s)).grid.subsampled).map(s => source-name.at(s)).join(", ") คำนวณจากกลุ่มตัวอย่างย่อยแบบสุ่ม) ซึ่งใช้งานจริงไม่ได้ ก็ได้เพียง #fmt(LB.at("kermany|all").grid.best) เมื่อเทียบกับ #pt("loso_full|kermany|all", "loso-sq") จากการฝึกสอนใหม่โดยไม่มีสิ่งชี้แนะจากขนาดภาพ แสดงว่าการปรับจุดตัดคงที่ร่วมกันทุกภาพแก้ข้อผิดพลาดที่ขึ้นกับขนาดภาพไม่ได้

#let lbk(s) = s + "|" + view3(s)
#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held-out Source], table.cell(rowspan: 2)[LOSO],
        table.cell(colspan: 2, align: center)[Cutoff Shift], table.cell(rowspan: 2)[AdaBN],
        table.cell(rowspan: 2)[Square Crop],
        [Cross-fit], [Ceiling],
        table.hline(),
      ),
      ..srcs
        .map(s => (
          [#source-name.at(s)],
          lb-pt(lbk(s), "none"),
          lb-dl(lbk(s), "bias-crossfit"),
          fmt(LB.at(lbk(s)).grid.best),
          ab-dl(lbk(s)),
          dl(k("loso_full", s), "loso-sq - loso-fill"),
        ))
        .flatten(),
      table.hline(),
    )
  },
  caption: [การปรับแก้หลังการฝึกสอนบนแหล่งข้อมูลที่ถูกแยกไว้ทั้งหมด (Macro-F1 %, โมเดลแบบ LOSO แบบ letterbox) โดยคอลัมน์วิธีการแก้ไขเป็นผลต่างแบบจับคู่เทียบกับ LOSO พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-posthoc>

วิธีที่สองคือ AdaBN @li2016adabn ซึ่งประมาณค่าสถิติของชั้น BatchNorm ใหม่จากภาพไม่มีป้ายข้อมูลของแหล่งข้อมูลที่ไม่เคยเห็น วิธีนี้เพิ่มคะแนนให้ Kermany #ab-dl("kermany|all") แต่น้อยกว่าการปรับเลื่อนจุดตัด และลดคะแนนของ OCTDL ลง (#ab-dl("octdl|all")) ซึ่งผลักดันภาพให้ถูกทำนายเป็น DME มากขึ้น ทั้งสองวิธีจึงไม่น่าเชื่อถือข้ามแหล่งข้อมูล และบน Kermany ไม่มีวิธีใดเข้าใกล้การฝึกสอนใหม่โดยไม่มีสิ่งชี้แนะจากขนาดภาพ
