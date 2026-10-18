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

เมื่อทำการประเมินผลบนแหล่งข้อมูลที่ใช้ในการฝึกสอน โมเดล ResNet-50 สามารถทำค่า Macro-F1 ได้ระหว่าง #pt(k("seen", "paima"), "per-dataset")–#pt(k("seen", "kermany"), "per-dataset") (@tab-seen; สำหรับชุดข้อมูล Duke เป็นการรวมผลลัพธ์จากทั้ง 5 ส่วน โดยแต่ละดวงตาประเมินด้วยโมเดลที่ไม่ได้ฝึกสอนด้วยดวงตานั้น) การฝึกสอนโมเดลเดี่ยวบนทั้ง 4 แหล่งข้อมูลร่วมกัน (โมเดลแบบ fused) แทบไม่สูญเสียประสิทธิภาพ โดยในทุกแหล่งข้อมูล โมเดลแบบ fused สูญเสียคะแนนไม่เกิน #whole(calc.min(..srcs.map(s => delta(k("seen", s), "fused-fill - per-dataset").lo))) จุดเมื่อเทียบกับโมเดลแบบรายชุดข้อมูล (ขอบเขตล่างของช่วงความเชื่อมั่นแบบจับคู่ร้อยละ 95) และการลดลงมีนัยสำคัญที่วัดได้เฉพาะใน Kermany เท่านั้น (#dl(k("seen", "kermany"), "fused-fill - per-dataset")) ในทำนองเดียวกัน วิธีการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัส (square crop) ซึ่งขจัดสิ่งชี้แนะจากขนาดภาพออกไป (@sec-square) ก็ไม่ส่งผลเสียต่อแหล่งข้อมูลที่เคยเห็นเช่นกัน โดยไม่มีแหล่งข้อมูลใดที่มีคะแนนลดลงเกินระดับสัญญาณรบกวนเมื่อเทียบกับโมเดลแบบ fused และชุดข้อมูล PAIMA ยังคงได้คะแนนเพิ่มขึ้นอีก #dl(k("seen", "paima"), "fused-sq - fused-fill")

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

ค่า recall แยกตามคลาส (@tab-recall) เผยให้เห็นลักษณะของข้อผิดพลาดที่เกิดขึ้น โดย _dry_ และ _wet_ คือค่า recall ภายในคลาส AMD ทั้งนี้ คลาส AMD ในชุดข้อมูล Duke เป็นชนิดแห้งทั้งหมด และไม่ได้แสดงระยะปานกลางของ OCTDL ทั้งนี้ PAIMA ไม่มีภาวะ DME ชุดข้อมูล PAIMA เป็นแหล่งข้อมูลที่ท้าทายที่สุดแม้จะเป็นแหล่งข้อมูลที่เคยเห็นก็ตาม (ได้ค่า Macro-F1 เท่ากับร้อยละ #pt(k("seen", "paima"), "fused-fill") สำหรับโมเดลแบบ fused) ซึ่งข้อผิดพลาดหลักเกิดขึ้นในคลาส AMD (มีค่า recall เพียงร้อยละ #pt(k("seen", "paima"), "fused-fill", name: "recall_amd")) และเมื่อพิจารณาเจาะลึกในกลุ่ม AMD พบว่าข้อผิดพลาดเกิดจากโรค AMD ชนิดแห้ง (dry AMD) เป็นหลัก กล่าวคือ โมเดลแบบ fused สามารถจำแนกภาพสแกนที่มีดรูเซน (drusen) ในชุดข้อมูล PAIMA ได้ถูกต้องเพียงร้อยละ #as-rec("test|paima|normal_amd", "dry", "fused-fill") เมื่อเทียบกับภาพถ่ายชนิดเปียก (wet AMD) ที่ทำได้ถึงร้อยละ #as-rec("test|paima|normal_amd", "wet", "fused-fill") โดยภาพถ่ายชนิดมีดรูเซนที่ทำนายผิดเกือบทั้งหมดถูกจำแนกว่าเป็นภาพปกติ (normal) โมเดลแบบรายชุดข้อมูล (per-dataset) ของชุดข้อมูล PAIMA ก็แสดงผลลัพธ์ในลักษณะเดียวกัน (ร้อยละ #as-rec("test|paima|normal_amd", "dry", "per-dataset") เทียบกับร้อยละ #as-rec("test|paima|normal_amd", "wet", "per-dataset")) ข้อจำกัดนี้จึงเกิดจากลักษณะเฉพาะของภาพสแกนชนิดดรูเซนในชุดข้อมูล PAIMA เอง ไม่ได้เกิดจากกระบวนการรวมข้อมูล (fusing) ภาพถ่ายชนิดมีดรูเซนในชุดข้อมูล OCTDL ได้รับการจำแนกถูกต้องน้อยกว่าชนิดเปียกเช่นกัน (ร้อยละ #as-rec("test|octdl|all", "dry", "fused-fill") เทียบกับร้อยละ #as-rec("test|octdl|all", "wet", "fused-fill") สำหรับโมเดลแบบ fused) ทว่าในชุดข้อมูลทดสอบ (test split) ของ OCTDL มีภาพถ่ายชนิดมีดรูเซนเพียง #AS.at("test|octdl|all").stages.dry.n_images ภาพจากผู้ป่วย #AS.at("test|octdl|all").stages.dry.n_groups รายเท่านั้น สำหรับภาพถ่ายชนิดมีดรูเซนในชุดข้อมูล Kermany นั้น ทั้งสองโมเดลสามารถจำแนกได้อย่างถูกต้องอย่างน้อยร้อยละ #fmt(calc.min(..("per-dataset", "fused-fill").map(a => AS.at("test|kermany|all").stages.dry.arms.at(a).recall))) ของกรณีทั้งหมด

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

ทุกแหล่งข้อมูลมีประสิทธิภาพลดลงเกินระดับสัญญาณรบกวนเมื่อเป็นแหล่งข้อมูลที่ไม่เคยเห็น (@tab-unseen) โดยคอลัมน์ทั้งแหล่งข้อมูลประเมินโมเดลแบบ LOSO บนภาพทั้งหมดของแหล่งข้อมูลที่ถูกแยกไว้ และชุดข้อมูล Duke ที่ถูกแยกไว้ประเมินด้วยโมเดลเดี่ยวบนดวงตาทั้ง #DS.duke.n_groups ดวง สำหรับชุดข้อมูล OCTDL, PAIMA และ Duke ส่วนต่างของค่า Macro-F1 อยู่ที่ #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) จุด ขณะที่ชุดข้อมูล Kermany มีส่วนต่างอยู่ที่ #dl(k("gap_test", "kermany"), "loso-fill - fused-fill")

สำหรับแหล่งข้อมูลประเภท 3 คลาส ส่วนต่างของค่า AUROC มีค่าน้อยกว่าส่วนต่างของค่า Macro-F1 อย่างมาก (@tab-auroc; สำหรับ Kermany #dl(k("gap_test", "kermany"), "loso-fill - fused-fill", name: "macro_auroc")) ซึ่งชี้ให้เห็นว่าโมเดลยังคงจัดอันดับคลาสที่ถูกต้องไว้ในลำดับต้น ๆ ได้ดีในภาพส่วนใหญ่ แต่การตัดสินใจมักตกไปอยู่อีกฝั่งของจุดตัดการตัดสินใจ ชุดข้อมูล PAIMA เป็นข้อยกเว้นเพียงหนึ่งเดียว โดยมีส่วนต่างของค่า AUROC อยู่ที่ #dl(k("gap_test", "paima"), "loso-fill - fused-fill", name: "macro_auroc") เมื่อเปรียบเทียบกับส่วนต่างของค่า Macro-F1 ที่ #dp(k("gap_test", "paima"), "loso-fill - fused-fill") ซึ่งสะท้อนถึงการเปลี่ยนแปลงในความสามารถด้านการจัดอันดับ ไม่ใช่เพียงแค่การคลาดเคลื่อนของจุดตัดเท่านั้น

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
      ..srcs.map(s => (
        [#source-name.at(s)],
        pt(k("gap_test", s), "fused-fill", name: "macro_auroc"),
        pt(k("gap_test", s), "loso-fill", name: "macro_auroc"),
        dl(k("gap_test", s), "loso-fill - fused-fill", name: "macro_auroc"),
      )).flatten(),
      table.hline(),
    )
  },
  caption: [Macro-AUROC ของแหล่งข้อมูลที่เคยเห็นเทียบกับไม่เคยเห็น (ชุดทดสอบ, letterbox) โดย _ส่วนต่าง_ คือผลต่างแบบจับคู่ พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-auroc>

ชุดข้อมูลทดสอบ (test set) ของ Kermany ไม่ได้เป็นตัวแทนที่ดีของชุดข้อมูลโดยรวม กล่าวคือ มีภาพจำนวนถึง #DS.kermany.test_sizes.at("1024x496") ภาพจากทั้งหมด #num(DS.kermany.n_test) ภาพที่มีขนาด 1024×496 พิกเซล ซึ่งเป็นขนาดที่มีสัดส่วนเพียงร้อยละ #pct1(size-share("kermany", "1024x496") * 100) ในชุดข้อมูล Kermany ทั้งหมด

เมื่อประเมินบนภาพทั้งหมดจำนวน #num(R.at("loso_full|kermany|all").n_images) ภาพ คะแนนของชุดข้อมูล Kermany ในกรณีที่เป็นแหล่งข้อมูลที่ถูกแยกไว้ (held-out) ทำได้เพียง #pt("loso_full|kermany|all", "loso-fill") ซึ่งต่ำกว่าคะแนน #pt("gap_test|kermany|all", "loso-fill") บนชุดข้อมูลทดสอบ (test split) อย่างมาก งานวิจัยนี้จึงใช้คะแนนแบบรวมทั้งแหล่งข้อมูล (full-source score) เป็นเกณฑ์หลักในการอ้างอิงข้อสรุปเกี่ยวกับชุดข้อมูล Kermany ในฐานะแหล่งข้อมูลที่ไม่เคยเห็น โดยในหัวข้อถัดไปจะแสดงให้เห็นว่าเหตุใดขนาดของภาพจึงส่งผลกระทบอย่างยิ่งยวด

== สิ่งชี้แนะทางลัดจากขนาดภาพ <sec-shortcut>

ในชุดข้อมูล Duke ขนาดของภาพสามารถระบุคลาสของข้อมูลได้ถึงร้อยละ #calc.round((int(duke-eyes("amd", size: "1024x496")) + int(duke-eyes("dme", size: "768x496"))) / DS.duke.n_groups * 100) ของดวงตาทั้งหมด (@sec-data) โมเดลที่ได้รับการฝึกสอนด้วยชุดข้อมูล Duke จึงสามารถเรียนรู้กฎทางลัดที่ว่า "ความกว้าง 768 พิกเซลหมายถึงภาวะ DME" ซึ่งเป็นกฎที่ถูกต้องภายในชุดข้อมูล Duke แต่เกิดความผิดพลาดเมื่อนำไปประเมินกับชุดข้อมูลอื่น การทดสอบ 3 ประการต่อไปนี้ช่วยยืนยันว่าโมเดลได้เรียนรู้กฎดังกล่าวจริง

=== การปรับเปลี่ยนเฉพาะขนาดของภาพทดสอบ

งานวิจัยนี้ได้นำโมเดลแบบ LOSO ที่แยกชุดข้อมูล Kermany ไว้ มาประเมินภาพถ่ายในชุดข้อมูล Kermany ซ้ำ โดยปรับเปลี่ยนเฉพาะความกว้างของภาพโดยไม่มีการฝึกสอนโมเดลใหม่ (@tab-size-test) การเติมขอบภาพ (padding) ในภาพปกติขนาด 512×496 พิกเซล ให้มีความกว้างเป็น 768 พิกเซล (ซึ่งเป็นความกว้างเฉพาะของภาวะ DME ในชุดข้อมูล Duke) ส่งผลให้สัดส่วนภาพที่ถูกทำนายว่าเป็นภาวะ DME เพิ่มขึ้นจากร้อยละ #pct1(KS.pad.at("512x496").at("normal->dme").at(0)) เป็นร้อยละ #pct1(KS.pad.at("512x496").at("normal->dme").at(1)) และการครอบตัด (crop) ภาพปกติขนาด 768×496 พิกเซล ให้เหลือความกว้าง 512 พิกเซล ให้ผลลัพธ์ในทิศทางตรงกันข้าม ปรากฏการณ์นี้ไม่ได้จำกัดอยู่เพียงแค่ชุดข้อมูล Kermany เท่านั้น โดยการเติมขอบภาพขนาด 512×496 พิกเซลของชุดข้อมูล Duke เพิ่มขึ้นอีก 16 แถว จนกลายเป็นขนาด 512×512 พิกเซล (ซึ่งเป็นขนาดที่พบเฉพาะภาวะ DME ใน Kermany) ให้ผลลัพธ์ในลักษณะเดียวกันในโมเดลที่แยกชุดข้อมูล Duke ไว้

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

ความกว้างของภาพถูกส่งผ่านไปยังโมเดลผ่านทางแถบขอบภาพแบบ letterbox เมื่อภาพสแกนขนาด 768×496 พิกเซล ถูกปรับขนาดให้พอดีกับกรอบสี่เหลี่ยมจัตุรัส จะเกิดแถบขอบเรียบที่ด้านบนและด้านล่างของเนื้อเยื่อ ซึ่งคิดเป็นพื้นที่ประมาณหนึ่งในสามของภาพทั้งหมด โดยภาพสแกนขนาด 768×496 พิกเซลที่เป็นภาวะ DME ทั้งหมดในชุดข้อมูล Duke ก็มีแถบขอบในลักษณะเดียวกันนี้ สำหรับโมเดลแบบแยกชุดข้อมูล Kermany ไว้ (LOSO) ทั้ง 5 โมเดล งานวิจัยนี้ได้ทำการสุ่มภาพขนาด 768×496 พิกเซล กลุ่มละ #gc-group-n("768 normal → DME") ภาพ (ผู้ป่วยละ 1 ภาพ) ประกอบด้วย ภาพปกติ (normal) และโรค AMD ที่โมเดลทำนายผิดพลาดเป็นภาวะ DME รวมทั้งภาพปกติและโรค AMD ที่โมเดลทำนายได้อย่างถูกต้อง จากนั้นได้ทำการแทรกแซงข้อมูล (intervention) โดยการแก้ไขพิกเซลเฉพาะบริเวณแถบขอบให้เป็นสีดำ หรือแทนที่ด้วยสัญญาณรบกวนแบบเกรน (speckle noise) ที่สุ่มมาจากบริเวณครึ่งที่มีความเข้มมืดของภาพสแกน แล้วจึงทำการประเมินผลคะแนนใหม่ (@tab-bands)

สำหรับโมเดลที่ 1, 2 และ 4 พบว่ามีข้อผิดพลาดหลงเหลืออยู่ไม่เกินร้อยละ #{
  let xs = (1, 2, 4).map(f => ("768 normal → DME", "768 AMD → DME").map(g => ("black", "speckle").map(b => gc-at(f, g, b)))).flatten()
  pct0(calc.max(..xs))
}% หลังจากเปลี่ยนแถบขอบทั้งสองรูปแบบ ขณะที่โมเดลที่ 3 และ 5 พบว่ามีข้อผิดพลาดหลงเหลืออยู่ไม่เกินร้อยละ #{
  let xs = (3, 5).map(f => ("768 normal → DME", "768 AMD → DME").map(g => gc-at(f, g, "speckle"))).flatten()
  pct0(calc.max(..xs))
}% เมื่อใช้แถบขอบแบบ speckle ซึ่งแสดงให้เห็นว่าแถบขอบภาพเป็นช่องทางหลัก แต่ไม่ใช่ช่องทางเดียวที่สัญญาณนี้ถูกส่งผ่านไปถึงโมเดล ในทางตรงกันข้าม #{
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

แผนภาพ Grad-CAM (เลเยอร์คอนโวลูชันสุดท้าย สำหรับคลาสที่ทำนาย) ช่วยแสดงให้เห็นกลไกเดียวกันนี้ในรูปแบบภาพตัวอย่าง (@fig-gradcam) โดยเมื่อทำการสุ่มค่าน้ำหนักในชั้นบนสุดของโครงข่ายใหม่ แผนภาพจะไม่สัมพันธ์กับแผนภาพของโมเดลที่ผ่านการฝึกสอนแล้วอีกต่อไป (ค่ามัธยฐานสหสัมพันธ์สเปียร์แมน #calc.round(GC.randomization_fold1_768_normal_to_dme.spearman_median, digits: 2) ในโมเดลที่ 1) จึงสะท้อนถึงสิ่งที่โมเดลเรียนรู้ อย่างไรก็ดี แผนภาพการระบุส่วนสำคัญยังอาจขัดแย้งกับผลลัพธ์ที่เกิดขึ้นจากการแทรกแซงข้อมูลจริงได้ @arun2021saliency @hooker2019roar งานวิจัยนี้จึงใช้การแทรกแซงบริเวณแถบขอบภาพเป็นหลักฐานเชิงประจักษ์ และใช้ Grad-CAM เป็นเพียงภาพประกอบเสริมเท่านั้น

#figure(
  image("../../figures/gradcam.png", width: 100%),
  caption: [Grad-CAM บนภาพปกติขนาด 768×496 จากชุดข้อมูล Kermany ที่ถูกแยกไว้ สำหรับโมเดล LOSO แบบ letterbox (บน) และ square-crop (ล่าง) โดยสองคอลัมน์สุดท้ายเป็นกลุ่มควบคุมที่ทั้งสองโมเดลทำนายเป็นปกติ],
) <fig-gradcam>

=== การฝึกสอนโดยไม่มีสิ่งชี้แนะจากขนาดภาพ

โมเดลแบบ square crop จะไม่เคยเห็นความกว้างของภาพสแกนเลย ซึ่งเมื่อแยกชุดข้อมูล Kermany ไว้ โมเดลนี้ทำคะแนนบนแหล่งข้อมูลทั้งหมดได้ #pt("loso_full|kermany|all", "loso-sq") เมื่อเทียบกับ #pt("loso_full|kermany|all", "loso-fill") (#dl("loso_full|kermany|all", "loso-sq - loso-fill")) และเมื่อประเมินบนชุดข้อมูลทดสอบ เปรียบเทียบกับโมเดลแบบ fused ของกลุ่มการทดลองเดียวกัน พบว่าส่วนต่างแคบลงประมาณร้อยละ #calc.round((1 - delta("gap_test|kermany|all", "loso-sq - fused-sq").point / delta("gap_test|kermany|all", "loso-fill - fused-fill").point) * 100)% จากเดิมที่ #dl("gap_test|kermany|all", "loso-fill - fused-fill") ประสิทธิภาพที่เพิ่มขึ้นเกิดขึ้นตรงกับจุดที่มีสิ่งชี้แนะพอดี (@fig-kermany-size): ที่ขนาด 768×496 ซึ่งเป็นความกว้างที่พบเฉพาะ DME ใน Duke การครอบตัดแบบ square crop ช่วยเพิ่มความแม่นยำได้มากที่สุดสำหรับภาพปกติและ AMD (ค่า recall ของภาพปกติเพิ่มจาก #pt("size_loso:768x496|kermany|all", "fill", name: "recall_normal") เป็น #pt("size_loso:768x496|kermany|all", "sq", name: "recall_normal")) และทำให้ค่า recall ของ DME ลดลง

#figure(
  kind: image,
  fig("kermany-size"),
  caption: [Recall ของชุดข้อมูล Kermany ที่ถูกแยกไว้ (ภาพทั้งหมด) จำแนกตามขนาดภาพและคลาสจริง สำหรับโมเดล LOSO แบบ letterbox และ square-crop พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <fig-kermany-size>

ทว่าการตัดสิ่งชี้แนะออกไม่ได้ช่วยในทุกแหล่งข้อมูล บนแหล่งข้อมูลอื่น ๆ ที่ถูกแยกไว้ การครอบตัดแบบ square crop สูญเสียความแม่นยำเกินระดับสัญญาณรบกวน (@tab-square): ได้แก่ ลดลง #span(("octdl", "duke", "paima").map(s => delta(k("loso_full", s), "loso-sq - loso-fill").point)) จุด Macro-F1 สำหรับ OCTDL, Duke และ PAIMA โดย PAIMA ยังสูญเสียค่า AUROC อีก #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill", name: "macro_auroc") ซึ่งเป็นการสูญเสียความสามารถด้านการจัดอันดับที่ผันแปรตามค่าซีดอย่างมาก (@sec-limits)

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

สำหรับ OCTDL และ Duke ผลเสียที่เกิดขึ้นเกี่ยวข้องกับภาวะ DME (@tab-square-dme) ในชุดข้อมูล OCTDL ค่า recall ของภาพปกติเปลี่ยนไป #dl("loso_full|octdl|all", "loso-sq - loso-fill", name: "recall_normal") เนื่องจากสัดส่วนภาพปกติที่ถูกทำนายเป็น DME เปลี่ยนจากร้อยละ #share("loso_full|octdl|all", "loso-fill", "normal", "dme")% เป็นร้อยละ #share("loso_full|octdl|all", "loso-sq", "normal", "dme")% ในขณะที่บน Duke ค่า recall ของภาวะ DME เปลี่ยนไป #dl("loso_full|duke|all", "loso-sq - loso-fill", name: "recall_dme") แต่ในมุมมอง Normal/AMD ซึ่งตัดภาพ DME และผลลัพธ์ DME ออก ผลเสียทั้งสองจุดหายไป กล่าวคือ เกิดการทำนายเป็น DME มากเกินไปใน OCTDL และทำนายเป็น DME น้อยเกินไปใน Duke

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 5,
      align: (left, left, right, right, right),
      table.header([แหล่งข้อมูลที่กันไว้], [ตัวชี้วัด], [Letterbox], [Square], [ผลต่าง], table.hline()),
      ..("octdl", "duke").map(s => {
        let key = "loso_full|" + s + "|"
        (
          table.cell(rowspan: 3)[#source-name.at(s)],
          [Normal/AMD macro-F1],
          pt(key + "normal_amd", "loso-fill"), pt(key + "normal_amd", "loso-sq"),
          dl(key + "normal_amd", "loso-sq - loso-fill"),
          [Normal recall],
          pt(key + "all", "loso-fill", name: "recall_normal"), pt(key + "all", "loso-sq", name: "recall_normal"),
          dl(key + "all", "loso-sq - loso-fill", name: "recall_normal"),
          [DME recall],
          pt(key + "all", "loso-fill", name: "recall_dme"), pt(key + "all", "loso-sq", name: "recall_dme"),
          dl(key + "all", "loso-sq - loso-fill", name: "recall_dme"),
          table.hline(stroke: 0.3pt + gray),
        )
      }).flatten(),
      table.hline(),
    )
  },
  caption: [OCTDL และ Duke ที่ถูกแยกไว้ บนแหล่งข้อมูลทั้งหมด: letterbox เทียบกับ square crop (%) รวมถึงมุมมอง Normal/AMD ที่ตัด DME ออก ผลต่างเป็นผลต่างแบบจับคู่ พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-square-dme>

ด้วยเหตุนี้ งานวิจัยนี้จึงกำหนดให้โมเดลแบบ letterbox เป็นการทดลองหลัก และจัดให้การครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัส (square crop) เป็นการทดลองตัดทอนองค์ประกอบ (ablation study) ซึ่งช่วยยืนยันว่าสิ่งชี้แนะทางลัดจากขนาดภาพเกิดขึ้นจริง และสามารถอธิบายส่วนต่างประสิทธิภาพในชุดข้อมูล Kermany ได้เป็นส่วนใหญ่ ทว่าไม่ใช่วิธีการแก้ปัญหาที่สามารถประยุกต์ใช้ได้เป็นการทั่วไป

ทั้งนี้ งานวิจัยนี้ได้กำหนดเกณฑ์การตีความดังกล่าวไว้ล่วงหน้าก่อนทราบผลการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัสของชุดข้อมูล OCTDL และ Duke ว่า วิธีการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัสจะถูกนำมาใช้เป็นรูปแบบหลักก็ต่อเมื่อไม่มีแหล่งข้อมูลใดสูญเสียประสิทธิภาพเกินระดับสัญญาณรบกวนเท่านั้น

== ส่วนต่างความแม่นยำที่ยังคงหลงเหลือ <sec-remaining>

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held-out Source], table.cell(rowspan: 2)[ส่วนต่าง 3 คลาส],
        table.cell(rowspan: 2)[DME recall], table.cell(colspan: 3, align: center)[Macro-F1 Normal/AMD],
        [Seen], [Unseen], [Gap],
        table.hline(),
      ),
      ..srcs
        .map(s => (
          [#source-name.at(s)],
          ..if s == "paima" { ([--], [--]) } else {
            (
              dl("gap_test|" + s + "|all", "loso-fill - fused-fill"),
              dl("gap_test|" + s + "|all", "loso-fill - fused-fill", name: "recall_dme"),
            )
          },
          pt("gap_test|" + s + "|normal_amd", "fused-fill"),
          pt("gap_test|" + s + "|normal_amd", "loso-fill"),
          dl("gap_test|" + s + "|normal_amd", "loso-fill - fused-fill"),
        ))
        .flatten(),
      table.hline(),
    )
  },
  caption: [ที่มาของส่วนต่างจากแหล่งข้อมูลที่ไม่เคยเห็น (unseen-source gap) (ประเมินโมเดลแบบ letterbox บนชุดข้อมูลทดสอบ (test split) คำนวณจากผลลัพธ์ของโมเดลแบบ LOSO ลบด้วยโมเดลแบบ fused มีหน่วยเป็นจุดเปอร์เซ็นต์ (percentage points) พร้อมช่วงความเชื่อมั่นแบบจับคู่ร้อยละ 95) ทั้งนี้ ชุดข้อมูล PAIMA ได้รับการประเมินผลเฉพาะการจำแนกระหว่างภาพปกติ (normal) กับโรค AMD ดังนั้น คะแนนในมุมมองภาพปกติ/โรค AMD (Normal/AMD view) จึงถือเป็นคะแนนหลักของชุดข้อมูลนี้ โดยคะแนนแบบ 3 คลาสแสดงไว้ใน @tab-unseen และค่า recall ของภาวะ DME แสดงไว้ใน @tab-recall-unseen ทั้งนี้ การตัดคลาสภาวะ DME ออก ช่วยลดส่วนต่างดังกล่าวลงได้เกือบทั้งหมดสำหรับชุดข้อมูล OCTDL และ Duke ส่วนชุดข้อมูล PAIMA ไม่มีข้อมูลในคลาส DME],
) <tab-remaining>

ส่วนต่างประสิทธิภาพที่ยังคงหลงเหลืออยู่เกือบทั้งหมดมีความเกี่ยวข้องกับภาวะ DME (@tab-remaining) โดยบนชุดข้อมูลทดสอบ (test split) ค่า recall ของภาวะ DME เมื่อประเมินในฐานะแหล่งข้อมูลที่ถูกแยกไว้ (held-out) ปรับตัวลดลงถึง #span(("kermany", "octdl", "duke").map(s => delta("gap_test|" + s + "|all", "loso-fill - fused-fill", name: "recall_dme").point)) จุดเปอร์เซ็นต์ สำหรับชุดข้อมูล Kermany, OCTDL และ Duke

ในมุมมองภาพปกติ/โรค AMD (Normal/AMD view) ส่วนต่างของชุดข้อมูล OCTDL และ Duke อยู่ในระดับสัญญาณรบกวนซึ่งใกล้เคียงกับศูนย์ (#dl("gap_test|octdl|normal_amd", "loso-fill - fused-fill") และ #dl("gap_test|duke|normal_amd", "loso-fill - fused-fill")) ในขณะที่ชุดข้อมูล Kermany ยังคงมีส่วนต่างในมุมมองภาพปกติ/โรค AMD ขนาดใหญ่ ซึ่งเกิดจากการที่ภาพถ่ายปกติ (normal) ถูกทำนายว่าเป็นโรค AMD กล่าวคือ เมื่อทำการแยกชุดข้อมูล Kermany ไว้ ค่า recall ของภาพถ่ายปกติลดลงจากร้อยละ #pt("gap_test|kermany|all", "fused-fill", name: "recall_normal") เหลือเพียงร้อยละ #pt("gap_test|kermany|all", "loso-fill", name: "recall_normal") (@tab-recall-unseen) และภาพถ่ายปกติที่ถูกทำนายผิดพลาดส่วนใหญ่ถูกระบุว่าเป็นโรค AMD

ซึ่งสอดคล้องกับสิ่งชี้แนะจากขนาดภาพที่ได้กล่าวไว้ข้างต้น (@fig-kermany-size) เนื่องจากภาพในชุดข้อมูลทดสอบส่วนใหญ่ของชุดข้อมูล Kermany มีขนาด 1024×496 พิกเซล ซึ่งเป็นขนาดเฉพาะของโรค AMD ในชุดข้อมูล Duke

สำหรับชุดข้อมูล PAIMA ซึ่งไม่มีข้อมูลในคลาส DME ยังคงมีส่วนต่างอยู่ที่ #dl("gap_test|paima|normal_amd", "loso-fill - fused-fill") โดยมีสาเหตุมาจากค่า recall ของโรค AMD และเป็นโรค AMD ชนิดแห้ง (dry AMD) เป็นหลักเช่นเดิม กล่าวคือ เมื่อทำการแยกชุดข้อมูล PAIMA ไว้ ค่า recall มีการเปลี่ยนแปลงไป #as-dl("test|paima|normal_amd", "dry", "loso-fill - fused-fill") สำหรับภาพสแกนชนิดมีดรูเซน (drusen) และ #as-dl("test|paima|normal_amd", "wet", "loso-fill - fused-fill") สำหรับภาพถ่ายชนิดเปียก (wet AMD)

#figure(
  recall-table(s => {
    let rows = (seen-row(s, "Fused (Seen)", "fused-fill"),)
    if s == "kermany" {
      rows.push(recall-row(s, "Fused (Unseen)", ZD.at("kermany|all").arms.at("zero-dme").metrics, none))
    }
    rows.push(recall-row(s, "LOSO (Held-out)", R.at(k("gap_test", s)).arms.at("loso-fill").metrics, "loso-fill"))
    rows
  }),
  caption: [ค่า recall แยกตามคลาสบนชุดข้อมูลทดสอบ (test split) ของแต่ละแหล่งข้อมูล เมื่อแหล่งข้อมูลนั้นเป็นแบบเคยเห็น (fused) และถูกแยกไว้ (LOSO) สำหรับโมเดลแบบ letterbox โดยรายละเอียดคอลัมน์เป็นไปตาม @tab-recall สำหรับชุดข้อมูล Kermany แสดงผลลัพธ์ของโมเดลแบบ fused ที่ฝึกสอนโดยไม่รวมภาวะ DME ของชุดข้อมูล Kermany ร่วมด้วย (@tab-zero-dme) ทั้งนี้ ค่าผลต่างที่อ้างถึงในเนื้อหามีช่วงความเชื่อมั่นแบบจับคู่ร้อยละ 95],
) <tab-recall-unseen>

สาเหตุที่เป็นไปได้คือ โมเดลแบบ LOSO แต่ละโมเดลได้รับการฝึกสอนด้วยข้อมูลภาวะ DME จากจำนวนแหล่งข้อมูลที่น้อยลง กล่าวคือ ชุดข้อมูล PAIMA ไม่มีข้อมูลภาวะ DME เลย ดังนั้น เมื่อทำการแยกชุดข้อมูล Kermany ไว้ โมเดลจึงเรียนรู้ลักษณะเฉพาะของภาวะ DME จากชุดข้อมูล OCTDL และ Duke เท่านั้น

งานวิจัยนี้ได้ทดสอบสมมติฐานดังกล่าวสำหรับชุดข้อมูล Kermany ด้วยโมเดลแบบ fused ที่ฝึกสอนบนทั้ง 4 แหล่งข้อมูล แต่ทำการตัดภาพถ่ายภาวะ DME ของชุดข้อมูล Kermany ออกทั้งหมด (ทั้งในชุดข้อมูลฝึกสอน (training set) และชุดข้อมูลตรวจสอบความถูกต้อง (validation set) โดยคลาสอื่นและแหล่งข้อมูลอื่นยังคงเดิม พร้อมทั้งฝึกสอนแบบ 5 ส่วน (5 folds) ตามชุดข้อมูล Duke ภายใต้ขั้นตอนเดียวกันกับโมเดลแบบ fused ปกติ

เงื่อนไขนี้ส่งผลให้ชุดข้อมูล Kermany กลายเป็นแหล่งข้อมูลที่เคยเห็น แต่ปราศจากข้อมูลภาวะ DME ของตนเอง การเปรียบเทียบกับโมเดลแบบ fused ปกติจึงช่วยแยกผลกระทบจากการขาดหายไปของข้อมูลภาวะ DME ออกมาได้อย่างชัดเจน ขณะเดียวกันการเปรียบเทียบกับโมเดลแบบ LOSO จะช่วยแยกผลกระทบอันเกิดจากการเป็นแหล่งข้อมูลที่ไม่เคยเห็น (unseen source)

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 6,
      align: (left, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Source], table.cell(colspan: 3, align: center)[Macro-F1],
        table.cell(colspan: 2, align: center)[DME recall],
        [Held-out], [Fused], [ผลต่าง], [Held-out], [Fused],
        table.hline(),
      ),
      ..srcs
        .map(s => {
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
        })
        .flatten(),
      table.hline(),
    )
  },
  caption: [ผลลัพธ์ของโมเดลแบบ fused ที่ฝึกสอนโดยตัดภาพถ่ายภาวะ DME ของชุดข้อมูล Kermany ออก (Held-out) และแบบปกติ (Fused) บนชุดข้อมูลทดสอบ (test split) ของแต่ละแหล่งข้อมูล ทั้งนี้ ค่าผลต่างเป็นค่าแบบจับคู่พร้อมช่วงความเชื่อมั่นร้อยละ 95 ในระดับผู้ป่วย],
) <tab-zero-dme>

ค่า recall แยกตามคลาสของโมเดล Kermany ทั้งสามรูปแบบแสดงไว้ใน @tab-recall-unseen การตัดข้อมูลภาวะ DME ของชุดข้อมูล Kermany ออกส่งผลให้คะแนนของชุดข้อมูล Kermany เปลี่ยนแปลงไป #zd-dl("kermany|all", "zero-dme - fused-fill") ซึ่งทั้งหมดเกิดจากค่า recall ของภาวะ DME ที่ลดลง (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_dme")) ขณะที่ค่า recall ของคลาสปกติ (normal) และโรค AMD ยังคงอยู่ในระดับสัญญาณรบกวน (#zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_normal"), #zd-dl("kermany|all", "zero-dme - fused-fill", name: "recall_amd"))

บนชุดภาพทดสอบเดียวกัน ผลกระทบนี้คิดเป็นสัดส่วนประมาณร้อยละ #zd-share() ของขนาดส่วนต่างเมื่อทำการแยกชุดข้อมูล Kermany ไว้ทั้งหมด (LOSO #zd-pt("kermany|all", "loso-fill"), ไม่รวม Kermany DME #zd-pt("kermany|all", "zero-dme"), fused #zd-pt("kermany|all", "fused-fill")) อย่างไรก็ดี โมเดลทั้งสองรูปแบบเกิดความผิดพลาดในลักษณะที่แตกต่างกัน ส่วนต่างประสิทธิภาพจึงไม่ได้แบ่งแยกตามสัดส่วนนี้โดยตรง

การขาดข้อมูลภาวะ DME จากอุปกรณ์สแกนเดียวกัน (in-device) เพียงอย่างเดียว ทำให้สูญเสียค่า recall ของภาวะ DME มากกว่าที่โมเดลแบบ LOSO สูญเสียเสียด้วยซ้ำ (ผลต่างของค่า recall ภาวะ DME ระหว่าง LOSO ลบด้วยแบบไม่มี DME เท่ากับ #zd-dl("kermany|all", "loso-fill - zero-dme", name: "recall_dme")) จึงสามารถอธิบายข้อผิดพลาดด้านภาวะ DME ของโมเดลแบบ LOSO ได้อย่างเพียงพอ

ความสูญเสียอีกส่วนหนึ่งของโมเดลแบบ LOSO คือค่า recall ของภาพถ่ายปกติ (normal) (#zd-dl("kermany|all", "loso-fill - zero-dme", name: "recall_normal") เมื่อเปรียบเทียบกับโมเดลที่ไม่รวม Kermany DME) ซึ่งเกิดจากการเป็นแหล่งข้อมูลที่ไม่เคยเห็นและสอดคล้องกับสิ่งชี้แนะจากขนาดภาพ การขาดข้อมูลภาวะ DME จึงเป็นสาเหตุที่มีอยู่จริง ทว่าอธิบายผลกระทบได้เพียงบางส่วนเท่านั้น

นอกจากนี้ ผลกระทบยังไม่ได้จำกัดอยู่เพียงแหล่งข้อมูลที่ถูกตัดข้อมูลภาวะ DME ออกไปเท่านั้น กล่าวคือ ค่า recall ของภาวะ DME ในชุดข้อมูล Duke มีการเปลี่ยนแปลงไปด้วยเช่นกัน (#zd-dl("duke|all", "zero-dme - fused-fill", name: "recall_dme")) ชี้ให้เห็นว่าข้อมูลภาวะ DME ของชุดข้อมูล Kermany มีส่วนช่วยให้โมเดลจำแนกภาวะ DME ของชุดข้อมูล Duke ได้ดีขึ้น อย่างไรก็ดี การเปลี่ยนแปลงนี้ประเมินจากข้อมูลเพียง #DS.duke.n_groups ดวงตา งานวิจัยนี้จึงถือเป็นเพียงหลักฐานสนับสนุนเท่านั้น ส่วนชุดข้อมูล OCTDL และ PAIMA อยู่ในระดับสัญญาณรบกวน

== วิธีการปรับแก้หลังการฝึกสอน (Post-hoc fixes) <sec-posthoc>

วิธีการปรับแก้หลังการฝึกสอน (post-hoc fixes) แบบมาตรฐาน 2 วิธียังคงมีส่วนต่างประสิทธิภาพหลงเหลืออยู่เกือบทั้งหมด (@tab-posthoc) โดยทั้งสองวิธีทำการปรับจูนพารามิเตอร์บนข้อมูลผู้ป่วยครึ่งหนึ่งของแหล่งข้อมูลที่ถูกแยกไว้ และนำไปประเมินผลบนข้อมูลผู้ป่วยอีกครึ่งหนึ่งที่เหลือ

วิธีแรกเป็นการทดสอบว่าส่วนต่างของประสิทธิภาพเกิดจากจุดตัดการตัดสินใจที่คลาดเคลื่อนหรือไม่ (@sec-unseen) โดยการปรับเลื่อนค่าเอนเอียง (bias) ของค่าความน่าจะเป็นเชิงลอการิทึม (log-probability) ในส่วนเอาต์พุตแยกตามแต่ละคลาส (cutoff shift) ซึ่งทำการปรับจูนด้วยป้ายข้อมูลของแหล่งข้อมูลนั้น ๆ เอง วิธีการนี้ส่งผลให้คะแนนของชุดข้อมูล Kermany มีการเปลี่ยนแปลงไป #lb-dl("kermany|all", "bias-crossfit") และชุดข้อมูล PAIMA เปลี่ยนแปลงไป #lb-dl("paima|normal_amd", "bias-crossfit") ขณะที่ชุดข้อมูล OCTDL (#lb-dl("octdl|all", "bias-crossfit")) และ Duke (#lb-dl("duke|all", "bias-crossfit")) ยังคงอยู่ในระดับสัญญาณรบกวน

แม้กระทั่งการปรับเลื่อนจุดตัดที่ดีที่สุดซึ่งคำนวณจากป้ายข้อมูลทั้งหมดของชุดข้อมูล Kermany (ซึ่งไม่สามารถทำได้ในโมเดลที่นำไปประยุกต์ใช้งานจริง) ก็ทำคะแนนได้เพียง #fmt(LB.at("kermany|all").grid.best) เมื่อเทียบกับ #pt("loso_full|kermany|all", "loso-sq") จากการฝึกสอนโมเดลใหม่โดยปราศจากสิ่งชี้แนะจากขนาดภาพ แสดงให้เห็นว่าการปรับจุดตัดแบบคงที่ที่ใช้ร่วมกันในทุกภาพ ไม่สามารถแก้ไขข้อผิดพลาดที่ขึ้นอยู่กับขนาดของภาพได้

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
  caption: [การปรับแก้หลังการฝึกสอน (Post-hoc fixes) บนแหล่งข้อมูลที่ถูกแยกไว้แบบรวมทั้งแหล่งข้อมูล (full source) (ค่า macro-F1, โมเดลแบบ LOSO แบบ letterbox; ชุดข้อมูล PAIMA ประเมินผลเฉพาะการจำแนกระหว่างภาพปกติ (normal) กับโรค AMD) โดยการปรับจุดตัดการตัดสินใจแบบ Cross-fit, เทคนิค AdaBN และการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัส (square crop) แสดงเป็นค่าผลต่างแบบจับคู่เทียบกับโมเดลแบบ LOSO พร้อมช่วงความเชื่อมั่นร้อยละ 95 ทั้งนี้ ค่าเพดานประสิทธิภาพสูงสุด (Ceiling) คือการปรับจุดตัดการตัดสินใจที่ดีที่สุดบนข้อมูลป้ายข้อมูลทั้งหมด (สำหรับชุดข้อมูล #srcs.filter(s => LB.at(lbk(s)).grid.subsampled).map(s => source-name.at(s)).join(", ") คำนวณจากกลุ่มตัวอย่างย่อยแบบสุ่ม)],
) <tab-posthoc>

วิธีการที่สองคือเทคนิค AdaBN @li2016adabn ซึ่งทำการประมาณค่าสถิติของชั้นแบตช์นอร์มัลไลเซชัน (BatchNorm) ใหม่จากภาพถ่ายที่ไม่มีป้ายข้อมูลของแหล่งข้อมูลที่ไม่เคยเห็น วิธีการนี้ส่งผลให้คะแนนของชุดข้อมูล Kermany มีการเปลี่ยนแปลงไป #ab-dl("kermany|all") ซึ่งน้อยกว่าการปรับเลื่อนจุดตัดการตัดสินใจ ขณะที่ชุดข้อมูล Duke ยังคงอยู่ในระดับสัญญาณรบกวน (#ab-dl("duke|all")) ทั้งยังทำให้ชุดข้อมูล OCTDL มีการเปลี่ยนแปลงไป #ab-dl("octdl|all") โดยส่งผลให้ภาพถ่ายถูกทำนายว่าเป็นภาวะ DME มากขึ้น และชุดข้อมูล PAIMA มีการเปลี่ยนแปลงไป #ab-dl("paima|normal_amd")

ดังนั้น ทั้งสองวิธีการจึงขาดความน่าเชื่อถือเมื่อนำไปประยุกต์ใช้ข้ามแหล่งข้อมูล และสำหรับชุดข้อมูล Kermany ไม่มีวิธีการใดที่ให้ผลลัพธ์ใกล้เคียงกับการฝึกสอนโมเดลใหม่โดยปราศจากสิ่งชี้แนะจากขนาดภาพเลย
