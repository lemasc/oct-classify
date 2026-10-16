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
  caption: [ค่า Macro-F1 บนชุดข้อมูลทดสอบของแต่ละแหล่งข้อมูล พร้อมช่วงความเชื่อมั่นร้อยละ 95 จากการสุ่มตัวอย่างซ้ำด้วยวิธีบูตสแตรป ในระดับผู้ป่วย สำหรับชุดข้อมูล PAIMA ได้รับการประเมินผลเฉพาะการจำแนกระหว่างภาพปกติ (normal) กับโรค AMD ส่วนชุดข้อมูล Duke เป็นการรวมผลลัพธ์จากทั้ง 5 ส่วน (5 folds) (รวมภาพ B-scan ทั้งหมด #num(DS.duke.n_images) ภาพ จากดวงตา #DS.duke.n_groups ดวง โดยดวงตาแต่ละดวงได้รับการประเมินด้วยโมเดลที่ไม่เคยใช้ดวงตาดังกล่าวในการฝึกสอน)],
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

== แหล่งข้อมูลที่ไม่เคยเห็น <sec-unseen>

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
  caption: [การเปรียบเทียบระหว่างแหล่งข้อมูลที่เคยเห็น (fused) กับแหล่งข้อมูลที่ไม่เคยเห็น (LOSO) สำหรับโมเดลแบบ letterbox โดยค่า Gap หมายถึงผลต่างแบบจับคู่ระหว่างโมเดลแบบ LOSO กับโมเดลแบบ fused บนชุดภาพทดสอบเดียวกัน พร้อมช่วงความเชื่อมั่นร้อยละ 95 ส่วนค่า Full source คือการประเมินผลโมเดลแบบ LOSO บนภาพทั้งหมดของแหล่งข้อมูลที่ถูกแยกไว้ สำหรับชุดข้อมูล Duke ในกรณีที่เป็นแหล่งข้อมูลที่ถูกแยกไว้ (held out) จะใช้โมเดลเดี่ยวที่ประเมินผลบนดวงตาทั้ง #DS.duke.n_groups ดวง],
) <tab-unseen>

#figure(
  kind: image,
  fig("gap"),
  caption: [ค่า Macro-F1 บนชุดข้อมูลทดสอบ (test split) ของแต่ละแหล่งข้อมูล เมื่อแหล่งข้อมูลนั้นเป็นแบบเคยเห็น (fused, จุดโปร่ง) และไม่เคยเห็น (LOSO, จุดทึบ) สำหรับโมเดลแบบ letterbox โดยเส้นแสดงช่วงความเชื่อมั่นร้อยละ 95 ของคะแนนบนแหล่งข้อมูลที่ไม่เคยเห็น (unseen) ส่วนผลต่างแบบจับคู่ (paired gaps) แสดงไว้ใน @tab-unseen โดยคะแนนทั้งสองรูปแบบได้รับการประเมินบนชุดภาพทดสอบเดียวกัน ทั้งนี้ สำหรับชุดข้อมูล Kermany ภาพส่วนใหญ่มีขนาด 1024×496 พิกเซล และได้คะแนนสูงกว่าชุดข้อมูล Kermany ในภาพรวมทั้งชุดข้อมูล (@tab-unseen, full source)],
) <fig-gap>

ทุกแหล่งข้อมูลมีประสิทธิภาพลดลงอย่างมีนัยสำคัญเหนือสัญญาณรบกวนเมื่อกลายเป็นแหล่งข้อมูลที่ไม่เคยเห็น (unseen source) (@tab-unseen, @fig-gap) โดยสำหรับชุดข้อมูล OCTDL, PAIMA และ Duke ส่วนต่างของค่า Macro-F1 อยู่ที่ #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) จุดเปอร์เซ็นต์ ขณะที่ชุดข้อมูล Kermany มีส่วนต่างอยู่ที่ #dl(k("gap_test", "kermany"), "loso-fill - fused-fill")

สำหรับแหล่งข้อมูลประเภท 3 คลาส ส่วนต่างของค่า AUROC มีค่าน้อยกว่าส่วนต่างของค่า Macro-F1 อย่างชัดเจน (#dl(k("gap_test", "kermany"), "loso-fill - fused-fill", name: "macro_auroc") สำหรับชุดข้อมูล Kermany, #dl(k("gap_test", "octdl"), "loso-fill - fused-fill", name: "macro_auroc") สำหรับชุดข้อมูล OCTDL และ #dl(k("gap_test", "duke"), "loso-fill - fused-fill", name: "macro_auroc") สำหรับชุดข้อมูล Duke) ซึ่งชี้ให้เห็นว่าโมเดลยังคงจัดอันดับคลาสที่ถูกต้องไว้ในลำดับต้น ๆ ได้ดีในภาพส่วนใหญ่ แต่คะแนนการทำนายมักตกไปอยู่อีกฝั่งของจุดตัดการตัดสินใจ (cutoff threshold)

ชุดข้อมูล PAIMA เป็นข้อยกเว้นเพียงหนึ่งเดียว โดยมีส่วนต่างของค่า AUROC อยู่ที่ #dl(k("gap_test", "paima"), "loso-fill - fused-fill", name: "macro_auroc") เมื่อเปรียบเทียบกับส่วนต่างของค่า Macro-F1 ที่ #dp(k("gap_test", "paima"), "loso-fill - fused-fill") ซึ่งสะท้อนถึงการเปลี่ยนแปลงในความสามารถด้านการจัดอันดับ (ranking) มิใช่เพียงแค่การคลาดเคลื่อนของจุดตัดการตัดสินใจเท่านั้น

ชุดข้อมูลทดสอบ (test set) ของ Kermany มิได้เป็นตัวแทนที่ดีของชุดข้อมูลโดยรวม กล่าวคือ มีภาพจำนวนถึง #DS.kermany.test_sizes.at("1024x496") ภาพจากทั้งหมด #num(DS.kermany.n_test) ภาพที่มีขนาด 1024×496 พิกเซล ซึ่งเป็นขนาดที่มีสัดส่วนเพียงร้อยละ #pct1(size-share("kermany", "1024x496") * 100) ในชุดข้อมูล Kermany ทั้งหมด

เมื่อประเมินบนภาพทั้งหมดจำนวน #num(R.at("loso_full|kermany|all").n_images) ภาพ คะแนนของชุดข้อมูล Kermany ในกรณีที่เป็นแหล่งข้อมูลที่ถูกแยกไว้ (held-out) ทำได้เพียง #pt("loso_full|kermany|all", "loso-fill") ซึ่งต่ำกว่าคะแนน #pt("gap_test|kermany|all", "loso-fill") บนชุดข้อมูลทดสอบ (test split) อย่างมาก งานวิจัยนี้จึงใช้คะแนนแบบรวมทั้งแหล่งข้อมูล (full-source score) เป็นเกณฑ์หลักในการอ้างอิงข้อสรุปเกี่ยวกับชุดข้อมูล Kermany ในฐานะแหล่งข้อมูลที่ไม่เคยเห็น โดยในหัวข้อถัดไปจะแสดงให้เห็นว่าเหตุใดขนาดของภาพจึงส่งผลกระทบอย่างยิ่งยวด

== สิ่งชี้แนะทางลัดจากขนาดภาพ <sec-shortcut>

ในชุดข้อมูล Duke ขนาดของภาพสามารถระบุคลาสของข้อมูลได้ถึงร้อยละ #calc.round((int(duke-eyes("amd", size: "1024x496")) + int(duke-eyes("dme", size: "768x496"))) / DS.duke.n_groups * 100) ของดวงตาทั้งหมด กล่าวคือ ภาพสแกนขนาด 768×496 พิกเซลทุกภาพเป็นภาวะ DME และภาพสแกนขนาด 1024×496 พิกเซลทุกภาพเป็นโรค AMD (@fig-size-class) โมเดลที่ได้รับการฝึกสอนด้วยชุดข้อมูล Duke จึงสามารถเรียนรู้กฎทางลัดที่ว่า "ความกว้าง 768 พิกเซลหมายถึงภาวะ DME" ซึ่งเป็นกฎที่ถูกต้องภายในชุดข้อมูล Duke แต่เกิดความผิดพลาดเมื่อนำไปประเมินกับชุดข้อมูลอื่น การทดสอบ 3 ประการต่อไปนี้ช่วยยืนยันว่าโมเดลได้เรียนรู้กฎดังกล่าวจริง

=== การปรับเปลี่ยนเฉพาะขนาดของภาพทดสอบ

งานวิจัยนี้ได้นำโมเดลแบบ LOSO ที่แยกชุดข้อมูล Kermany ไว้ มาประเมินภาพถ่ายในชุดข้อมูล Kermany ซ้ำ โดยปรับเปลี่ยนเฉพาะความกว้างของภาพโดยไม่มีการฝึกสอนโมเดลใหม่ (@tab-size-test) การเติมขอบภาพ (padding) ในภาพปกติขนาด 512×496 พิกเซล ให้มีความกว้างเป็น 768 พิกเซล (ซึ่งเป็นความกว้างเฉพาะของภาวะ DME ในชุดข้อมูล Duke) ส่งผลให้สัดส่วนภาพที่ถูกทำนายว่าเป็นภาวะ DME พุ่งสูงขึ้นจากร้อยละ #pct1(KS.pad.at("512x496").at("normal->dme").at(0)) เป็นร้อยละ #pct1(KS.pad.at("512x496").at("normal->dme").at(1)) ในทางตรงกันข้าม การครอบตัด (crop) ภาพปกติขนาด 768×496 พิกเซล ให้เหลือความกว้าง 512 พิกเซล ให้ผลลัพธ์ในทิศทางตรงกันข้าม โดยสัดส่วนภาพที่ถูกทำนายว่าเป็นภาวะ DME ลดลงจากร้อยละ #pct1(KS.crop.at("768x496").at("normal->dme").at(0)) เหลือเพียงร้อยละ #pct1(KS.crop.at("768x496").at("normal->dme").at(1))

ปรากฏการณ์นี้มิได้จำกัดอยู่เพียงแค่ชุดข้อมูล Kermany เท่านั้น โดยการเติมขอบภาพขนาด 512×496 พิกเซลของชุดข้อมูล Duke เพิ่มขึ้นอีก 16 แถว จนกลายเป็นขนาด 512×512 พิกเซล (ซึ่งเป็นขนาดที่พบเฉพาะภาวะ DME ในชุดข้อมูล Kermany) ส่งผลให้ภาพปกติของชุดข้อมูล Duke ถึงร้อยละ #calc.round(HS.at("duke/pad512x512").all.at("normal->dme").at(1)) ถูกทำนายผิดพลาดว่าเป็นภาวะ DME ในโมเดลที่แยกชุดข้อมูล Duke ไว้

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
  caption: [สัดส่วนของภาพถ่ายปกติ (normal) ที่ถูกทำนายว่าเป็นภาวะ DME ก่อนและหลังการปรับเปลี่ยนเฉพาะขนาดของภาพทดสอบ โดยใช้โมเดลแบบ LOSO ประเมินผลบนแหล่งข้อมูลที่ถูกแยกไว้ (held-out) โดยไม่มีการฝึกสอนใหม่ ทั้งนี้ ผลต่างเป็นค่าแบบจับคู่พร้อมช่วงความเชื่อมั่นร้อยละ 95],
) <tab-size-test>

=== ตำแหน่งที่สัญญาณขนาดภาพส่งผล

ความกว้างของภาพถูกส่งผ่านไปยังโมเดลผ่านทางแถบขอบภาพแบบ letterbox เมื่อภาพสแกนขนาด 768×496 พิกเซล ถูกปรับขนาดให้พอดีกับกรอบสี่เหลี่ยมจัตุรัส จะเกิดแถบขอบเรียบที่ด้านบนและด้านล่างของเนื้อเยื่อ ซึ่งคิดเป็นพื้นที่ประมาณหนึ่งในสามของภาพทั้งหมด โดยภาพสแกนขนาด 768×496 พิกเซลที่เป็นภาวะ DME ทั้งหมดในชุดข้อมูล Duke ก็มีแถบขอบในลักษณะเดียวกันนี้ สำหรับโมเดลแบบแยกชุดข้อมูล Kermany ไว้ (LOSO) ทั้ง 5 โมเดล งานวิจัยนี้ได้ทำการสุ่มภาพขนาด 768×496 พิกเซล กลุ่มละ #gc-group-n("768 normal → DME") ภาพ (ผู้ป่วยละ 1 ภาพ) ประกอบด้วย ภาพปกติ (normal) และโรค AMD ที่โมเดลทำนายผิดพลาดเป็นภาวะ DME รวมทั้งภาพปกติและโรค AMD ที่โมเดลทำนายได้อย่างถูกต้อง จากนั้นได้ทำการแทรกแซงข้อมูล (intervention) โดยการแก้ไขพิกเซลเฉพาะบริเวณแถบขอบให้เป็นสีดำ หรือแทนที่ด้วยสัญญาณรบกวนแบบเกรน (speckle noise) ที่สุ่มมาจากบริเวณครึ่งที่มีความเข้มมืดของภาพสแกน แล้วจึงทำการประเมินผลคะแนนใหม่ (@tab-bands)

สำหรับโมเดลที่ 1, 2 และ 4 พบว่ามีข้อผิดพลาดหลงเหลืออยู่ไม่เกินร้อยละ #{
  let xs = (1, 2, 4).map(f => ("768 normal → DME", "768 AMD → DME").map(g => ("black", "speckle").map(b => gc-at(f, g, b)))).flatten()
  pct0(calc.max(..xs))
} หลังจากเปลี่ยนแถบขอบทั้งสองรูปแบบ ขณะที่โมเดลที่ 3 และ 5 พบว่ายังมีข้อผิดพลาดในกลุ่มโรค AMD หลงเหลืออยู่ร้อยละ #pct0(gc-at(3, "768 AMD → DME", "speckle")) และ #pct0(gc-at(5, "768 AMD → DME", "speckle")) และข้อผิดพลาดในกลุ่มภาพปกติหลงเหลืออยู่ร้อยละ #pct0(gc-at(3, "768 normal → DME", "speckle")) และ #pct0(gc-at(5, "768 normal → DME", "speckle")) ตามลำดับ เมื่อใช้แถบขอบแบบสัญญาณรบกวน speckle ซึ่งแสดงให้เห็นว่าแถบขอบภาพเป็นช่องทางหลัก แต่ไม่ใช่ช่องทางเดียวที่สัญญาณนี้ถูกส่งผ่านไปถึงโมเดล ในทางตรงกันข้าม #{
  let rows = GC.intervention_by_fold.filter(r => r.group in ("768 normal → normal", "768 AMD → AMD") and r.at("band fill") != "trained")
  let total = rows.map(r => r.n).sum()
  let changed = rows.map(r => calc.round(r.n * (1 - r.at("called true class")))).sum()
  let to-dme = rows.map(r => calc.round(r.n * r.at("called DME"))).sum()
  [ภาพที่เดิมเคยทำนายได้อย่างถูกต้องแทบไม่ได้รับผลกระทบ โดยจากกรณีทดสอบทั้งหมด #total กรณี (การปรับเปลี่ยนแถบขอบ 2 รูปแบบ ในโมเดลทั้ง 5 โมเดล) มีเพียง #to-dme ภาพที่เปลี่ยนผลการทำนายเป็นภาวะ DME และมีเพียง #changed ภาพเท่านั้นที่ผลการทำนายเปลี่ยนไปจากคลาสเดิม]
}

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 7,
      align: (left, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Model], table.cell(colspan: 2, align: center)[Normal ทำนายเป็น DME],
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
  caption: [ภาพถ่ายขนาด 768×496 พิกเซล จากชุดข้อมูล Kermany ที่ถูกแยกไว้ (held-out) ซึ่งโมเดล LOSO-Kermany ทำนายผิดพลาดเป็นภาวะ DME: สัดส่วน (ร้อยละ) ที่ยังคงถูกทำนายว่าเป็นภาวะ DME หลังจากแทนที่แถบขอบภาพแบบ letterbox ด้วยสีดำหรือสัญญาณรบกวนแบบเกรน (speckle noise) ขณะที่สัดส่วนบนแถบขอบ Grad-CAM แสดงสัดส่วน (ร้อยละ) ของแผนภาพที่ครอบคลุมบริเวณแถบขอบ สำหรับภาพถ่ายปกติ (normal) ที่ถูกทำนายเป็นภาวะ DME (ทำนายผิดพลาด) และที่ถูกทำนายเป็นภาพถ่ายปกติ (ทำนายถูกต้อง)],
) <tab-bands>

แผนภาพ Grad-CAM ช่วยแสดงให้เห็นถึงกลไกนี้เช่นเดียวกัน (@fig-gradcam) อย่างไรก็ดี มีข้อควรระวังว่า สำหรับภาพถ่ายที่ทำนายได้อย่างถูกต้อง ทุกโมเดลจะมีสัดส่วนของแผนภาพที่ครอบคลุมบริเวณแถบขอบอยู่เพียงร้อยละ #pct0(calc.min(..range(1, 6).map(f => gc-band("768 normal → normal", f))))--#pct0(calc.max(..range(1, 6).map(f => gc-band("768 normal → normal", f)))) ทว่าสำหรับภาพถ่ายที่ทำนายผิดพลาด สัดส่วนดังกล่าวกลับมีความผันแปรอยู่ในช่วงกว้างตั้งแต่ร้อยละ #pct0(calc.min(..range(1, 6).map(f => gc-band("768 normal → DME", f)))) ถึง #pct0(calc.max(..range(1, 6).map(f => gc-band("768 normal → DME", f)))) และมิได้สอดคล้องกับผลการแทรกแซงข้อมูลโดยตรง ตัวอย่างเช่น โมเดลที่ 4 มีสัดส่วนแผนภาพที่ตกบนแถบขอบเพียงร้อยละ #pct0(gc-band("768 normal → DME", 4)) แต่กลับมีข้อผิดพลาดในกลุ่มภาพปกติ (normal) หลงเหลืออยู่เพียงร้อยละ #pct0(gc-at(4, "768 normal → DME", "speckle")) เมื่อทดแทนด้วยแถบขอบแบบสัญญาณรบกวน speckle (@tab-bands)

แผนภาพเหล่านี้สะท้อนถึงสิ่งที่โมเดลเรียนรู้ มิใช่โครงสร้างทางกายภาพของภาพถ่าย กล่าวคือ เมื่อทำการสุ่มกำหนดค่าน้ำหนักในชั้นบนของโครงข่ายใหม่ (re-initialisation) แผนภาพที่ได้จะไม่สัมพันธ์กับแผนภาพของโมเดลที่ผ่านการฝึกสอนแล้วอีกต่อไป (ค่ามัธยฐานของค่าสหสัมพันธ์แบบสเปียร์แมน [Spearman correlation] เท่ากับ #calc.round(GC.randomization_fold1_768_normal_to_dme.spearman_median, digits: 2) ในโมเดลที่ 1) ดังนั้น งานวิจัยนี้จึงใช้การทดลองแทรกแซงบริเวณแถบขอบเป็นหลักฐานเชิงประจักษ์หลัก และใช้แผนภาพ Grad-CAM เป็นเพียงภาพประกอบเสริมเท่านั้น

#figure(
  image("../../figures/gradcam.png", width: 100%),
  caption: [แผนภาพ Grad-CAM (เลเยอร์คอนโวลูชันสุดท้าย สำหรับคลาสที่ทำนาย) บนภาพ normal ขนาด 768×496 จาก Kermany ที่กันไว้ตามมุมมองของโมเดล LOSO-Kermany แบบ letterbox (แถวบน) และ square-crop (แถวล่าง) โดยแสดงคลาสที่ทำนายไว้เหนือแต่ละช่อง สองคอลัมน์สุดท้ายเป็นกลุ่มควบคุมที่ทั้งสองโมเดลทำนายเป็น normal ถูกต้อง],
) <fig-gradcam>

=== การฝึกสอนโดยไม่มีสิ่งชี้แนะจากขนาดภาพ

โมเดลแบบครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัส (square crop) จะไม่สามารถรับรู้ความกว้างของภาพสแกนได้เลย ซึ่งเมื่อทำการแยกชุดข้อมูล Kermany ไว้ โมเดลนี้สามารถทำคะแนนรวมทั้งแหล่งข้อมูล (full source) ได้ร้อยละ #pt("loso_full|kermany|all", "loso-sq") เมื่อเปรียบเทียบกับร้อยละ #pt("loso_full|kermany|all", "loso-fill") ของโมเดลแบบ letterbox (เพิ่มขึ้น #dl("loso_full|kermany|all", "loso-sq - loso-fill")) และเมื่อประเมินผลบนชุดข้อมูลทดสอบ (test split) เปรียบเทียบกับโมเดลแบบ fused ในกลุ่มการทดลองเดียวกัน พบว่าส่วนต่างประสิทธิภาพลดลงจาก #dl("gap_test|kermany|all", "loso-fill - fused-fill") เหลือเพียง #dl("gap_test|kermany|all", "loso-sq - fused-sq") หรือแคบลงประมาณร้อยละ #calc.round((1 - delta("gap_test|kermany|all", "loso-sq - fused-sq").point / delta("gap_test|kermany|all", "loso-fill - fused-fill").point) * 100)

ประสิทธิภาพที่ปรับตัวดีขึ้นนี้เกิดขึ้นตรงกับตำแหน่งที่มีสิ่งชี้แนะทางลัดพอดี (@fig-kermany-size) โดยสำหรับภาพถ่ายปกติ (normal) ขนาด 768×496 พิกเซล ค่า recall เพิ่มขึ้นจากร้อยละ #pt("size_loso:768x496|kermany|all", "fill", name: "recall_normal") เป็นร้อยละ #pt("size_loso:768x496|kermany|all", "sq", name: "recall_normal") และสัดส่วนภาพที่ถูกทำนายผิดพลาดว่าเป็นภาวะ DME ลดลงจากร้อยละ #share("size_loso:768x496|kermany|all", "fill", "normal", "dme") เหลือเพียงร้อยละ #share("size_loso:768x496|kermany|all", "sq", "normal", "dme")

#figure(
  kind: image,
  fig("kermany-size"),
  caption: [การประเมินผลบน Kermany ที่กันไว้ (ภาพทั้งหมด) จำแนกตามขนาดภาพและคลาสจริง: Recall ของโมเดล LOSO แบบ letterbox และ square-crop พร้อมช่วงความเชื่อมั่น 95% การตัดภาพแบบ square crop ช่วยเพิ่มความแม่นยำได้มากที่สุดสำหรับภาพ normal และ AMD ขนาด 768×496 ซึ่งเป็นความกว้างที่พบเฉพาะคลาส DME ใน Duke และทำให้ recall ของ DME ลดลงในขนาดดังกล่าว],
) <fig-kermany-size>

อย่างไรก็ตาม การขจัดสิ่งชี้แนะจากขนาดภาพมิได้ส่งผลดีในทุกกรณี กล่าวคือ สำหรับแหล่งข้อมูลอื่น ๆ ที่ถูกแยกไว้ (held-out sources) วิธีการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัส (square crop) มิได้ช่วยให้ผลลัพธ์ปรับตัวดีขึ้น และในบางกรณีกลับมีผลลัพธ์ที่ลดลง (@tab-square) ในชุดข้อมูล OCTDL ค่า recall ของภาพถ่ายปกติ (normal) มีการเปลี่ยนแปลงไป #dl("loso_full|octdl|all", "loso-sq - loso-fill", name: "recall_normal") โดยสัดส่วนภาพถ่ายปกติที่ถูกทำนายว่าเป็นภาวะ DME เปลี่ยนจากร้อยละ #share("loso_full|octdl|all", "loso-fill", "normal", "dme") เป็นร้อยละ #share("loso_full|octdl|all", "loso-sq", "normal", "dme") ขณะที่บนชุดข้อมูล Duke ค่า recall ของภาวะ DME มีการเปลี่ยนแปลงไป #dl("loso_full|duke|all", "loso-sq - loso-fill", name: "recall_dme") เมื่อพิจารณาในมุมมองภาพปกติ/โรค AMD (Normal/AMD view) ซึ่งทำการตัดคลาสภาวะ DME ออก พบว่าการลดลงของประสิทธิภาพทั้งสองจุดดังกล่าวได้หายไป (#dl("loso_full|octdl|normal_amd", "loso-sq - loso-fill") และ #dl("loso_full|duke|normal_amd", "loso-sq - loso-fill")) ชี้ให้เห็นว่าผลกระทบเชิงลบของการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัสในแหล่งข้อมูลเหล่านี้มีความเกี่ยวข้องกับภาวะ DME โดยตรง กล่าวคือ เกิดการทำนายว่าเป็นภาวะ DME มากเกินไป (over-prediction) ในชุดข้อมูล OCTDL และการทำนายว่าเป็นภาวะ DME น้อยเกินไป (under-prediction) ในชุดข้อมูล Duke สำหรับชุดข้อมูล PAIMA คะแนนแมโคร เอฟ1 (macro-F1) มีการเปลี่ยนแปลงไป #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill") และค่า AUROC มีการเปลี่ยนแปลงไป #dl("loso_full|paima|normal_amd", "loso-sq - loso-fill", name: "macro_auroc") ซึ่งเป็นการสูญเสียความสามารถในการจัดอันดับ (ranking capability) ที่มีความผันแปรตามค่าซีด (seed) อย่างมาก (@sec-limits)

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 8,
      align: (left, right, right, right, right, right, right, right),
      table.header(
        table.cell(rowspan: 2)[Held-out Source], table.cell(colspan: 3, align: center)[Macro-F1],
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
  caption: [ประเมินโมเดลแบบ LOSO บนแหล่งข้อมูลที่ถูกแยกไว้แบบรวมทั้งแหล่งข้อมูล (full source) โดยเปรียบเทียบระหว่างรูปแบบที่มีสิ่งชี้แนะจากขนาดภาพ (letterbox) และไม่มีสิ่งชี้แนะจากขนาดภาพ (square crop) (ร้อยละ) โดยค่าผลต่างเป็นค่าแบบจับคู่พร้อมช่วงความเชื่อมั่นร้อยละ 95 สำหรับ PAIMA ประเมินผลเฉพาะ normal กับ AMD ดังนั้นมุมมอง Normal/AMD จึงเป็นคะแนนหลักของชุดข้อมูลนี้],
) <tab-square>

ด้วยเหตุนี้ งานวิจัยนี้จึงกำหนดให้โมเดลแบบ letterbox เป็นการทดลองหลัก และจัดให้การครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัส (square crop) เป็นการทดลองตัดทอนองค์ประกอบ (ablation study) ซึ่งช่วยยืนยันว่าสิ่งชี้แนะทางลัดจากขนาดภาพเกิดขึ้นจริง และสามารถอธิบายส่วนต่างประสิทธิภาพในชุดข้อมูล Kermany ได้เป็นส่วนใหญ่ ทว่ามิใช่วิธีการแก้ปัญหาที่สามารถประยุกต์ใช้ได้เป็นการทั่วไป

ทั้งนี้ งานวิจัยนี้ได้กำหนดเกณฑ์การตีความดังกล่าวไว้ล่วงหน้าก่อนทราบผลการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัสของชุดข้อมูล OCTDL และ Duke ว่า วิธีการครอบตัดภาพเป็นสี่เหลี่ยมจัตุรัสจะถูกนำมาใช้เป็นรูปแบบหลักก็ต่อเมื่อไม่มีแหล่งข้อมูลใดสูญเสียประสิทธิภาพเกินระดับสัญญาณรบกวนเท่านั้น

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
