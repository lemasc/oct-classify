#import "../../lib.typ": *
#import "../../i18n/th.typ": source-name, split-name

= ข้อมูล <sec-data>

== ชุดข้อมูล OCT

งานวิจัยนี้ใช้ชุดข้อมูลภาพ OCT ของจอตาที่เผยแพร่สาธารณะสี่ชุด โดยถือแต่ละชุดเป็นแหล่งข้อมูลหนึ่งแหล่ง แหล่งข้อมูลเหล่านี้ต่างกันในด้านสถานพยาบาล ประชากรกลุ่มตัวอย่าง ข้อกำหนดในการจัดเก็บภาพ และการกำกับป้ายข้อมูล (labelling) แต่โดยส่วนใหญ่ไม่ต่างกันในด้านอุปกรณ์ โดยชุดข้อมูล Kermany @kermany2018 (จากศูนย์การแพทย์ห้าแห่งในสหรัฐอเมริกาและจีน), Duke @srinivasan2014 (จากศูนย์การแพทย์สามแห่งในสหรัฐอเมริกา) และ PAIMA @sotoudehpaima2022 (จากโรงพยาบาลหนึ่งแห่งในประเทศอิหร่าน) ต่างบันทึกภาพด้วยเครื่อง spectral-domain OCT ของ Heidelberg Engineering (รุ่น Spectralis ใน Kermany และ Duke ส่วน PAIMA ระบุเพียงชื่อผู้ผลิต) มีเพียง OCTDL @kulyabin2024octdl ที่มาจากผู้ผลิตรายอื่น (Optovue Avanti RTVue XR) ดังนั้นเมื่อ OCTDL เป็นแหล่งข้อมูลที่ไม่เคยเห็น (unseen source) จึงครอบคลุมอุปกรณ์ที่ไม่เคยพบมาก่อนด้วย

#figure(
  image("../../figures/examples.png", width: 100%),
  caption: [ภาพ B-scan หนึ่งภาพต่อหนึ่งแหล่งข้อมูลและต่อหนึ่งคลาส วาดที่สเกลพิกเซลเดียวกัน (ไม่เทียบเท่าในเชิงกายภาพ) และระบุขนาดเป็นพิกเซล],
) <fig-examples>

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 9,
      align: (left, right, right, right, right, right, right, right, left),
      table.header(
        table.cell(rowspan: 2)[Source], table.cell(rowspan: 2)[Images], table.cell(rowspan: 2)[Groups],
        table.cell(rowspan: 2)[Normal],
        table.cell(rowspan: 2)[AMD],
        table.cell(colspan: 2, align: center)[AMD Stage],
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
  caption: [แหล่งข้อมูลทั้งสี่แหล่งหลังจากการขจัดข้อมูลซ้ำ พร้อมจำนวนภาพแต่ละคลาส สัดส่วนระยะของโรค AMD และการแบ่งชุดข้อมูล
    #super[a]ภาพถ่ายโรค AMD อีกร้อยละ #stage-share("octdl", "intermediate") ที่เหลือของชุดข้อมูล OCTDL อยู่ในระยะปานกลาง (intermediate stage) ที่สงสัยว่ามีภาวะ MNV],
) <tab-data>

== การจัดกลุ่มคลาสของข้อมูล

งานวิจัยนี้รวมป้ายข้อมูล (labels) จากทุกแหล่งเป็น 3 คลาส ได้แก่ ปกติ (Normal), โรคจอตาเสื่อมตามอายุ (age-related macular degeneration: AMD) และภาวะจุดรับภาพบวมน้ำจากโรคเบาหวาน (diabetic macular edema: DME) ซึ่งเป็นภาวะแทรกซ้อนรุนแรงของโรคเบาหวานขึ้นจอตา (Diabetic Retinopathy) คลาส AMD รวม 2 ระยะที่ลักษณะบนภาพ B-scan ต่างกัน (@sec-diseases, @fig-examples) ได้แก่ โรค AMD ชนิดแห้ง (dry AMD) ซึ่งพบดรูเซน (drusen) สะสมอยู่ใต้ชั้นจอตา และโรค AMD ชนิดเปียก (wet AMD) ซึ่งเกิดภาวะมีหลอดเลือดสร้างใหม่ในชั้นคอรอยด์ (choroidal neovascularization: CNV หรือเรียกว่า MNV ในชุดข้อมูล OCTDL)

แต่ละชุดข้อมูลมีสัดส่วนของสองระยะนี้แตกต่างกันมาก (@tab-data) กล่าวคือ AMD ชนิดแห้งคิดเป็นประมาณ 1 ใน 5 ของภาพ AMD ใน Kermany และ OCTDL, ร้อยละ #stage-share("paima", "dry") ใน PAIMA และทั้งหมดใน Duke ซึ่งมีเฉพาะดวงตาที่เป็น AMD ชนิดแห้ง @srinivasan2014 PAIMA มีเพียงภาพปกติและ AMD การไม่มี DME เป็นลักษณะเชิงโครงสร้างของชุดข้อมูล ไม่ใช่การกำกับป้ายเชิงลบ (masked loss, @sec-method) และประเมินเฉพาะการจำแนกระหว่างภาพปกติกับ AMD

== หน่วยของการกำกับป้ายข้อมูล

Duke กำกับป้ายข้อมูลในระดับดวงตา (volume) และส่งต่อผลวินิจฉัยไปยังภาพ B-scan ทุกภาพ ขณะที่แหล่งอื่นกำกับป้ายในระดับภาพ B-scan โดยตรง เพื่อให้เปรียบเทียบกันได้ งานวิจัยนี้จึงประเมินทุกแหล่งในระดับภาพ B-scan อย่างไรก็ตาม ดวงตา #DS.duke.n_groups ดวงของ Duke เป็นหน่วยในการแบ่งชุดข้อมูล (splitting) และการสุ่มซ้ำแบบบูตสแตรป (bootstrap resampling) ช่วงค่าที่ได้ (intervals) จึงสะท้อนดวงตาที่เป็นอิสระต่อกันจำนวน #DS.duke.n_groups ดวง ไม่ใช่ภาพถ่ายจำนวน #num(DS.duke.n_images) ภาพ#footnote[Duke วัดผลในระดับดวงตาได้เช่นกัน โดยเฉลี่ยความน่าจะเป็นจากภาพ B-scan ในแต่ละดวงตา ซึ่งในระดับนี้ โมเดลแบบ per-dataset, fused, fused แบบ square-crop, LOSO และ LOSO แบบ square-crop ทำนายผิดไป #eye-wrong("per-dataset"), #eye-wrong("fused-fill"), #eye-wrong("fused-sq"), #eye-wrong("loso-fill") และ #eye-wrong("loso-sq") จากทั้งหมด #EYE.n_eyes ข้าง (Macro-F1 เท่ากับ #eye-f1("per-dataset"), #eye-f1("fused-fill"), #eye-f1("fused-sq"), #eye-f1("loso-fill") และ #eye-f1("loso-sq") ตามลำดับ) เรายังใช้ระดับ B-scan เป็นหน่วยวัดหลัก เพื่อประเมินทั้งสี่ชุดข้อมูลภายใต้เกณฑ์เดียวกัน]

== การคัดกรองข้อมูลซ้ำและการแบ่งชุดข้อมูล

ภาพที่ซ้ำกันทุกประการ (exact duplicates) และภาพที่ใกล้เคียงกันมากจากการตรวจด้วย perceptual hash (distance เท่ากับศูนย์) ซึ่งมาจากต่างผู้ป่วย จะถูกนำออกหรือเชื่อมให้อยู่ในชุดแบ่งเดียวกัน กระบวนการนี้ขจัดภาพซ้ำที่มีป้ายข้อมูลเดียวกันได้ถึง #num(DS.kermany.dedup.same_label_copies_removed) ภาพในชุดข้อมูล Kermany เพียงชุดเดียว

การแบ่งชุดข้อมูลทำในระดับผู้ป่วย (หรือระดับดวงตา/volume) เพื่อไม่ให้ผู้ป่วยรายใดปรากฏในชุดแบ่งมากกว่าหนึ่งชุด เพราะการแบ่งข้อมูล OCT ในระดับภาพเดี่ยวทำให้ความแม่นยำในการทดสอบสูงเกินจริงถึงร้อยละ 5–30 @tampu2022 Kermany ใช้ชุดทดสอบเดิมที่ผู้สร้างให้มา (supplied test split) และแบ่งส่วนที่เหลือ 85/15 เป็นชุด train และ validation ส่วน OCTDL และ PAIMA แบ่ง 70/15/15 เป็นชุด train, validation และ test สำหรับ Duke ซึ่งมีข้อมูลเพียง #DS.duke.n_groups ดวงตา ใช้การตรวจสอบความถูกต้องแบบ 5 ส่วน (5-fold cross-validation) โดยมี #DS.duke.test_eyes_per_fold.first() ดวงตาทดสอบต่อ fold (#DS.duke.test_eyes_per_class_per_fold.first() ดวงตาต่อแต่ละคลาส)

== มิติและสัดส่วนของภาพ

แหล่งข้อมูลต่างกันในด้านขนาดภาพ (@fig-size-class) โดยภาพทั้งหมดใน PAIMA มีขนาด 768×496 พิกเซล; Kermany มีขนาดหลัก 5 ขนาด ได้แก่ 512×496, 768×496, 1024×496, 1536×496 และ 512×512 พิกเซล (ขนาดสุดท้ายพบเฉพาะใน DME); ภาพใน OCTDL ถูกครอบตัดหลายขนาด โดยส่วนใหญ่ประมาณ 880×320 พิกเซล

Duke มีขนาดภาพ 3 ขนาด โดย 2 ขนาดเชื่อมโยงกับป้ายข้อมูลเพียงประเภทเดียว ได้แก่ ภาพขนาด 768×496 พิกเซลทุกภาพใน Duke เป็น DME และภาพขนาด 1024×496 พิกเซลทุกภาพเป็น AMD ความเชื่อมโยงนี้เกิดจากการตั้งค่าการสแกนบนอุปกรณ์ชนิดเดียวกัน ไม่ได้เกิดจากตัวโรค กล่าวคือ ในข้อกำหนดการจัดเก็บภาพของ Duke @srinivasan2014 ข้อมูลปริมาตรภาพ (volumes) สภาวะปกติทั้ง #duke-eyes("normal", size: "512x496") ชุดมีจำนวน 512 A-scans ต่อหนึ่งภาพ B-scan, ข้อมูลภาพ AMD จำนวน #duke-eyes("amd", size: "1024x496") จาก #duke-eyes("amd") ชุดมีจำนวน 1,024 A-scans และข้อมูลภาพภาวะ DME จำนวน #duke-eyes("dme", size: "768x496") จาก #duke-eyes("dme") ชุดมีจำนวน 768 A-scans โมเดลที่ฝึกสอนด้วย Duke จึงเรียนรู้ขนาดภาพเป็นสิ่งชี้แนะป้ายข้อมูล (label/shortcut cue) ได้ ซึ่งการวิเคราะห์ใน @sec-shortcut แสดงว่าโมเดลเรียนรู้สิ่งชี้แนะทางลัดนี้จริง

#figure(
  kind: image,
  fig("size-class"),
  caption: [สัดส่วนคลาสในแต่ละขนาดภาพ สำหรับชุดข้อมูล OCTDL ภาพถูกครอบตัดมาเป็นหลายขนาดจึงแสดงรวมไว้ในแถวเดียว ทั้งนี้ได้ตัดภาพใน Kermany ที่มีขนาดพบน้อยจำนวน #kermany-rare ภาพออก],
) <fig-size-class>
