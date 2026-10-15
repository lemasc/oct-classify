#import "../../lib.typ": *
#import "../../i18n/th.typ": source-name, split-name

= ข้อมูล <sec-data>

เราใช้ชุดข้อมูลภาพ retinal OCT สาธารณะ 4 ชุดข้อมูล โดยกำหนดให้แต่ละชุดข้อมูลเป็นหนึ่ง _แหล่งข้อมูล (source)_ ชุดข้อมูลเหล่านี้มีความแตกต่างกันทั้งในด้านสถานพยาบาลที่เก็บข้อมูล กลุ่มประชากร ระเบียบวิธีเก็บภาพ (acquisition protocol) และเกณฑ์การกำกับป้ายกำกับ (labelling) แต่ส่วนใหญ่ใช้เครื่องสแกนจากผู้ผลิตรายเดียวกัน

ชุดข้อมูล Kermany @kermany2018 (จาก 5 ศูนย์ในสหรัฐฯ และจีน), Duke @srinivasan2014 (จาก 3 ศูนย์ในสหรัฐฯ) และ PAIMA @sotoudehpaima2022 (จากโรงพยาบาล 1 แห่งในอิหร่าน) ล้วนถ่ายด้วยเครื่อง spectral-domain OCT ของ Heidelberg Engineering (โดย Kermany และ Duke ระบุรุ่นเป็น Spectralis ขณะที่บทความของ PAIMA ระบุเพียงชื่อผู้ผลิต) มีเพียง OCTDL @kulyabin2024octdl เท่านั้นที่มาจากผู้ผลิตรายอื่น (Optovue Avanti RTVue XR) ดังนั้นการทดสอบบน _unseen source_ จึงหมายถึงการประเมินบนชุดข้อมูลที่ไม่เคยเห็นในการเทรน โดยกรณีของ OCTDL จะมีความท้าทายพิเศษที่เป็น _unseen device_ (เครื่องสแกนต่างผู้ผลิต) ร่วมด้วย @fig-examples แสดงภาพตัวอย่างสแกนของแต่ละแหล่งข้อมูลและแต่ละคลาส

#figure(
  image("../../figures/examples.png", width: 100%),
  caption: [ตัวอย่างภาพ B-scan หนึ่งภาพต่อแต่ละแหล่งข้อมูลและแต่ละคลาส แสดงตามสเกลพิกเซลเดียวกัน (ไม่ใช่สเกลขนาดจริงทางกายภาพ) พร้อมระบุขนาดของภาพเป็นพิกเซล ทั้งนี้ AMD แบบแห้งและแบบเปียกถูกยุบรวมเป็นคลาสเดียวกันสำหรับการเทรน],
) <fig-examples>

#figure(
  {
    set text(size: (9.5 / 11) * 1em)
    table(
      columns: 9,
      align: (left, right, right, right, right, right, right, right, left),
      table.header(
        table.cell(rowspan: 2)[แหล่งข้อมูล], table.cell(rowspan: 2)[จำนวนภาพ], table.cell(rowspan: 2)[กลุ่ม],
        table.cell(rowspan: 2)[Normal (ปกติ)],
        table.cell(rowspan: 2)[AMD],
        table.cell(colspan: 2, align: center)[ระยะของ AMD],
        table.cell(rowspan: 2)[DME], table.cell(rowspan: 2)[การแบ่งข้อมูล],
        [Dry (แห้ง)], [Wet (เปียก)],
        table.hline(),
      ),
      ..srcs
        .map(s => {
          let d = DS.at(s)
          let stage(st) = if stage-share(s, st) == none [--] else if s == "duke" [*100%*] else [#stage-share(s, st)%]
          (
            [#source-name.at(s)],
            [#num(d.n_images)],
            [#num(d.n_groups) #if d.label_unit == "eye" [ตา] else [ผู้ป่วย]],
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
  caption: [ชุดข้อมูลทั้งสี่แหล่งหลังตัดข้อมูลซ้ำ (deduplication) กำหนดให้กลุ่มคือหน่วยที่จัดสรรให้อยู่ใน split เดียวกันและใช้สุ่มซ้ำในการทำ bootstrap; ชุดข้อมูล Duke กำกับป้ายกำกับระดับดวงตาทั้งดวง (whole eye) ส่วนชุดข้อมูลอื่นกำกับระดับภาพ B-scan เดี่ยว คอลัมน์ _ระยะของ AMD_ แสดงสัดส่วนภาพ AMD ในแต่ละชุดข้อมูลที่เป็นแบบแห้ง (drusen) หรือแบบเปียก (CNV/MNV) ซึ่งทั้งสองระยะจะถูกยุบรวมเป็นคลาสเดียวกันในการเทรน #super[a]สำหรับ OCTDL ภาพ AMD อีก #stage-share("octdl", "intermediate")% ที่เหลือเป็นระยะปานกลาง (intermediate stage) ที่สงสัยว่ามี MNV],
) <tab-data>

== การจัดกลุ่มคลาส (Taxonomy)

เรายุบรวมป้ายกำกับจากแต่ละชุดข้อมูลออกเป็น 3 คลาส ได้แก่ _ปกติ (normal)_, _AMD_ (จอประสาทตาเสื่อมตามอายุ; age-related macular degeneration) และ _DME_ (จุดภาพชัดจอตาบวมจากเบาหวาน; diabetic macular edema) ทั้งนี้คลาส AMD ได้รวมสองระยะของโรคที่มีลักษณะบนภาพ B-scan แตกต่างกันอย่างชัดเจนเข้าด้วยกัน (@fig-examples) ได้แก่ dry AMD ซึ่งมี drusen สะสมอยู่ใต้จอประสาทตา และ wet AMD ซึ่งเกิดการงอกผิดปกติของหลอดเลือดใหม่ใต้คอรอยด์ (choroidal neovascularization หรือ CNV โดยในชุดข้อมูล OCTDL เรียกว่า MNV)

แต่ละชุดข้อมูลมีสัดส่วนของสองระยะนี้ผสมกันแตกต่างกันมาก (@tab-data) กล่าวคือ dry AMD คิดเป็นประมาณ 1 ใน 5 ของภาพ AMD ใน Kermany และ OCTDL คิดเป็น #stage-share("paima", "dry")% ใน PAIMA และคิดเป็นทั้งหมดใน Duke ซึ่งมีเฉพาะตาที่เป็น dry AMD เท่านั้น @srinivasan2014 นอกจากนี้ PAIMA มีเพียงภาพปกติและ AMD โดยการไม่มีภาพ DME ถือเป็นข้อจำกัดเชิงโครงสร้างของชุดข้อมูล ไม่ใช่การระบุว่าเป็นผลลบ (negative label) ภาพจาก PAIMA จึงไม่ถูกนำไปคำนวณ loss ในส่วนของเอาต์พุต DME ระหว่างการเทรน (ผ่าน masked loss, @sec-method) และการประเมินผลสำหรับ PAIMA จะวัดผลเฉพาะระหว่าง normal กับ AMD เท่านั้น

== หน่วยของป้ายกำกับ (Label unit)

ชุดข้อมูล Duke กำกับป้ายกำกับในระดับดวงตาทั้งดวง (volume) แล้วกระจายผลการวินิจฉัยนั้นไปยังภาพ B-scan ทุกภาพในตาข้างนั้น ในขณะที่ชุดข้อมูลอื่นกำกับป้ายกำกับทีละภาพ B-scan โดยตรง เพื่อให้การประเมินสามารถเปรียบเทียบกันได้ เราจึงวัดผลทุกชุดข้อมูลในระดับ B-scan แต่จะใช้ตาจำนวน #DS.duke.n_groups ข้างของ Duke เป็นหน่วยในการแบ่ง split และการสุ่มซ้ำแบบ bootstrap ช่วงความเชื่อมั่นของ Duke จึงสะท้อนความเป็นอิสระของตา #DS.duke.n_groups ข้าง ไม่ใช่ภาพ #num(DS.duke.n_images) ภาพ#footnote[ชุดข้อมูล Duke ยังสามารถวัดผลในระดับดวงตาได้ด้วย โดยหาค่าเฉลี่ยของความน่าจะเป็นจากภาพ B-scan ในแต่ละดวงตา ซึ่งในระดับดวงตานี้ โมเดลแบบ per-dataset, fused, fused แบบ square-crop, LOSO และ LOSO แบบ square-crop ทำนายผิดไป #eye-wrong("per-dataset"), #eye-wrong("fused-fill"), #eye-wrong("fused-sq"), #eye-wrong("loso-fill") และ #eye-wrong("loso-sq") จากทั้งหมด #EYE.n_eyes ข้าง (macro-F1 เท่ากับ #eye-f1("per-dataset"), #eye-f1("fused-fill"), #eye-f1("fused-sq"), #eye-f1("loso-fill") และ #eye-f1("loso-sq") ตามลำดับ) เรายังคงเลือกใช้ระดับ B-scan เป็นหน่วยวัดหลัก เพื่อให้สามารถประเมินผลทั้งสี่ชุดข้อมูลภายใต้เกณฑ์เดียวกัน]

== การคัดกรองข้อมูลซ้ำและการแบ่งชุดข้อมูล (Deduplication and splits)

ภาพที่ซ้ำกันทุกประการ (exact duplicates) และภาพที่ใกล้เคียงกันมากจากการตรวจสอบด้วย perceptual hash (distance เท่ากับศูนย์) ซึ่งมาจากต่างผู้ป่วย จะถูกนำออกหรือเชื่อมโยงให้อยู่ในหน่วย split เดียวกัน ซึ่งเฉพาะใน Kermany ก็สามารถตัดภาพซ้ำที่มีป้ายกำกับเดียวกันออกไปได้ถึง #num(DS.kermany.dedup.same_label_copies_removed) ภาพ

การแบ่ง split จะทำในระดับผู้ป่วย (หรือระดับดวงตา/volume) เพื่อป้องกันไม่ให้มีข้อมูลของผู้ป่วยรายเดียวกันปรากฏในหลาย split สำหรับ Kermany ยังคงใช้ชุด test เดิมที่ให้มา ส่วนที่เหลือแบ่งเป็นชุด train และ validation ในสัดส่วน 85/15 ขณะที่ OCTDL และ PAIMA ถูกแบ่งเป็น train, validation และ test ในสัดส่วน 70/15/15 ส่วน Duke ซึ่งมีข้อมูลเพียง #DS.duke.n_groups ตา จะใช้ 5-fold cross-validation (มีตาสำหรับทดสอบ #DS.duke.test_eyes_per_fold.first() ข้างต่อ fold หรือ #DS.duke.test_eyes_per_class_per_fold.first() ข้างต่อคลาส)

== มิติและสัดส่วนของภาพ (Image geometry)

แต่ละชุดข้อมูลมีขนาดภาพที่แตกต่างกัน (@fig-size-class) โดย PAIMA มีขนาด 768×496 พิกเซลทั้งหมด, Kermany มี 5 ขนาดหลัก (512×496, 768×496, 1024×496, 1536×496 และ 512×512 ซึ่งขนาดหลังสุดนี้พบเฉพาะใน DME เท่านั้น), OCTDL เป็นภาพที่ถูกครอบตัด (crop) มาในหลากหลายขนาด ส่วนใหญ่อยู่ที่ประมาณ 880×320 พิกเซล ส่วน Duke มี 3 ขนาด และ 2 ขนาดในนั้นมีความสัมพันธ์แบบหนึ่งต่อหนึ่งกับป้ายกำกับโรค กล่าวคือ ภาพขนาด 768×496 ทุกภาพใน Duke ล้วนเป็น DME และภาพขนาด 1024×496 ทุกภาพล้วนเป็น AMD

ความสัมพันธ์นี้เกิดจากการตั้งค่าการสแกนบนเครื่องตรวจเครื่องเดียวกัน ไม่ได้เกิดจากตัวโรคโดยตรง ในระเบียบวิธีเก็บภาพของ Duke @srinivasan2014 ข้อมูล volume ปกติทั้ง #duke-eyes("normal", size: "512x496") volume ถูกสแกนโดยมี A-scan 512 เส้นต่อหนึ่ง B-scan ขณะที่ volume ของ AMD จำนวน #duke-eyes("amd", size: "1024x496") จากทั้งหมด #duke-eyes("amd") volume มี A-scan 1,024 เส้น และ volume ของ DME จำนวน #duke-eyes("dme", size: "768x496") จากทั้งหมด #duke-eyes("dme") volume มี A-scan 768 เส้น ส่งผลให้โมเดลที่เทรนบนชุดข้อมูล Duke สามารถเรียนรู้ขนาดภาพเพื่อใช้เป็นสัญญาณบอกคลาส (shortcut cue) ซึ่งใน @sec-shortcut จะแสดงให้เห็นว่าโมเดลเรียนรู้ทางลัดนี้จริง

#figure(
  kind: image,
  fig("size-class"),
  caption: [สัดส่วนคลาสในแต่ละขนาดภาพ สำหรับชุดข้อมูล Duke ภาพขนาด 768×496 ทุกภาพเป็น DME และขนาด 1024×496 ทุกภาพเป็น AMD; ใน Kermany ภาพขนาด 512×512 ทุกภาพเป็น DME สำหรับ OCTDL ภาพถูกครอบตัดมาเป็นหลายขนาดจึงแสดงรวมไว้ในแถวเดียว ทั้งนี้ได้ตัดภาพใน Kermany ที่มีขนาดพบน้อยจำนวน #kermany-rare ภาพออก],
) <fig-size-class>
