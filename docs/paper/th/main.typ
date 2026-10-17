// Thai companion. Build (from docs/paper): typst compile --root ../.. th/main.typ
// A section is taken from th/sections/ when listed in `translated` (i18n/th.typ), else from the English ../sections/.
// `--input final=1` hides the "not translated" markers.
#import "../template.typ": paper, front
#import "../lib.typ": *
#import "../i18n/th.typ"

#show: paper.with(title: th.title, lang: "th")

#let final = sys.inputs.at("final", default: "0") == "1"

#let section(name) = {
  if name in th.translated {
    include "sections/" + name + ".typ"
  } else {
    [
      #set text(lang: "en")
      #if not final { pending[ยังไม่ได้แปล: #name] }
      #include "../sections/" + name + ".typ"
    ]
  }
}

#front[
#align(center)[
  #text(size: 1.45em, weight: "bold")[#th.title]
  #v(0.5em)
  #todo[authors, affiliation]
]

#heading(outlined: false, numbering: none)[บทคัดย่อ]
ตัวจำแนกภาพ OCT จอตามักได้รับการทดสอบบนชุดข้อมูลเดียวกับที่ใช้ฝึกสอนเท่านั้น งานวิจัยนี้วัดประเมินว่าตัวจำแนกดังกล่าวจะยังคงรักษาความแม่นยำไว้ได้มากน้อยเพียงใดบนชุดข้อมูลที่ไม่เคยเห็นมาก่อน และวิเคราะห์หาสาเหตุที่ทำให้สูญเสียความแม่นยำไป จากการใช้ชุดข้อมูลสาธารณะ 4 ชุด (Kermany, OCTDL, PAIMA และ Duke) ร่วมกับแบบจำลอง ResNet-50 ที่ผ่านการฝึกสอนล่วงหน้าด้วย ImageNet เพื่อจำแนกภาพ B-scan (ภาพตัดขวาง 2 มิติของจอตา) ออกเป็น 3 คลาส ได้แก่ ปกติ, AMD หรือ DME งานวิจัยนี้เปรียบเทียบระหว่างโมเดลที่ประเมินบนแหล่งข้อมูลที่ใช้ฝึกสอน กับโมเดลแบบกันหนึ่งแหล่งข้อมูลออก (leave-one-source-out) โดยแบ่งชุดข้อมูลในระดับผู้ป่วยและรายงานผลต่างร่วมกับช่วงความเชื่อมั่นแบบจับคู่ด้วยการบูตสแตรป

บนแหล่งข้อมูลที่เคยเห็น โมเดลทำคะแนน macro-F1 ได้ #span(("kermany", "octdl", "paima", "duke").map(s => metric(k("seen", s), "per-dataset").point)) ทว่าทุกแหล่งข้อมูลกลับสูญเสียความแม่นยำเกินระดับสัญญาณรบกวนเมื่อเป็นแหล่งข้อมูลที่ไม่เคยเห็น โดยคะแนนลดลง #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) จุดสำหรับชุดข้อมูล OCTDL, PAIMA และ Duke และลดลงประมาณ #whole(delta(k("gap_test", "kermany"), "loso-fill - fused-fill").point) จุดสำหรับชุดข้อมูล Kermany บนชุดทดสอบ แม้ว่า 3 ใน 4 แหล่งข้อมูลจะใช้เครื่องสแกนจากผู้ผลิตรายเดียวกันก็ตาม ส่วนต่างประสิทธิภาพส่วนใหญ่ของ Kermany เกิดจากการเรียนรู้ทางลัดที่นำเข้าจากชุดข้อมูลอื่น กล่าวคือ ในชุดข้อมูล Duke ขนาดของภาพสามารถบ่งชี้ป้ายข้อมูลได้ และโมเดลที่ฝึกสอนร่วมกับ Duke ได้นำกฎนี้ไปปรับใช้กับ Kermany ผ่านแถบถมขอบภาพ (padding bands) ที่เพิ่มเข้ามาเมื่อปรับภาพสแกนแนวกว้างให้เข้ากับอินพุตสี่เหลี่ยมจัตุรัส
การฝึกสอนโดยไม่มีสิ่งชี้แนะจากขนาดภาพช่วยเพิ่มคะแนน macro-F1 ของชุดข้อมูล Kermany ที่ถูกแยกไว้จาก #pt("loso_full|kermany|all", "loso-fill") เป็น #pt("loso_full|kermany|all", "loso-sq") ทว่ากลับทำให้ประสิทธิภาพบนแหล่งข้อมูลที่ไม่เคยเห็นอื่น ๆ ลดลง คลาสที่เป็นสาเหตุของส่วนต่างแตกต่างกันไปตามแหล่งข้อมูล ได้แก่ ค่า recall ของภาวะ DME สำหรับ OCTDL และ Duke, ค่า recall ของระดับปกติสำหรับ Kermany และค่า recall ของโรค AMD สำหรับ PAIMA การปรับจุดตัดการตัดสินใจหรือการใช้ AdaBN ยังคงเหลือส่วนต่างส่วนใหญ่อยู่เช่นเดิม ทั้งนี้ การเปรียบเทียบการตั้งค่าสแกนกับป้ายข้อมูลก่อนการรวมชุดข้อมูลเข้าด้วยกัน จะช่วยเปิดเผยการเรียนรู้ทางลัดนี้ได้ตั้งแต่ก่อนเริ่มการฝึกสอน

]

#for name in ("introduction", "background", "data", "method", "results", "discussion") { section(name) }

#bibliography("../refs.bib")
