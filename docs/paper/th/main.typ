// Thai companion. Build (from docs/paper): typst compile --root ../.. th/main.typ
// A section is taken from th/sections/ when listed in `translated` (i18n/th.typ), else from the English ../sections/.
// `--input final=1` hides the "not translated" markers.
#import "../template.typ": front, front-gap, front-note, page-notes, paper
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
    #v(0.2em)
    #set text(weight: "bold")
    ศักดิธัช ธนาสดใส#super[1]
    และ
    พิชญ์สิณี วงศ์ศุภธรรม#super[2] \
    คณะเทคโนโลยีสารสนเทศ สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง \
    เลขที่ 1 ซอยฉลองกรุง 1 แขวงลาดกระบัง เขตลาดกระบัง กรุงเทพฯ 10520
  ]
  #v(1em)

  #set align(left)
  #heading(outlined: false, numbering: none)[บทคัดย่อ]
  #v(0.2em)
  #set par(justify: false, first-line-indent: 1.5em)
  ตัวจำแนกภาพ OCT จอตามักได้รับการทดสอบบนชุดข้อมูลเดียวกับที่ใช้ฝึกสอน งานวิจัยนี้จึงวัดว่าตัวจำแนกยังคงความแม่นยำไว้ได้เพียงใดบนชุดข้อมูลที่ไม่เคยเห็น และวิเคราะห์สาเหตุที่ลดลง โดยใช้ชุดข้อมูลสาธารณะ 4 ชุด (Kermany, OCTDL, PAIMA และ Duke) ร่วมกับ ResNet-50 ที่ผ่านการฝึกสอนล่วงหน้าด้วย ImageNet เพื่อจำแนกภาพ B-scan (ภาพตัดขวาง 2 มิติของจอตา) เป็น 3 คลาส ได้แก่ ปกติ, AMD และ DME เปรียบเทียบโมเดลที่ประเมินบนแหล่งข้อมูลที่ใช้ฝึกสอนกับโมเดลแบบกันหนึ่งแหล่งข้อมูลออก (leave-one-source-out) โดยแบ่งข้อมูลระดับผู้ป่วยและรายงานผลต่างพร้อมช่วงความเชื่อมั่นบูตสแตรปแบบจับคู่ บนแหล่งข้อมูลที่เคยเห็น โมเดลได้ macro-F1 ที่ #span(("kermany", "octdl", "paima", "duke").map(s => metric(k("seen", s), "per-dataset").point)) ทว่าทุกแหล่งข้อมูลสูญเสียความแม่นยำเกินระดับสัญญาณรบกวนเมื่อไม่เคยเห็น โดยคะแนนลดลง #span(("octdl", "paima", "duke").map(s => delta(k("gap_test", s), "loso-fill - fused-fill").point)) จุดสำหรับ OCTDL, PAIMA และ Duke และประมาณ #whole(delta(k("gap_test", "kermany"), "loso-fill - fused-fill").point) จุดสำหรับ Kermany บนชุดทดสอบ แม้ 3 ใน 4 แหล่งข้อมูลจะใช้เครื่องสแกนจากผู้ผลิตรายเดียวกัน ส่วนต่างของ Kermany ส่วนใหญ่เกิดจากการเรียนรู้ทางลัดที่นำเข้าจากชุดข้อมูลอื่น กล่าวคือ ขนาดภาพใน Duke บ่งชี้ป้ายข้อมูลได้ และโมเดลนำกฎนี้ไปใช้กับ Kermany ผ่านแถบถมขอบภาพ (padding bands) การฝึกโดยไม่มีทางลัดจากขนาดภาพเพิ่มคะแนน macro-F1 ของ Kermany จาก #pt("loso_full|kermany|all", "loso-fill") เป็น #pt("loso_full|kermany|all", "loso-sq") แต่ทำให้แหล่งข้อมูลอื่นลดลง ส่วนการปรับจุดตัดการตัดสินใจหรือ AdaBN ยังเหลือส่วนต่างส่วนใหญ่ จึงเป็นข้อที่อาจนำมาพัฒนาต่อไปในอนาคต

  #v(0.6em)
  #par(
    first-line-indent: 0em,
  )[*คำสำคัญ:* ภาพถ่ายตัดขวางจอตาด้วยคลื่นแสงสะท้อน, โครงข่ายประสาทแบบคอนโวลูชัน, การวางนัยทั่วไปข้ามชุดข้อมูล]
  #v(front-gap)
]

#page-notes(
  front-note[1][อีเมล: 67070173\@kmitl.ac.th],
  front-note[2][อีเมล: 67070260\@kmitl.ac.th],
)

#pagebreak()

#for name in ("introduction", "background", "data", "method", "results", "discussion") { section(name) }

#bibliography("../refs.bib")
