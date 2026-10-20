#import "../flow.typ": letterbox, square

// Thai strings. Thai section files `#import` this after lib.typ, so these names shadow the English ones there.

// Sections that are written in Thai. Anything not listed falls back to the English file in ../sections/.
#let translated = ("introduction", "background", "data", "method", "results", "discussion")

#let title = [การวิเคราะห์การจำแนกประเภทภาพถ่ายตัดขวางจอตาด้วยคลื่นแสงสะท้อนข้ามชุดข้อมูลด้วยโครงข่ายประสาทแบบคอนโวลูชัน]

// Names shown in tables. Keys match `source-name` / `split-name` in lib.typ and sections/data.typ.
#let source-name = (kermany: "Kermany", octdl: "OCTDL", paima: "PAIMA", duke: "Duke")
#let split-name = (kermany: "supplied test", octdl: "70/15/15", paima: "70/15/15", duke: "5-fold CV")

// Labels of the study overview (fig-method, flow.typ).
#let flow-labels = (
  stages: (
    ([ข้อมูล], [4 แหล่งข้อมูล 3 คลาส; คัดภาพซ้ำออก; แบ่งชุดข้อมูลระดับผู้ป่วย]),
    ([เตรียมภาพ], [#letterbox[letterbox] (ค่าตั้งต้น) หรือ #square[square crop] (@fig-padding)]),
    ([ฝึกสอน], [ResNet-50 (ImageNet); per-dataset, fused, LOSO (@fig-design)]),
    ([ประเมินผล], [macro-F1 ราย B-scan, paired bootstrap; ส่วนต่าง = LOSO − fused]),
    ([วิเคราะห์สาเหตุ], [ปรับขนาดภาพทดสอบ; แทนที่แถบขอบ; Grad-CAM]),
  ),
  loop: [พบสิ่งชี้แนะจากขนาดภาพ:\ เพิ่มกลุ่มการทดลอง\ square crop],
)
