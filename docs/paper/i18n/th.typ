// Thai strings. Thai section files `#import` this after lib.typ, so these names shadow the English ones there.

// Sections that are written in Thai. Anything not listed falls back to the English file in ../sections/.
#let translated = ("background", "data", "method", "results", "discussion")

#let title = [การวางนัยทั่วไปข้ามชุดข้อมูลของตัวจำแนกภาพ OCT จอประสาทตาด้วย CNN]
#let abstract-title = [บทคัดย่อ]
#let abstract = none // none: keep the English abstract

// Names shown in tables. Keys match `source-name` / `split-name` in lib.typ and sections/data.typ.
#let source-name = (kermany: "Kermany", octdl: "OCTDL", paima: "PAIMA", duke: "Duke")
#let split-name = (kermany: "supplied test", octdl: "70/15/15", paima: "70/15/15", duke: "5-fold CV")
