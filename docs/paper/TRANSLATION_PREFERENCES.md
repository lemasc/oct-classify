# Thai Translation Preferences & Handoff Guide

This document captures editorial conventions, workflow rules, and translation preferences established during the Thai translation of the paper in `docs/paper/`.

---

## 1. Interaction & Workflow Preferences

- **Chat Language:** Respond primarily in English in the chat (Thai examples are welcome when discussing specific phrases). Keep Thai exclusively for the translated Typst files.
- **Review in File:** Do not paste full translation drafts into the chat. Translate directly into `th/sections/<section>.typ`, verify Typst compilation, and commit or leave for review so the user can inspect the changes with `git diff`.
- **Pacing:** Proceed strictly **section-by-section**, never all in one pass.
- **Iteration cycle:**
  1. Translate the section and add its name to `translated` in `docs/paper/i18n/th.typ`.
  2. Verify compilation: `typst compile --root ../.. th/main.typ`.
  3. Commit the initial draft.
  4. The user reviews and refines directly in the `.typ` file.
  5. Inspect the user's `git diff`, discuss any terminology decisions, and amend the commit before moving to the next section.

---

## 2. Text Structure & Formatting Rules

- **Paragraph Breaks:** Thai does not use sentence-ending periods. Long, dense English paragraphs feel overwhelming when translated directly. **Actively break long paragraphs into shorter 2–4 sentence units** for readability.
- **Section Hierarchy:** Use Typst subheadings (`==`, `===`) generously to separate distinct concepts (for example, promoting *Data preparation* and *Model and training* to separate level-2 headings rather than burying them together).
- **Hyperparameters & Settings:** For model hyperparameters, optimizer settings, and training schedules, use a concise **bulleted list** (`- *Optimizer...*: ...`) instead of running prose or formal floating tables.
- **Run-in Headings:** When using inline bold titles before prose, append a colon (`:`) to mark a clean visual boundary (e.g. `*ส่วนต่างความแม่นยำ (The gap):* ...`).
- **Typst Code & References:** Preserve all Typst functions (`#num(...)`, `#pt(...)`, `#stage-share(...)`), citations (`@srinivasan2014`), and labels (`<sec-...>`, `<fig-...>`, `<tab-...>`) unchanged. Move helper calls to wherever Thai word order requires them.

---

## 3. Terminology & Vocabulary Conventions

- **Target Audience:** Machine learning and computer vision researchers, not clinical medical doctors.
- **Class Labels:** English-first or bilingual with Thai in parentheses:
  - `Normal (ปกติ)`, `AMD`, `DME`
  - AMD stages: `Dry (แห้ง)`, `Wet (เปียก)`
- **Dataset Splits:** Do **not** translate `train`, `validation`, and `test` into Thai words. Use:
  - `ชุด train`, `ชุด validation`, `ชุด test`
  - `test split`, `5-fold cross-validation`
- **Experimental Arms & Model Variants:** Keep primary ML terms in English:
  - `per-dataset` (one model trained and tested on its own source)
  - `fused` (single model trained across all sources)
  - `leave-one-source-out (LOSO)` (trained on 3 sources, tested on held-out source)
  - `unseen-source gap` (score drop between fused and LOSO)
  - `full-source` (evaluation across all images of the held-out source)
  - `square-crop` vs `letterbox`
  - `fill-aware padding`
  - `cluster bootstrap`
- **Domain & Device Shift:**
  - `unseen source`: A dataset unseen during training.
  - `unseen device`: A hardware scanner vendor unseen during training (specifically Optovue for OCTDL, whereas Kermany, Duke, and PAIMA all share the Heidelberg family).
  - Keep both terms in English, providing brief contextual explanation on first use.
- **Explainable AI (XAI):**
  - **Attribution:** `การระบุส่วนที่ส่งผลต่อการทำนาย (Attribution)`
  - **Intervention:** `การแทรกแซงข้อมูล (Intervention)`
  - **Attribution map:** `แผนภาพ attribution` / `แผนภาพ Grad-CAM`
- **Data Units:**
  - Duke: `ตา` or `ดวงตา` (volume / eye)
  - Other sources: `ผู้ป่วย` (patients) and `ภาพ B-scan` (individual slices)

---

## 4. Current Translation Status

| Section | English File | Thai File | Status |
| :--- | :--- | :--- | :--- |
| **Introduction** | `sections/introduction.typ` | `th/sections/introduction.typ` | Deferred by author (stub only) |
| **Data** | `sections/data.typ` | `th/sections/data.typ` | **Done & committed** (`e104c2f`) |
| **Method** | `sections/method.typ` | `th/sections/method.typ` | **Done & committed** (`0a46097`) |
| **Results** | `sections/results.typ` | `th/sections/results.typ` | Wording migrated from the colleague's DOCX draft (`docs/paper.typ`) |
| **Discussion** | `sections/discussion.typ` | `th/sections/discussion.typ` | Wording migrated from the colleague's DOCX draft |
| **Abstract** | `main.typ` | `i18n/th.typ` | Deferred by author |

To resume in the next session, begin with **Section 5: Discussion** (`sections/discussion.typ` -> `th/sections/discussion.typ`) once Results is finalized.
