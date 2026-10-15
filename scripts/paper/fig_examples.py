"""F1: one B-scan per source and class, all drawn at the same pixel scale so that image size differences show.

AMD is shown as its two stages, dry (drusen) and wet (CNV/MNV), which every source merges into one AMD class for
training. Stage comes from the source labels: Kermany and PAIMA DRUSEN vs CNV, OCTDL subcategory early (drusen) vs late
(MNV); Duke's AMD is dry only. Duke's AMD and DME examples are its single-label sizes (1024x496 AMD, 768x496 DME).
Scans were picked by eye from seeded random contact sheets of different patients for clear, typical pathology; Duke
scans are central slices of their volumes. Images are shown as stored, with a thin frame marking the canvas so that
white rotation fill stays visible. Run from the repository root:
`uv run --with matplotlib python scripts/paper/fig_examples.py`
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import style

ROOT = {"kermany": "datasets/kermany", "octdl": "datasets/octdl", "paima": "datasets/paima", "duke": "datasets/duke1"}
DEVICE = {"kermany": "Spectralis", "octdl": "RTVue XR", "paima": "Heidelberg", "duke": "Spectralis"}
COLUMNS = [("normal", "Normal"), ("dry", "Dry (drusen)"), ("wet", "Wet (CNV)"), ("dme", "DME")]
MISSING = {("paima", "dme"): "no DME images", ("duke", "wet"): "no wet AMD"}
PICKS = {
    "kermany": {"normal": "train/NORMAL/NORMAL-6190347-3.jpeg", "dry": "train/DRUSEN/DRUSEN-4878077-53.jpeg",
                "wet": "train/CNV/CNV-7732521-77.jpeg", "dme": "train/DME/DME-4643364-13.jpeg"},
    "octdl": {"normal": "NO/no_7052103_7.jpg", "dry": "AMD/amd_3441858_4.jpg", "wet": "AMD/amd_1513396_2.jpg",
              "dme": "DME/dme_8800161_1.jpg"},
    "paima": {"normal": "NORMAL/28/OS/005_Normal.tif", "dry": "DRUSEN/136/OD/016_Drusen.tif",
              "wet": "CNV/9/009_CNV.tif"},
    "duke": {"normal": "NORMAL10/TIFFs/8bitTIFFs/050.tif", "dry": "AMD14/TIFFs/8bitTIFFs/037.tif",
             "dme": "DME14/TIFFs/8bitTIFFs/030.tif"},
}
LABEL_W, GAP_X, GAP_Y, CAPTION_H, HEADER_H = 0.78, 0.06, 0.06, 0.15, 0.42  # inches


def main() -> None:
    style.apply()
    images = {(s, c): np.asarray(Image.open(style.ROOT / ROOT[s] / p).convert("L"))
              for s, picks in PICKS.items() for c, p in picks.items()}
    keys = [k for k, _ in COLUMNS]
    col_px = [max(im.shape[1] for (_, c), im in images.items() if c == k) for k in keys]
    row_px = [max(im.shape[0] for (s, _), im in images.items() if s == src) for src in PICKS]
    scale = (style.TEXT_WIDTH_IN - LABEL_W - GAP_X * (len(keys) - 1) - 0.02) / sum(col_px)  # inches per pixel
    col_x = np.cumsum([LABEL_W] + [w * scale + GAP_X for w in col_px[:-1]])
    col_mid = [x + w * scale / 2 for x, w in zip(col_x, col_px)]
    row_h = [h * scale + CAPTION_H + GAP_Y for h in row_px]
    row_y = np.cumsum([HEADER_H] + row_h[:-1])
    width, height = style.TEXT_WIDTH_IN, HEADER_H + sum(row_h)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    for c, (key, name) in enumerate(COLUMNS):
        ax.text(col_mid[c], HEADER_H - 0.06, name, ha="center", va="bottom", fontsize=8,
                fontweight="normal" if key in ("dry", "wet") else "bold")
    # dry and wet are one training class: bracket over both columns
    left, right = col_x[1], col_x[2] + col_px[2] * scale
    bracket_y = HEADER_H - 0.24
    ax.plot([left, left, right, right], [bracket_y + 0.04, bracket_y, bracket_y, bracket_y + 0.04],
            color=style.INK_2, linewidth=0.7)
    ax.text((left + right) / 2, bracket_y - 0.03, "AMD (one class in training)", ha="center", va="bottom",
            fontsize=8, fontweight="bold")
    for r, src in enumerate(PICKS):
        top, box_h = row_y[r], row_px[r] * scale
        ax.text(LABEL_W - 0.1, top + box_h / 2, style.SOURCE_NAME[src], ha="right", va="bottom", fontsize=8,
                fontweight="bold")
        ax.text(LABEL_W - 0.1, top + box_h / 2 + 0.03, DEVICE[src], ha="right", va="top", fontsize=7,
                color=style.INK_2)
        for c, key in enumerate(keys):
            im = images.get((src, key))
            if im is None:
                ax.text(col_mid[c], top + box_h / 2, MISSING[(src, key)], ha="center", va="center", fontsize=7,
                        color=style.INK_3, style="italic")
                continue
            h, w = im.shape[0] * scale, im.shape[1] * scale
            x0, y0 = col_mid[c] - w / 2, top + (box_h - h) / 2
            ax.imshow(im, cmap="gray", vmin=0, vmax=255, extent=(x0, x0 + w, y0 + h, y0), interpolation="lanczos")
            ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, edgecolor=style.INK_3, linewidth=0.4))
            ax.text(col_mid[c], y0 + h + 0.03, f"{im.shape[1]}×{im.shape[0]}", ha="center", va="top", fontsize=6.5,
                    color=style.INK_2)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    path = style.FIGURES / "examples.png"
    fig.savefig(path, dpi=300)
    print(f"wrote {path.relative_to(style.ROOT)}")


if __name__ == "__main__":
    main()
