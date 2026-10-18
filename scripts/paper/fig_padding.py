"""F4: how a B-scan becomes the 224x224 model input, letterbox vs square crop, for Duke's three image sizes.

Top row: raw Duke B-scans at the same pixel scale, so their widths differ as stored. Bottom row: the model input made
by the training preprocessing itself (`preprocess_image` with fill-aware padding; for square crop, the centred window
of `square_crop` first, as at test time). Fitting a wider scan into the square leaves taller flat bands above and below
the tissue, and in Duke width follows the label (512 normal, 768 DME, 1024 AMD), so band height does too. The square
crop column takes the same DME scan as the letterbox DME column. Scans are the Duke picks of fig_examples. Run from the
repository root:
`uv run --with matplotlib python scripts/paper/fig_padding.py`
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from oct_classify.data.geometry import square_crop
from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image
import style

DUKE = style.ROOT / "datasets/duke1"
PICKS = {"normal": "NORMAL10/TIFFs/8bitTIFFs/050.tif", "dme": "DME14/TIFFs/8bitTIFFs/030.tif",
         "amd": "AMD14/TIFFs/8bitTIFFs/037.tif"}  # narrowest to widest
SPEC = PreprocessingSpec(padding="fill_aware")
SIDE = SPEC.image_size
GAP_X, ARM_GAP, HEADER_H, ROW_TITLE_H, ARROW_H, NOTE_H = 0.14, 0.34, 0.44, 0.17, 0.2, 0.2  # inches


def as_gray(array: np.ndarray) -> np.ndarray:
    return np.clip(array[..., 0], 0, 1)


def main() -> None:
    style.apply()
    raw = {c: Image.open(DUKE / p) for c, p in PICKS.items()}
    # (header, header colour, raw image, model input, square crop window or None)
    columns = [(f"{style.CLASS_NAME[c]}", style.CLASS_COLOR[c], im, preprocess_image(im, SPEC), None)
               for c, im in raw.items()]
    dme = raw["dme"]
    crop = square_crop(dme)
    side = min(dme.size)
    window = ((dme.width - side) // 2, (dme.height - side) // 2, side)
    columns.append(("DME", style.CLASS_COLOR["dme"], dme, preprocess_image(crop, SPEC), window))

    widths = [c[2].width for c in columns]
    scale = (style.TEXT_WIDTH_IN - GAP_X * (len(columns) - 2) - ARM_GAP - 0.02) / sum(widths)  # inches per pixel
    col_x, x = [], 0.0
    for i, w in enumerate(widths):
        col_x.append(x)
        x += w * scale + (ARM_GAP if i == len(columns) - 2 else GAP_X)
    col_mid = [x + w * scale / 2 for x, w in zip(col_x, widths)]
    canvas = min(widths) * scale  # the narrowest raw scan sets the input size, so every input fits its column
    raw_h = max(c[2].height for c in columns) * scale
    raw_top = HEADER_H + ROW_TITLE_H
    in_top = raw_top + raw_h + NOTE_H + ARROW_H + ROW_TITLE_H
    width, height = style.TEXT_WIDTH_IN, in_top + canvas + NOTE_H

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    # arm headers over the letterbox columns and the square crop column
    divider = col_x[-1] - ARM_GAP / 2
    for (left, right), arm in [((0, divider - ARM_GAP / 2), "fill"), ((col_x[-1], width), "sq")]:
        ax.text((left + right) / 2, 0.02, style.ARM_NAME[arm], ha="center", va="top", fontsize=8.5,
                fontweight="bold", color=style.ARM_COLOR[arm])
        ax.plot([left, right], [0.2, 0.2], color=style.ARM_COLOR[arm], linewidth=1.2, solid_capstyle="butt")
    ax.plot([divider, divider], [0.05, height - 0.02], color=style.GRID, linewidth=0.8)
    for title, y in [("Raw B-scan (same pixel scale)", raw_top), ("Model input (224×224)", in_top)]:
        ax.text(0, y - 0.05, title, ha="left", va="bottom", fontsize=7, color=style.INK_2, style="italic")

    for (name, color, im, model_in, crop_window), x0, mid in zip(columns, col_x, col_mid):
        w, h = im.width * scale, im.height * scale
        ax.text(mid, HEADER_H - 0.02, name, ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)
        ax.imshow(np.asarray(im.convert("L")), cmap="gray", vmin=0, vmax=255, extent=(x0, x0 + w, raw_top + h, raw_top),
                  interpolation="lanczos")
        ax.add_patch(plt.Rectangle((x0, raw_top), w, h, fill=False, edgecolor=style.INK_3, linewidth=0.4))
        ax.text(mid, raw_top + h + 0.03, f"{im.width}×{im.height}", ha="center", va="top", fontsize=6.5,
                color=style.INK_2)
        if crop_window is not None:
            left, top, side = crop_window
            ax.add_patch(plt.Rectangle((x0 + left * scale, raw_top + top * scale), side * scale, side * scale,
                                       fill=False, edgecolor=style.ARM_COLOR["sq"], linewidth=1.1, linestyle=(0, (3, 1.5))))
        arrow_y = raw_top + raw_h + NOTE_H
        ax.annotate("", xy=(mid, arrow_y + ARROW_H - 0.02), xytext=(mid, arrow_y + 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=style.INK_3, linewidth=0.8, mutation_scale=7))

        c0 = mid - canvas / 2
        ax.imshow(as_gray(model_in), cmap="gray", vmin=0, vmax=1, extent=(c0, c0 + canvas, in_top + canvas, in_top),
                  interpolation="lanczos")
        ax.add_patch(plt.Rectangle((c0, in_top), canvas, canvas, fill=False, edgecolor=style.INK_3, linewidth=0.4))
        if crop_window is not None:
            note = "no bands"
        else:
            tissue = round(im.height * SIDE / im.width)  # rows of tissue after fitting the width to the canvas
            top_band = (SIDE - tissue) // 2
            for y, rows in [(0, top_band), (top_band + tissue, SIDE - tissue - top_band)]:
                ax.add_patch(plt.Rectangle((c0, in_top + y * canvas / SIDE), canvas, rows * canvas / SIDE, fill=False,
                                           edgecolor=style.ARM_COLOR["fill"], linewidth=0.9))
            note = f"bands {round((SIDE - tissue) / SIDE * 100)}%"
        ax.text(mid, in_top + canvas + 0.03, note, ha="center", va="top", fontsize=6.5, color=style.INK_2)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    style.save(fig, "padding")


if __name__ == "__main__":
    main()
