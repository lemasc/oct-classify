"""F0: how an OCT volume is built, from one Duke normal eye (NORMAL10, the eye shown in fig_examples).

(a) one A-scan: the brightness of one image column against depth; (b) the B-scan through the fovea (slice 50 of 97),
512 A-scans side by side; (c) the volume, B-scans stacked across the macula, drawn from the B-scan of (b) backwards (slices 1-49 omitted). Scan size
is from Srinivasan et al. 2014, Table 1: every Duke normal volume covers about 6 x 6 mm with 512 A-scans per B-scan;
axial pixel spacing is 3.87 um. Panels (a) and (b) are cropped in depth to the band that holds the retina, and drawn
at equal pixel scale, so they are stretched about 3x in depth relative to their width; the scale bar, drawn like the one Heidelberg burns into its exports, has two 200 um arms. The A-scan is plotted raw (thin)
and with a 7-pixel running mean (bold). Run from the repository root:
`uv run --with matplotlib python scripts/paper/fig_scan.py`
"""

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

import style

VOLUME = style.ROOT / "datasets/duke1/NORMAL10/TIFFs/8bitTIFFs"
N_SLICES, B_SLICE, A_COL = 97, 50, 150  # slices are 1-based; A_COL is the image column shown as the A-scan
ROWS = slice(90, 340)  # depth crop that keeps the retina and choroid
FOVEA = (258, 170)  # (column, row) of the foveal pit (deepest point of the inner retinal surface) in slice 50
SCAN_MM, AXIAL_UM = 6.0, 3.87
STACK = [97, 89, 81, 73, 65, 57, 50]  # back to front; the front one is B_SLICE, so slices 1-49 are not drawn
HIGHLIGHT = "#f2b134"
PX = 2.2 / 512  # inches per pixel in (a) and (b)
A_W, GAP, TOP = 0.75, 0.32, 0.5  # inches
STACK_OFFSET = 0.42  # inches, total diagonal offset of the volume's slices


def load(i: int) -> np.ndarray:
    name = f"0{i}.tif" if i < 10 else f"{i:03d}.tif"  # files are 01..09, then 010..097
    return np.asarray(Image.open(VOLUME / name).convert("L"))


def main() -> None:
    style.apply()
    bscan = load(B_SLICE)
    crop = bscan[ROWS]
    h_px, w_px = crop.shape
    b_w, b_h = w_px * PX, h_px * PX
    b_x = A_W + GAP
    v_x = b_x + b_w + GAP + 0.45
    v_w = style.TEXT_WIDTH_IN - v_x
    height = TOP + b_h + 0.3

    fig = plt.figure(figsize=(style.TEXT_WIDTH_IN, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, style.TEXT_WIDTH_IN)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    def title(x, text):
        ax.text(x, TOP - 0.22, text, ha="left", va="bottom", fontsize=8.5, fontweight="bold")

    # (b) B-scan with the A-scan column marked
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, extent=(b_x, b_x + b_w, TOP + b_h, TOP), interpolation="lanczos")
    ax.add_patch(plt.Rectangle((b_x, TOP), b_w, b_h, fill=False, edgecolor=style.INK_3, linewidth=0.4))
    col_x = b_x + (A_COL + 0.5) * PX
    ax.plot([col_x, col_x], [TOP, TOP + b_h], color=HIGHLIGHT, linewidth=0.9)
    fx, fy = b_x + FOVEA[0] * PX, TOP + (FOVEA[1] - ROWS.start) * PX
    ax.annotate("fovea", (fx + 0.02, fy - 0.03), (fx + 0.4, fy - 0.3), color="white", fontsize=7, ha="left",
                va="center", arrowprops=dict(arrowstyle="-", color="white", linewidth=0.6, shrinkB=1))
    ax.text(col_x, TOP - 0.03, "one A-scan", ha="center", va="bottom", fontsize=7, color=style.INK_2)
    title(b_x - 0.0, "(b) B-scan")
    dim_y = TOP + b_h + 0.1
    ax.annotate("", (b_x, dim_y), (b_x + b_w, dim_y),
                arrowprops=dict(arrowstyle="<->", color=style.INK_2, linewidth=0.6, shrinkA=0, shrinkB=0))
    ax.text(b_x + b_w / 2, dim_y + 0.05, f"512 A-scans side by side (≈{SCAN_MM:.0f} mm)", ha="center", va="top",
            fontsize=7, color=style.INK_2)
    # scale bar in the style Heidelberg burns into exports (as on PAIMA scans): an L with two 200 um arms in the lower
    # left, white with a dark outline, one label inside the L
    bar_um, outline = 200, [patheffects.withStroke(linewidth=1.6, foreground=style.INK)]
    bar_h = bar_um / AXIAL_UM * PX
    bar_w = bar_um / (SCAN_MM * 1000 / w_px) * PX
    bar_x, bar_bottom = b_x + 10 * PX, TOP + b_h - 8 * PX
    ax.plot([bar_x, bar_x, bar_x + bar_w], [bar_bottom - bar_h, bar_bottom, bar_bottom], color="white",
            linewidth=0.9, solid_joinstyle="miter", solid_capstyle="butt", path_effects=outline)
    ax.text(bar_x + 3 * PX, bar_bottom - 3 * PX, f"{bar_um} µm", ha="left", va="bottom", fontsize=5.5,
            family="DejaVu Sans", color="white", path_effects=outline)

    # (a) A-scan: brightness of that column against depth, on the B-scan's depth scale
    col = crop[:, A_COL].astype(float)
    smooth = np.convolve(col, np.ones(7) / 7, mode="same")
    depth = TOP + (np.arange(h_px) + 0.5) * PX
    to_x = lambda v: 0.08 + v / 255 * (A_W - 0.14)
    ax.plot(to_x(col), depth, color=style.INK_3, linewidth=0.3)
    ax.plot(to_x(smooth), depth, color=style.INK, linewidth=0.9)
    ax.plot([0.08, 0.08], [TOP, TOP + b_h], color=style.INK_3, linewidth=0.6)
    ax.annotate("", (b_x - 0.04, TOP + b_h * 0.12), (col_x, TOP + b_h * 0.12),
                arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, linewidth=0.8, shrinkA=0, shrinkB=0))
    ax.text(0.08, TOP + b_h + 0.1, "brightness →", ha="left", va="top", fontsize=6.5, color=style.INK_2)
    ax.text(0.04, TOP + b_h / 2, "depth ↓", ha="right", va="center", fontsize=6.5, color=style.INK_2)
    title(0, "(a) A-scan")

    # (c) volume: slices offset along a diagonal, cut open at the B-scan of (b)
    order = STACK
    s_w = v_w - STACK_OFFSET
    s_h = s_w * h_px / w_px
    dx = dy = STACK_OFFSET / (len(order) - 1)
    y0 = TOP
    for k, i in enumerate(order):
        x, y = v_x + k * dx, y0 + k * dy
        ax.imshow(load(i)[ROWS], cmap="gray", vmin=0, vmax=255, extent=(x, x + s_w, y + s_h, y),
                  interpolation="lanczos", zorder=k)
        edge, lw = (HIGHLIGHT, 1.0) if i == B_SLICE else (style.INK_3, 0.4)
        ax.add_patch(plt.Rectangle((x, y), s_w, s_h, fill=False, edgecolor=edge, linewidth=lw, zorder=k + 0.5))
    ax.annotate("", (v_x - 0.08, y0 + s_h * 0.6), (v_x - 0.08 + STACK_OFFSET, y0 + s_h * 0.6 + STACK_OFFSET),
                arrowprops=dict(arrowstyle="->", color=style.INK_2, linewidth=0.6, shrinkA=0, shrinkB=0))
    ax.text(v_x - 0.08, y0 + s_h * 0.6 + STACK_OFFSET / 2, "next\nB-scans ", ha="right", va="center",
            fontsize=6.5, color=style.INK_2, linespacing=1.1)
    ax.text(v_x + STACK_OFFSET + s_w, y0 + STACK_OFFSET + s_h + 0.1,
            f"(b) is B-scan {B_SLICE} of {N_SLICES}, spread over ≈{SCAN_MM:.0f} mm", ha="right", va="top", fontsize=7,
            color=style.INK_2)
    title(v_x, "(c) Volume")

    style.save(fig, "scan")


if __name__ == "__main__":
    main()
