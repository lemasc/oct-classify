"""F3: the three training variants as a grid of models x sources (schematic, no data).

Each cell says what a model does with a source: trains on its train split and is scored on its test split, trains
only, or never sees it and is scored on all its images (held out). Model counts are on the right; which variants also
have a square-crop arm is left to the caption. Run from the repository root:
`uv run --with matplotlib python scripts/paper/fig_design.py`
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import style

SOURCES = ["kermany", "octdl", "paima", "duke"]
SEEN, TRAIN, HELD = "seen", "train", "held"

# (group, row label or None for a single-row group, cell state per source, number of models)
ROWS = [
    ("Per-dataset", None, [SEEN] * 4, "1 per source"),
    ("Fused", None, [SEEN] * 4, "5"),
    *[
        ("LOSO", f"{style.SOURCE_NAME[h]} held out", [HELD if s == h else TRAIN for s in SOURCES],
         "1" if h == "duke" else "5")
        for h in SOURCES
    ],
]
CELL_W, CELL_H, GAP_X, GAP_Y, GROUP_GAP = 2.0, 0.62, 0.12, 0.1, 0.3
TEST_W = 0.36  # share of a seen cell drawn as its test split
LABEL_X = -0.18  # right edge of row labels
GROUP_X = -2.95  # x of the bracket and rotated label of a multi-row group
NOTE_X = len(SOURCES) * (CELL_W + GAP_X) + 0.2
ROUND = "round,pad=0,rounding_size=0.06"


def cell(ax, x, y, state):
    mid = y + CELL_H / 2
    if state == HELD:
        ax.add_patch(FancyBboxPatch((x, y), CELL_W, CELL_H, boxstyle=ROUND, facecolor="white",
                                    edgecolor=style.INK, linewidth=1.3))
        ax.text(x + CELL_W / 2, y + CELL_H * 0.35, "held out", ha="center", va="center", fontsize=7,
                fontweight="bold")
        ax.text(x + CELL_W / 2, y + CELL_H * 0.72, "all images", ha="center", va="center", fontsize=6.3,
                color=style.INK_2)
        return
    ax.add_patch(FancyBboxPatch((x, y), CELL_W, CELL_H, boxstyle=ROUND, facecolor=style.GRID, linewidth=0))
    if state == TRAIN:
        ax.text(x + CELL_W / 2, mid, "train", ha="center", va="center", fontsize=7)
        return
    # seen: train split on the left, test split as a dark strip on the right
    split = x + CELL_W * (1 - TEST_W)
    ax.add_patch(FancyBboxPatch((split, y), CELL_W * TEST_W, CELL_H, boxstyle=ROUND, facecolor=style.INK_3,
                                linewidth=0))
    ax.text((x + split) / 2, mid, "train", ha="center", va="center", fontsize=7)
    ax.text(split + CELL_W * TEST_W / 2, mid, "test", ha="center", va="center", fontsize=6.5, color="white")


def main() -> None:
    style.apply()
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH_IN, 2.9))
    y, previous, group_rows = 0.0, None, {}
    for group, label, states, models in ROWS:
        if previous is not None and group != previous:
            y += GROUP_GAP
        for i, state in enumerate(states):
            cell(ax, i * (CELL_W + GAP_X), y, state)
        mid = y + CELL_H / 2
        if label is None:  # single-row group: the group name is the row label
            ax.text(LABEL_X, mid, group, ha="right", va="center", fontsize=8, fontweight="bold")
        else:
            ax.text(LABEL_X, mid, label, ha="right", va="center", fontsize=7.5, color=style.INK_2)
            group_rows.setdefault(group, []).append(y)
        ax.text(NOTE_X, mid, models, ha="left", va="center", fontsize=7, color=style.INK_2)
        previous = group
        y += CELL_H + GAP_Y
    for group, ys in group_rows.items():
        top, bottom = ys[0] + 0.04, ys[-1] + CELL_H - 0.04
        ax.plot([GROUP_X] * 2, [top, bottom], color=style.INK_3, linewidth=0.8)
        ax.text(GROUP_X - 0.12, (top + bottom) / 2, group, ha="right", va="center", rotation=90, fontsize=8,
                fontweight="bold")
    header_y = -0.2
    for i, s in enumerate(SOURCES):
        ax.text(i * (CELL_W + GAP_X) + CELL_W / 2, header_y, style.SOURCE_NAME[s], ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.text(NOTE_X, header_y, "Models", ha="left", va="bottom", fontsize=7.5, fontweight="bold", color=style.INK_2)
    ax.set_xlim(GROUP_X - 0.5, NOTE_X + 1.3)
    ax.set_ylim(y, header_y - 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    style.save(fig, "design")


if __name__ == "__main__":
    main()
