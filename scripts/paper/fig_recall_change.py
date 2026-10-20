"""F9: change in per-class recall when each source is held out (LOSO minus fused), letterbox, test split.

Cells are coloured only when the paired 95% interval excludes zero (beyond noise); the rest stay neutral gray.
Losses are red, gains blue, on one symmetric scale so cells compare across sources.
"""

import json
from decimal import ROUND_HALF_UP, Decimal

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle

import style

SOURCES = ["kermany", "octdl", "paima", "duke"]
COLUMNS = [("normal", "Normal"), ("amd", "AMD"), ("dry", "dry"), ("wet", "wet"), ("dme", "DME")]
PAIR = "loso-fill - fused-fill"
LIMIT = 35  # percentage points at the ends of the scale; Kermany normal (about −32) is the largest change
NEUTRAL = "#f0efec"  # diverging midpoint (reference palette)
# Diverging pair from the reference palette: red arm for losses, blue arm for gains, gray midpoint.
CMAP = LinearSegmentedColormap.from_list(
    "recall_change", [(0, "#a12a2a"), (0.25, "#e34948"), (0.5, NEUTRAL), (0.75, "#3987e5"), (1, "#1c5cab")])
STAGES = json.loads((style.ROOT / "data/paper/amd-stage/results.json").read_text())


def cell(results: dict, source: str, column: str) -> dict | None:
    """Paired delta {point, lo, hi} as fractions, or None where the class does not apply."""
    view = "normal_amd" if source == "paima" else "all"
    deltas = results[f"gap_test|{source}|{view}"]["deltas"][PAIR]
    if column in ("normal", "amd", "dme"):
        return deltas.get(f"recall_{column}")
    if source == "duke":  # Duke's AMD is all dry
        return deltas["recall_amd"] if column == "dry" else None
    return STAGES[f"test|{source}|{view}"]["deltas"][column][PAIR]


def fmt(x: float) -> str:
    """Signed percentage with one decimal, rounded half away from zero like `fmt` in lib.typ (−6.25 -> −6.3)."""
    v = float(Decimal(repr(x * 100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return ("−" if v < 0 else "+" if v > 0 else "") + f"{abs(v):.1f}"


def ink_on(rgb: tuple) -> str:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "white" if luminance < 0.45 else style.INK


def main() -> None:
    style.apply()
    results = style.results()
    norm = Normalize(-LIMIT, LIMIT)
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH_IN, 2.35))
    gap = 0.04  # surface gap between cells, in cell units
    for row, source in enumerate(SOURCES):
        for col, (column, _) in enumerate(COLUMNS):
            d = cell(results, source, column)
            x, y = col + gap / 2, row + gap / 2
            if d is None:
                ax.add_patch(Rectangle((x, y), 1 - gap, 1 - gap, color="white", ec=style.GRID, lw=0.6))
                ax.text(col + 0.5, row + 0.5, "–", ha="center", va="center", color=style.INK_3, fontsize=8)
                continue
            beyond = d["lo"] > 0 or d["hi"] < 0
            rgb = to_rgb(CMAP(norm(d["point"] * 100))) if beyond else to_rgb(NEUTRAL)
            ax.add_patch(Rectangle((x, y), 1 - gap, 1 - gap, color=rgb, lw=0))
            ink = ink_on(rgb) if beyond else style.INK_2
            ax.text(col + 0.5, row + 0.42, fmt(d["point"]), ha="center", va="center", color=ink,
                    fontsize=9, fontweight="bold" if beyond else "normal")
            ax.text(col + 0.5, row + 0.72, f"[{fmt(d['lo'])}, {fmt(d['hi'])}]", ha="center", va="center",
                    color=ink, fontsize=6.3)
    ax.set_xlim(0, len(COLUMNS))
    ax.set_ylim(len(SOURCES), 0)
    ax.set_yticks([r + 0.5 for r in range(len(SOURCES))], [style.SOURCE_NAME[s] for s in SOURCES])
    ax.set_xticks([c + 0.5 for c in range(len(COLUMNS))], [name for _, name in COLUMNS])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Group the two AMD stages under one header.
    ax.annotate("", xy=(2.05, -0.38), xytext=(3.95, -0.38), xycoords="data", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=style.INK_3, lw=0.6))
    ax.text(3, -0.48, "AMD stage", ha="center", va="bottom", fontsize=7.5, color=style.INK_2)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, fraction=0.025, pad=0.02,
                       ticks=[-30, -15, 0, 15, 30])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=7)
    bar.ax.set_yticklabels(["−30", "−15", "0", "+15", "+30"])
    bar.set_label("Recall change (points)", fontsize=7.5, color=style.INK_2)
    fig.text(0.01, -0.02, "Gray: 95% interval includes zero.   –: class not in this source.", fontsize=7,
             color=style.INK_2)
    style.save(fig, "recall-change")


if __name__ == "__main__":
    main()
