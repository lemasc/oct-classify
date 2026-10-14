"""F5: held-out Kermany (full source), recall per true class and image size, fill-aware vs square crop, 95% CIs."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style

SIZES = ["512x496", "768x496", "1024x496", "1536x496", "512x512"]
DUKE_CUE = {"768x496": "Duke: DME only", "1024x496": "Duke: AMD only"}


def main() -> None:
    style.apply()
    results = style.results()
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH_IN, 3.9))
    ticks, labels, y = [], [], 0.0
    for size in SIZES:
        entry = results[f"size_loso:{size}|kermany|all"]
        confusion = entry["arms"]["fill"]["confusion"]
        start = y
        for index, cls in enumerate(["normal", "amd", "dme"]):
            if sum(confusion[index]) == 0:
                continue
            fill = entry["arms"]["fill"]["metrics"][f"recall_{cls}"]
            square = entry["arms"]["sq"]["metrics"][f"recall_{cls}"]
            ax.plot([fill["point"] * 100, square["point"] * 100], [y, y], color=style.GRID, lw=3, zorder=1,
                    solid_capstyle="round")
            for arm, m in [("fill", fill), ("sq", square)]:
                color = style.ARM_COLOR[arm]
                ax.plot([m["lo"] * 100, m["hi"] * 100], [y, y], color=color, lw=1, zorder=2)
                ax.plot(m["point"] * 100, y, "o", ms=6, color=color, zorder=3)
            ticks.append(y)
            labels.append(f"{size.replace('x', '×')}  {style.CLASS_NAME[cls]}")
            y += 1
        if size in DUKE_CUE:
            ax.text(100.8, (start + y - 1) / 2, DUKE_CUE[size], va="center", fontsize=6.8, color=style.INK,
                    fontweight="bold")
        y += 0.5
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(45, 100)
    ax.set_xlabel("Recall on held-out Kermany (%)")
    ax.xaxis.grid(True, color=style.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color=style.ARM_COLOR[a], ms=6, label=style.ARM_NAME[a])
               for a in ["fill", "sq"]]
    ax.legend(handles=handles, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    style.save(fig, "kermany-size")


if __name__ == "__main__":
    main()
