"""F4: seen (fused) vs unseen (LOSO) macro-F1 on each source's test split, fill-aware and square crop, 95% CIs."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style

SOURCES = ["kermany", "octdl", "paima", "duke"]


def main() -> None:
    style.apply()
    results = style.results()
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH_IN, 3.0))
    ticks, labels = [], []
    y = 0.0
    for source in SOURCES:
        view = "normal_amd" if source == "paima" else "all"
        entry = results[f"gap_test|{source}|{view}"]["arms"]
        for arm in ["fill", "sq"]:
            seen = entry[f"fused-{arm}"]["metrics"]["macro_f1"]
            unseen = entry[f"loso-{arm}"]["metrics"]["macro_f1"]
            color = style.ARM_COLOR[arm]
            ax.plot([unseen["point"] * 100, seen["point"] * 100], [y, y], color=style.GRID, lw=3, zorder=1,
                    solid_capstyle="round")
            # Interval only on the unseen marker; seen intervals are in Table 2 and the paired gaps in Table 3.
            ax.plot([unseen["lo"] * 100, unseen["hi"] * 100], [y, y], color=color, lw=1, zorder=2)
            for m, filled in [(seen, False), (unseen, True)]:
                ax.plot(m["point"] * 100, y, "o", ms=6.5, zorder=3, mec=color, mew=1.4,
                        mfc=color if filled else "white")
            ax.text(unseen["lo"] * 100 - 0.6, y, f"{unseen['point'] * 100:.1f}", ha="right", va="center",
                    fontsize=6.8, color=style.INK_2)
            ticks.append(y)
            labels.append(f"{style.SOURCE_NAME[source]}  {style.ARM_NAME[arm].lower()}")
            y += 1
        y += 0.5
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(70, 100.5)
    ax.set_xlabel("Macro-F1 on the source's test split (%)")
    ax.xaxis.grid(True, color=style.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=style.INK_2, mew=1.4, ms=6.5, label="Seen (fused)"),
        Line2D([], [], marker="o", ls="", mfc=style.INK_2, mec=style.INK_2, ms=6.5, label="Unseen (LOSO)"),
        Line2D([], [], color=style.ARM_COLOR["fill"], lw=2, label=style.ARM_NAME["fill"]),
        Line2D([], [], color=style.ARM_COLOR["sq"], lw=2, label="Square crop"),
    ]
    ax.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=1.4)
    style.save(fig, "gap")


if __name__ == "__main__":
    main()
