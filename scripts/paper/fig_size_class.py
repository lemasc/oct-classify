"""F2: image size x class in each source (artifacts/eda/features.csv). Shows that Duke's 768 and 1024 widths are single-label."""

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

import style

ORDER = {
    "duke": ["512x496", "768x496", "1024x496"],
    "kermany": ["512x496", "512x512", "768x496", "1024x496", "1536x496"],
    "paima": ["768x496"],
    "octdl": [],
}


def rows() -> list[tuple[str, str, Counter]]:
    table = pd.read_csv(style.ROOT / "artifacts/eda/features.csv", usecols=["source", "width", "height", "label"])
    table["size"] = table["width"].astype(str) + "x" + table["height"].astype(str)
    out = []
    for source in ["duke", "kermany", "paima", "octdl"]:
        part = table[table["source"] == source]
        named = ORDER[source]
        for size in named:
            out.append((source, size.replace("x", "×"), Counter(part.loc[part["size"] == size, "label"])))
        if not named:  # OCTDL crops vary image by image; one row for the whole source
            out.append((source, "varied sizes", Counter(part["label"])))
    return out


def main() -> None:
    style.apply()
    data = rows()
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH_IN, 3.6))
    y_positions, y = [], 0.0
    previous = None
    for source, _, _ in data:
        if previous is not None and source != previous:
            y += 0.6  # gap between sources
        y_positions.append(y)
        y += 1
        previous = source
    for (source, size, counts), y in zip(data, y_positions, strict=True):
        total = sum(counts.values())
        left = 0.0
        for cls in ["normal", "amd", "dme"]:
            share = counts.get(cls, 0) / total
            if share == 0:
                continue
            ax.barh(y, share, left=left, height=0.72, color=style.CLASS_COLOR[cls], edgecolor="white", linewidth=1.0)
            if share == 1:
                text = f"{style.CLASS_NAME[cls]} only (single label)"
            else:
                text = f"{share:.0%}" if share >= 0.07 else ""
            ink = style.INK if cls == "dme" else "white"  # aqua is too light for white text
            ax.text(left + share / 2, y, text, ha="center", va="center", fontsize=6.8, color=ink,
                    fontweight="bold" if share == 1 else "normal")
            left += share
        ax.text(1.015, y, f"n = {total:,}", va="center", fontsize=7, color=style.INK_2)
    ax.set_yticks(y_positions, [f"{style.SOURCE_NAME[s]}  {size}" for s, size, _ in data])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of images by class")
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.CLASS_COLOR[c]) for c in ["normal", "amd", "dme"]]
    ax.legend(handles, [style.CLASS_NAME[c] for c in ["normal", "amd", "dme"]], ncol=3, loc="lower left",
              bbox_to_anchor=(0, 1.0), handlelength=1.0, columnspacing=1.2)
    ax.spines["left"].set_visible(False)
    style.save(fig, "size-class")


if __name__ == "__main__":
    main()
