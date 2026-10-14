"""Shared figure style for the paper (static print figures, light surface).

Colours come from the validated reference palette (dataviz skill):
- classes: normal / AMD / DME = slots 1-3, validated all-pairs; DME aqua is < 3:1 on white, so bars carry direct labels.
- arms: fill-aware violet, square crop green, validated as a pair; seen (fused) is a neutral hollow marker.
"""

import json
from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs/paper/figures"
RESULTS = ROOT / "data/paper/results.json"

INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
CLASS_COLOR = {"normal": "#2a78d6", "amd": "#eb6834", "dme": "#1baf7a"}
CLASS_NAME = {"normal": "Normal", "amd": "AMD", "dme": "DME"}
ARM_COLOR = {"fill": "#6250d6", "sq": "#008300"}
ARM_NAME = {"fill": "Fill-aware", "sq": "Square crop"}
SOURCE_NAME = {"kermany": "Kermany", "octdl": "OCTDL", "paima": "PAIMA", "duke": "Duke"}

TEXT_WIDTH_IN = 6.3  # A4 with 2.5 cm margins


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": INK_3,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0,
        "axes.grid": False,
        "legend.frameon": False,
        "svg.fonttype": "path",
        "figure.dpi": 150,
    })


def results() -> dict:
    return json.loads(RESULTS.read_text())


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {path.relative_to(ROOT)}")
