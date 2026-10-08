"""Detection and repair of white acquisition fill (rotation/crop padding) in OCT images, no torch."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

WHITE_THRESHOLD = 250  # all channels at or above: saturated fill (rotation/crop padding)
BLACK_THRESHOLD = 3  # all channels at or below: empty fill (JPEG ringing tolerated)
MIN_FILL_AREA = 300  # px; smaller border-touching blobs are speckle, not padding
EDGE_GROW = 3  # px; JPEG/antialiasing leaves 200-249 pixels just outside the saturated fill
EDGE_FLOOR = 150  # grown pixels must be at least this bright to count as fill edge
BACKGROUND_RING = 40  # px; the fill is re-drawn from the tissue-side pixels this close to it
MAX_FILL_FRACTION = 0.6  # above this, the image is not trusted to be "tissue plus padding"


def _border_connected(mask: np.ndarray) -> np.ndarray:
    """Keep only mask components that touch the image border and are large enough."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.zeros(count + 1, dtype=bool)
    keep[border] = True
    keep &= areas >= MIN_FILL_AREA
    keep[0] = False
    return keep[labels]


def fill_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of border-connected saturated (white) and empty (black) fill in an HWC image."""
    white = _border_connected((rgb >= WHITE_THRESHOLD).all(axis=2))
    black = _border_connected((rgb <= BLACK_THRESHOLD).all(axis=2))
    return white, black


def white_fill_mask(rgb: np.ndarray) -> np.ndarray:
    """White fill including its JPEG edge; empty when absent or implausibly large.

    Only white is treated as padding. Black regions can be genuine background (e.g. PAIMA's zeroed
    deep layers), so they stay untouched.
    """
    white, _ = fill_masks(rgb)
    if not white.any():
        return white
    gray = rgb.mean(axis=2)
    # Grow only through bright pixels so tissue next to the fill is not eaten.
    fill = white | (ndimage.binary_dilation(white, iterations=EDGE_GROW) & (gray >= EDGE_FLOOR))
    return fill if fill.mean() <= MAX_FILL_FRACTION else np.zeros_like(fill)


def redraw_fill(rgb: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Replace `fill` with pixels drawn from the nearby tissue-side background (deterministic)."""
    if not fill.any():
        return rgb
    ring = ndimage.binary_dilation(fill, iterations=BACKGROUND_RING) & ~fill
    donors = rgb[ring if ring.any() else ~fill]
    rng = np.random.default_rng(0)
    out = rgb.copy()
    out[fill] = donors[rng.integers(0, len(donors), size=int(fill.sum()))]
    return out
