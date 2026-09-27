"""Rough retina and padding masks for judging where attribution maps place their mass.

The mask is a heuristic: it separates the bright retinal band from the dark vitreous/choroid
background and from the letterbox padding added by `preprocess_image`. It does not segment
layers, so use it for coarse "on retina versus off retina" statistics only.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def canvas_box(width: int, height: int, image_size: int) -> tuple[int, int, int, int]:
    """Return `(left, top, width, height)` of the resized image inside the square canvas.

    Mirrors `oct_classify.data.preprocessing.preprocess_image`.
    """
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return (
        (image_size - resized_width) // 2,
        (image_size - resized_height) // 2,
        resized_width,
        resized_height,
    )


def content_mask(width: int, height: int, image_size: int) -> np.ndarray:
    """True inside the resized image, False on the letterbox padding."""
    left, top, box_width, box_height = canvas_box(width, height, image_size)
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[top : top + box_height, left : left + box_width] = True
    return mask


def _touching_top_or_bottom(labels: np.ndarray, content: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(content.any(axis=1))
    if len(rows) == 0:
        return np.zeros(0, dtype=labels.dtype)
    edges = np.concatenate([labels[rows[0]], labels[rows[-1]]])
    return np.unique(edges[edges > 0])


def fill_mask(
    image: np.ndarray,
    content: np.ndarray,
    *,
    window: int = 5,
    tolerance: float = 0.03,
    min_area_fraction: float = 0.001,
) -> np.ndarray:
    """Artificial fill inside the image content, e.g. Kermany's white rotation wedges.

    Saturated white regions touching the top or bottom content edge qualify (the RPE can saturate
    too, but it runs to the left and right edges). Black regions must also be flat, because the
    real vitreous is dark but speckled.
    """
    mean = ndimage.uniform_filter(image.astype(np.float64), window)
    variance = ndimage.uniform_filter(np.square(image.astype(np.float64)), window) - np.square(mean)
    flat = np.sqrt(np.maximum(variance, 0)) < tolerance
    candidates = ((image >= 1 - tolerance) | ((image <= tolerance) & flat)) & content
    labels, count = ndimage.label(candidates)
    if not count:
        return np.zeros_like(content)
    touching = _touching_top_or_bottom(labels, content)
    sizes = ndimage.sum(candidates, labels, index=touching)
    keep = touching[sizes >= min_area_fraction * content.sum()]
    return ndimage.binary_dilation(np.isin(labels, keep), iterations=2) & content


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    histogram, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_low = np.cumsum(histogram)
    weight_high = weight_low[-1] - weight_low
    sum_low = np.cumsum(histogram * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_low = sum_low / weight_low
        mean_high = (sum_low[-1] - sum_low) / weight_high
        between = weight_low * weight_high * np.square(mean_low - mean_high)
    return float(centers[np.nanargmax(between)])


def retina_mask(
    image: np.ndarray,
    content: np.ndarray,
    *,
    sigma: float = 2.0,
    margin: int = 4,
    below_rpe: int = 10,
    min_component_fraction: float = 0.05,
    opening: int = 5,
    boundary_degree: int = 4,
    boundary_tolerance: float = 16.0,
) -> np.ndarray:
    """Estimate the retinal band of a preprocessed grayscale B-scan in `[0, 1]`.

    `content` should already exclude padding and fill wedges (see `fill_mask`). Tissue is the
    smoothed content above an Otsu threshold, reduced to components holding at least
    `min_component_fraction` of it. A second Otsu threshold within the tissue marks the
    hyper-reflective layers (nerve fibre layer, RPE). Per column the band runs from the first tissue
    row to `below_rpe` pixels under the last hyper-reflective row, which keeps sub-RPE lesions but
    drops most of the moderately bright choroid. Each boundary is fitted with a robust polynomial
    across columns, and columns far from the fit (e.g. a bright edge artifact captured as tissue)
    take the fitted row instead.
    """
    if image.shape != content.shape:
        raise ValueError("image and content mask must have the same shape.")
    if not content.any():
        return np.zeros_like(content)
    # Fill excluded pixels so smoothing does not bleed padding or white wedges into the content.
    filled = np.where(content, image, np.median(image[content])).astype(np.float64)
    smoothed = ndimage.gaussian_filter(filled, sigma)
    tissue = (smoothed > otsu_threshold(smoothed[content])) & content
    # Opening drops thin bright lines (image borders, wedge edges) that would join the tissue.
    tissue = ndimage.binary_opening(tissue, structure=np.ones((opening, opening), dtype=bool))
    labels, count = ndimage.label(tissue)
    if count:
        sizes = ndimage.sum(tissue, labels, index=np.arange(1, count + 1))
        keep = np.flatnonzero(sizes >= min_component_fraction * sizes.sum()) + 1
        # Small pieces on the top or bottom edge are fill or border artifacts, not retina.
        edge_pieces = _touching_top_or_bottom(labels, content)
        small = edge_pieces[sizes[edge_pieces - 1] < 0.25 * sizes.max()]
        tissue = np.isin(labels, np.setdiff1d(keep, small))
    if not tissue.any():
        return np.zeros_like(content)
    hyper = tissue & (smoothed > otsu_threshold(smoothed[tissue]))
    height = image.shape[0]
    rows = np.arange(height)[:, None]
    has_tissue = tissue.any(axis=0)
    top = tissue.argmax(axis=0)
    last_tissue = height - 1 - tissue[::-1].argmax(axis=0)
    last_hyper = height - 1 - hyper[::-1].argmax(axis=0)
    bottom = np.where(hyper.any(axis=0), np.minimum(last_tissue, last_hyper + below_rpe), last_tissue)
    columns = np.flatnonzero(has_tissue)
    if len(columns) > boundary_degree + 1:
        top[columns] = _robust_boundary(columns, top[columns], boundary_degree, boundary_tolerance)
        bottom[columns] = _robust_boundary(
            columns, bottom[columns], boundary_degree, boundary_tolerance
        )
    band = (rows >= top - margin) & (rows <= bottom + margin) & has_tissue[None, :]
    return band & content


def _robust_boundary(
    columns: np.ndarray, rows: np.ndarray, degree: int, tolerance: float, iterations: int = 4
) -> np.ndarray:
    x = (columns - columns.mean()) / max(np.ptp(columns), 1)
    y = rows.astype(np.float64)
    keep = np.ones(len(x), dtype=bool)
    for _ in range(iterations):
        fit = np.polyval(np.polyfit(x[keep], y[keep], degree), x)
        residual = np.abs(y - fit)
        limit = max(tolerance, 3 * 1.4826 * float(np.median(residual[keep])))
        updated = residual <= limit
        if updated.sum() <= degree + 1 or np.array_equal(updated, keep):
            break
        keep = updated
    return np.where(residual <= limit, y, fit).round().astype(rows.dtype)


def mass_fraction(cam: np.ndarray, mask: np.ndarray) -> float | None:
    """Share of non-negative attribution mass inside `mask`; undefined for an all-zero map."""
    total = float(cam.sum())
    if total <= 0:
        return None
    return float(cam[mask].sum() / total)


def plausible_retina(retina: np.ndarray, content: np.ndarray) -> bool:
    """Flag masks whose coverage cannot be a retinal band, so summaries can exclude them."""
    fraction = retina.sum() / max(content.sum(), 1)
    return bool(0.05 <= fraction <= 0.6)
