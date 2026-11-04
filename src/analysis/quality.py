"""Image-quality numbers used to show what N4 bias correction changed.

All of them compare grey and white matter intensities inside a tissue label map:
- CJV (coefficient of joint variation, as in MRIQC): (sd_GM + sd_WM) / |mean_WM - mean_GM|.
  Lower is better; a bias field widens both peaks and pushes it up.
- CNR (contrast-to-noise ratio): |mean_WM - mean_GM| / sqrt(sd_GM^2 + sd_WM^2). Higher is better.
- SNR of white matter: mean_WM / sd_WM. Higher is better.
"""

from __future__ import annotations

import numpy as np

from analysis.tissue import TISSUE_LABEL


def _tissue_values(image: np.ndarray, labels: np.ndarray, tissue: str) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)[np.asarray(labels) == TISSUE_LABEL[tissue]]


def contrast_metrics(image: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """CJV, CNR and white-matter SNR of `image` inside the GM and WM of `labels`."""
    gm = _tissue_values(image, labels, "gm")
    wm = _tissue_values(image, labels, "wm")
    if gm.size < 2 or wm.size < 2:
        raise ValueError("need at least two grey-matter and two white-matter voxels")
    mean_gap = abs(wm.mean() - gm.mean())
    spread = np.sqrt(gm.var() + wm.var())
    return {
        "cjv": round(float((gm.std() + wm.std()) / mean_gap), 4) if mean_gap else float("inf"),
        "cnr": round(float(mean_gap / spread), 4) if spread else float("inf"),
        "snr_wm": round(float(wm.mean() / wm.std()), 4) if wm.std() else float("inf"),
    }


def bias_field_range(bias: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """How far the N4 field strays from flat inside the brain, as percentiles of the field / its median.

    A field of 1.10 at the 99th percentile means the brightest corner of the
    brain was about 10% too bright before correction.
    """
    values = np.asarray(bias, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    relative = values / np.median(values)
    low, high = np.percentile(relative, [1, 99])
    return {"bias_p01": round(float(low), 4), "bias_p99": round(float(high), 4)}
