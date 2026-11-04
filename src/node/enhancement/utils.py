"""Intensity clean-up applied before K-means. Background voxels (value 0) are left untouched."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import medfilt


def denoise(volume: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter; `kernel_size=1` leaves the values unchanged."""
    return medfilt(volume, kernel_size)


def rescale_intensity(
    volume: np.ndarray, percentils: Sequence[float] = (0.5, 99.5), bins_num: int = 256
) -> np.ndarray:
    """Stretch the brain's intensities between two percentiles.

    With `bins_num > 0` the result is integer levels 1..bins_num-1; with
    `bins_num == 0` it is a float scale where the percentiles map to 0 and 1.
    """
    obj_volume = volume[volume > 0]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / np.float32(max_value - min_value)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume = np.clip(obj_volume, 1, bins_num - 1)

    rescaled = volume.astype(obj_volume.dtype)
    rescaled[volume > 0] = obj_volume
    return rescaled


def equalize_hist(volume: np.ndarray, bins_num: int = 256) -> np.ndarray:
    """Histogram equalisation of the non-zero voxels."""
    obj_volume = volume[volume > 0]
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    equalized = volume.copy()
    equalized[volume > 0] = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    return equalized
