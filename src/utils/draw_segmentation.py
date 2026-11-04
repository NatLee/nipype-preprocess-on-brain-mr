"""Draw a segmentation map over an anatomical image with nilearn (axial slices)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from nilearn import image  # noqa: E402
from nilearn.plotting import plot_stat_map  # noqa: E402

# Axial slices in template millimetres: z = -50, -30, ..., 30.
CUT_COORDS = list(range(-50, 50, 20))


def normalize(data: np.ndarray) -> np.ndarray:
    """Scale to 0..1 (a constant image becomes all zeros instead of dividing by zero)."""
    low, high = float(np.min(data)), float(np.max(data))
    if high == low:
        return np.zeros_like(data, dtype=np.float64)
    return (data - low) / (high - low)


def draw_segmentation(
    input_nii_path: str,
    input_seg_nii_path: str,
    output_png_path: str,
    threshold: float = 0.6,
    title: str = "Prob. map",
) -> None:
    """Plot the segmentation (scaled to 0..1) over `input_nii_path`, which must share its space."""
    input_nii = image.load_img(input_nii_path)
    input_seg_nii = image.load_img(input_seg_nii_path)
    normalized_seg_nii = nib.Nifti1Image(
        normalize(input_seg_nii.get_fdata()), input_seg_nii.affine, input_seg_nii.header
    )
    plot_stat_map(
        stat_map_img=normalized_seg_nii,
        title=title,
        cmap=plt.cm.magma,
        threshold=threshold,
        bg_img=input_nii,  # bg_img is the background image on top of which we plot the stat_map
        display_mode="z",
        cut_coords=CUT_COORDS,
        dim=-1,
        output_file=output_png_path,
    )
    plt.close("all")
