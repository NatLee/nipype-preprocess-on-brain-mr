"""What every segmentation step hands to the rest of the workflow.

Whichever method is chosen as the primary segmentation, the downstream steps
read the same fields: a tissue label map in the shared coding and one map per
tissue (a probability for FAST and Atropos, the weighted image for K-means).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
from nipype.interfaces.base import File, TraitedSpec

from analysis.tissue import TISSUES, classes_by_intensity
from utils.load_nii import load_nii
from utils.save_nii import save_nii


class TissueSegmentationOutputSpec(TraitedSpec):
    tissue_labels_file = File(exists=True, desc="Tissue label map: 0 background, 1 CSF, 2 GM, 3 WM")
    csf_map_file = File(exists=True, desc="Per-voxel CSF map (probability, or K-means' weighted image)")
    gm_map_file = File(exists=True, desc="Per-voxel grey-matter map")
    wm_map_file = File(exists=True, desc="Per-voxel white-matter map")


class ToolSegmentationOutputSpec(TissueSegmentationOutputSpec):
    raw_labels_file = File(exists=True, desc="The label map exactly as the tool wrote it")


def tissue_outputs(
    raw_labels_file: str | Path,
    intensity_file: str | Path,
    class_maps: Sequence[str | Path],
    output_file: Path,
) -> dict[str, str]:
    """Tissue labels and per-tissue maps from a tool that numbers its classes 1..k.

    `class_maps[c - 1]` is the tool's map for class value c. Classes are matched
    to tissues by brightness (darkest = CSF), so the tool's own numbering does
    not matter.
    """
    raw = nib.load(str(raw_labels_file)).get_fdata().round().astype(int)
    image, affine = load_nii(str(intensity_file))
    order = classes_by_intensity(raw, image)
    labels = np.zeros(raw.shape, dtype=np.uint8)
    for new_label, value in enumerate(order, start=1):
        labels[raw == value] = new_label
    save_nii(labels, str(output_file), affine)
    outputs = {"tissue_labels_file": Path(output_file).as_posix()}
    for tissue, value in zip(TISSUES, order):
        outputs[f"{tissue}_map_file"] = Path(class_maps[value - 1]).as_posix()
    return outputs
