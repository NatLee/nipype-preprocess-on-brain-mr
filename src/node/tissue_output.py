"""Shared output of the extra segmentation methods (FSL FAST, ANTs Atropos)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
from nipype.interfaces.base import File, TraitedSpec

from analysis.tissue import order_labels_by_intensity
from utils.load_nii import load_nii
from utils.save_nii import save_nii


class TissueLabelsOutputSpec(TraitedSpec):
    tissue_labels_file = File(exists=True, desc="Tissue label map: 0 background, 1 CSF, 2 GM, 3 WM")
    raw_labels_file = File(exists=True, desc="The label map exactly as the tool wrote it")


def save_tissue_labels(raw_labels_file: str | Path, intensity_file: str | Path, output_file: Path) -> Path:
    """Write the tool's labels renumbered to the shared tissue coding (darkest class = CSF)."""
    labels = nib.load(str(raw_labels_file)).get_fdata().round().astype(int)
    image, affine = load_nii(str(intensity_file))
    save_nii(order_labels_by_intensity(labels, image), str(output_file), affine)
    return output_file
