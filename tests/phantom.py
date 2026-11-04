"""A synthetic "brain" for tests: nested ellipsoids with T1-like contrast.

White matter in the middle (brightest), a wavy grey-matter shell around it,
then a thin CSF shell (darkest); zero outside, like a skull-stripped scan.
The true label of every voxel is known, so segmentation and volume code can
be checked against exact answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

# T1-weighted ordering: CSF dark, GM mid, WM bright (arbitrary scanner units).
INTENSITY = {1: 300.0, 2: 600.0, 3: 900.0}


@dataclass
class Phantom:
    image: np.ndarray  # float32 intensities
    labels: np.ndarray  # uint8: 0 background, 1 CSF, 2 GM, 3 WM
    affine: np.ndarray

    @property
    def zooms(self) -> tuple[float, float, float]:
        return tuple(float(z) for z in np.abs(np.diag(self.affine)[:3]))

    def save(self, folder: Path, name: str = "phantom") -> tuple[Path, Path]:
        image_path = folder / f"{name}.nii.gz"
        labels_path = folder / f"{name}_labels.nii.gz"
        nib.save(nib.Nifti1Image(self.image, self.affine), str(image_path))
        nib.save(nib.Nifti1Image(self.labels, self.affine), str(labels_path))
        return image_path, labels_path


def make_phantom(
    shape: tuple[int, int, int] = (48, 56, 48),
    voxel_mm: float = 2.0,
    noise_sd: float = 25.0,
    bias: float = 0.0,
    seed: int = 0,
) -> Phantom:
    """Build the phantom. `bias` adds a linear intensity drift of +-bias across x (0.2 = +-20%)."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij"))
    radius = np.sqrt((grid**2).sum(axis=0))
    theta = np.arctan2(grid[1], grid[0])
    phi = np.arccos(np.clip(grid[2] / np.maximum(radius, 1e-6), -1, 1))
    folds = 1 + 0.08 * np.sin(6 * theta) * np.sin(5 * phi)  # wavy GM/WM boundary, like gyri

    labels = np.zeros(shape, dtype=np.uint8)
    labels[radius < 0.85] = 1
    labels[radius < 0.75] = 2
    labels[radius < 0.5 * folds] = 3

    image = np.zeros(shape, dtype=np.float32)
    for label, value in INTENSITY.items():
        image[labels == label] = value
    image += rng.normal(0, noise_sd, shape).astype(np.float32) * (labels > 0)
    if bias:
        image *= (1 + bias * grid[0]).astype(np.float32)
    image[labels == 0] = 0
    image = np.clip(image, 0, None)

    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    affine[:3, 3] = -np.array(shape) * voxel_mm / 2
    return Phantom(image=image, labels=labels, affine=affine)
