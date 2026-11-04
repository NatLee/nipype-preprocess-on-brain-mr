"""Tissue volumes from a label map.

The label maps live in template space (after the affine registration), so raw
voxel counts describe a brain that has been stretched to the template's size.
FLIRT's matrix says by how much: the absolute determinant of its 3x3 part is
the volume ratio template / native. Dividing by it gives an estimate of the
volumes in the scan's own space, which is what you would compare across people.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.tissue import TISSUE_LABEL, TISSUES

MM3_PER_ML = 1000.0


def voxel_volume_ml(zooms) -> float:
    """Volume of one voxel in millilitres from its edge lengths in millimetres."""
    return float(np.prod(np.asarray(zooms[:3], dtype=float))) / MM3_PER_ML


def read_fsl_matrix(path: str | Path) -> np.ndarray:
    """Read a 4x4 FSL/FLIRT affine matrix (plain text)."""
    matrix = np.loadtxt(str(path))
    if matrix.shape != (4, 4):
        raise ValueError(f"{path} is not a 4x4 affine matrix")
    return matrix


def registration_scale(matrix: np.ndarray) -> float:
    """Volume ratio template / native of an affine transform (|det| of its linear part)."""
    scale = abs(float(np.linalg.det(np.asarray(matrix, dtype=float)[:3, :3])))
    if scale == 0:
        raise ValueError("the registration matrix is singular")
    return scale


def tissue_volumes(labels: np.ndarray, voxel_ml: float, scale: float = 1.0) -> dict[str, float]:
    """Volumes (mL) of CSF, GM and WM, their sum (TIV) and each tissue's share of it.

    `<tissue>_ml` are in the label map's space; `native_<tissue>_ml` divide by
    `scale` (see `registration_scale`). Fractions are the same in both spaces.
    """
    labels = np.asarray(labels)
    row: dict[str, float] = {}
    for tissue in TISSUES:
        row[f"{tissue}_ml"] = float(np.count_nonzero(labels == TISSUE_LABEL[tissue])) * voxel_ml
    row["tiv_ml"] = sum(row[f"{tissue}_ml"] for tissue in TISSUES)
    for tissue in TISSUES:
        row[f"{tissue}_fraction"] = row[f"{tissue}_ml"] / row["tiv_ml"] if row["tiv_ml"] else 0.0
    for key in [f"{tissue}_ml" for tissue in TISSUES] + ["tiv_ml"]:
        row[f"native_{key}"] = row[key] / scale
    row["registration_scale"] = scale
    return {key: round(value, 4) for key, value in row.items()}
