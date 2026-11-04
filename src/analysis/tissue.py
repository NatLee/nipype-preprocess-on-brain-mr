"""Tissue label conventions and agreement between segmentations.

Every segmentation the pipeline writes as a "tissue label map" uses the same
coding, the one FSL FAST uses for T1-weighted scans: 0 background, then the
classes from darkest to brightest - 1 CSF, 2 grey matter, 3 white matter.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np

# Ordered by brightness on a T1-weighted scan; label value = position + 1.
TISSUES = ("csf", "gm", "wm")
TISSUE_NAMES = {"csf": "CSF", "gm": "Grey matter", "wm": "White matter"}
TISSUE_LABEL = {tissue: index + 1 for index, tissue in enumerate(TISSUES)}


def classes_by_intensity(labels: np.ndarray, image: np.ndarray) -> list[int]:
    """The non-zero class values of `labels`, from the darkest to the brightest in `image`."""
    labels = np.asarray(labels)
    classes = [int(value) for value in np.unique(labels) if value != 0]
    means = [float(np.mean(image[labels == value])) for value in classes]
    return [classes[index] for index in np.argsort(means)]


def order_labels_by_intensity(labels: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Renumber the non-zero classes of `labels` 1..k from darkest to brightest in `image`.

    Clustering methods number their classes arbitrarily; this makes the
    numbering mean the same tissue whatever method produced it.
    """
    labels = np.asarray(labels)
    ordered = np.zeros(labels.shape, dtype=np.uint8)
    for new_label, value in enumerate(classes_by_intensity(labels, image), start=1):
        ordered[labels == value] = new_label
    return ordered


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two boolean masks: 1 means identical, 0 means no overlap."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return 1.0
    return 2.0 * int(np.logical_and(a, b).sum()) / total


def agreement(label_maps: Mapping[str, np.ndarray]) -> list[dict]:
    """Dice per tissue for every pair of segmentation methods."""
    rows = []
    for (name_a, map_a), (name_b, map_b) in combinations(label_maps.items(), 2):
        if map_a.shape != map_b.shape:
            raise ValueError(f"{name_a} and {name_b} label maps have different shapes")
        for tissue in TISSUES:
            label = TISSUE_LABEL[tissue]
            rows.append(
                {
                    "method_a": name_a,
                    "method_b": name_b,
                    "tissue": tissue,
                    "dice": round(dice(map_a == label, map_b == label), 4),
                }
            )
    return rows

# Segmentation methods the pipeline can run, in display order.
METHOD_NAMES = {"kmeans": "K-means", "fast": "FSL FAST", "atropos": "ANTs Atropos"}
# The primary segmentation feeds the probability maps, final_output/, the QC
# numbers and the 3D surface (python workflow.py --segmentation). FAST by
# default: the user chose it on 2026-11-04 over K-means, the original method,
# because K-means has no spatial or intensity model and moves the grey/white
# border (see README, "Why FSL FAST is the default").
DEFAULT_PRIMARY_METHOD = "fast"
