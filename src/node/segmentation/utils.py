"""K-means tissue clustering on voxel intensities (the pipeline's original segmentation)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from analysis.tissue import order_labels_by_intensity


def extract_features(data: np.ndarray) -> np.ndarray:
    """One row per non-zero voxel: [intensity, x, y, z]."""
    x_idx, y_idx, z_idx = np.nonzero(data > 0)
    return np.column_stack([data[x_idx, y_idx, z_idx], x_idx, y_idx, z_idx]).astype(np.float64)


def kmeans_cluster(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster the non-zero voxels by intensity; returns labels 1..n_clusters (0 = background).

    Cluster numbers are arbitrary (as returned by scikit-learn); use
    `get_target_labels` or `tissue_labels` to know which cluster is which tissue.
    """
    foreground = data > 0
    intensities = data[foreground].reshape(-1, 1).astype(np.float64)
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        verbose=0,
        random_state=42,
        max_iter=100,
        tol=1e-6,
    ).fit(intensities)

    labels = np.zeros(data.shape)
    labels[foreground] = kmeans_model.labels_ + 1
    return labels


def get_target_labels(labels: np.ndarray, data: np.ndarray) -> list[int]:
    """Cluster numbers of [GM, WM, CSF]: middle, brightest and darkest mean intensity."""
    clusters = [value for value in np.unique(labels) if value != 0]
    mean_intensities = [float(np.mean(data[labels == value])) for value in clusters]
    order = np.argsort(mean_intensities)
    csf, gm, wm = (int(clusters[i]) for i in (order[0], order[len(order) // 2], order[-1]))
    return [gm, wm, csf]


def tissue_labels(labels: np.ndarray, data: np.ndarray) -> np.ndarray:
    """K-means clusters renumbered to the shared tissue coding (1 CSF, 2 GM, 3 WM)."""
    return order_labels_by_intensity(labels, data)


def segment(labels: np.ndarray, data: np.ndarray, target: int) -> np.ndarray:
    """The image with the target tissue at full intensity and every other voxel dimmed to a third."""
    mask = np.where(labels == target, 1.0, 0.333).astype(np.float32)
    return np.multiply(data.astype(np.float32), mask)
