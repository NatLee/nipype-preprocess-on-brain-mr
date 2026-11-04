"""The rewritten enhancement and K-means code must give exactly what the 2024 code gave.

The functions prefixed `original_` are copied from the last release before the
refactor (commit 19faf14) and act as the reference.
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans

from node.enhancement.utils import denoise, equalize_hist, rescale_intensity
from node.segmentation.utils import extract_features, get_target_labels, kmeans_cluster, segment, tissue_labels

# --------------------------------------------------- reference (commit 19faf14) ---


def original_rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])
    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1
    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def original_equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]
    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def original_kmeans_cluster(data, n_clusters):
    x_idx, y_idx, z_idx = np.where(data > 0)
    features = np.array([[data[x, y, z], x, y, z] for x, y, z in zip(x_idx, y_idx, z_idx)])
    intensities = features[..., 0].reshape((-1, 1))
    model = KMeans(n_clusters=n_clusters, init="k-means++", verbose=0, random_state=42, max_iter=100, tol=1e-6)
    model.fit(intensities)
    labels = np.zeros(data.shape)
    for label, feature in zip(model.labels_, features):
        labels[int(feature[1]), int(feature[2]), int(feature[3])] = label + 1
    return labels


def original_get_target_labels(labels, data):
    labels_set = np.unique(labels)
    mean_intensities = [np.mean(data[np.where(labels == label)]) for label in labels_set[1:]]
    return [
        mean_intensities.index(np.median(mean_intensities)) + 1,
        mean_intensities.index(np.max(mean_intensities)) + 1,
        mean_intensities.index(np.min(mean_intensities)) + 1,
    ]


def original_segment(labels, data, target):
    mask = np.copy(labels).astype(np.float32)
    mask[np.where(mask != target)] = 0.333
    mask[np.where(mask == target)] = 1.0
    return np.multiply(data.astype(np.float32), mask)


# ---------------------------------------------------------------------- tests ---


@pytest.mark.parametrize("bins_num", [256, 64, 0])
def test_rescale_intensity_matches_original(phantom, bins_num):
    expected = original_rescale_intensity(phantom.image.copy(), bins_num=bins_num)
    np.testing.assert_array_equal(rescale_intensity(phantom.image, bins_num=bins_num), expected)


def test_equalize_hist_matches_original_and_leaves_input_alone(phantom):
    rescaled = rescale_intensity(phantom.image)
    before = rescaled.copy()
    expected = original_equalize_hist(rescaled.copy())
    np.testing.assert_array_equal(equalize_hist(rescaled), expected)
    np.testing.assert_array_equal(rescaled, before)  # the old version modified its argument


def test_denoise_with_kernel_one_is_identity(phantom):
    np.testing.assert_array_equal(denoise(phantom.image, 1), phantom.image)


def test_kmeans_matches_original_loop_version():
    small = make_small_volume()
    np.testing.assert_array_equal(kmeans_cluster(small, 3), original_kmeans_cluster(small, 3))


def test_target_labels_and_segment_match_original():
    small = make_small_volume()
    labels = kmeans_cluster(small, 3)
    targets = get_target_labels(labels, small)
    assert targets == original_get_target_labels(labels, small)
    for target in targets:
        np.testing.assert_array_equal(segment(labels, small, target), original_segment(labels, small, target))


def test_extract_features_rows(phantom):
    features = extract_features(phantom.image)
    assert features.shape == (int((phantom.image > 0).sum()), 4)
    x, y, z = features[0, 1:].astype(int)
    assert features[0, 0] == phantom.image[x, y, z]


def test_tissue_labels_recover_the_phantom(phantom):
    from analysis.tissue import dice

    labels = tissue_labels(kmeans_cluster(phantom.image, 3), phantom.image)
    for value in (1, 2, 3):
        assert dice(labels == value, phantom.labels == value) > 0.95


def make_small_volume():
    """Phantom cropped small enough for the voxel-by-voxel reference to run quickly."""
    from phantom import make_phantom

    return make_phantom(shape=(20, 24, 20), voxel_mm=4.0).image
