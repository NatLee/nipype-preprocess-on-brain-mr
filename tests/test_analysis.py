"""Tissue conventions, volumes and image-quality numbers on the synthetic phantom."""

import numpy as np
import pytest

from analysis.quality import bias_field_range, contrast_metrics
from analysis.tissue import (
    METHOD_NAMES,
    TISSUE_LABEL,
    TISSUES,
    agreement,
    dice,
    order_labels_by_intensity,
    reference_method,
)
from analysis.volumetrics import read_fsl_matrix, registration_scale, tissue_volumes, voxel_volume_ml
from phantom import make_phantom


def test_label_coding_follows_t1_brightness():
    assert TISSUES == ("csf", "gm", "wm")
    assert [TISSUE_LABEL[t] for t in TISSUES] == [1, 2, 3]


def test_order_labels_by_intensity_renumbers_dark_to_bright(phantom):
    scrambled = np.choose(phantom.labels, [0, 7, 2, 5])  # CSF->7, GM->2, WM->5
    np.testing.assert_array_equal(order_labels_by_intensity(scrambled, phantom.image), phantom.labels)


def test_dice_edge_cases():
    a = np.array([1, 1, 0, 0], bool)
    assert dice(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    assert dice(np.zeros(4, bool), np.zeros(4, bool)) == 1.0
    assert dice(a, np.array([1, 0, 0, 0], bool)) == pytest.approx(2 / 3)


def test_agreement_has_one_row_per_pair_and_tissue(phantom):
    rows = agreement({"kmeans": phantom.labels, "fast": phantom.labels, "atropos": phantom.labels})
    assert len(rows) == 3 * 3
    assert all(r["dice"] == 1.0 for r in rows)
    with pytest.raises(ValueError, match="different shapes"):
        agreement({"a": phantom.labels, "b": phantom.labels[:-1]})


def test_reference_method_prefers_fast_then_atropos_then_kmeans():
    assert reference_method(METHOD_NAMES) == "fast"
    assert reference_method({"kmeans", "atropos"}) == "atropos"
    assert reference_method({"kmeans"}) == "kmeans"
    with pytest.raises(ValueError):
        reference_method(set())


def test_voxel_volume_and_registration_scale(tmp_path):
    assert voxel_volume_ml((2.0, 2.0, 2.0)) == pytest.approx(0.008)
    matrix = np.diag([1.1, 1.1, 1.1, 1.0])
    matrix[:3, 3] = [5, -3, 2]  # translation does not change volume
    path = tmp_path / "x.mat"
    np.savetxt(path, matrix)
    assert registration_scale(read_fsl_matrix(path)) == pytest.approx(1.331)
    np.savetxt(path, np.eye(3))
    with pytest.raises(ValueError, match="4x4"):
        read_fsl_matrix(path)
    with pytest.raises(ValueError, match="singular"):
        registration_scale(np.zeros((4, 4)))


def test_tissue_volumes_counts_voxels_and_scales_to_native():
    labels = np.array([0, 1, 2, 2, 3, 3, 3, 3])  # 1 CSF, 2 GM, 4 WM voxels
    row = tissue_volumes(labels, voxel_ml=0.5, scale=2.0)
    assert (row["csf_ml"], row["gm_ml"], row["wm_ml"], row["tiv_ml"]) == (0.5, 1.0, 2.0, 3.5)
    assert row["wm_fraction"] == pytest.approx(4 / 7, abs=1e-4)
    assert row["native_tiv_ml"] == 1.75
    assert row["registration_scale"] == 2.0


def test_a_bias_field_inflates_cjv():
    flat = make_phantom(bias=0.0)
    biased = make_phantom(bias=0.3)
    assert contrast_metrics(biased.image, biased.labels)["cjv"] > contrast_metrics(flat.image, flat.labels)["cjv"]
    clean = contrast_metrics(flat.image, flat.labels)
    assert clean["cnr"] > 5 and clean["snr_wm"] > 20


def test_contrast_metrics_need_both_tissues(phantom):
    with pytest.raises(ValueError, match="grey-matter"):
        contrast_metrics(phantom.image, np.where(phantom.labels == 2, 0, phantom.labels))


def test_bias_field_range_of_a_flat_field_is_one(phantom):
    field = np.full(phantom.labels.shape, 3.0)
    assert bias_field_range(field, phantom.labels > 0) == {"bias_p01": 1.0, "bias_p99": 1.0}
