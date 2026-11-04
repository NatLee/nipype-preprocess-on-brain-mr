"""The nipype nodes that run pure Python, executed for real on phantoms."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from node.analysis.interface import AnalysisInterface
from node.cohort_summary.interface import CohortSummaryInterface
from node.draw_segmentation.interface import DrawSegmentationInterface
from node.enhancement.interface import EnhancementInterface
from node.enhancement.utils import equalize_hist, rescale_intensity
from node.final_output.interface import OrganizeFinalOutputInterface
from node.qc_report.interface import QCReportInterface
from node.render_3d.interface import Render3DInterface
from node.segmentation.interface import SegmentationInterface
from node.segmentation.utils import kmeans_cluster, tissue_labels
from node.zip.interface import ZipOutputInterface
from phantom import make_phantom

REGISTRATION_SCALE = 1.2**3  # the fake FLIRT matrix below scales every axis by 1.2


@pytest.fixture
def subject(tmp_path):
    """A subject folder holding the files the later steps read, built from phantoms."""
    folder = tmp_path / "output" / "sub-xx"
    folder.mkdir(parents=True)
    flat = make_phantom()
    biased = make_phantom(bias=0.2)
    files = {}
    for name, data in {
        "registered": flat.image,
        "before_n4": biased.image,
        "after_n4": flat.image,
        "bias": np.where(flat.labels > 0, biased.image / np.maximum(flat.image, 1), 1.0),
        "mask": (flat.labels > 0).astype(np.uint8),
        "labels": flat.labels,
    }.items():
        files[name] = folder / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.float32), flat.affine), files[name])
    files["matrix"] = folder / "reg.mat"
    np.savetxt(files["matrix"], np.diag([1.2, 1.2, 1.2, 1.0]))
    files["folder"] = folder
    return files


def test_enhancement_and_kmeans_nodes_keep_their_file_names(subject):
    folder = subject["folder"]
    enhanced = EnhancementInterface(input_file=str(subject["after_n4"]), output_folder=str(folder)).run()
    assert Path(enhanced.outputs.output_file) == folder / "enhancement" / "after_n4.nii.gz"

    # K-means reads the corrected brain itself (as in the workflow), not the equalised view.
    seg = SegmentationInterface(input_file=str(subject["after_n4"]), output_folder=str(folder)).run()
    for tissue in ("gm", "wm", "csf"):
        labels_file = Path(getattr(seg.outputs, f"{tissue}_labels_output_file"))
        assert labels_file.name == f"after_n4.nii_{tissue}_labels.nii.gz"  # the pre-refactor names
        assert Path(getattr(seg.outputs, f"{tissue}_segmented_output_file")).exists()
        # The shared map output the downstream steps read is K-means' original weighted image.
        assert getattr(seg.outputs, f"{tissue}_map_file") == getattr(seg.outputs, f"{tissue}_segmented_output_file")
    labels = nib.load(seg.outputs.tissue_labels_file).get_fdata()
    truth = nib.load(subject["labels"]).get_fdata()
    assert (labels == truth).mean() > 0.97


def tissue_shares(labels: np.ndarray) -> np.ndarray:
    brain = labels > 0
    return np.array([(labels == value).sum() / brain.sum() for value in (1, 2, 3)])


def test_kmeans_tissue_shares_follow_the_anatomy_not_thirds(tmp_path):
    """Regression (user-approved change): K-means used to cluster the histogram-equalised image.

    Equalisation flattens the histogram, so the three clusters came out the same
    size whatever the brain looked like (about 33% each on the demo scans). On
    the corrected intensities the shares follow the real tissue volumes.
    """
    lopsided = make_phantom(radii=(0.78, 0.74, 0.45))  # about 15% CSF, 66% GM, 19% WM
    truth = tissue_shares(lopsided.labels)
    image_file, _ = lopsided.save(tmp_path)
    result = SegmentationInterface(input_file=str(image_file), output_folder=str(tmp_path / "out")).run()
    found = tissue_shares(np.asanyarray(nib.load(result.outputs.tissue_labels_file).dataobj))
    np.testing.assert_allclose(found, truth, atol=0.03)

    # The old input pulls the shares towards equal thirds, far from the truth.
    equalised = equalize_hist(rescale_intensity(lopsided.image))
    old = tissue_shares(tissue_labels(kmeans_cluster(equalised, 3), equalised))
    assert np.abs(old - 1 / 3).max() < np.abs(truth - 1 / 3).max() / 3
    assert np.abs(old - truth).max() > 0.2


def test_zip_node_gzips_the_input(subject, tmp_path):
    plain = tmp_path / "scan_RAS.nii"
    nib.save(nib.load(subject["registered"]), plain)
    result = ZipOutputInterface(input_file=str(plain), output_folder=str(subject["folder"])).run()
    assert Path(result.outputs.output_file).name == "scan_RAS.nii.gz"
    np.testing.assert_array_equal(nib.load(result.outputs.output_file).get_fdata(), nib.load(plain).get_fdata())


def test_draw_segmentation_still_accepts_the_old_input_name(subject):
    common = {"segmented_input_file": str(subject["after_n4"]), "output_folder": str(subject["folder"])}
    new = DrawSegmentationInterface(background_file=str(subject["registered"]), title="GM@map", **common).run()
    assert Path(new.outputs.output_file).name == "GM@map.png"
    with pytest.warns(UserWarning, match="background_file"):
        old = DrawSegmentationInterface(acpc_input_file=str(subject["registered"]), title="WM@map", **common)
    assert old.inputs.background_file == str(subject["registered"])
    assert Path(old.run().outputs.output_file).exists()
    assert (subject["folder"] / "draw_segmentation" / "GM@map.png").exists()  # the sibling map survives


def run_analysis(subject, primary="fast"):
    return AnalysisInterface(
        output_folder=str(subject["folder"]),
        primary_method=primary,
        kmeans_labels=str(subject["labels"]),
        fast_labels=str(subject["labels"]),
        matrix_file=str(subject["matrix"]),
        before_n4=str(subject["before_n4"]),
        after_n4=str(subject["after_n4"]),
        bias_field=str(subject["bias"]),
        brain_mask=str(subject["mask"]),
    ).run()


def test_analysis_writes_volumes_agreement_and_quality(subject):
    result = run_analysis(subject)
    volumes = pd.read_csv(result.outputs.volumetrics_file)
    assert volumes["method"].tolist() == ["kmeans", "fast"]
    assert volumes["registration_scale"].iloc[0] == pytest.approx(REGISTRATION_SCALE)
    assert volumes["native_tiv_ml"].iloc[0] == pytest.approx(volumes["tiv_ml"].iloc[0] / REGISTRATION_SCALE, rel=1e-3)
    assert set(pd.read_csv(result.outputs.agreement_file)["dice"]) == {1.0}
    quality = json.loads(Path(result.outputs.quality_file).read_text())
    assert quality["primary_method"] == "fast"
    assert quality["after_n4"]["cjv"] < quality["before_n4"]["cjv"]


def test_report_render_and_cohort_pages(subject):
    folder = subject["folder"]
    analysis = run_analysis(subject)
    render = Render3DInterface(
        output_folder=str(folder), label_file=str(subject["labels"]), n_frames=4, size_px=160
    ).run()
    assert Path(render.outputs.gif_file).stat().st_size > 0
    assert (Path(render.outputs.html_file).parent / "plotly.min.js").exists()

    report = QCReportInterface(
        output_folder=str(folder),
        input_file=str(subject["registered"]),
        acpc_file=str(subject["registered"]),
        acpc_png_file=str(render.outputs.gif_file),
        registered_file=str(subject["registered"]),
        brain_file=str(subject["before_n4"]),
        brain_mask=str(subject["mask"]),
        corrected_file=str(subject["after_n4"]),
        bias_field=str(subject["bias"]),
        enhanced_file=str(subject["after_n4"]),
        kmeans_labels=str(subject["labels"]),
        fast_labels=str(subject["labels"]),
        volumetrics_file=analysis.outputs.volumetrics_file,
        agreement_file=analysis.outputs.agreement_file,
        quality_file=analysis.outputs.quality_file,
        gif_file=render.outputs.gif_file,
        html_file=render.outputs.html_file,
    ).run()
    page = Path(report.outputs.report_file)
    html = page.read_text()
    assert page == folder / "report" / "index.html"
    assert "sub-xx" in html and "K-means vs FSL FAST" in html
    for figure in ("pipeline", "segmentation", "bias_correction", "volumes", "agreement"):
        assert (folder / "report" / "figures" / f"{figure}.png").exists()
    assert 'src="../render3d/brain_3d.gif"' in html

    cohort = CohortSummaryInterface(
        output_root=str(folder.parent),
        primary_method="fast",
        volumetrics_files=[analysis.outputs.volumetrics_file],
        agreement_files=[analysis.outputs.agreement_file],
        quality_files=[analysis.outputs.quality_file],
        report_files=[str(page)],
        gif_files=[render.outputs.gif_file],
    ).run()
    index = Path(cohort.outputs.index_file).read_text()
    assert 'href="sub-xx/report/index.html"' in index
    assert len(pd.read_csv(cohort.outputs.volumetrics_file)) == 2


def test_qc_numbers_follow_the_chosen_primary_method(subject):
    assert json.loads(Path(run_analysis(subject, "kmeans").outputs.quality_file).read_text())["primary_method"] == "kmeans"
    with pytest.raises(ValueError, match="primary segmentation \\(atropos\\)"):
        run_analysis(subject, "atropos")  # no Atropos labels were given


def test_tool_maps_are_matched_to_tissues_by_brightness(phantom, tmp_path):
    """FAST/Atropos number their classes arbitrarily; maps must follow the same order as the labels."""
    from node.tissue_output import tissue_outputs

    raw = np.choose(phantom.labels, [0, 3, 1, 2])  # tool class 3 = CSF, 1 = GM, 2 = WM
    raw_file = tmp_path / "raw.nii.gz"
    nib.save(nib.Nifti1Image(raw.astype(np.int16), phantom.affine), raw_file)
    image_file = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(phantom.image, phantom.affine), image_file)
    class_maps = [tmp_path / f"class_{c}.nii.gz" for c in (1, 2, 3)]  # map of class value c
    outputs = tissue_outputs(raw_file, image_file, class_maps, tmp_path / "labels.nii.gz")
    assert outputs["csf_map_file"].endswith("class_3.nii.gz")
    assert outputs["gm_map_file"].endswith("class_1.nii.gz")
    assert outputs["wm_map_file"].endswith("class_2.nii.gz")
    np.testing.assert_array_equal(np.asanyarray(nib.load(outputs["tissue_labels_file"]).dataobj), phantom.labels)


def test_final_output_copies_the_usual_files_and_optional_extras(subject, tmp_path):
    pngs = {}
    for name in ("acpc", "gm", "wm", "csf"):
        pngs[name] = tmp_path / f"{name}.png"
        pngs[name].write_bytes(b"png")
    result = OrganizeFinalOutputInterface(
        output_folder=str(subject["folder"]),
        acpc_output_file=str(subject["registered"]),
        acpc_output_png_file=str(pngs["acpc"]),
        gm_png_file=str(pngs["gm"]),
        wm_png_file=str(pngs["wm"]),
        csf_png_file=str(pngs["csf"]),
        volumetrics_file=str(subject["matrix"]),
    ).run()
    names = sorted(p.name for p in Path(result.outputs.output_folder).iterdir())
    assert names == sorted(
        ["registered.nii.gz", "acpc.png", "GM_Segmentation.png", "WM_Segmentation.png", "CSF_Segmentation.png", "reg.mat"]
    )
