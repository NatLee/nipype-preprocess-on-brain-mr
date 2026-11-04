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
from node.final_output.interface import OrganizeFinalOutputInterface
from node.qc_report.interface import QCReportInterface
from node.render_3d.interface import Render3DInterface
from node.segmentation.interface import SegmentationInterface
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

    seg = SegmentationInterface(input_file=enhanced.outputs.output_file, output_folder=str(folder)).run()
    for tissue in ("gm", "wm", "csf"):
        labels_file = Path(getattr(seg.outputs, f"{tissue}_labels_output_file"))
        assert labels_file.name == f"after_n4.nii_{tissue}_labels.nii.gz"  # the pre-refactor names
        assert Path(getattr(seg.outputs, f"{tissue}_segmented_output_file")).exists()
    labels = nib.load(seg.outputs.tissue_labels_output_file).get_fdata()
    truth = nib.load(subject["labels"]).get_fdata()
    # Histogram equalisation stretches the noise too, so a few boundary voxels flip class.
    assert (labels == truth).mean() > 0.9


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


def run_analysis(subject):
    return AnalysisInterface(
        output_folder=str(subject["folder"]),
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
    assert quality["tissue_masks_from"] == "fast"
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
        volumetrics_files=[analysis.outputs.volumetrics_file],
        agreement_files=[analysis.outputs.agreement_file],
        quality_files=[analysis.outputs.quality_file],
        report_files=[str(page)],
        gif_files=[render.outputs.gif_file],
    ).run()
    index = Path(cohort.outputs.index_file).read_text()
    assert 'href="sub-xx/report/index.html"' in index
    assert len(pd.read_csv(cohort.outputs.volumetrics_file)) == 2


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
