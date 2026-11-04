"""The whole workflow on a real scan, at 2 mm (8x fewer voxels than the default 1 mm template).

Needs FSL, ANTs and acpcdetect, so it only runs inside the Docker image:

    python fetch_demo_data.py --out ./data --subjects 1     # or any T1w scan in ./data
    docker build --target test -t nipype-preprocess-on-brain-mr:test .
    docker run --rm -v "$PWD/data:/data:ro" nipype-preprocess-on-brain-mr:test python -m pytest -m neuro /tests
"""

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

import config
import workflow
from utils.paths import is_nifti, nii_stem

DATA_DIR = Path(os.environ.get("E2E_DATA_DIR", str(config.DEFAULT_DATA_DIR)))
# Set E2E_OUTPUT_DIR to keep the results (CI uploads them); otherwise a temporary folder is used.
OUTPUT_DIR = os.environ.get("E2E_OUTPUT_DIR")
TOOLS = ("flirt", "bet", "fast", "fslreorient2std", "N4BiasFieldCorrection", "Atropos")

pytestmark = [
    pytest.mark.neuro,
    pytest.mark.skipif(
        not os.environ.get("FSLDIR") or any(shutil.which(t) is None for t in TOOLS) or not config.ACPCDETECT_BIN.exists(),
        reason="FSL, ANTs or acpcdetect missing (run inside the Docker image)",
    ),
    pytest.mark.skipif(
        not DATA_DIR.is_dir() or not any(is_nifti(p) for p in DATA_DIR.iterdir()),
        reason=f"no scan in {DATA_DIR}",
    ),
]

# Every file the workflow wrote before the refactor, relative to the subject folder ({s} = scan name).
ORIGINAL_OUTPUTS = [
    "acpc/{s}_RAS.nii", "acpc/{s}_RAS.png", "acpc/{s}_FSL.mat", "acpc/{s}_ACPC.txt",
    "zip/{s}_RAS.nii.gz", "orient2std/{s}_RAS.nii.gz", "registration/{s}_RAS.nii.gz",
    "skull_stripping/{s}_RAS.nii.gz", "bias_correction/{s}_RAS.nii.gz", "enhancement/{s}_RAS.nii.gz",
    "segmentation/{s}_RAS.nii_gm_labels.nii.gz", "segmentation/{s}_RAS.nii_wm_segmented.nii.gz",
    "segmentation/{s}_RAS.nii_csf_segmented.nii.gz",
    "draw_segmentation/GM@map.png", "draw_segmentation/WM@map.png", "draw_segmentation/CSF@map.png",
    "final_output/{s}_RAS.nii", "final_output/{s}_RAS.png", "final_output/GM_Segmentation.png",
    "final_output/WM_Segmentation.png", "final_output/CSF_Segmentation.png",
]
NEW_OUTPUTS = [
    "acpc/{s}_ACPC_sagittal.png", "registration/{s}_RAS.mat", "skull_stripping/{s}_RAS_mask.nii.gz",
    "bias_correction/{s}_RAS_bias.nii.gz", "segmentation/{s}_RAS_tissue_labels.nii.gz",
    "segmentation_fast/{s}_RAS_tissue_labels.nii.gz", "segmentation_atropos/{s}_RAS_tissue_labels.nii.gz",
    "analysis/volumetrics.csv", "analysis/agreement.csv", "analysis/quality.json",
    "render3d/brain_3d.gif", "render3d/brain_3d.html", "report/index.html",
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    scan = sorted(p for p in DATA_DIR.iterdir() if is_nifti(p))[0]
    data = tmp_path_factory.mktemp("data")
    shutil.copy(scan, data / scan.name)
    output = Path(OUTPUT_DIR) if OUTPUT_DIR else tmp_path_factory.mktemp("output")
    reference = config.fsl_standard_dir() / "MNI152_T1_2mm.nii.gz"
    code = workflow.main(["--data-dir", str(data), "--output-dir", str(output), "--reference", str(reference)])
    return code, output, nii_stem(scan)


def test_workflow_succeeds(run):
    code, output, _ = run
    assert code == 0
    assert not list((output / "_work").glob("crash/*"))


@pytest.mark.parametrize("pattern", ORIGINAL_OUTPUTS + NEW_OUTPUTS)
def test_every_output_file_is_written(run, pattern):
    _, output, scan = run
    path = output / scan / pattern.format(s=scan)
    assert path.is_file() and path.stat().st_size > 0, path


def test_run_level_outputs(run):
    _, output, _ = run
    for name in ("graph.png", "graph_detailed.png", "index.html", "cohort/volumetrics.csv"):
        assert (output / name).stat().st_size > 0, name


def test_volumes_are_plausible_for_an_adult_brain(run):
    _, output, scan = run
    volumes = pd.read_csv(output / scan / "analysis" / "volumetrics.csv").set_index("method")
    assert list(volumes.index) == ["kmeans", "fast", "atropos"]
    for method, row in volumes.iterrows():
        assert 1000 < row["native_tiv_ml"] < 2200, method
        assert 0.3 < row["gm_fraction"] < 0.55, method
        assert 0.2 < row["wm_fraction"] < 0.45, method


def test_methods_agree_and_n4_helps(run):
    _, output, scan = run
    agreement = pd.read_csv(output / scan / "analysis" / "agreement.csv")
    fast_atropos = agreement[(agreement.method_a == "fast") & (agreement.method_b == "atropos")]
    assert (fast_atropos["dice"] > 0.8).all()
    quality = json.loads((output / scan / "analysis" / "quality.json").read_text())
    assert quality["after_n4"]["cjv"] < quality["before_n4"]["cjv"]


def test_kmeans_shares_follow_fast_not_thirds(run):
    """Regression (user-approved change): K-means used to split every brain into equal thirds.

    It clustered the histogram-equalised image, which put CSF at about 33% on the
    demo scans while FAST found about 21%. It now clusters the corrected brain.
    """
    _, output, scan = run
    volumes = pd.read_csv(output / scan / "analysis" / "volumetrics.csv").set_index("method")
    assert volumes.loc["kmeans", "csf_fraction"] < 0.3
    assert abs(volumes.loc["kmeans", "csf_fraction"] - volumes.loc["fast", "csf_fraction"]) < 0.06
    agreement = pd.read_csv(output / scan / "analysis" / "agreement.csv")
    kmeans_fast = agreement[(agreement.method_a == "kmeans") & (agreement.method_b == "fast")]
    assert (kmeans_fast["dice"] > 0.85).all()
