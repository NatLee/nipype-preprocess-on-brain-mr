"""acpcdetect wrapper: input conversion, clear errors, and the node's outputs.

A small shell script stands in for the real acpcdetect binary, so these
tests run anywhere with a POSIX shell (Linux CI, the Docker image).
"""

import os
import stat
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import config
from node.acpc_detect.interface import ACPCDetectInterface
from node.acpc_detect.utils import ART_MODEL_FILES, acpc_detect, check_art_installation

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake acpcdetect is a shell script")

# Like the real tool: needs $ARTHOME's model, takes "-i <file>.nii" last, writes <file>_RAS.nii.
FAKE_ACPCDETECT = r"""#!/bin/sh
[ -f "$ARTHOME/T1acpc.mdl" ] || { echo "I cannot open the model file" >&2; exit 1; }
for last; do :; done
[ -n "$FAIL" ] && { echo "landmark detection failed" >&2; exit 3; }
cp "$last" "${last%.nii}_RAS.nii"
echo fake > "${last%.nii}_ACPC_sagittal.png"
"""


@pytest.fixture
def fake_art(tmp_path, monkeypatch):
    home = tmp_path / "art"
    (home / "bin").mkdir(parents=True)
    for name in ART_MODEL_FILES:
        (home / name).write_text("model")
    binary = home / "bin" / "acpcdetect"
    binary.write_text(FAKE_ACPCDETECT)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(config, "ART_HOME", home)
    monkeypatch.setattr(config, "ACPCDETECT_BIN", binary)
    return home


def save_scan(phantom, path: Path) -> Path:
    nib.save(nib.Nifti1Image(phantom.image.astype(np.int16), phantom.affine), path)
    return path


def test_missing_model_file_is_reported_before_running(fake_art, phantom, tmp_path):
    (fake_art / "PILbrain.nii").unlink()
    with pytest.raises(FileNotFoundError, match="PILbrain.nii"):
        check_art_installation()
    with pytest.raises(FileNotFoundError, match="ARTHOME"):
        acpc_detect(save_scan(phantom, tmp_path / "s.nii"), tmp_path / "out")


def test_tool_failure_raises_with_its_message(fake_art, phantom, tmp_path, monkeypatch):
    monkeypatch.setenv("FAIL", "1")
    with pytest.raises(RuntimeError, match="exit code 3.*landmark detection failed"):
        acpc_detect(save_scan(phantom, tmp_path / "s.nii"), tmp_path / "out")


def test_node_accepts_nii_gz_and_reports_outputs(fake_art, phantom, tmp_path):
    scan = save_scan(phantom, tmp_path / "sub-07_T1w.nii.gz")
    out = tmp_path / "out" / "sub-07_T1w"
    out.mkdir(parents=True)
    result = ACPCDetectInterface(input_file=str(scan), output_folder=str(out)).run()

    acpc = out / "acpc"
    assert Path(result.outputs.output_file) == acpc / "sub-07_T1w_RAS.nii"
    assert Path(result.outputs.output_png_file).stat().st_size > 0
    assert Path(result.outputs.landmarks_png_file).name == "sub-07_T1w_ACPC_sagittal.png"
    assert (acpc / "sub-07_T1w.nii").exists()  # the converted input is kept, as before
    np.testing.assert_array_equal(
        np.asanyarray(nib.load(result.outputs.output_file).dataobj), phantom.image.astype(np.int16)
    )


def test_arthome_is_passed_to_the_tool_not_set_globally(fake_art, phantom, tmp_path, monkeypatch):
    monkeypatch.delenv("ARTHOME", raising=False)
    acpc_detect(save_scan(phantom, tmp_path / "s.nii"), tmp_path / "out")
    assert "ARTHOME" not in os.environ
