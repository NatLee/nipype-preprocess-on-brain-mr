import json
import re

import nibabel as nib
import numpy as np
import pytest

import fetch_demo_data
from node.acpc_detect.utils import to_int16, write_acpcdetect_input
from utils.paths import clear_dir, is_nifti, nii_stem, prepare_step_dir


@pytest.mark.parametrize(
    "name, stem",
    [("sub-01_T1w.nii.gz", "sub-01_T1w"), ("scan.nii", "scan"), ("A.B.NII.GZ", "A.B"), ("notes.txt", "notes")],
)
def test_nii_stem(name, stem):
    assert nii_stem(f"/some/dir/{name}") == stem


def test_is_nifti():
    assert is_nifti("a.nii") and is_nifti("a.NII.GZ")
    assert not is_nifti("a.gz") and not is_nifti("a.nii.bak")


def test_prepare_step_dir_empties_an_existing_folder(tmp_path):
    step = tmp_path / "subject" / "zip"
    (step / "old").mkdir(parents=True)
    (step / "old" / "x.txt").write_text("stale")
    (step / "y.txt").write_text("stale")
    assert prepare_step_dir(tmp_path / "subject", "zip") == step
    assert list(step.iterdir()) == []
    clear_dir(step)  # emptying an empty folder is fine


def test_to_int16_keeps_values_that_fit_and_rescales_the_rest():
    np.testing.assert_array_equal(to_int16(np.array([0.0, 12.0, 300.0])), [0, 12, 300])
    rescaled = to_int16(np.array([0.0, 0.5, 1.0]))
    assert rescaled.dtype == np.int16 and rescaled[-1] == 32767 and rescaled[1] == 16384
    assert to_int16(np.full(3, 7.5)).tolist() == [0, 0, 0]
    assert to_int16(np.array([0.0, 1e6])).max() == 32767


def test_acpcdetect_input_from_each_format(tmp_path, phantom):
    int16 = nib.Nifti1Image(phantom.image.astype(np.int16), phantom.affine)
    plain = tmp_path / "plain.nii"
    nib.save(int16, plain)
    write_acpcdetect_input(plain, tmp_path / "copy.nii")
    assert (tmp_path / "copy.nii").read_bytes() == plain.read_bytes()  # same bytes as before the refactor

    gz = tmp_path / "packed.nii.gz"
    nib.save(int16, gz)
    write_acpcdetect_input(gz, tmp_path / "unpacked.nii")
    unpacked = nib.load(tmp_path / "unpacked.nii")
    assert unpacked.get_data_dtype() == np.int16
    np.testing.assert_array_equal(np.asanyarray(unpacked.dataobj), phantom.image.astype(np.int16))

    floats = tmp_path / "float.nii.gz"
    nib.save(nib.Nifti1Image(phantom.image / 1000.0, phantom.affine), floats)
    write_acpcdetect_input(floats, tmp_path / "from_float.nii")
    converted = nib.load(tmp_path / "from_float.nii")
    assert converted.get_data_dtype() == np.int16
    assert np.asanyarray(converted.dataobj).max() == 32767


def test_demo_manifest_is_complete():
    manifest = json.loads(fetch_demo_data.MANIFEST.read_text())
    assert "CC0" in manifest["license"]
    assert manifest["files"]
    for entry in manifest["files"]:
        assert entry["path"].endswith("_T1w.nii.gz")
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])


def test_fetch_downloads_once_and_rejects_a_wrong_checksum(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"brain" * 1000)
    good = fetch_demo_data.sha256(source)
    target = tmp_path / "out" / "scan.nii.gz"
    target.parent.mkdir()
    assert fetch_demo_data.fetch(source.as_uri(), target, good) is True
    assert target.read_bytes() == source.read_bytes()
    assert fetch_demo_data.fetch(source.as_uri(), target, good) is False  # already there
    with pytest.raises(RuntimeError, match="checksum"):
        fetch_demo_data.fetch(source.as_uri(), tmp_path / "out" / "other.nii.gz", "0" * 64)
    assert not (tmp_path / "out" / "other.nii.gz").exists()
    assert not list(target.parent.glob("*.part"))
