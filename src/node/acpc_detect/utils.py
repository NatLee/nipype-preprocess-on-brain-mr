"""Run ART's `acpcdetect` on one scan.

acpcdetect finds the mid-sagittal plane and the anterior / posterior
commissures (AC, PC) and writes a copy of the scan oriented so the AC-PC line
is horizontal. It only reads uncompressed `.nii` files stored as 16-bit
integers, so the input is converted into a temporary folder first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from loguru import logger

import config
from utils.paths import nii_stem, prepare_step_dir

STEP_NAME = "acpc"

# -no-tilt-correction: keep the voxel grid, store the tilt in the header (no resampling yet)
# -center-AC:          put the AC at the centre of the field of view
# -noppm:              keep only the PNG copies of the landmark previews (made with netpbm)
ACPCDETECT_ARGS = ("-no-tilt-correction", "-center-AC", "-noppm")
# Files acpcdetect reads from $ARTHOME. A missing PILbrain.nii does not make it
# fail: it prints a warning, exits 0 and writes a wrongly oriented scan.
ART_MODEL_FILES = ("T1acpc.mdl", "orion.mdl", "PILbrain.nii")

# acpcdetect accepts "short or unsigned short" voxels only (see its documentation).
SUPPORTED_DTYPES = (np.dtype(np.int16), np.dtype(np.uint16))
INT16_MAX = np.iinfo(np.int16).max


def to_int16(data: np.ndarray) -> np.ndarray:
    """Convert voxel values to int16, rescaling only when they do not fit as they are."""
    finite = np.nan_to_num(np.asarray(data, dtype=np.float64))
    is_whole = np.array_equal(finite, np.round(finite))
    if is_whole and finite.min() >= np.iinfo(np.int16).min and finite.max() <= INT16_MAX:
        return finite.astype(np.int16)
    low, high = float(finite.min()), float(finite.max())
    if high == low:
        return np.zeros(finite.shape, dtype=np.int16)
    return np.round((finite - low) / (high - low) * INT16_MAX).astype(np.int16)


def write_acpcdetect_input(source: Path, target: Path) -> None:
    """Write `source` (.nii or .nii.gz, any numeric type) as an uncompressed int16-compatible `.nii`."""
    image = nib.load(str(source))
    if source.name.lower().endswith(".nii") and image.get_data_dtype() in SUPPORTED_DTYPES:
        shutil.copy(source, target)  # already in the right format: copy the bytes unchanged
        return
    if image.get_data_dtype() in SUPPORTED_DTYPES:
        data = np.asanyarray(image.dataobj)
    else:
        data = to_int16(image.get_fdata())
    header = image.header.copy()
    header.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, image.affine, header), str(target))


def check_art_installation() -> None:
    """Fail early, with a clear message, when acpcdetect or one of its model files is missing."""
    missing = [str(config.ACPCDETECT_BIN)] if not config.ACPCDETECT_BIN.exists() else []
    missing += [str(config.ART_HOME / name) for name in ART_MODEL_FILES if not (config.ART_HOME / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"ART files not found: {', '.join(missing)}. "
            "Set ARTHOME (and ACPCDETECT_BIN), or run inside the Docker image."
        )


def acpc_detect(nii_file_path: Path, output_path: Path) -> Path:
    """Run acpcdetect and move everything it wrote into `<output_path>/acpc/`."""
    nii_file_path = Path(nii_file_path)
    logger.info(f"ACPC detection on: {nii_file_path}")
    check_art_installation()

    output_folder = prepare_step_dir(output_path, STEP_NAME)
    logger.info(f"ACPC output path: {output_folder}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_nii_file_path = temp_dir_path / f"{nii_stem(nii_file_path)}.nii"
        write_acpcdetect_input(nii_file_path, temp_nii_file_path)

        command = [str(config.ACPCDETECT_BIN), *ACPCDETECT_ARGS, "-i", str(temp_nii_file_path)]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={**os.environ, "ARTHOME": str(config.ART_HOME)},
            check=False,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()[-2000:]
            raise RuntimeError(
                f"acpcdetect failed on {nii_file_path.name} (exit code {result.returncode}): {details}"
            )

        for file in temp_dir_path.iterdir():
            shutil.move(str(file), str(output_folder / file.name))

    return output_folder
