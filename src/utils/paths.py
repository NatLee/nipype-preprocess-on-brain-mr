"""File-name helpers shared by every step."""

from __future__ import annotations

import shutil
from pathlib import Path

# Longest suffix first, so "scan.nii.gz" is not mistaken for "scan.nii" + ".gz".
NIFTI_SUFFIXES = (".nii.gz", ".nii")


def is_nifti(path: str | Path) -> bool:
    """True for `.nii` and `.nii.gz` files (case-insensitive)."""
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def nii_stem(path: str | Path) -> str:
    """File name without its NIfTI suffix: `sub-01_T1w.nii.gz` -> `sub-01_T1w`."""
    name = Path(path).name
    for suffix in NIFTI_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def clear_dir(folder: Path) -> None:
    """Delete everything inside `folder`, keeping the folder itself."""
    for item in folder.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def prepare_step_dir(output_folder: str | Path, step_name: str) -> Path:
    """Create `<output_folder>/<step_name>` and empty it, so a rerun never mixes old files in."""
    folder = Path(output_folder) / step_name
    folder.mkdir(parents=True, exist_ok=True)
    clear_dir(folder)
    return folder
