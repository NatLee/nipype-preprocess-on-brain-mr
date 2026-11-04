"""Settings shared by the workflow and its nodes.

Anything that depends on where a tool or a template lives is resolved here,
from environment variables, with the paths of the Docker image as defaults.
The nodes import these names instead of hard-coding locations.
"""

from __future__ import annotations

import os
from pathlib import Path

# Where `python workflow.py` looks for scans and writes results by default
# (the mount points used in the README's `docker run` commands).
DEFAULT_DATA_DIR = Path("/data")
DEFAULT_OUTPUT_DIR = Path("/output")

# ART (Automatic Registration Toolbox). `acpcdetect` reads its models
# (T1acpc.mdl, orion.mdl) from $ARTHOME.
ART_HOME = Path(os.environ.get("ARTHOME", "/utils/acpcdetect_v2.1_LinuxCentOS6.7"))
ACPCDETECT_BIN = Path(os.environ.get("ACPCDETECT_BIN", str(ART_HOME / "bin" / "acpcdetect")))

# FSL's standard-space template the scans are registered to.
DEFAULT_REFERENCE_NAME = "MNI152_T1_1mm.nii.gz"


def fsl_standard_dir() -> Path:
    """Return `$FSLDIR/data/standard`, or raise a readable error when FSL is missing."""
    fsldir = os.environ.get("FSLDIR")
    if not fsldir:
        raise RuntimeError(
            "FSLDIR is not set, so the default MNI152 template cannot be found. "
            "Run the pipeline inside the Docker image, or pass --reference <template.nii.gz>."
        )
    return Path(fsldir) / "data" / "standard"


def default_reference() -> Path:
    """The template FLIRT registers to when no reference is given."""
    return fsl_standard_dir() / DEFAULT_REFERENCE_NAME
