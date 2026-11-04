import nibabel as nib
import numpy as np


def load_nii(path: str):
    """Voxel data (as stored, no scaling copy) and the affine of a NIfTI file."""
    nii = nib.load(str(path))
    return np.asanyarray(nii.dataobj), nii.affine
