import nibabel as nib


def save_nii(data, path, affine):
    """Write `data` with `affine` to `path` (.nii or .nii.gz)."""
    nib.save(nib.Nifti1Image(data, affine), str(path))
