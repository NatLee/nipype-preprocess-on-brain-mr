import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402

from utils.load_nii import load_nii  # noqa: E402


def save_nii_as_png(nii_path, png_path):
    """
    Save the middle slice along the first axis of a NIfTI file as a PNG image
    (the mid-sagittal slice for an RAS-oriented scan).

    Args:
    nii_path (str): Path to the NIfTI file.
    png_path (str): Path where the PNG file will be saved.
    """
    data, _ = load_nii(nii_path)
    selected_slice = data[data.shape[0] // 2, :, :].squeeze()

    plt.imshow(selected_slice.T, cmap="gray", origin="lower")
    plt.axis("off")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    logger.info(f"Saved PNG file to {png_path}")
