"""Make the two pictures on the README cover from a real pipeline output (sub-01, 1 mm template).

After the quick-start run, from the repository root:

    docker run --rm -v "$PWD/output:/output:ro" -v "$PWD/docs/images:/assets" \
        nipype-preprocess-on-brain-mr python /assets/cover_assets.py

then render cover.html at 1280x640 (see the comment at its top).
"""

import sys

sys.path.insert(0, "/src")  # the pipeline code inside the image

import matplotlib.pyplot as plt  # noqa: E402

from analysis import figures, render3d  # noqa: E402

SUBJECT = "/output/sub-01_T1w"
LABELS = f"{SUBJECT}/segmentation_fast/sub-01_T1w_RAS_tissue_labels.nii.gz"
BRAIN = f"{SUBJECT}/bias_correction/sub-01_T1w_RAS.nii.gz"

surface = render3d.brain_surface(LABELS)
frames = render3d.turntable_frames(surface, n_frames=12, size_px=600, elev=18)
frames[10].save("/assets/cover-brain.png", optimize=True)  # 300 degrees: seen from the front left

data, zooms = figures.load_canonical(BRAIN)
labels = figures.load_canonical(LABELS)[0].round().astype(int)
vmin, vmax = figures.display_range(data)
aspect = zooms[1] / zooms[0]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), facecolor=figures.THEME["surface"])
for ax, z in zip(axes, figures.axial_slices(data > 0, 3, span=(0.35, 0.75))):
    ax.imshow(data[:, :, z].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect=aspect)
    ax.imshow(figures.label_rgba(labels[:, :, z].T), origin="lower", aspect=aspect, interpolation="nearest")
    ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03)
fig.savefig("/assets/cover-slices.png", dpi=100, facecolor=figures.THEME["surface"])
