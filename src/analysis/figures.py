"""Static figures for the per-subject QC report and the cohort page.

One dark theme for every figure: MRI slices read best on a dark ground, and
the report pages use the same colours. Tissue colours follow the label coding
in `analysis.tissue` and never change meaning between figures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

from analysis.tissue import METHOD_NAMES, TISSUE_LABEL, TISSUE_NAMES, TISSUES  # noqa: E402

# Theme (dark surface). Tissue colours are the first three slots of the
# validated categorical palette, stepped for a dark surface.
THEME = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "ink": "#f0efec",
    "ink_secondary": "#c3c2b7",
    "ink_muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
}
TISSUE_COLORS = {"csf": "#3987e5", "gm": "#d95926", "wm": "#199e70"}
# Diverging map for the bias field: cool = too dark, warm = too bright, grey = flat.
BIAS_CMAP = LinearSegmentedColormap.from_list("bias", ["#3987e5", THEME["axis"], "#e66767"])
DPI = 110
OVERLAY_ALPHA = 0.55
FONT = ["DejaVu Sans"]  # ships with matplotlib, so text renders the same everywhere


def _style() -> dict:
    return {
        "figure.facecolor": THEME["surface"],
        "axes.facecolor": THEME["surface"],
        "savefig.facecolor": THEME["surface"],
        "text.color": THEME["ink"],
        "axes.labelcolor": THEME["ink_secondary"],
        "axes.edgecolor": THEME["axis"],
        "xtick.color": THEME["ink_muted"],
        "ytick.color": THEME["ink_muted"],
        "axes.titlecolor": THEME["ink"],
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.family": FONT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": THEME["grid"],
        "grid.linewidth": 1,
    }


def _save(fig: plt.Figure, out_file: str | Path) -> Path:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_file


# ----------------------------------------------------------------- volumes ---


def load_canonical(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """3D voxel data reoriented to RAS (so slices display the same way) and the voxel sizes.

    Trailing length-1 dimensions are dropped: acpcdetect writes its output as x*y*z*1.
    """
    image = nib.as_closest_canonical(nib.squeeze_image(nib.load(str(path))))
    return np.asanyarray(image.dataobj).astype(np.float32), np.asarray(image.header.get_zooms()[:3])


def display_range(data: np.ndarray) -> tuple[float, float]:
    """Display range: 1st-99.5th percentile of the non-zero voxels."""
    values = data[data > 0]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [1, 99.5])
    return float(low), float(high if high > low else low + 1)


def _centre(data: np.ndarray) -> tuple[int, int, int]:
    """Slice indices through the centre of mass of the non-zero voxels."""
    mask = data > 0
    if not mask.any():
        return tuple(s // 2 for s in data.shape)
    return tuple(int(round(c)) for c in ndimage.center_of_mass(mask))


def _views(data: np.ndarray, zooms: np.ndarray, centre: Sequence[int]):
    """Sagittal, coronal and axial slices ready for imshow (origin lower), with aspect ratios."""
    x, y, z = centre
    return [
        (data[x, :, :].T, zooms[2] / zooms[1]),
        (data[:, y, :].T, zooms[2] / zooms[0]),
        (data[:, :, z].T, zooms[1] / zooms[0]),
    ]


def _blank(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ------------------------------------------------------------------ figures ---


def pipeline_mosaic(steps: Sequence[tuple[str, str | Path]], out_file: str | Path) -> Path:
    """Filmstrip of the scan after each step: one column per step, sagittal/coronal/axial rows."""
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(3, len(steps), figsize=(2.3 * len(steps), 7.2), squeeze=False)
        for col, (title, path) in enumerate(steps):
            data, zooms = load_canonical(path)
            vmin, vmax = display_range(data)
            for row, (plane, aspect) in enumerate(_views(data, zooms, _centre(data))):
                ax = axes[row, col]
                ax.imshow(plane, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect=aspect)
                _blank(ax)
            axes[0, col].set_title(title, fontsize=10, pad=6)
        for row, name in enumerate(("Sagittal", "Coronal", "Axial")):
            axes[row, 0].set_ylabel(name, color=THEME["ink_muted"], fontsize=9)
        fig.subplots_adjust(wspace=0.04, hspace=0.04)
        return _save(fig, out_file)


def label_rgba(labels: np.ndarray) -> np.ndarray:
    rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
    for tissue in TISSUES:
        color = matplotlib.colors.to_rgba(TISSUE_COLORS[tissue], OVERLAY_ALPHA)
        rgba[labels == TISSUE_LABEL[tissue]] = color
    return rgba


def _tissue_legend(fig: plt.Figure, **kwargs) -> None:
    handles = [Patch(color=TISSUE_COLORS[t], label=TISSUE_NAMES[t]) for t in TISSUES]
    fig.legend(handles=handles, ncol=len(handles), handlelength=1.2, **kwargs)


def axial_slices(mask: np.ndarray, count: int, span: tuple[float, float] = (0.2, 0.8)) -> list[int]:
    """`count` evenly spaced axial slice indices inside the mask's vertical extent."""
    z_indices = np.flatnonzero(mask.any(axis=(0, 1)))
    if z_indices.size == 0:
        return [mask.shape[2] // 2] * count
    low, high = z_indices[0], z_indices[-1]
    return [int(round(low + f * (high - low))) for f in np.linspace(span[0], span[1], count)]


def segmentation_overlays(
    background: str | Path, label_maps: Mapping[str, str | Path], out_file: str | Path, n_slices: int = 6
) -> Path:
    """One row per method: axial slices with the tissue labels painted over the brain."""
    data, zooms = load_canonical(background)
    vmin, vmax = display_range(data)
    slices = axial_slices(data > 0, n_slices)
    aspect = zooms[1] / zooms[0]
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            len(label_maps), n_slices, figsize=(1.9 * n_slices, 2.2 * len(label_maps) + 0.5), squeeze=False
        )
        for row, (method, path) in enumerate(label_maps.items()):
            labels, _ = load_canonical(path)
            labels = labels.round().astype(int)
            for col, z in enumerate(slices):
                ax = axes[row, col]
                ax.imshow(data[:, :, z].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect=aspect)
                ax.imshow(label_rgba(labels[:, :, z].T), origin="lower", aspect=aspect, interpolation="nearest")
                _blank(ax)
            axes[row, 0].set_ylabel(method, fontsize=10, color=THEME["ink"])
        fig.subplots_adjust(wspace=0.03, hspace=0.05, top=0.9)
        _tissue_legend(fig, loc="upper center", bbox_to_anchor=(0.5, 0.995))
        return _save(fig, out_file)


HISTOGRAM_RANGE = 1.6  # x axis: 0 to 1.6 times the white-matter median
HISTOGRAM_BINS = 128


def _wm_median(image: np.ndarray, labels: np.ndarray) -> float:
    wm = image[labels == TISSUE_LABEL["wm"]]
    return float(np.median(wm)) if wm.size else 1.0


def histogram_edges(image: np.ndarray, scale: float) -> np.ndarray:
    """Bin edges on the relative axis (intensity / scale).

    When the voxels hold whole numbers (an int16 scan), every bin spans the
    same number of integer levels; equal-width bins would catch 3 levels in
    some bins and 4 in others and draw a comb instead of a histogram.
    """
    if not np.array_equal(image, np.round(image)):
        return np.linspace(0, HISTOGRAM_RANGE, HISTOGRAM_BINS + 1)
    step = max(1.0, float(np.round(HISTOGRAM_RANGE * scale / HISTOGRAM_BINS)))
    return (np.arange(0, HISTOGRAM_RANGE * scale + step, step) - 0.5) / scale


def n4_figure(
    before: str | Path,
    after: str | Path,
    bias_field: str | Path,
    mask: str | Path,
    labels: str | Path,
    out_file: str | Path,
    metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> Path:
    """The field N4 removed (left) and the tissue histograms before/after it (right)."""
    pre, zooms = load_canonical(before)
    post, _ = load_canonical(after)
    bias, _ = load_canonical(bias_field)
    brain = load_canonical(mask)[0] > 0
    tissue = load_canonical(labels)[0].round().astype(int)

    relative_bias = np.where(brain, bias / np.median(bias[brain]), np.nan)
    spread = float(np.nanpercentile(np.abs(relative_bias - 1), 99)) or 0.01
    centre = _centre(brain.astype(np.float32))

    with plt.rc_context(_style()):
        fig = plt.figure(figsize=(11, 4.6))
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6], wspace=0.25, hspace=0.45)
        for col, (plane, aspect) in enumerate(_views(relative_bias, zooms, centre)[1:]):
            ax = fig.add_subplot(grid[:, col])
            shown = ax.imshow(
                (plane - 1) * 100, cmap=BIAS_CMAP, vmin=-spread * 100, vmax=spread * 100, origin="lower", aspect=aspect
            )
            _blank(ax)
            ax.set_title(("Coronal", "Axial")[col], fontsize=9, color=THEME["ink_muted"])
        colorbar = fig.colorbar(shown, ax=fig.axes[:2], orientation="horizontal", fraction=0.05, pad=0.04)
        colorbar.set_label("Bias field vs. brain median (%)", color=THEME["ink_secondary"])
        colorbar.outline.set_visible(False)

        for row, (name, image) in enumerate((("Before N4", pre), ("After N4", post))):
            ax = fig.add_subplot(grid[row, 2])
            scale = _wm_median(image, tissue)
            values, bins = image / scale, histogram_edges(image[brain], scale)
            ax.hist(values[brain], bins=bins, color=THEME["ink_muted"], alpha=0.18, label="Whole brain")
            for t in TISSUES:
                ax.hist(
                    values[tissue == TISSUE_LABEL[t]], bins=bins, histtype="step", linewidth=2,
                    color=TISSUE_COLORS[t], label=TISSUE_NAMES[t],
                )
            title = name
            if metrics and name in metrics:
                title += f"   CJV {metrics[name]['cjv']:.3f} · CNR {metrics[name]['cnr']:.2f}"
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            if row == 1:
                ax.set_xlabel("Intensity relative to the white-matter median")
            else:
                ax.legend(loc="upper left", ncol=2)
        return _save(fig, out_file)


def _title_and_legend(ax: plt.Axes, title: str) -> None:
    """Title on the first line, the legend on its own line under it, both left-aligned."""
    ax.set_title(title, loc="left", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.3, handlelength=1.2)


def _fmt_ml(value: float) -> str:
    return f"{value:,.0f}"


def volume_bars(volumes: Sequence[Mapping], out_file: str | Path, title: str) -> Path:
    """Stacked horizontal bar per row of `volumes` (native mL), one segment per tissue."""
    labels = [row["label"] for row in volumes]
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(volumes) + 1.3))
        y = np.arange(len(volumes))[::-1]
        left = np.zeros(len(volumes))
        for tissue in TISSUES:
            widths = np.array([row[f"native_{tissue}_ml"] for row in volumes])
            ax.barh(y, widths, left=left, height=0.5, color=TISSUE_COLORS[tissue],
                    edgecolor=THEME["surface"], linewidth=2, label=TISSUE_NAMES[tissue])
            for yi, start, width, row in zip(y, left, widths, volumes):
                share = row[f"{tissue}_fraction"] * 100
                if width > 0.09 * max(r["native_tiv_ml"] for r in volumes):
                    ax.text(start + width * 0.04, yi, f"{share:.0f}%", va="center", ha="left",
                            fontsize=8, color="#ffffff")
            left += widths
        for yi, total in zip(y, left):
            ax.text(total * 1.01, yi, f"{_fmt_ml(total)} mL", va="center", fontsize=8, color=THEME["ink_secondary"])
        ax.set_yticks(y, labels)
        ax.set_xlim(0, left.max() * 1.16)
        ax.set_xlabel("Native-space volume (mL)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, colors=THEME["ink_secondary"])
        _title_and_legend(ax, title)
        return _save(fig, out_file)


def agreement_dots(rows: Sequence[Mapping], out_file: str | Path) -> Path:
    """Dice per tissue for each pair of methods: one row per pair, one dot per tissue.

    The three tissues sit on slightly offset lines inside a row so equal values
    do not hide each other.
    """
    pairs = list(dict.fromkeys((r["method_a"], r["method_b"]) for r in rows))
    offsets = dict(zip(TISSUES, np.linspace(0.2, -0.2, len(TISSUES))))
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(pairs) + 1.3))
        y_of = {pair: i for i, pair in enumerate(reversed(pairs))}
        for tissue in TISSUES:
            points = [(r["dice"], y_of[(r["method_a"], r["method_b"])] + offsets[tissue])
                      for r in rows if r["tissue"] == tissue]
            if points:
                xs, ys = zip(*points)
                ax.scatter(xs, ys, s=70, color=TISSUE_COLORS[tissue], edgecolor=THEME["surface"],
                           linewidth=2, zorder=3, label=TISSUE_NAMES[tissue])
        ax.set_yticks(list(y_of.values()), [f"{METHOD_NAMES[a]} vs {METHOD_NAMES[b]}" for a, b in y_of])
        low = min([r["dice"] for r in rows] + [0.5])
        ax.set_xlim(max(0.0, low - 0.05), 1.0)
        ax.set_ylim(-0.6, len(pairs) - 0.4)
        ax.set_xlabel("Dice overlap (1 = identical voxels)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, colors=THEME["ink_secondary"])
        _title_and_legend(ax, "How much the methods agree")
        return _save(fig, out_file)

