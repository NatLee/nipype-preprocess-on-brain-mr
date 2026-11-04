"""3D brain surface: marching cubes on the segmentation, then a turntable GIF and an interactive page.

The surface is the outer boundary of grey + white matter. Each vertex is
coloured by its "grey-matter depth": the distance (mm) from that point of the
surface to the nearest white matter. Thin ridges of cortex show up light,
places where the surface bridges a fold show up dark, so the folding pattern
is visible even in a small picture.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage import measure  # noqa: E402

from analysis.figures import THEME  # noqa: E402
from analysis.tissue import TISSUE_LABEL  # noqa: E402

# One hue (the grey-matter orange) from light to dark: deeper grey matter = darker.
DEPTH_CMAP = LinearSegmentedColormap.from_list("gm_depth", ["#ffe9dc", "#f4a37a", "#d95926", "#8f3412"])
# Simple lamp model: ambient light everywhere, diffuse light on faces turned towards
# the lamp, and a small white highlight so the folds read as 3D.
AMBIENT = 0.3
DIFFUSE = 0.85
SPECULAR = 0.15
ZOOM = 1.35  # how much of the frame the brain fills (matplotlib leaves room for a rotating cube)


@dataclass
class Surface:
    vertices: np.ndarray  # (n, 3) world coordinates in mm
    faces: np.ndarray  # (m, 3) vertex indices
    values: np.ndarray  # (n,) grey-matter depth in mm
    normals: np.ndarray  # (n, 3) unit normals pointing out of the brain, world space


def brain_surface(label_file: str | Path, smoothing_mm: float = 1.5, mesh_spacing_mm: float = 2.0) -> Surface:
    """Marching cubes on (GM + WM) of a tissue label map, coloured by grey-matter depth.

    `mesh_spacing_mm` sets how far apart the mesh's corners are. At the default
    2 mm a whole brain is about 120,000 triangles, which matplotlib turns into
    a 36-frame GIF in under a minute; a 1 mm mesh has four times as many
    triangles, takes about three minutes and looks the same at GIF size.
    """
    image = nib.load(str(label_file))
    labels = np.asanyarray(image.dataobj).round().astype(np.int16)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
    step_size = max(1, round(mesh_spacing_mm / zooms.min()))

    brain = np.isin(labels, (TISSUE_LABEL["gm"], TISSUE_LABEL["wm"]))
    if not brain.any():
        raise ValueError(f"{label_file} has no grey or white matter to draw")
    brain = ndimage.binary_fill_holes(brain)
    smooth = ndimage.gaussian_filter(brain.astype(np.float32), sigma=smoothing_mm / zooms)
    padded = np.pad(smooth, 1)  # closes the surface where the brain touches the volume's edge
    # Normals follow the intensity gradient downhill, i.e. out of the brain.
    verts, faces, normals, _ = measure.marching_cubes(padded, level=0.5, step_size=step_size)
    verts -= 1

    # Distance (mm) from every voxel to the nearest white-matter voxel, read at each vertex.
    wm = labels == TISSUE_LABEL["wm"]
    depth = ndimage.distance_transform_edt(~wm, sampling=zooms) if wm.any() else np.zeros(labels.shape)
    values = ndimage.map_coordinates(depth, verts.T, order=1, mode="nearest")

    world = nib.affines.apply_affine(image.affine, verts)
    # Normals transform with the inverse transpose of the voxel-to-world matrix.
    world_normals = normals @ np.linalg.inv(image.affine[:3, :3])
    world_normals /= np.linalg.norm(world_normals, axis=1, keepdims=True)
    return Surface(
        vertices=world.astype(np.float32),
        faces=faces.astype(np.int32),
        values=values.astype(np.float32),
        normals=world_normals.astype(np.float32),
    )


def _face_normals(surface: Surface) -> np.ndarray:
    """Outward unit normal of each face: the mean of its corners' normals."""
    normals = surface.normals[surface.faces].mean(axis=1)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length == 0, 1, length)


def _light_direction(azim_deg: float, elev_deg: float) -> np.ndarray:
    """Unit vector from the scene towards matplotlib's camera, nudged up and to the left."""
    azim, elev = np.radians(azim_deg + 25), np.radians(elev_deg + 25)
    return np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])


def depth_range(surface: Surface) -> tuple[float, float]:
    """Colour scale limits shared by the GIF and the interactive page (2nd-98th percentile)."""
    low, high = np.percentile(surface.values, [2, 98])
    return float(low), float(max(high, low + 1e-3))


def turntable_frames(surface: Surface, n_frames: int = 36, size_px: int = 480, elev: float = 15.0) -> list[Image.Image]:
    """Render the surface rotating once around the vertical axis."""
    normals = _face_normals(surface)
    face_values = surface.values[surface.faces].mean(axis=1)
    base = DEPTH_CMAP(Normalize(*depth_range(surface), clip=True)(face_values))[:, :3]
    lo, hi = surface.vertices.min(axis=0), surface.vertices.max(axis=0)
    centre, radius = (lo + hi) / 2, (hi - lo).max() / 2

    frames = []
    with plt.rc_context({"figure.facecolor": THEME["surface"], "savefig.facecolor": THEME["surface"]}):
        fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.set_facecolor(THEME["surface"])
        # No antialiasing: with tens of thousands of tiny faces it draws visible seams.
        mesh = Poly3DCollection(surface.vertices[surface.faces], linewidths=0, edgecolors="none", antialiased=False)
        ax.add_collection3d(mesh)
        for setter, axis in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
            setter(centre[axis] - radius, centre[axis] + radius)
        ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
        ax.set_axis_off()
        for azim in np.linspace(0, 360, n_frames, endpoint=False):
            ax.view_init(elev=elev, azim=azim)
            lambert = np.clip(normals @ _light_direction(azim, elev), 0, 1)
            colors = base * (AMBIENT + DIFFUSE * lambert)[:, None] + SPECULAR * lambert[:, None] ** 8
            mesh.set_facecolor(np.clip(colors, 0, 1))
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100)
            buffer.seek(0)
            frames.append(Image.open(buffer).convert("RGB"))
        plt.close(fig)
    return frames


def save_gif(frames: list[Image.Image], out_file: str | Path, frame_ms: int = 90) -> Path:
    """Animated GIF that loops forever; one shared palette keeps it small and flicker-free."""
    out_file = Path(out_file)
    palette = frames[len(frames) // 4].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    quantized[0].save(
        out_file, save_all=True, append_images=quantized[1:], duration=frame_ms, loop=0, optimize=True, disposal=1
    )
    return out_file


def save_interactive(surface: Surface, out_file: str | Path, title: str) -> Path:
    """Standalone HTML page you can rotate and zoom in a browser (plotly.js is written next to it)."""
    import plotly.graph_objects as go  # only this function needs plotly

    vmin, vmax = depth_range(surface)
    stops = np.linspace(0, 1, 9)
    colorscale = [[float(s), matplotlib.colors.to_hex(DEPTH_CMAP(s))] for s in stops]
    x, y, z = surface.vertices.T
    i, j, k = surface.faces.T
    mesh = go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        intensity=surface.values, cmin=vmin, cmax=vmax, colorscale=colorscale,
        colorbar={"title": {"text": "Grey-matter depth (mm)"}, "thickness": 14, "len": 0.6},
        lighting={"ambient": AMBIENT, "diffuse": 0.8, "specular": 0.15, "roughness": 0.6},
        lightposition={"x": 1000, "y": 1000, "z": 2000},
        hovertemplate="x %{x:.0f} · y %{y:.0f} · z %{z:.0f} mm<br>depth %{intensity:.1f} mm<extra></extra>",
    )
    figure = go.Figure(mesh)
    hidden_axis = {"visible": False}
    figure.update_layout(
        title={"text": title, "x": 0.02, "font": {"size": 16}},
        paper_bgcolor=THEME["surface"],
        font={"color": THEME["ink"], "family": "system-ui, -apple-system, Segoe UI, sans-serif"},
        scene={"xaxis": hidden_axis, "yaxis": hidden_axis, "zaxis": hidden_axis, "aspectmode": "data",
               "camera": {"eye": {"x": -1.6, "y": 0.4, "z": 0.5}}},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    out_file = Path(out_file)
    figure.write_html(str(out_file), include_plotlyjs="directory", full_html=True)
    return out_file
