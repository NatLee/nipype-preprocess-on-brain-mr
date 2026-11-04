import numpy as np
import pytest
from PIL import Image

from analysis import render3d


@pytest.fixture
def surface(phantom_files):
    _, labels = phantom_files
    return render3d.brain_surface(labels)


def test_surface_is_in_world_mm_with_outward_normals(surface, phantom):
    assert surface.vertices.shape[1] == 3 and surface.faces.shape[1] == 3
    assert len(surface.values) == len(surface.vertices) == len(surface.normals)
    centre = surface.vertices.mean(axis=0)
    outward = np.einsum("ij,ij->i", surface.normals, surface.vertices - centre)
    assert (outward > 0).mean() > 0.95
    # The phantom is centred on the origin, so the surface must be too (world coordinates, not voxels).
    assert np.abs(centre).max() < 5


def test_grey_matter_depth_is_positive_and_plausible(surface):
    # Phantom: WM radius about 0.5, outer GM radius 0.75 of a 48-voxel x 2 mm box -> roughly 12 mm of GM.
    assert surface.values.min() >= 0
    assert 5 < np.median(surface.values) < 20


def test_turntable_gif_loops_with_one_frame_per_angle(surface, tmp_path):
    frames = render3d.turntable_frames(surface, n_frames=5, size_px=120)
    assert len(frames) == 5 and frames[0].size == (120, 120)
    gif = render3d.save_gif(frames, tmp_path / "brain.gif")
    with Image.open(gif) as image:
        assert image.n_frames == 5
        assert image.info.get("loop") == 0


def test_interactive_page_is_standalone(surface, tmp_path):
    page = render3d.save_interactive(surface, tmp_path / "brain.html", title="phantom")
    html = page.read_text()
    assert '<script src="plotly.min.js"' in html or 'src="plotly.min.js"' in html
    assert (tmp_path / "plotly.min.js").stat().st_size > 1_000_000


def test_empty_label_map_is_a_clear_error(tmp_path, phantom):
    import nibabel as nib

    empty = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.uint8), phantom.affine), empty)
    with pytest.raises(ValueError, match="no grey or white matter"):
        render3d.brain_surface(empty)
