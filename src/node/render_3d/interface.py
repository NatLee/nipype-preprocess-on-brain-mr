from pathlib import Path

from loguru import logger
from nipype.interfaces.base import BaseInterfaceInputSpec, Directory, File, SimpleInterface, TraitedSpec, traits

from analysis.render3d import brain_surface, save_gif, save_interactive, turntable_frames
from utils.paths import prepare_step_dir

STEP_NAME = "render3d"


class Render3DInputSpec(BaseInterfaceInputSpec):
    output_folder = Directory(mandatory=True, desc="Subject output folder")
    label_file = File(exists=True, mandatory=True, desc="Tissue label map the surface is built from")
    n_frames = traits.Int(36, usedefault=True, desc="Frames in the turntable GIF (one full turn)")
    size_px = traits.Int(480, usedefault=True, desc="Width and height of the GIF in pixels")


class Render3DOutputSpec(TraitedSpec):
    gif_file = File(exists=True, desc="Rotating view of the brain surface")
    html_file = File(exists=True, desc="Interactive 3D viewer (open in a browser)")


class Render3DInterface(SimpleInterface):
    """Marching-cubes brain surface: a turntable GIF and an interactive HTML viewer."""

    input_spec = Render3DInputSpec
    output_spec = Render3DOutputSpec

    def _run_interface(self, runtime):
        folder = prepare_step_dir(self.inputs.output_folder, STEP_NAME)
        subject = Path(self.inputs.output_folder).name
        surface = brain_surface(self.inputs.label_file)
        logger.info(f"3D surface of {subject}: {len(surface.faces):,} triangles")

        gif_file = save_gif(
            turntable_frames(surface, n_frames=self.inputs.n_frames, size_px=self.inputs.size_px),
            folder / "brain_3d.gif",
        )
        html_file = save_interactive(surface, folder / "brain_3d.html", title=f"{subject} - brain surface")

        self._results["gif_file"] = gif_file.as_posix()
        self._results["html_file"] = html_file.as_posix()
        return runtime
