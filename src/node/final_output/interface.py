import shutil
from pathlib import Path

from loguru import logger
from nipype.interfaces.base import BaseInterfaceInputSpec, Directory, File, SimpleInterface, TraitedSpec

from utils.paths import prepare_step_dir

STEP_NAME = "final_output"
# Fixed names for the three probability maps, whatever their titles are.
MAP_NAMES = {"gm_png_file": "GM_Segmentation.png", "wm_png_file": "WM_Segmentation.png",
             "csf_png_file": "CSF_Segmentation.png"}
# Optional extras copied under their own file names when the workflow provides them.
EXTRA_INPUTS = ("volumetrics_file", "brain_3d_gif_file")


class FinalOutputInputSpec(BaseInterfaceInputSpec):
    acpc_output_file = File(exists=True, desc="ACPC corrected NIfTI file (.nii)", mandatory=True)
    acpc_output_png_file = File(exists=True, desc="ACPC corrected PNG file", mandatory=True)
    gm_png_file = File(exists=True, desc="Gray matter segmentation PNG file", mandatory=True)
    wm_png_file = File(exists=True, desc="White matter segmentation PNG file", mandatory=True)
    csf_png_file = File(exists=True, desc="CSF segmentation PNG file", mandatory=True)
    output_folder = Directory(desc="Final output directory", mandatory=True)
    volumetrics_file = File(exists=True, desc="Tissue volumes CSV")
    brain_3d_gif_file = File(exists=True, desc="Rotating 3D brain GIF")


class FinalOutputOutputSpec(TraitedSpec):
    output_folder = Directory(exists=True, desc="Directory containing organized final outputs")


class OrganizeFinalOutputInterface(SimpleInterface):
    """Copy the files most people want into one folder."""

    input_spec = FinalOutputInputSpec
    output_spec = FinalOutputOutputSpec

    def _run_interface(self, runtime):
        output_dir = prepare_step_dir(self.inputs.output_folder, STEP_NAME)
        logger.info(f"Organizing final output in {output_dir}")

        for name in ("acpc_output_file", "acpc_output_png_file"):
            source = Path(getattr(self.inputs, name))
            shutil.copy(source, output_dir / source.name)
        for name, target in MAP_NAMES.items():
            shutil.copy(getattr(self.inputs, name), output_dir / target)
        for name in EXTRA_INPUTS:
            if value := getattr(self.inputs, name):
                shutil.copy(value, output_dir / Path(value).name)

        self._results["output_folder"] = output_dir.as_posix()
        return runtime
