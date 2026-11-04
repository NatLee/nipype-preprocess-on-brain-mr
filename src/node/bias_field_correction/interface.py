from pathlib import Path

from loguru import logger
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.base import File, TraitedSpec

from node.base import StepInputSpec, StepInterface
from utils.paths import nii_stem


class BiasFieldCorrectionInputSpec(StepInputSpec):
    pass


class BiasFieldCorrectionOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="Path to the bias corrected image")
    bias_field_file = File(exists=True, desc="The smooth intensity drift N4 removed (multiplicative field)")


class BiasFieldCorrectionInterface(StepInterface):
    """Remove the scanner's slow intensity drift with ANTs N4."""

    input_spec = BiasFieldCorrectionInputSpec
    output_spec = BiasFieldCorrectionOutputSpec
    step_name = "bias_correction"

    def _run_interface(self, runtime):
        input_file = self.inputs.input_file
        output_folder = self._step_dir()
        output_file = output_folder / Path(input_file).name
        bias_file = output_folder / f"{nii_stem(input_file)}_bias.nii.gz"

        logger.info(f"N4ITK on: {input_file}")
        logger.info(f"Output: {output_file}")

        N4BiasFieldCorrection(
            input_image=str(input_file),
            output_image=str(output_file),
            save_bias=True,
            bias_image=str(bias_file),
            dimension=3,
            n_iterations=[100, 100, 60, 40],
            shrink_factor=3,
            convergence_threshold=1e-4,
            bspline_fitting_distance=300,
        ).run()

        self._results["output_file"] = output_file.as_posix()
        self._results["bias_field_file"] = bias_file.as_posix()
        return runtime
