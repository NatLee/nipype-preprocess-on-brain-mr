from pathlib import Path

from loguru import logger
from nipype.interfaces.base import File, TraitedSpec
from nipype.interfaces.fsl import FLIRT

from config import default_reference
from node.base import StepInputSpec, StepInterface
from utils.paths import nii_stem


class FLIRTInputSpec(StepInputSpec):
    ref_file = File(exists=True, desc="Reference image (default: FSL's MNI152_T1_1mm)")


class FLIRTOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="Path to the registered image")
    out_matrix_file = File(exists=True, desc="Affine matrix from the scan to the reference (FSL format)")


class FLIRTInterface(StepInterface):
    """Affine (12 degrees of freedom) registration to the reference template with FSL FLIRT."""

    input_spec = FLIRTInputSpec
    output_spec = FLIRTOutputSpec
    step_name = "registration"

    def _run_interface(self, runtime):
        input_file = self.inputs.input_file
        output_reg_folder = self._step_dir()
        output_file = output_reg_folder / Path(input_file).name
        matrix_file = output_reg_folder / f"{nii_stem(input_file)}.mat"
        ref_file = self.inputs.ref_file if self.inputs.ref_file else str(default_reference())

        logger.info(f"Registering {input_file} to {ref_file}")
        logger.info(f"Output file: {output_file}")

        FLIRT(
            in_file=input_file,
            reference=Path(ref_file).as_posix(),
            out_file=str(output_file),
            out_matrix_file=str(matrix_file),
            bins=256,
            cost_func="corratio",
            searchr_x=[0, 0],
            searchr_y=[0, 0],
            searchr_z=[0, 0],
            dof=12,
            interp="spline",
        ).run()

        self._results["output_file"] = output_file.as_posix()
        self._results["out_matrix_file"] = matrix_file.as_posix()
        return runtime
