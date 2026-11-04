from pathlib import Path

from loguru import logger
from nipype.interfaces.base import File, TraitedSpec
from nipype.interfaces.fsl import Reorient2Std

from node.base import StepInputSpec, StepInterface


class Reorient2StdInputSpec(StepInputSpec):
    pass


class Reorient2StdOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="Reoriented output file")


class Reorient2StdInterface(StepInterface):
    """Reorder the voxel axes to match the MNI152 template (FSL fslreorient2std)."""

    input_spec = Reorient2StdInputSpec
    output_spec = Reorient2StdOutputSpec
    step_name = "orient2std"

    def _run_interface(self, runtime):
        input_file = self.inputs.input_file
        output_file = self._step_dir() / Path(input_file).name

        logger.info(f"Reorienting {input_file} to standard space")
        logger.info(f"Output file: {output_file}")
        Reorient2Std(in_file=input_file, out_file=str(output_file)).run()

        self._results["output_file"] = output_file.as_posix()
        return runtime
