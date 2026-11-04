from pathlib import Path

from nipype.interfaces.base import File, TraitedSpec, traits
from nipype.interfaces.fsl import BET

from node.base import StepInputSpec, StepInterface


class SkullStrippingInputSpec(StepInputSpec):
    frac = traits.Float(0.5, usedefault=True, desc="Fractional intensity threshold")


class SkullStrippingOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="Path to the extracted brain image")
    mask_file = File(exists=True, desc="Binary brain mask written by BET")


class SkullStrippingInterface(StepInterface):
    """Remove skull and scalp with FSL BET (robust centre estimation)."""

    input_spec = SkullStrippingInputSpec
    output_spec = SkullStrippingOutputSpec
    step_name = "skull_stripping"

    def _run_interface(self, runtime):
        input_file = self.inputs.input_file
        output_file = self._step_dir() / Path(input_file).name

        result = BET(
            in_file=input_file, out_file=str(output_file), frac=self.inputs.frac, robust=True, mask=True
        ).run()

        self._results["output_file"] = result.outputs.out_file
        self._results["mask_file"] = result.outputs.mask_file
        return runtime
