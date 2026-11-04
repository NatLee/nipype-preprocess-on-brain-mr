"""Common pieces of the pipeline's nipype interfaces.

Every step reads one image, writes its results into
`<subject output folder>/<step_name>/` and passes the paths on. The input
spec and the "create and empty my folder" logic live here once.
"""

from __future__ import annotations

from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, Directory, File, SimpleInterface

from utils.paths import prepare_step_dir


class StepInputSpec(BaseInterfaceInputSpec):
    input_file = File(exists=True, mandatory=True, desc="Input image (.nii or .nii.gz)")
    output_folder = Directory(
        mandatory=True,
        desc="Subject output folder; the step writes into a sub-folder named after the step",
    )


class StepInterface(SimpleInterface):
    """A SimpleInterface that owns one sub-folder of the subject's output folder."""

    step_name: str = ""

    def _step_dir(self) -> Path:
        if not self.step_name:
            raise NotImplementedError(f"{type(self).__name__} must set step_name")
        return prepare_step_dir(self.inputs.output_folder, self.step_name)
