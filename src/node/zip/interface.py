import gzip
import shutil
from pathlib import Path

from loguru import logger
from nipype.interfaces.base import File, TraitedSpec

from node.base import StepInputSpec, StepInterface


class ZipOutputInputSpec(StepInputSpec):
    pass


class ZipOutputOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="Output compressed file")


class ZipOutputInterface(StepInterface):
    """Gzip the AC-PC oriented scan so the FSL steps get a .nii.gz."""

    input_spec = ZipOutputInputSpec
    output_spec = ZipOutputOutputSpec
    step_name = "zip"

    def _run_interface(self, runtime):
        input_file = Path(self.inputs.input_file)
        output_file = self._step_dir() / f"{input_file.name}.gz"

        logger.info(f"Compressing {input_file} to {output_file}")
        with open(input_file, "rb") as f_in, gzip.open(output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        self._results["output_file"] = output_file.as_posix()
        return runtime
