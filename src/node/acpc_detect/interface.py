from pathlib import Path

from nipype.interfaces.base import File, TraitedSpec

from node.acpc_detect.utils import STEP_NAME, acpc_detect
from node.base import StepInputSpec, StepInterface
from utils.paths import nii_stem
from utils.save_nii_as_png import save_nii_as_png


class ACPCDetectInputSpec(StepInputSpec):
    pass


class ACPCDetectOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="AC-PC oriented scan (<name>_RAS.nii)")
    output_png_file = File(exists=True, desc="Middle sagittal slice of the oriented scan")
    landmarks_png_file = File(desc="acpcdetect's sagittal preview with the detected AC, PC and mid-plane")


class ACPCDetectInterface(StepInterface):
    input_spec = ACPCDetectInputSpec
    output_spec = ACPCDetectOutputSpec
    step_name = STEP_NAME  # acpc_detect() creates and fills this folder

    def _run_interface(self, runtime):
        input_file = Path(self.inputs.input_file)
        output_folder = acpc_detect(input_file, Path(self.inputs.output_folder))

        stem = nii_stem(input_file)
        output_file = output_folder / f"{stem}_RAS.nii"
        if not output_file.exists():
            raise RuntimeError(f"acpcdetect finished but did not write {output_file.name}")
        output_png_file = output_folder / f"{stem}_RAS.png"
        save_nii_as_png(output_file, output_png_file)

        self._results["output_file"] = output_file.as_posix()
        self._results["output_png_file"] = output_png_file.as_posix()
        landmarks = output_folder / f"{stem}_ACPC_sagittal.png"
        if landmarks.exists():
            self._results["landmarks_png_file"] = landmarks.as_posix()
        return runtime
