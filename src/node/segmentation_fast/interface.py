from loguru import logger
from nipype.interfaces.fsl import FAST

from node.base import StepInputSpec, StepInterface
from node.tissue_output import ToolSegmentationOutputSpec, tissue_outputs
from utils.paths import nii_stem


class FASTSegmentationInputSpec(StepInputSpec):
    pass


class FASTSegmentationInterface(StepInterface):
    """Tissue segmentation with FSL FAST (a hidden Markov random field model with bias correction)."""

    input_spec = FASTSegmentationInputSpec
    output_spec = ToolSegmentationOutputSpec
    step_name = "segmentation_fast"

    def _run_interface(self, runtime):
        input_file = self.inputs.input_file
        folder = self._step_dir()
        stem = nii_stem(input_file)
        logger.info(f"FSL FAST on: {input_file}")

        # nipype names FAST outputs relative to the working directory, so run it inside the step folder.
        result = FAST(
            in_files=input_file,
            out_basename=stem,
            img_type=1,  # T1-weighted
            number_classes=3,
        ).run(cwd=str(folder))

        # FAST's partial-volume map pve_<c-1> belongs to class c of its label map.
        raw_labels = result.outputs.tissue_class_map
        self._results["raw_labels_file"] = raw_labels
        self._results.update(
            tissue_outputs(
                raw_labels, input_file, result.outputs.partial_volume_files, folder / f"{stem}_tissue_labels.nii.gz"
            )
        )
        return runtime
