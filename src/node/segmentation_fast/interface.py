from loguru import logger
from nipype.interfaces.fsl import FAST

from node.base import StepInputSpec, StepInterface
from node.tissue_output import TissueLabelsOutputSpec, save_tissue_labels
from utils.paths import nii_stem


class FASTSegmentationInputSpec(StepInputSpec):
    pass


class FASTSegmentationInterface(StepInterface):
    """Tissue segmentation with FSL FAST (a hidden Markov random field model with bias correction)."""

    input_spec = FASTSegmentationInputSpec
    output_spec = TissueLabelsOutputSpec
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

        raw_labels = result.outputs.tissue_class_map
        self._results["raw_labels_file"] = raw_labels
        self._results["tissue_labels_file"] = save_tissue_labels(
            raw_labels, input_file, folder / f"{stem}_tissue_labels.nii.gz"
        ).as_posix()
        return runtime
