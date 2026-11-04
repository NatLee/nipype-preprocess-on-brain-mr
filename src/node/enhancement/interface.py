from pathlib import Path

from loguru import logger
from nipype.interfaces.base import File, TraitedSpec, traits

from node.base import StepInputSpec, StepInterface
from node.enhancement.utils import DEFAULT_PERCENTILES, denoise, equalize_hist, rescale_intensity
from utils.load_nii import load_nii
from utils.save_nii import save_nii


class EnhancementInputSpec(StepInputSpec):
    kernel_size = traits.Int(
        1, usedefault=True, desc="Kernel size for denoising. If the size is too large, the image will be blurred"
    )
    percentiles = traits.List(list(DEFAULT_PERCENTILES), usedefault=True, desc="Percentiles for intensity rescaling")
    bins_num = traits.Int(256, usedefault=True, desc="Number of bins for histogram equalization")
    eh = traits.Bool(True, usedefault=True, desc="Enable histogram equalization")


class EnhancementOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="Path to the enhanced image")


class EnhancementInterface(StepInterface):
    """Median filter, percentile rescaling and histogram equalisation, for viewing the brain with more contrast."""

    input_spec = EnhancementInputSpec
    output_spec = EnhancementOutputSpec
    step_name = "enhancement"

    def _run_interface(self, runtime):
        input_file = self.inputs.input_file
        enhanced_image_path = self._step_dir() / Path(input_file).name

        logger.info(f"Preprocess on: {input_file}")
        logger.info(f"Output: {enhanced_image_path}")

        volume, affine = load_nii(input_file)
        volume = denoise(volume, self.inputs.kernel_size)
        volume = rescale_intensity(volume, self.inputs.percentiles, self.inputs.bins_num)
        if self.inputs.eh:
            volume = equalize_hist(volume, self.inputs.bins_num)
        save_nii(volume, str(enhanced_image_path), affine)

        self._results["output_file"] = enhanced_image_path.as_posix()
        return runtime
