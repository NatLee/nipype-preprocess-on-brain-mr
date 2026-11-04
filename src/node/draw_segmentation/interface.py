from pathlib import Path

from loguru import logger
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from utils.draw_segmentation import draw_segmentation

STEP_NAME = "draw_segmentation"


class DrawSegmentationInputSpec(BaseInterfaceInputSpec):
    background_file = File(
        exists=True, mandatory=True, desc="Anatomical image in the same space as the segmentation (.nii or .nii.gz)"
    )
    # Old name of `background_file`. nipype's own deprecation handling warns and moves the value over.
    acpc_input_file = File(exists=True, deprecated="2.0", new_name="background_file", desc="Use background_file")
    segmented_input_file = File(exists=True, desc="Segmented Source image path (.nii.gz or .nii)", mandatory=True)
    output_folder = Directory(desc="Output folder for the segmented image", mandatory=True)
    threshold = traits.Float(0.68, usedefault=True, desc="Threshold for the segmentation image (0-1)")
    title = traits.String("prob-map", usedefault=True, desc="Title for the segmentation image")


class DrawSegmentationOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="Path to png image")


class DrawSegmentationInterface(SimpleInterface):
    """Probability map of one tissue drawn with nilearn; three of these share one folder."""

    input_spec = DrawSegmentationInputSpec
    output_spec = DrawSegmentationOutputSpec

    def _run_interface(self, runtime):
        title = self.inputs.title
        output_draw_folder = Path(self.inputs.output_folder) / STEP_NAME
        output_draw_folder.mkdir(parents=True, exist_ok=True)
        # The folder is shared by the GM, WM and CSF nodes, so only this node's file is replaced.
        output_png_path = output_draw_folder / f"{title}.png"
        output_png_path.unlink(missing_ok=True)

        logger.info(f"Running draw segmentation for {self.inputs.segmented_input_file} to {output_png_path}")
        draw_segmentation(
            input_nii_path=self.inputs.background_file,
            input_seg_nii_path=self.inputs.segmented_input_file,
            output_png_path=output_png_path.absolute().as_posix(),
            threshold=self.inputs.threshold,
            title=title,
        )

        self._results["output_file"] = output_png_path.as_posix()
        return runtime
