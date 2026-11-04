from loguru import logger
from nipype.interfaces.base import File, TraitedSpec, traits

from node.base import StepInputSpec, StepInterface
from node.enhancement.utils import DEFAULT_PERCENTILES, rescale_intensity
from node.segmentation.utils import get_target_labels, kmeans_cluster, segment, tissue_labels
from utils.load_nii import load_nii
from utils.paths import nii_stem
from utils.save_nii import save_nii

# Split the brain into 3 clusters (GM, WM, CSF)
N_CLUSTERS = 3


class SegmentationInputSpec(StepInputSpec):
    percentiles = traits.List(
        list(DEFAULT_PERCENTILES),
        usedefault=True,
        desc="Intensity percentiles the brain is clipped and rescaled to before clustering, "
        "so a few very bright voxels (vessels, dura) cannot claim a cluster of their own",
    )


class SegmentationOutputSpec(TraitedSpec):
    gm_segmented_output_file = File(exists=True, desc="Path to GM segmented image")
    wm_segmented_output_file = File(exists=True, desc="Path to WM segmented image")
    csf_segmented_output_file = File(exists=True, desc="Path to CSF segmented image")

    gm_labels_output_file = File(exists=True, desc="Path to GM labels image")
    wm_labels_output_file = File(exists=True, desc="Path to WM labels image")
    csf_labels_output_file = File(exists=True, desc="Path to CSF labels image")

    tissue_labels_output_file = File(exists=True, desc="Tissue label map: 0 background, 1 CSF, 2 GM, 3 WM")


class SegmentationInterface(StepInterface):
    """K-means on intensity: the pipeline's original GM / WM / CSF segmentation.

    The input is the bias-corrected brain, the same image FAST and Atropos get.
    Its intensities are clipped and rescaled exactly like the enhancement step
    does, but not histogram-equalised: equalisation flattens the histogram, and
    K-means on a flat histogram cuts every brain into three equal thirds.
    """

    input_spec = SegmentationInputSpec
    output_spec = SegmentationOutputSpec
    step_name = "segmentation"

    def _run_interface(self, runtime):
        input_file = self.inputs.input_file
        output_segmentation_folder = self._step_dir()
        # Path(...).stem of "x.nii.gz" is "x.nii"; the original file names keep that ".nii" part.
        legacy_stem = f"{nii_stem(input_file)}.nii"

        logger.info(f"Segment on: {input_file}")
        logger.info(f"Output folder: {output_segmentation_folder}")

        data, affine = load_nii(input_file)
        data = rescale_intensity(data, self.inputs.percentiles)
        labels = kmeans_cluster(data, N_CLUSTERS)
        targets = dict(zip(("gm", "wm", "csf"), get_target_labels(labels, data)))

        for tissue, target in targets.items():
            labels_path = output_segmentation_folder / f"{legacy_stem}_{tissue}_labels.nii.gz"
            segmented_path = output_segmentation_folder / f"{legacy_stem}_{tissue}_segmented.nii.gz"
            save_nii(labels, str(labels_path), affine)
            save_nii(segment(labels, data, target), str(segmented_path), affine)
            self._results[f"{tissue}_labels_output_file"] = labels_path.as_posix()
            self._results[f"{tissue}_segmented_output_file"] = segmented_path.as_posix()

        tissue_path = output_segmentation_folder / f"{nii_stem(input_file)}_tissue_labels.nii.gz"
        save_nii(tissue_labels(labels, data), str(tissue_path), affine)
        self._results["tissue_labels_output_file"] = tissue_path.as_posix()
        return runtime
