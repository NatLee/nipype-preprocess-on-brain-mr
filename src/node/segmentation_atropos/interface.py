from loguru import logger
from nipype.interfaces.ants import Atropos
from nipype.interfaces.base import File

from node.base import StepInputSpec, StepInterface
from node.tissue_output import ToolSegmentationOutputSpec, tissue_outputs
from utils.paths import nii_stem


class AtroposSegmentationInputSpec(StepInputSpec):
    mask_file = File(exists=True, mandatory=True, desc="Brain mask (from BET) limiting where Atropos looks")


class AtroposSegmentationInterface(StepInterface):
    """Tissue segmentation with ANTs Atropos (Gaussian mixture + Markov random field)."""

    input_spec = AtroposSegmentationInputSpec
    output_spec = ToolSegmentationOutputSpec
    step_name = "segmentation_atropos"

    def _run_interface(self, runtime):
        input_file = self.inputs.input_file
        folder = self._step_dir()
        stem = nii_stem(input_file)
        logger.info(f"ANTs Atropos on: {input_file}")

        result = Atropos(
            dimension=3,
            intensity_images=[input_file],
            mask_image=self.inputs.mask_file,
            # Otsu thresholds start the classes between the histogram peaks. A K-means start
            # can spend one class on the few very bright voxels BET leaves in (vessels, dura).
            initialization="Otsu",
            number_of_tissue_classes=3,
            likelihood_model="Gaussian",
            mrf_smoothing_factor=0.1,
            mrf_radius=[1, 1, 1],
            n_iterations=5,
            convergence_threshold=0.001,
            use_random_seed=False,  # a fixed seed, so the same scan always gives the same labels
            out_classified_image_name=str(folder / f"{stem}_atropos_labels.nii.gz"),
            save_posteriors=True,
            output_posteriors_name_template=f"{stem}_atropos_posterior_%02d.nii.gz",
        ).run(cwd=str(folder))

        # Atropos writes one posterior probability map per class, in class order 1..k.
        raw_labels = result.outputs.classified_image
        self._results["raw_labels_file"] = raw_labels
        self._results.update(
            tissue_outputs(raw_labels, input_file, result.outputs.posteriors, folder / f"{stem}_tissue_labels.nii.gz")
        )
        return runtime
