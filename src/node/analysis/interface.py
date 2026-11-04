import json
from pathlib import Path

import nibabel as nib
import pandas as pd
from loguru import logger
from nipype.interfaces.base import BaseInterfaceInputSpec, Directory, File, SimpleInterface, TraitedSpec

from analysis.quality import bias_field_range, contrast_metrics
from analysis.tissue import METHOD_NAMES, agreement, reference_method
from analysis.volumetrics import read_fsl_matrix, registration_scale, tissue_volumes, voxel_volume_ml
from utils.load_nii import load_nii
from utils.paths import prepare_step_dir

STEP_NAME = "analysis"


class AnalysisInputSpec(BaseInterfaceInputSpec):
    output_folder = Directory(mandatory=True, desc="Subject output folder")
    kmeans_labels = File(exists=True, mandatory=True, desc="K-means tissue label map")
    fast_labels = File(exists=True, desc="FSL FAST tissue label map (optional)")
    atropos_labels = File(exists=True, desc="ANTs Atropos tissue label map (optional)")
    matrix_file = File(exists=True, mandatory=True, desc="FLIRT matrix scan -> template")
    before_n4 = File(exists=True, mandatory=True, desc="Skull-stripped image before bias correction")
    after_n4 = File(exists=True, mandatory=True, desc="Bias-corrected image")
    bias_field = File(exists=True, mandatory=True, desc="Bias field N4 removed")
    brain_mask = File(exists=True, mandatory=True, desc="BET brain mask")


class AnalysisOutputSpec(TraitedSpec):
    volumetrics_file = File(exists=True, desc="CSV: tissue volumes per segmentation method")
    agreement_file = File(exists=True, desc="CSV: Dice overlap per tissue for every pair of methods")
    quality_file = File(exists=True, desc="JSON: CJV / CNR / SNR before and after N4, bias-field range")


class AnalysisInterface(SimpleInterface):
    """Volumes, method agreement and image-quality numbers for one subject."""

    input_spec = AnalysisInputSpec
    output_spec = AnalysisOutputSpec

    def _label_maps(self) -> dict[str, Path]:
        maps = {}
        for method in METHOD_NAMES:
            value = getattr(self.inputs, f"{method}_labels")
            if value:
                maps[method] = Path(value)
        return maps

    def _run_interface(self, runtime):
        folder = prepare_step_dir(self.inputs.output_folder, STEP_NAME)
        subject = Path(self.inputs.output_folder).name
        label_files = self._label_maps()
        logger.info(f"Analysis of {subject}: methods {', '.join(label_files)}")

        scale = registration_scale(read_fsl_matrix(self.inputs.matrix_file))
        label_data = {}
        rows = []
        for method, path in label_files.items():
            image = nib.load(str(path))
            label_data[method] = load_nii(str(path))[0]
            volumes = tissue_volumes(label_data[method], voxel_volume_ml(image.header.get_zooms()), scale)
            rows.append({"subject": subject, "method": method, "method_name": METHOD_NAMES[method], **volumes})
        volumetrics_file = folder / "volumetrics.csv"
        pd.DataFrame(rows).to_csv(volumetrics_file, index=False)

        agreement_file = folder / "agreement.csv"
        pd.DataFrame(agreement(label_data), columns=["method_a", "method_b", "tissue", "dice"]).assign(
            subject=subject
        ).to_csv(agreement_file, index=False)

        reference = reference_method(label_data)
        before, _ = load_nii(self.inputs.before_n4)
        after, _ = load_nii(self.inputs.after_n4)
        bias, _ = load_nii(self.inputs.bias_field)
        mask, _ = load_nii(self.inputs.brain_mask)
        quality = {
            "subject": subject,
            "tissue_masks_from": reference,
            "before_n4": contrast_metrics(before, label_data[reference]),
            "after_n4": contrast_metrics(after, label_data[reference]),
            **bias_field_range(bias, mask > 0),
            "registration_scale": round(scale, 4),
        }
        quality_file = folder / "quality.json"
        quality_file.write_text(json.dumps(quality, indent=2))

        self._results["volumetrics_file"] = volumetrics_file.as_posix()
        self._results["agreement_file"] = agreement_file.as_posix()
        self._results["quality_file"] = quality_file.as_posix()
        return runtime
