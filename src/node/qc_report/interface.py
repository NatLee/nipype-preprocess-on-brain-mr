import json
from pathlib import Path

import nipype
import pandas as pd
from loguru import logger
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
)

from analysis import figures
from analysis.report import relative, render_page
from analysis.tissue import METHOD_NAMES
from utils.paths import prepare_step_dir

STEP_NAME = "report"


class QCReportInputSpec(BaseInterfaceInputSpec):
    output_folder = Directory(mandatory=True, desc="Subject output folder")
    input_file = File(exists=True, mandatory=True, desc="The original scan")
    acpc_file = File(exists=True, mandatory=True, desc="AC-PC oriented scan")
    acpc_png_file = File(exists=True, mandatory=True, desc="Middle sagittal slice of the AC-PC scan")
    landmarks_png_file = File(desc="acpcdetect's landmark preview")
    registered_file = File(exists=True, mandatory=True, desc="Scan registered to the template")
    brain_file = File(exists=True, mandatory=True, desc="Skull-stripped scan (before N4)")
    brain_mask = File(exists=True, mandatory=True, desc="BET brain mask")
    corrected_file = File(exists=True, mandatory=True, desc="Bias-corrected brain")
    bias_field = File(exists=True, mandatory=True, desc="Bias field N4 removed")
    enhanced_file = File(exists=True, mandatory=True, desc="Enhanced brain (K-means input)")
    kmeans_labels = File(exists=True, mandatory=True, desc="K-means tissue label map")
    fast_labels = File(exists=True, desc="FSL FAST tissue label map")
    atropos_labels = File(exists=True, desc="ANTs Atropos tissue label map")
    volumetrics_file = File(exists=True, mandatory=True, desc="volumetrics.csv from the analysis step")
    agreement_file = File(exists=True, mandatory=True, desc="agreement.csv from the analysis step")
    quality_file = File(exists=True, mandatory=True, desc="quality.json from the analysis step")
    map_png_files = InputMultiObject(File(exists=True), desc="Probability-map PNGs drawn with nilearn")
    gif_file = File(desc="Rotating 3D brain (optional)")
    html_file = File(desc="Interactive 3D viewer (optional)")


class QCReportOutputSpec(TraitedSpec):
    report_file = File(exists=True, desc="The subject's HTML report")


def _pairs_table(agreement: pd.DataFrame) -> list[dict]:
    """One row per method pair with the Dice of each tissue as columns."""
    rows = []
    for (method_a, method_b), group in agreement.groupby(["method_a", "method_b"], sort=False):
        row = {"label": f"{METHOD_NAMES[method_a]} vs {METHOD_NAMES[method_b]}"}
        row.update(dict(zip(group["tissue"], group["dice"])))
        rows.append(row)
    return rows


class QCReportInterface(SimpleInterface):
    """One HTML page per subject with every figure and number the pipeline produced."""

    input_spec = QCReportInputSpec
    output_spec = QCReportOutputSpec

    def _label_maps(self) -> dict[str, str]:
        return {
            METHOD_NAMES[method]: value
            for method in METHOD_NAMES
            if (value := getattr(self.inputs, f"{method}_labels"))
        }

    def _run_interface(self, runtime):
        folder = prepare_step_dir(self.inputs.output_folder, STEP_NAME)
        figure_dir = folder / "figures"
        subject = Path(self.inputs.output_folder).name
        logger.info(f"QC report for {subject}")

        volumes = pd.read_csv(self.inputs.volumetrics_file)
        agreement = pd.read_csv(self.inputs.agreement_file)
        quality = json.loads(Path(self.inputs.quality_file).read_text())
        reference = quality["tissue_masks_from"]
        label_maps = self._label_maps()

        made = {
            "pipeline": figures.pipeline_mosaic(
                [
                    ("Original", self.inputs.input_file),
                    ("AC-PC", self.inputs.acpc_file),
                    ("Registered", self.inputs.registered_file),
                    ("Skull stripped", self.inputs.brain_file),
                    ("N4 corrected", self.inputs.corrected_file),
                    ("Enhanced", self.inputs.enhanced_file),
                ],
                figure_dir / "pipeline.png",
            ),
            "overlays": figures.segmentation_overlays(
                self.inputs.corrected_file, label_maps, figure_dir / "segmentation.png"
            ),
            "n4": figures.n4_figure(
                self.inputs.brain_file,
                self.inputs.corrected_file,
                self.inputs.bias_field,
                self.inputs.brain_mask,
                getattr(self.inputs, f"{reference}_labels"),
                figure_dir / "bias_correction.png",
                metrics={"Before N4": quality["before_n4"], "After N4": quality["after_n4"]},
            ),
            "volumes": figures.volume_bars(
                [{"label": row["method_name"], **row} for row in volumes.to_dict("records")],
                figure_dir / "volumes.png",
                title="Tissue volumes by method",
            ),
        }
        if len(agreement):
            made["agreement"] = figures.agreement_dots(agreement.to_dict("records"), figure_dir / "agreement.png")

        page = folder / "index.html"
        links = {name: relative(path, folder) for name, path in made.items()}
        links.update(
            landmarks=relative(self.inputs.landmarks_png_file or None, folder),
            acpc_png=relative(self.inputs.acpc_png_file, folder),
            gif=relative(self.inputs.gif_file or None, folder),
            html=relative(self.inputs.html_file or None, folder),
            maps=[(Path(p).stem, relative(p, folder)) for p in (self.inputs.map_png_files or [])],
        )
        render_page(
            "subject_report.html.j2",
            page,
            {
                "subject": subject,
                "input_name": Path(self.inputs.input_file).name,
                "nipype_version": nipype.__version__,
                "reference": reference,
                "volumes": volumes.to_dict("records"),
                "agreement": _pairs_table(agreement),
                "quality": quality,
                "figures": links,
                "index_href": relative(Path(self.inputs.output_folder).parent / "index.html", folder),
                "files": [
                    ("Volumes per method (CSV)", relative(self.inputs.volumetrics_file, folder)),
                    ("Method agreement (CSV)", relative(self.inputs.agreement_file, folder)),
                    ("Image-quality numbers (JSON)", relative(self.inputs.quality_file, folder)),
                ],
            },
        )
        self._results["report_file"] = page.as_posix()
        return runtime
