import json
from pathlib import Path

import pandas as pd
from loguru import logger
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from analysis import figures
from analysis.report import relative, render_page
from analysis.tissue import METHOD_NAMES
from utils.paths import prepare_step_dir

FOLDER_NAME = "cohort"


class CohortSummaryInputSpec(BaseInterfaceInputSpec):
    output_root = Directory(exists=True, mandatory=True, desc="The run's output folder (parent of the subjects)")
    primary_method = traits.Enum(*METHOD_NAMES, mandatory=True, desc="Segmentation the summary table and chart use")
    volumetrics_files = InputMultiObject(File(exists=True), mandatory=True, desc="volumetrics.csv of every subject")
    agreement_files = InputMultiObject(File(exists=True), mandatory=True, desc="agreement.csv of every subject")
    quality_files = InputMultiObject(File(exists=True), mandatory=True, desc="quality.json of every subject")
    report_files = InputMultiObject(File(exists=True), mandatory=True, desc="HTML report of every subject")
    gif_files = InputMultiObject(File(exists=True), desc="3D GIF of every subject (optional)")


class CohortSummaryOutputSpec(TraitedSpec):
    index_file = File(exists=True, desc="<output>/index.html linking every subject")
    volumetrics_file = File(exists=True, desc="<output>/cohort/volumetrics.csv with every subject and method")


class CohortSummaryInterface(SimpleInterface):
    """Join the per-subject results into one table, one chart and the index page."""

    input_spec = CohortSummaryInputSpec
    output_spec = CohortSummaryOutputSpec

    def _run_interface(self, runtime):
        root = Path(self.inputs.output_root)
        folder = prepare_step_dir(root, FOLDER_NAME)
        logger.info(f"Cohort summary of {len(self.inputs.report_files)} subject(s)")

        volumes = pd.concat([pd.read_csv(f) for f in self.inputs.volumetrics_files], ignore_index=True)
        # Rows already come subject by subject (input order), methods in METHOD_NAMES order.
        agreement = pd.concat([pd.read_csv(f) for f in self.inputs.agreement_files], ignore_index=True)
        quality = {q["subject"]: q for q in (json.loads(Path(f).read_text()) for f in self.inputs.quality_files)}
        volumetrics_file = folder / "volumetrics.csv"
        agreement_file = folder / "agreement.csv"
        volumes.to_csv(volumetrics_file, index=False)
        agreement.to_csv(agreement_file, index=False)

        primary = self.inputs.primary_method
        primary_rows = volumes[volumes["method"] == primary]
        reports = {Path(f).parent.parent.name: f for f in self.inputs.report_files}
        subjects = [
            {
                **row,
                "cjv_before": quality[row["subject"]]["before_n4"]["cjv"],
                "cjv_after": quality[row["subject"]]["after_n4"]["cjv"],
                "report": relative(reports[row["subject"]], root),
            }
            for row in primary_rows.to_dict("records")
        ]
        chart = figures.volume_bars(
            [{"label": s["subject"], **s} for s in subjects],
            folder / "volumes.png",
            title=f"Tissue volumes per subject ({METHOD_NAMES[primary]})",
        )
        index_file = render_page(
            "index.html.j2",
            root / "index.html",
            {
                "subjects": subjects,
                "primary": primary,
                "figures": {"volumes": relative(chart, root)},
                "gifs": [(Path(g).parent.parent.name, relative(g, root)) for g in (self.inputs.gif_files or [])],
                "files": [
                    ("Volumes, every subject and method (CSV)", relative(volumetrics_file, root)),
                    ("Method agreement, every subject (CSV)", relative(agreement_file, root)),
                ],
            },
        )
        self._results["index_file"] = index_file.as_posix()
        self._results["volumetrics_file"] = volumetrics_file.as_posix()
        return runtime
