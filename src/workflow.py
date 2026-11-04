"""Brain MRI preprocessing with nipype.

Every T1-weighted scan in the data folder goes through the same chain of
nipype nodes (AC-PC alignment, registration to MNI152, skull stripping, N4
bias correction, enhancement, tissue segmentation), and every step writes
its result to `<output>/<scan name>/<step>/`. On top of that the workflow
compares three segmentation methods, measures tissue volumes, renders the
brain in 3D and writes an HTML report per scan plus an index page.

Run `python workflow.py --help` for the options.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from nipype import Function
from nipype.interfaces.utility import Merge
from nipype.pipeline import JoinNode, Node, Workflow

import config
from analysis.tissue import DEFAULT_PRIMARY_METHOD, METHOD_NAMES
from node.acpc_detect.interface import ACPCDetectInterface
from node.analysis.interface import AnalysisInterface
from node.bias_field_correction.interface import BiasFieldCorrectionInterface
from node.cohort_summary.interface import CohortSummaryInterface
from node.draw_segmentation.interface import DrawSegmentationInterface
from node.enhancement.interface import EnhancementInterface
from node.final_output.interface import OrganizeFinalOutputInterface
from node.orient2std.interface import Reorient2StdInterface
from node.qc_report.interface import QCReportInterface
from node.registration.interface import FLIRTInterface
from node.render_3d.interface import Render3DInterface
from node.segmentation.interface import SegmentationInterface
from node.segmentation_atropos.interface import AtroposSegmentationInterface
from node.segmentation_fast.interface import FASTSegmentationInterface
from node.skull_stripping.interface import SkullStrippingInterface
from node.zip.interface import ZipOutputInterface
from utils.paths import clear_dir, is_nifti, nii_stem

WORKFLOW_NAME = "preprocess_workflow"
# Each segmentation method: its node name and interface. Everything downstream reads the
# shared outputs (tissue_labels_file, <tissue>_map_file), so adding a method means one row here.
SEGMENTATION_NODES = {
    "kmeans": ("segmentation", SegmentationInterface),
    "fast": ("segmentation_fast", FASTSegmentationInterface),
    "atropos": ("segmentation_atropos", AtroposSegmentationInterface),
}
# The three tissue maps drawn with nilearn: (tissue, title). The titles are also the PNG names.
SEGMENTATION_MAPS = (("gm", "GM@map"), ("wm", "WM@map"), ("csf", "CSF@map"))


# ------------------------------------------------------------------ inputs ---


def discover_inputs(data_dir: Path) -> list[Path]:
    """Every .nii / .nii.gz file directly inside `data_dir`, sorted by name."""
    return sorted(p for p in Path(data_dir).iterdir() if p.is_file() and is_nifti(p))


def prepare_output_folders(input_files: Sequence[Path], output_dir: Path) -> list[tuple[Path, Path]]:
    """Create (and empty) one output folder per scan, named after the scan."""
    names = [nii_stem(f) for f in input_files]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Two input files map to the same output folder: {', '.join(duplicates)}")
    pairs = []
    for input_file, name in zip(input_files, names):
        folder = Path(output_dir) / name
        folder.mkdir(parents=True, exist_ok=True)
        clear_dir(folder)
        pairs.append((input_file, folder))
    return pairs


def process_pair(pair):
    input_file, output_folder = pair
    return input_file, output_folder


# ---------------------------------------------------------------- workflow ---


def build_workflow(
    pairs: Sequence[tuple[Path, Path]],
    output_dir: Path,
    *,
    reference: Path | None = None,
    primary: str = DEFAULT_PRIMARY_METHOD,
    methods: Sequence[str] = tuple(METHOD_NAMES),
    render_3d: bool = True,
    base_dir: Path | None = None,
) -> Workflow:
    """Wire the nodes.

    `primary` is the segmentation the tissue maps, final_output/, the QC numbers
    and the 3D surface use; `methods` are the segmentations run next to it for
    the comparison (the primary one always runs).
    """
    unknown = ({primary} | set(methods)) - set(SEGMENTATION_NODES)
    if unknown:
        raise ValueError(f"Unknown segmentation method(s): {', '.join(sorted(unknown))}")
    run_methods = [m for m in SEGMENTATION_NODES if m in set(methods) | {primary}]
    pairs = [(str(i), str(o)) for i, o in pairs]

    workflow = Workflow(name=WORKFLOW_NAME, base_dir=str(base_dir) if base_dir else None)

    process_pair_node = Node(
        Function(input_names=["pair"], output_names=["input_file", "output_folder"], function=process_pair),
        name="process_pair",
    )
    process_pair_node.iterables = [("pair", pairs)]

    acpc_node = Node(ACPCDetectInterface(), name="t1_acpc_detect")
    zip_node = Node(ZipOutputInterface(), name="zip_output")
    orient2std_node = Node(Reorient2StdInterface(), name="orient2std")
    registration_node = Node(FLIRTInterface(), name="registration")
    if reference:
        registration_node.inputs.ref_file = str(reference)
    skull_stripping_node = Node(SkullStrippingInterface(), name="skull_stripping")
    bias_field_correction_node = Node(BiasFieldCorrectionInterface(), name="bias_field_correction")
    enhancement_node = Node(EnhancementInterface(), name="enhancement")
    segmentation_nodes = {m: Node(SEGMENTATION_NODES[m][1](), name=SEGMENTATION_NODES[m][0]) for m in run_methods}
    primary_node = segmentation_nodes[primary]
    final_output_node = Node(OrganizeFinalOutputInterface(), name="final_output")
    analysis_node = Node(AnalysisInterface(primary_method=primary), name="analysis")
    report_node = Node(QCReportInterface(), name="qc_report")
    cohort_node = JoinNode(
        CohortSummaryInterface(),
        joinsource="process_pair",
        # A join field must be connected, so the GIFs are only joined when they are drawn.
        joinfield=["volumetrics_files", "agreement_files", "quality_files", "report_files"]
        + (["gif_files"] if render_3d else []),
        name="cohort_summary",
    )
    cohort_node.inputs.output_root = str(output_dir)
    cohort_node.inputs.primary_method = primary

    # Every per-subject node writes into the subject's folder.
    per_subject = [
        acpc_node, zip_node, orient2std_node, registration_node, skull_stripping_node,
        bias_field_correction_node, enhancement_node, final_output_node, analysis_node, report_node,
        *segmentation_nodes.values(),
    ]
    for node in per_subject:
        workflow.connect(process_pair_node, "output_folder", node, "output_folder")

    # The original chain: each step feeds the next one.
    workflow.connect(process_pair_node, "input_file", acpc_node, "input_file")
    chain = [
        acpc_node, zip_node, orient2std_node, registration_node, skull_stripping_node,
        bias_field_correction_node, enhancement_node,
    ]
    for upstream, downstream in zip(chain, chain[1:]):
        workflow.connect(upstream, "output_file", downstream, "input_file")
    # Every segmentation reads the bias-corrected brain. (K-means used to read the
    # histogram-equalised image, which split every brain into three equal thirds; that
    # image is still made, for the report.)
    for method, node in segmentation_nodes.items():
        workflow.connect(bias_field_correction_node, "output_file", node, "input_file")
        workflow.connect(node, "tissue_labels_file", analysis_node, f"{method}_labels")
        workflow.connect(node, "tissue_labels_file", report_node, f"{method}_labels")
    if "atropos" in segmentation_nodes:
        workflow.connect(skull_stripping_node, "mask_file", segmentation_nodes["atropos"], "mask_file")

    # The primary segmentation's tissue maps, drawn with nilearn over the registered scan
    # (the same space as the segmentation).
    map_merge = Node(Merge(len(SEGMENTATION_MAPS)), name="collect_maps")
    for index, (tissue, title) in enumerate(SEGMENTATION_MAPS, start=1):
        draw_node = Node(DrawSegmentationInterface(title=title), name=f"draw_{tissue}_segmentation")
        workflow.connect(process_pair_node, "output_folder", draw_node, "output_folder")
        workflow.connect(registration_node, "output_file", draw_node, "background_file")
        workflow.connect(primary_node, f"{tissue}_map_file", draw_node, "segmented_input_file")
        workflow.connect(draw_node, "output_file", final_output_node, f"{tissue}_png_file")
        workflow.connect(draw_node, "output_file", map_merge, f"in{index}")

    workflow.connect(acpc_node, "output_file", final_output_node, "acpc_output_file")
    workflow.connect(acpc_node, "output_png_file", final_output_node, "acpc_output_png_file")

    # Volumes, agreement and image-quality numbers.
    workflow.connect(registration_node, "out_matrix_file", analysis_node, "matrix_file")
    workflow.connect(skull_stripping_node, "output_file", analysis_node, "before_n4")
    workflow.connect(skull_stripping_node, "mask_file", analysis_node, "brain_mask")
    workflow.connect(bias_field_correction_node, "output_file", analysis_node, "after_n4")
    workflow.connect(bias_field_correction_node, "bias_field_file", analysis_node, "bias_field")

    # The per-subject report.
    report_inputs = [
        (process_pair_node, "input_file", "input_file"),
        (acpc_node, "output_file", "acpc_file"),
        (acpc_node, "output_png_file", "acpc_png_file"),
        (acpc_node, "landmarks_png_file", "landmarks_png_file"),
        (registration_node, "output_file", "registered_file"),
        (skull_stripping_node, "output_file", "brain_file"),
        (skull_stripping_node, "mask_file", "brain_mask"),
        (bias_field_correction_node, "output_file", "corrected_file"),
        (bias_field_correction_node, "bias_field_file", "bias_field"),
        (enhancement_node, "output_file", "enhanced_file"),
        (analysis_node, "volumetrics_file", "volumetrics_file"),
        (analysis_node, "agreement_file", "agreement_file"),
        (analysis_node, "quality_file", "quality_file"),
        (map_merge, "out", "map_png_files"),
    ]
    for node, field, target in report_inputs:
        workflow.connect(node, field, report_node, target)

    if render_3d:
        render_node = Node(Render3DInterface(), name="render_3d")
        workflow.connect(process_pair_node, "output_folder", render_node, "output_folder")
        workflow.connect(primary_node, "tissue_labels_file", render_node, "label_file")
        workflow.connect(render_node, "gif_file", report_node, "gif_file")
        workflow.connect(render_node, "html_file", report_node, "html_file")
        workflow.connect(render_node, "gif_file", cohort_node, "gif_files")
        workflow.connect(render_node, "gif_file", final_output_node, "brain_3d_gif_file")

    workflow.connect(analysis_node, "volumetrics_file", final_output_node, "volumetrics_file")
    for field in ("volumetrics_file", "agreement_file", "quality_file"):
        workflow.connect(analysis_node, field, cohort_node, field.replace("_file", "_files"))
    workflow.connect(report_node, "report_file", cohort_node, "report_files")
    return workflow


def write_graphs(workflow: Workflow, output_dir: Path) -> Path:
    """Draw the workflow (graph.png and graph_detailed.png) into the output folder."""
    dot_file = Path(output_dir) / "graph.dot"
    workflow.write_graph(dotfilename=str(dot_file), graph2use="flat", format="png", simple_form=True)
    return dot_file.with_name("graph_detailed.png")


# --------------------------------------------------------------------- CLI ---


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python workflow.py",
        description=(
            "Preprocess every T1-weighted brain MRI (.nii / .nii.gz) in DATA_DIR:\n"
            "AC-PC alignment, registration to MNI152, skull stripping, N4 bias correction,\n"
            "enhancement and GM/WM/CSF segmentation; then tissue volumes, a 3D render and\n"
            "an HTML report per scan, plus OUTPUT_DIR/index.html for the whole run."
        ),
        epilog=(
            "examples:\n"
            "  python workflow.py                              # /data -> /output, everything on\n"
            "  python workflow.py --segmentation kmeans --methods --no-3d   # the original K-means only, no 3D\n"
            "  python workflow.py --reference $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz   # 8x fewer voxels"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR,
                        help="folder with the input scans (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR,
                        help="where results go, one sub-folder per scan (default: %(default)s)")
    parser.add_argument("--reference", type=Path, default=None,
                        help=f"template to register to (default: $FSLDIR/data/standard/{config.DEFAULT_REFERENCE_NAME})")
    parser.add_argument("--segmentation", choices=list(METHOD_NAMES), default=DEFAULT_PRIMARY_METHOD,
                        help="primary tissue segmentation, used for the tissue maps, final_output/, the QC "
                             "numbers and the 3D brain (default: %(default)s)")
    parser.add_argument("--methods", nargs="*", choices=list(METHOD_NAMES), default=list(METHOD_NAMES),
                        metavar="METHOD",
                        help=f"segmentations to run and compare: {' '.join(METHOD_NAMES)} "
                             "(default: all; the --segmentation one always runs)")
    parser.add_argument("--no-3d", dest="render_3d", action="store_false", help="skip the 3D brain render")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="scans/steps to run in parallel (default: %(default)s, one at a time)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="nipype's working folder and crash files (default: OUTPUT_DIR/_work, emptied each run)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_dir.is_dir():
        logger.error(f"Data folder not found: {args.data_dir}")
        return 2
    input_files = discover_inputs(args.data_dir)
    if not input_files:
        logger.error(f"No .nii or .nii.gz files in {args.data_dir}")
        return 2
    if args.reference is not None and not args.reference.is_file():
        logger.error(f"Reference template not found: {args.reference}")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or args.output_dir / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    pairs = prepare_output_folders(input_files, args.output_dir)
    logger.info(f"{len(pairs)} scan(s): {', '.join(p[0].name for p in pairs)}")
    workflow = build_workflow(
        pairs, args.output_dir, reference=args.reference, primary=args.segmentation, methods=args.methods,
        render_3d=args.render_3d, base_dir=work_dir,
    )
    workflow.config["execution"]["crashdump_dir"] = str(work_dir / "crash")
    graph = write_graphs(workflow, args.output_dir)
    logger.info(f"Workflow graph: {graph}")

    started = time.monotonic()
    if args.n_procs > 1:
        workflow.run(plugin="MultiProc", plugin_args={"n_procs": args.n_procs})
    else:
        workflow.run()
    logger.info(f"Done in {time.monotonic() - started:.0f} s. Open {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
