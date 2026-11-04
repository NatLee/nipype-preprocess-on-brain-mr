"""How workflow.py finds inputs, wires the nodes and parses its options (nothing is executed)."""

import shutil
from pathlib import Path

import pytest

import workflow


@pytest.fixture
def pairs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("b.nii.gz", "a.nii", "notes.txt", "c.nii.bak"):
        (data / name).write_bytes(b"")
    (data / "sub").mkdir()
    return workflow.prepare_output_folders(workflow.discover_inputs(data), tmp_path / "output")


def test_inputs_are_nifti_files_sorted_by_name(pairs, tmp_path):
    assert [p[0].name for p in pairs] == ["a.nii", "b.nii.gz"]
    assert [p[1] for p in pairs] == [tmp_path / "output" / "a", tmp_path / "output" / "b"]


def test_two_scans_with_the_same_name_are_refused(tmp_path):
    with pytest.raises(ValueError, match="same output folder: x"):
        workflow.prepare_output_folders([tmp_path / "x.nii", tmp_path / "x.nii.gz"], tmp_path / "out")


def test_full_workflow_has_every_original_and_new_node(pairs, tmp_path):
    wf = workflow.build_workflow(pairs, tmp_path / "output")
    names = set(wf.list_node_names())
    original = {
        "process_pair", "t1_acpc_detect", "zip_output", "orient2std", "registration", "skull_stripping",
        "bias_field_correction", "enhancement", "segmentation", "draw_gm_segmentation",
        "draw_wm_segmentation", "draw_csf_segmentation", "final_output",
    }
    new = {"segmentation_fast", "segmentation_atropos", "analysis", "render_3d", "qc_report", "cohort_summary"}
    assert original | new <= names


def edges(wf, tmp_path) -> set[tuple[str, str]]:
    """Node-to-node connections, read from the dot file nipype writes (no graphviz needed)."""
    dot = tmp_path / "hierarchy.dot"
    wf.write_hierarchical_dotfile(dotfilename=str(dot), simple_form=True)
    prefix = f"{workflow.WORKFLOW_NAME}_"
    found = set()
    for line in dot.read_text().splitlines():
        if "->" in line:
            source, target = (part.strip(" ;").removeprefix(prefix) for part in line.split("->"))
            found.add((source, target))
    return found


def test_probability_maps_are_drawn_over_the_registered_scan(pairs, tmp_path):
    """Regression: the maps live in template space, so their background must too (it was the AC-PC scan)."""
    connections = edges(workflow.build_workflow(pairs, tmp_path / "output"), tmp_path)
    for tissue in ("gm", "wm", "csf"):
        assert ("registration", f"draw_{tissue}_segmentation") in connections
        assert ("t1_acpc_detect", f"draw_{tissue}_segmentation") not in connections


def test_the_original_chain_is_kept(pairs, tmp_path):
    connections = edges(workflow.build_workflow(pairs, tmp_path / "output"), tmp_path)
    chain = ["process_pair", "t1_acpc_detect", "zip_output", "orient2std", "registration", "skull_stripping",
             "bias_field_correction", "enhancement", "segmentation"]
    for upstream, downstream in zip(chain, chain[1:]):
        assert (upstream, downstream) in connections
    assert ("t1_acpc_detect", "final_output") in connections


def test_methods_and_3d_can_be_switched_off(pairs, tmp_path):
    wf = workflow.build_workflow(pairs, tmp_path / "output", methods=[], render_3d=False)
    names = set(wf.list_node_names())
    assert not names & {"segmentation_fast", "segmentation_atropos", "render_3d"}
    assert "analysis" in names
    with pytest.raises(ValueError, match="Unknown segmentation"):
        workflow.build_workflow(pairs, tmp_path / "output", methods=["spm"])


def test_reference_template_reaches_flirt(pairs, tmp_path):
    template = tmp_path / "template.nii.gz"
    template.write_bytes(b"")  # FLIRT's ref_file must exist
    wf = workflow.build_workflow(pairs, tmp_path / "output", reference=template)
    assert wf.get_node("registration").inputs.ref_file == str(template)


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz is not installed")
def test_graphs_are_drawn_into_the_output_folder(pairs, tmp_path):
    output = tmp_path / "output"
    detailed = workflow.write_graphs(workflow.build_workflow(pairs, output), output)
    assert detailed == output / "graph_detailed.png"
    assert detailed.stat().st_size > 10_000 and (output / "graph.png").exists()


def test_defaults_match_the_original_script():
    args = workflow.parse_args([])
    assert (args.data_dir, args.output_dir) == (Path("/data"), Path("/output"))
    assert args.methods == ["fast", "atropos"] and args.render_3d and args.n_procs == 1


def test_main_explains_bad_input_folders(tmp_path):
    assert workflow.main(["--data-dir", str(tmp_path / "missing")]) == 2
    assert workflow.main(["--data-dir", str(tmp_path)]) == 2  # exists but has no scans
    (tmp_path / "s.nii").write_bytes(b"")
    assert workflow.main(["--data-dir", str(tmp_path), "--reference", str(tmp_path / "nope.nii.gz")]) == 2
