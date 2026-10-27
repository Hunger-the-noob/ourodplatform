"""D3/D4 的故障数据、可重复构建、进度及持久化历史回归。"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from od_platform.common import paths
from od_platform.common.progress import iter_progress, progress_reporting
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.data_validation.checks.label_format import _validate_one_line
from od_platform.data_validation.history import compare_reports, read_report
from od_platform.data_validation.labels import parse_label_line
from od_platform.data_validation.registry import (
    CheckContext,
    CheckSeverity,
    get_check,
    list_check_names,
)
from od_platform.data_validation.service import validate_dataset
from od_platform.data_validation.snapshot import DatasetSnapshot, build_snapshot

pytestmark = pytest.mark.usefixtures("isolated_data_workspace")


def test_progress_is_opt_in_and_resets_after_failure(capsys):
    assert list(iter_progress(range(2), "quiet")) == [0, 1]
    assert capsys.readouterr().err == ""
    with pytest.raises(RuntimeError), progress_reporting(True):
        assert list(iter_progress(range(2), "visible")) == [0, 1]
        raise RuntimeError("stop")
    output = capsys.readouterr().err
    assert "visible" in output and "100%" in output
    assert list(iter_progress(range(2), "quiet again")) == [0, 1]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("dataset", "format_name", "task", "classes"),
    [
        ("demo", "pascal_voc", "detect", None),
        ("demococo", "coco", "detect", None),
        ("demococo", "coco", "segment", None),
        ("demoyolo", "yolo", "detect", ["head", "反光衣"]),
    ],
)
def test_pipeline_progress_preserves_outputs(dataset, format_name, task, classes, capsys):
    pipeline = DatasetPipeline(dataset, format_name, task=task, classes=classes)
    before = pipeline.run()
    processed = paths.dataset_processed_dir(dataset)
    contents = {
        str(path.relative_to(processed)): path.read_bytes()
        for path in processed.rglob("*")
        if path.is_file()
    }
    after = pipeline.run(show_progress=True)
    assert before["contract_fingerprint"] == after["contract_fingerprint"]
    assert before["data_fingerprint"] == after["data_fingerprint"]
    assert contents == {
        str(path.relative_to(processed)): path.read_bytes()
        for path in processed.rglob("*")
        if path.is_file()
    }
    assert sum(after["counts"].values()) == len(list((pipeline.raw_images).glob("*.jpg")))
    assert "D3 convert" in capsys.readouterr().err
    ctx = CheckContext(Path(after["yaml"]), build_snapshot(Path(after["yaml"])))
    assert get_check("image_integrity").func(ctx).severity == CheckSeverity.PASS


@pytest.mark.parametrize(
    ("line", "task", "kind"),
    [
        ("0 0.5 0.5 0.2 0.2", "detect", None),
        ("1 0.5 0.5 0.2 0.2", "detect", None),
        ("2 0.5 0.5 0.2 0.2", "detect", "class_id_out_of_range"),
        ("-1 0.5 0.5 0.2 0.2", "detect", "class_id_out_of_range"),
        ("1.5 0.5 0.5 0.2 0.2", "detect", "parse_error"),
        ("0 0.5 0.5 0.2", "detect", "field_count_mismatch"),
        ("0 1.01 0.5 0.2 0.2", "detect", "coord_out_of_range"),
        ("0 -0.01 0.5 0.2 0.2", "detect", "coord_out_of_range"),
        ("0 nan 0.5 0.2 0.2", "detect", "non_finite_coordinate"),
        ("0 0.5 inf 0.2 0.2", "detect", "non_finite_coordinate"),
        ("0 0.5 0.5 -inf 0.2", "detect", "non_finite_coordinate"),
        ("0 0 0 1 0 0 1", "segment", None),
        ("2 0 0 1 0 0 1", "segment", "class_id_out_of_range"),
        ("0 0 0 1.2 0 0 1", "segment", "coord_out_of_range"),
        ("0 0 0 nan 0 0 1", "segment", "non_finite_coordinate"),
        ("0 0 0 1 0", "segment", "polygon_too_few_points"),
    ],
)
def test_label_format_boundaries(line, task, kind):
    assert _validate_one_line(line, task, nc=2)[1] == kind


def test_non_finite_coordinates_do_not_reach_geometry():
    assert parse_label_line("0 nan 0.5 0.2 0.2", "detect")[0] is None


def test_image_integrity_reports_all_bad_images(tmp_path):
    valid = tmp_path / "valid.jpg"
    Image.new("RGB", (32, 32), "white").save(valid)
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"this is not an image")
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(valid.read_bytes()[:-10])
    disappeared = tmp_path / "disappeared.jpg"
    snapshot = DatasetSnapshot(
        tmp_path / "data.yaml",
        images_per_split={
            "train": (valid, empty, fake, truncated, disappeared),
        },
    )
    result = get_check("image_integrity").func(CheckContext(snapshot.yaml_path, snapshot))
    assert result.severity == CheckSeverity.ERROR
    assert result.details["n_bad"] == 4
    assert result.details["total_images"] == 5
    assert {Path(item["image"]).name for item in result.details["bad_images_preview"]} == {
        "empty.jpg",
        "fake.png",
        "truncated.jpg",
        "disappeared.jpg",
    }


def test_undecodable_label_is_a_structured_error(tmp_path):
    label = tmp_path / "bad.txt"
    label.write_bytes(b"\xff\xfe\x80")
    snapshot = DatasetSnapshot(
        tmp_path / "data.yaml",
        nc=2,
        images_per_split={"train": ()},
        label_files_per_split={"train": (label,)},
    )
    result = get_check("label_format").func(CheckContext(snapshot.yaml_path, snapshot))
    assert result.severity == CheckSeverity.ERROR
    assert result.details["error_kinds"] == {"label_read_error": 1}
    assert result.details["n_read_errors"] == 1
    assert result.details["n_bad"] == 0


def test_validation_history_discovers_new_and_fixed_problem(demo_yaml, capsys):
    first = validate_dataset(demo_yaml, run_id="history-before", compare_previous=False)
    image = next(paths.dataset_processed_dir("demo").glob("images/train/*.jpg"))
    original = image.read_bytes()
    image.write_bytes(b"broken")
    try:
        broken = validate_dataset(demo_yaml, run_id="history-broken", show_progress=True)
    finally:
        image.write_bytes(original)
    assert broken.comparison["baseline_run_id"] == first.run_id
    assert "image_integrity" in {item["check"] for item in broken.comparison["new_issues"]}
    output = capsys.readouterr().err
    assert "D4 manifest SHA256" in output and "100%" in output
    fixed = validate_dataset(demo_yaml, run_id="history-fixed")
    assert fixed.comparison["baseline_run_id"] == broken.run_id
    assert "image_integrity" in {item["check"] for item in fixed.comparison["resolved_issues"]}
    report_dir = paths.validation_run_dir(fixed.run_id)
    saved = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["comparison"] == fixed.comparison
    assert (report_dir / "comparison.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_history_never_overwrites_or_crosses_dataset_task(demo_yaml):
    validate_dataset(demo_yaml, run_id="baseline", compare_previous=False)
    baseline = paths.validation_run_dir("baseline") / "report.json"
    before = baseline.read_bytes()
    with pytest.raises(ValueError, match="拒绝覆盖"):
        validate_dataset(demo_yaml, run_id="baseline")
    assert baseline.read_bytes() == before
    with pytest.raises(ValueError, match="task 不一致"):
        validate_dataset(demo_yaml, task_type="segment", compare_to=baseline, write_report=False)


def test_auto_history_skips_corrupt_reports_and_other_datasets(demo_yaml, tmp_path):
    reference = validate_dataset(demo_yaml, run_id="valid-reference", compare_previous=False)
    corrupt = paths.validation_run_dir("corrupt") / "report.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{", encoding="utf-8")
    foreign = reference.to_dict()
    foreign["dataset"]["yaml_path"] = str(tmp_path / "other.yaml")
    other = paths.validation_run_dir("foreign") / "report.json"
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps(foreign), encoding="utf-8")
    result = validate_dataset(demo_yaml, write_report=False)
    assert result.comparison["baseline_run_id"] == reference.run_id
    assert (
        validate_dataset(demo_yaml, write_report=False, compare_previous=False).comparison is None
    )
    with pytest.raises(ValueError, match="无法读取历史报告"):
        validate_dataset(demo_yaml, write_report=False, compare_to=corrupt)


def test_comparison_keeps_changed_counts_and_removed_checks_separate(tmp_path):
    def result(name, severity, count=1):
        return {
            "name": name,
            "severity": severity,
            "summary": "example",
            "details": {"n_bad": count},
        }

    old = {
        "run_id": "previous",
        "dataset": {"task": "detect"},
        "results": [
            result("fixed", "ERROR"),
            result("changed", "WARNING"),
            result("removed", "ERROR"),
            result("same", "INFO"),
        ],
    }
    current = {
        "run_id": "current",
        "dataset": old["dataset"],
        "results": [
            result("fixed", "PASS", 0),
            result("changed", "WARNING", 2),
            result("added", "ERROR"),
            result("same", "INFO"),
        ],
    }
    comparison = compare_reports(old, current, baseline=tmp_path / "previous.json")
    assert [item["check"] for item in comparison["resolved_issues"]] == ["fixed"]
    assert [item["check"] for item in comparison["new_issues"]] == ["added"]
    assert [item["check"] for item in comparison["changed_issues"]] == ["changed"]
    assert comparison["removed_checks"] == ["removed"]
    assert comparison["added_checks"] == ["added"]
    assert len(comparison["unchanged_issues"]) == 1


@pytest.mark.parametrize("bad_document", [[], {}, {"run_id": "x", "dataset": {}, "results": [{}]}])
def test_malformed_history_is_rejected(tmp_path, bad_document):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(bad_document), encoding="utf-8")
    with pytest.raises(ValueError, match="无法读取历史报告"):
        read_report(path)


def test_manifest_hash_and_move_detection(demo_yaml, tmp_path, monkeypatch):
    from od_platform.data_validation.checks import manifest_lineage

    doc = yaml.safe_load(demo_yaml.read_text(encoding="utf-8"))
    manifest_path = paths.ROOT_DIR / doc["odp_meta"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = next(sample for sample in manifest["samples"] if sample["split"] == "train")
    snapshot = build_snapshot(demo_yaml)
    label = next(
        label for label in snapshot.label_files_per_split["train"] if label.stem == sample["stem"]
    )
    assert manifest_lineage._sha256(label) == hashlib.sha256(label.read_bytes()).hexdigest()
    moved_snapshot = copy.deepcopy(snapshot)
    moved_snapshot.label_files_per_split["train"] = tuple(
        path for path in moved_snapshot.label_files_per_split["train"] if path != label
    )
    moved_snapshot.label_files_per_split["val"] += (label,)
    moved = get_check("manifest_lineage").func(CheckContext(demo_yaml, moved_snapshot))
    assert moved.details["n_moved"] == 1
    assert moved.details["n_changed"] == 0

    def unreadable(_path):
        raise PermissionError("permission denied")

    monkeypatch.setattr(manifest_lineage, "_sha256", unreadable)
    result = get_check("manifest_lineage").func(CheckContext(demo_yaml, snapshot))
    assert result.severity == CheckSeverity.ERROR
    assert result.details["n_unreadable"] == len(manifest["samples"])


def test_all_quality_checks_are_registered():
    assert len(list_check_names()) == 16
    assert {"image_integrity", "label_format", "bbox_within_image", "manifest_lineage"} <= set(
        list_check_names()
    )


@pytest.mark.parametrize("no_progress", [False, True])
def test_transform_cli_and_validation_exit_codes(demo_yaml, capsys, no_progress):
    from od_platform.cli.transform_data import main as transform
    from od_platform.cli.validate_data import main as validate

    arguments = ["--dataset", "demo", "--format", "pascal_voc"]
    if no_progress:
        arguments.append("--no-progress")
    assert transform(arguments) == 0
    output = capsys.readouterr().err
    assert ("D3 convert VOC" in output) is not no_progress
    arguments = [str(demo_yaml), "--no-report", "--no-compare"]
    if no_progress:
        arguments.append("--no-progress")
    assert validate(arguments) == 1  # 演示数据的刻意越界框为 WARNING。
    output = capsys.readouterr().err
    assert ("D4 manifest SHA256" in output) is not no_progress
    assert validate([str(demo_yaml), "--compare-to", "missing-report.json", "--no-progress"]) == 2


@pytest.mark.parametrize(
    "line",
    [
        "0 0.01 0.5 0.2 0.2",
        "0 0.99 0.5 0.2 0.2",
        "0 0.5 0.01 0.2 0.2",
        "0 0.5 0.99 0.2 0.2",
    ],
)
def test_bbox_edges_outside_image_are_reported(tmp_path, line):
    label = tmp_path / "edge.txt"
    label.write_text(line, encoding="utf-8")
    snapshot = DatasetSnapshot(
        tmp_path / "data.yaml",
        nc=2,
        images_per_split={"train": ()},
        label_files_per_split={"train": (label,)},
    )
    result = get_check("bbox_within_image").func(CheckContext(snapshot.yaml_path, snapshot))
    assert result.severity == CheckSeverity.ERROR
    assert result.details["n_out"] == 1
