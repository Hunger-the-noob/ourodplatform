"""新增：数据工程功能回归；全部测试使用临时数据与真实解码器。"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from PIL import Image

from od_platform.common import paths, stamp
from od_platform.common.progress import ConsoleProgress, emit_progress, progress_scope, track
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.data_validation.checks.label_format import _validate_one_line
from od_platform.data_validation.history import compare_reports, report_history
from od_platform.data_validation.registry import CheckContext, get_check
from od_platform.data_validation.service import validate_dataset
from od_platform.data_validation.snapshot import build_snapshot


@pytest.fixture
def dataset_yaml(tmp_path, monkeypatch):
    root = tmp_path / "processed"
    for kind in ("images", "labels"):
        (root / kind / "train").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32, 32), (20, 80, 120)).save(root / "images" / "train" / f"s{i}.jpg")
        (root / "labels" / "train" / f"s{i}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    config = tmp_path / "sample.yaml"
    config.write_text(
        yaml.safe_dump({"path": str(root), "train": "images/train", "nc": 1, "names": ["object"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "VALIDATION_RUNS_DIR", tmp_path / "validation")
    return config


def run_check(config, name):
    return get_check(name).func(CheckContext(config, build_snapshot(config)))


@pytest.mark.parametrize(
    "line,kind",
    [
        ("1 0.5 0.5 0.2 0.2", "class_id_out_of_range"),
        ("-1 0.5 0.5 0.2 0.2", "class_id_out_of_range"),
        ("0 1.1 0.5 0.2 0.2", "coord_out_of_range"),
        ("0 0.5 -0.1 0.2 0.2", "coord_out_of_range"),
        ("0 nan 0.5 0.2 0.2", "parse_error"),
        ("0 0.5 inf 0.2 0.2", "parse_error"),
        ("0 0.5 0.5 -inf 0.2", "parse_error"),
        ("0 0.5 0.5 0 0.2", "non_positive_box_size"),
        ("0.5 0.5 0.5 0.2 0.2", "parse_error"),
        ("0 0.5 0.5 0.2", "field_count_mismatch"),
    ],
)
def test_label_invalid_values(line, kind):
    assert _validate_one_line(line, "detect", 1)[1] == kind


def test_class_and_coordinate_boundaries():
    assert _validate_one_line("0 0.5 0.5 1 1", "detect", 1)[1] is None
    assert _validate_one_line("1 0 0 1 0 1 1", "segment", 2)[1] is None
    assert _validate_one_line("2 0 0 1 0 1 1", "segment", 2)[1] == "class_id_out_of_range"


@pytest.mark.parametrize("damage", [b"", b"not an image", b"\xff\xd8\xff\xe0"])
def test_image_integrity_detects_corruption(dataset_yaml, damage):
    snap = build_snapshot(dataset_yaml)
    snap.images_per_split["train"][0].write_bytes(damage)
    result = run_check(dataset_yaml, "image_integrity")
    assert result.severity == "ERROR"
    assert result.details["n_bad"] == 1
    assert len(result.issues) == 1


def test_image_integrity_decodes_pixels_and_reports_progress(dataset_yaml):
    events = []
    with progress_scope(events.append):
        result = run_check(dataset_yaml, "image_integrity")
    assert result.severity == "PASS"
    assert events[-1] == {"stage": "image_integrity", "completed": 3, "total": 3, "percent": 100.0}


def test_image_integrity_rejects_truncated_pixel_data(dataset_yaml):
    image = build_snapshot(dataset_yaml).images_per_split["train"][0]
    image.write_bytes(image.read_bytes()[:-20])
    assert run_check(dataset_yaml, "image_integrity").severity == "ERROR"


def test_unreadable_labels_do_not_count_as_fixed_issues():
    previous = {
        "run_id": "old",
        "results": [
            {
                "name": "label_format",
                "severity": "ERROR",
                "issues": [
                    {"split": "train", "file": "x.txt", "line_no": 1, "kind": "coord_out_of_range"}
                ],
            }
        ],
    }
    current = {
        "results": [
            {
                "name": "label_format",
                "severity": "ERROR",
                "issues": [{"split": "train", "file": "x.txt", "kind": "read_error"}],
            }
        ]
    }
    delta = compare_reports(current, previous)
    assert delta["resolved"] == []
    assert delta["new"][0]["kind"] == "read_error"


def test_history_reestablishes_baseline_for_older_check_granularity():
    previous = {
        "run_id": "old",
        "results": [{"name": "label_format", "severity": "ERROR", "summary": "bad"}],
    }
    current = {
        "results": [
            {
                "name": "label_format",
                "severity": "ERROR",
                "issues": [{"file": "x.txt", "kind": "coord_out_of_range"}],
            }
        ]
    }
    delta = compare_reports(current, previous)
    assert delta["new"] == delta["resolved"] == []


def test_unicode_label_read_error_is_a_finding(dataset_yaml):
    build_snapshot(dataset_yaml).label_files_per_split["train"][0].write_bytes(b"\xff\xfe")
    result = run_check(dataset_yaml, "label_format")
    assert result.severity == "ERROR"
    assert result.issues[0]["kind"] == "read_error"


def test_bbox_edges_can_overflow_with_normalized_coordinates(dataset_yaml):
    label = build_snapshot(dataset_yaml).label_files_per_split["train"][0]
    label.write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")
    assert run_check(dataset_yaml, "label_format").severity == "PASS"
    assert run_check(dataset_yaml, "bbox_within_image").details["n_out"] == 1


def test_history_tracks_new_and_fixed_images(dataset_yaml):
    first = validate_dataset(dataset_yaml)
    assert first.comparison["baseline_run_id"] is None
    image = build_snapshot(dataset_yaml).images_per_split["train"][0]
    original = image.read_bytes()
    image.write_bytes(b"broken")
    second = validate_dataset(dataset_yaml)
    assert second.comparison["baseline_run_id"] == first.run_id
    assert any(i["check"] == "image_integrity" for i in second.comparison["new"])
    image.write_bytes(original)
    third = validate_dataset(dataset_yaml)
    assert any(i["check"] == "image_integrity" for i in third.comparison["resolved"])
    reports = report_history(dataset_yaml)
    assert len({r["run_id"] for r in reports}) == 3
    assert reports[0]["run_id"] == third.run_id
    assert not list(paths.VALIDATION_RUNS_DIR.glob("*/*.tmp"))


def test_history_isolates_datasets_and_ignores_broken_json(dataset_yaml):
    first = validate_dataset(dataset_yaml)
    other = dataset_yaml.with_name("other.yaml")
    other.write_bytes(dataset_yaml.read_bytes())
    assert validate_dataset(other).comparison["baseline_run_id"] is None
    broken = paths.VALIDATION_RUNS_DIR / "broken" / "report.json"
    broken.parent.mkdir()
    broken.write_text("{", encoding="utf-8")
    assert report_history(dataset_yaml)[0]["run_id"] == first.run_id


def test_history_does_not_truncate_or_confuse_same_issue_counts():
    def report(start):
        return {
            "run_id": str(start),
            "results": [
                {
                    "name": "image_integrity",
                    "severity": "ERROR",
                    "issues": [
                        {"file": f"{i}.jpg", "kind": "decode_error"}
                        for i in range(start, start + 150)
                    ],
                }
            ],
        }

    delta = compare_reports(report(1), report(0))
    assert len(delta["new"]) == len(delta["resolved"]) == 1
    assert delta["persistent_count"] == 149


def test_crashed_check_is_not_reported_as_fixed():
    previous = {
        "run_id": "before",
        "results": [{"name": "test", "severity": "ERROR", "summary": "bad"}],
    }
    current = {
        "results": [
            {
                "name": "test",
                "severity": "ERROR",
                "summary": "crash",
                "details": {"exception": "error"},
            }
        ]
    }
    assert compare_reports(current, previous)["resolved"] == []


def test_no_report_and_duplicate_run_ids(dataset_yaml):
    validate_dataset(dataset_yaml, write_report=False)
    assert not paths.VALIDATION_RUNS_DIR.exists()
    validate_dataset(dataset_yaml, run_id="fixed")
    with pytest.raises(ValueError, match="已存在"):
        validate_dataset(dataset_yaml, run_id="fixed")
    with pytest.raises(ValueError, match="run_id"):
        validate_dataset(dataset_yaml, run_id="../outside")


def test_thread_progress_isolation_and_exception_cleanup():
    def worker(name):
        events = []
        with progress_scope(events.append):
            list(track(range(4), name))
        return events

    with ThreadPoolExecutor(max_workers=2) as executor:
        for name, events in zip(("one", "two"), executor.map(worker, ("one", "two")), strict=True):
            assert {e["stage"] for e in events} == {name}
            assert events[-1]["completed"] == 4
    events = []
    with pytest.raises(RuntimeError), progress_scope(events.append):
        raise RuntimeError("abort")
    emit_progress("outside")
    assert events == []


def test_console_progress_redirected_output():
    output = io.StringIO()
    ConsoleProgress(output)({"stage": "example", "completed": 3, "total": 3, "percent": 100})
    assert "3/3 100.0%" in output.getvalue()


def test_pipeline_to_validation_progress_and_reproducibility(tmp_path, monkeypatch):
    for key, value in {
        "ROOT_DIR": tmp_path,
        "RAW_DATA_DIR": tmp_path / "raw",
        "PROCESSED_DATA_DIR": tmp_path / "processed",
        "RUNS_DIR": tmp_path / "runs",
        "DATASET_CONFIGS_DIR": tmp_path / "configs",
        "VALIDATION_RUNS_DIR": tmp_path / "validation",
    }.items():
        monkeypatch.setattr(paths, key, value)
    monkeypatch.setattr(stamp, "LINEAGE_DIR", tmp_path / "lineage")
    raw = paths.raw_dataset_root("tiny")
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir()
    for i in range(12):
        Image.new("RGB", (32, 32)).save(raw / "images" / f"{i}.jpg")
        (raw / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    events = []
    first = DatasetPipeline("tiny", "yolo", classes=["item"], progress=events.append).run()
    second = DatasetPipeline("tiny", "yolo", classes=["item"]).run()
    assert sum(first["counts"].values()) == 12
    assert first["contract_fingerprint"] == second["contract_fingerprint"]
    assert any(e["stage"] == "convert:yolo" and e["completed"] == 12 for e in events)
    assert any(e["stage"].startswith("materialize:") for e in events)
    assert events[-1]["stage"] == "complete"
    events.clear()
    report = validate_dataset(first["yaml"], progress=events.append)
    assert report.overall_severity in {"PASS", "INFO"}
    assert any(e["stage"] == "manifest_lineage" and e["completed"] == 12 for e in events)
    assert json.loads(
        (paths.VALIDATION_RUNS_DIR / report.run_id / "report.json").read_text(encoding="utf-8")
    )
