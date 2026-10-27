from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml
from django.conf import settings

PROJECT_ROOT = settings.BASE_DIR.parents[1]
TRAINING_RUNS_DIR = PROJECT_ROOT / "runs" / "training"
TRAINED_MODELS_DIR = PROJECT_ROOT / "models" / "trained"
DATASET_CONFIGS_DIR = PROJECT_ROOT / "apps" / "platform" / "configs" / "datasets"
VALIDATION_RUNS_DIR = PROJECT_ROOT / "runs" / "data_validation"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_training_run() -> Path | None:
    if not TRAINING_RUNS_DIR.exists():
        return None
    completed = [
        path
        for path in TRAINING_RUNS_DIR.iterdir()
        if path.is_dir() and (path / "odp_audit.json").exists()
    ]
    candidates = completed or [
        path
        for path in TRAINING_RUNS_DIR.iterdir()
        if path.is_dir()
        and (
            (path / "results.csv").exists()
            or (path / "args.yaml").exists()
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_results_row(run_dir: Path) -> dict[str, float]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return {}
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError:
        return {}
    if not rows:
        return {}
    cleaned: dict[str, float] = {}
    for raw_key, raw_value in rows[-1].items():
        if raw_key is None or raw_value in (None, ""):
            continue
        try:
            cleaned[raw_key.strip()] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return cleaned


def _results_history(run_dir: Path) -> list[dict[str, Any]]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return []
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError:
        return []
    history: list[dict[str, Any]] = []
    for row in rows:
        cleaned: dict[str, float] = {}
        for raw_key, raw_value in row.items():
            if raw_key is None or raw_value in (None, ""):
                continue
            try:
                cleaned[raw_key.strip()] = float(raw_value)
            except (TypeError, ValueError):
                continue
        history.append(
            {
                "epoch": int(cleaned.get("epoch", len(history) + 1)),
                "precision": cleaned.get("metrics/precision(B)"),
                "recall": cleaned.get("metrics/recall(B)"),
                "map50": cleaned.get("metrics/mAP50(B)"),
                "map50_95": cleaned.get("metrics/mAP50-95(B)"),
                "train_box_loss": cleaned.get("train/box_loss"),
                "train_cls_loss": cleaned.get("train/cls_loss"),
                "train_dfl_loss": cleaned.get("train/dfl_loss"),
                "val_box_loss": cleaned.get("val/box_loss"),
                "val_cls_loss": cleaned.get("val/cls_loss"),
                "val_dfl_loss": cleaned.get("val/dfl_loss"),
            }
        )
    return history


def _metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return round(float(value), 5)
    return None


def _latest_validation(dataset_name: str) -> dict[str, Any]:
    if not VALIDATION_RUNS_DIR.exists():
        return {}
    candidates = sorted(
        VALIDATION_RUNS_DIR.glob("*/report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for report_path in candidates:
        report = _read_json(report_path)
        yaml_path = str(report.get("yaml_path", "")).replace("\\", "/")
        if dataset_name not in yaml_path:
            continue
        return {
            "run_id": report.get("run_id"),
            "severity": report.get("overall_severity"),
            "duration_seconds": report.get("duration_seconds"),
        }
    return {}


def get_model_overview() -> dict[str, Any]:
    run_dir = _latest_training_run()
    if run_dir is None:
        return {
            "status": "not_found",
            "message": "尚未找到训练结果。",
        }

    audit = _read_json(run_dir / "odp_audit.json")
    args = _read_yaml(run_dir / "args.yaml")
    latest_row = _latest_results_row(run_dir)
    history = _results_history(run_dir)
    completed = bool(audit)

    data_yaml_path = Path(str(audit.get("data_yaml", args.get("data", ""))))
    data_yaml = _read_yaml(data_yaml_path)
    metadata = data_yaml.get("odp_meta", {}) or {}
    split = metadata.get("split", {}) or {}
    dataset_name = metadata.get("dataset_name") or data_yaml_path.stem or "未命名数据集"
    names = data_yaml.get("names", {}) or {}
    if isinstance(names, list):
        names = dict(enumerate(names))

    audit_metrics = (audit.get("metrics", {}) or {}).get("overall", {}) or {}
    class_maps = (audit.get("metrics", {}) or {}).get("class_map_50_95", {}) or {}
    speed_ms = (audit.get("metrics", {}) or {}).get("speed_ms", {}) or {}
    config_values = (audit.get("config", {}) or {}).get("values", {}) or {}

    best_path = run_dir / "weights" / "best.pt"
    archived_weights = sorted(
        TRAINED_MODELS_DIR.glob("*-best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if TRAINED_MODELS_DIR.exists() else []
    archived_best = archived_weights[0] if archived_weights else None

    curve_path = run_dir / "results.png"
    return {
        "status": "completed" if completed else "training",
        "run_id": audit.get("run_id") or args.get("name") or run_dir.name,
        "experiment_name": audit.get("experiment_name") or args.get("name"),
        "created_at": audit.get("created_at"),
        "dataset": {
            "name": dataset_name,
            "classes": [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))],
            "class_count": int(data_yaml.get("nc", len(names)) or len(names)),
            "splits": split.get("counts", {}),
            "strategy": split.get("strategy"),
            "seed": split.get("seed"),
            "validation": _latest_validation(dataset_name),
        },
        "training": {
            "epochs": config_values.get("epochs", args.get("epochs")),
            "imgsz": config_values.get("imgsz", args.get("imgsz")),
            "batch": config_values.get("batch", args.get("batch")),
            "device": config_values.get("device", args.get("device")),
            "base_model": config_values.get("model", args.get("model")),
        },
        "metrics": {
            "precision": _metric(audit_metrics, "metrics/precision(B)", "precision"),
            "recall": _metric(audit_metrics, "metrics/recall(B)", "recall"),
            "map50": _metric(audit_metrics, "metrics/mAP50(B)", "mAP50"),
            "map50_95": _metric(audit_metrics, "metrics/mAP50-95(B)", "mAP50-95"),
            "fitness": _metric(audit_metrics, "fitness"),
        },
        "loss": {
            key.split("/", 1)[-1]: round(value, 5)
            for key, value in latest_row.items()
            if key.startswith("train/") and key.endswith("_loss")
        },
        "speed_ms": {
            key: round(float(value), 3)
            for key, value in speed_ms.items()
            if isinstance(value, (int, float))
        },
        "class_map_50_95": class_maps,
        "metrics_history": history,
        "evaluation": _latest_evaluation(),
        "weights": {
            "run_best": best_path.name if best_path.exists() else None,
            "archived": archived_best.name if archived_best else None,
            "size_mb": round(archived_best.stat().st_size / 1024 / 1024, 2)
            if archived_best
            else None,
        },
        "curve_url": "/api/model/training-curve/" if curve_path.exists() else None,
        "run_dir_name": run_dir.name,
    }


def latest_training_curve() -> Path | None:
    run_dir = _latest_training_run()
    if run_dir is None:
        return None
    curve = run_dir / "results.png"
    return curve if curve.exists() else None


def _latest_evaluation() -> dict[str, Any] | None:
    from .evaluation import latest_evaluation

    return latest_evaluation()
