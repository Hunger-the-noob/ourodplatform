from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from django.conf import settings

from .recognition import latest_trained_weight

PROJECT_ROOT = settings.BASE_DIR.parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "runs" / "model_evaluation"
DATASET_YAML = (
    PROJECT_ROOT
    / "apps"
    / "platform"
    / "configs"
    / "datasets"
    / "Rubbish_data_set.yaml"
)


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return round(float(value), 5) if isinstance(value, (int, float)) else None


def latest_evaluation() -> dict[str, Any] | None:
    if not EVALUATION_ROOT.exists():
        return None
    candidates = sorted(
        EVALUATION_ROOT.glob("*/eval.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    payload["run_dir"] = str(candidates[0].parent.name)
    payload["artifacts"] = {
        name: f"/api/model/evaluation-artifact/{name}/"
        for name in (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "BoxF1_curve.png",
        )
        if (candidates[0].parent / name).exists()
    }
    return payload


def run_test_evaluation(*, imgsz: int = 416, device: str = "cpu") -> dict[str, Any]:
    from ultralytics import YOLO

    if not DATASET_YAML.exists():
        raise FileNotFoundError("未找到 Rubbish_data_set 数据集配置。")
    weight = latest_trained_weight()
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = EVALUATION_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weight))
    results = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=imgsz,
        device=device,
        plots=True,
        project=str(EVALUATION_ROOT),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )
    results_dict = getattr(results, "results_dict", {}) or {}
    names = getattr(results, "names", {}) or {}
    maps = getattr(getattr(results, "box", None), "maps", None)
    class_maps: dict[str, float] = {}
    size = int(getattr(maps, "size", 0) or 0)
    if size:
        for index, name in names.items():
            if int(index) < size:
                class_maps[str(name)] = round(float(maps[index]), 5)

    payload = {
        "run_id": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "weight": weight.name,
        "split": "test",
        "imgsz": imgsz,
        "device": device,
        "metrics": {
            "precision": _metric(results_dict, "metrics/precision(B)"),
            "recall": _metric(results_dict, "metrics/recall(B)"),
            "map50": _metric(results_dict, "metrics/mAP50(B)"),
            "map50_95": _metric(results_dict, "metrics/mAP50-95(B)"),
        },
        "class_map_50_95": class_maps,
    }
    (run_dir / "eval.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return latest_evaluation() or payload


def evaluation_artifact(name: str) -> Path | None:
    if name not in {
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
    }:
        return None
    if not EVALUATION_ROOT.exists():
        return None
    candidates = sorted(
        [path for path in EVALUATION_ROOT.glob(f"*/{name}") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
