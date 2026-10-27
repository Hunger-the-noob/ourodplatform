from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from django.conf import settings

from .model_info import _latest_validation

PROJECT_ROOT = settings.BASE_DIR.parents[1]
DATASET_NAME = "Rubbish_data_set"
DATASET_YAML = (
    PROJECT_ROOT / "apps" / "platform" / "configs" / "datasets"
    / f"{DATASET_NAME}.yaml"
)


def _read_dataset_yaml() -> dict[str, Any]:
    if not DATASET_YAML.exists():
        return {}
    try:
        data = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def get_dataset_understanding() -> dict[str, Any]:
    data = _read_dataset_yaml()
    metadata = data.get("odp_meta", {}) or {}
    split_meta = metadata.get("split", {}) or {}
    counts = split_meta.get("counts", {}) or {}
    names = data.get("names", {}) or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    classes = [
        str(value)
        for _, value in sorted(names.items(), key=lambda item: int(item[0]))
    ]

    root = Path(data.get("path", ""))
    class_counts = Counter()
    samples: list[dict[str, str]] = []
    split_stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        image_files = sorted(images_dir.glob("*.*")) if images_dir.exists() else []
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        split_class_counts = Counter()
        box_count = 0
        for label_file in label_files:
            try:
                lines = label_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                box_count += 1
                class_name = str(names.get(class_id, f"class_{class_id}"))
                class_counts[class_name] += 1
                split_class_counts[class_name] += 1
        split_stats[split] = {
            "images": len(image_files),
            "boxes": box_count,
            "class_counts": dict(split_class_counts),
        }
        for image_file in image_files[:3]:
            samples.append(
                {
                    "split": split,
                    "name": image_file.name,
                    "url": (
                        f"/api/data/sample/{split}/{image_file.name}"
                    ),
                }
            )

    return {
        "name": DATASET_NAME,
        "classes": classes,
        "class_count": int(data.get("nc", len(classes)) or len(classes)),
        "splits": {
            split: {
                "images": counts.get(split, split_stats[split]["images"]),
                "boxes": split_stats[split]["boxes"],
            }
            for split in ("train", "val", "test")
        },
        "class_counts": dict(class_counts),
        "split_class_counts": {
            split: split_stats[split]["class_counts"]
            for split in ("train", "val", "test")
        },
        "total_images": sum(value["images"] for value in split_stats.values()),
        "total_boxes": sum(value["boxes"] for value in split_stats.values()),
        "strategy": split_meta.get("strategy"),
        "seed": split_meta.get("seed"),
        "validation": _latest_validation(DATASET_NAME),
        "samples": samples,
        "yaml_path": str(DATASET_YAML),
    }


def dataset_sample_path(split: str, filename: str) -> Path | None:
    if split not in {"train", "val", "test"}:
        return None
    data = _read_dataset_yaml()
    root = Path(data.get("path", ""))
    candidate = root / "images" / split / Path(filename).name
    resolved_root = (root / "images" / split).resolve()
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None
