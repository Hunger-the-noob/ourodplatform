from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from django.conf import settings

from od_platform.common.constants import IMAGE_EXTENSIONS
from od_platform.data_validation.labels import parse_label_line

# 新增：按名称选取本机数据配置，并读取与当前配置匹配的质检报告。
from .data_catalog import dataset_yaml
from .data_reports import latest_validation as _latest_validation

PROJECT_ROOT = settings.BASE_DIR.parents[1]
DATASET_NAME = "Rubbish_data_set"
DATASET_YAML = PROJECT_ROOT / "apps" / "platform" / "configs" / "datasets" / f"{DATASET_NAME}.yaml"


def _read_dataset_yaml(name: str | None = None) -> dict[str, Any]:
    # 新增：不传参仍读取原 DATASET_YAML；指定名称时读取对应配置。
    yaml_path = DATASET_YAML if name is None else dataset_yaml(name)
    if not yaml_path.exists():
        return {}
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def get_dataset_understanding(name: str | None = None) -> dict[str, Any]:
    # 新增：原统计入口保留，在原有流程中增加数据集选择。
    yaml_path = DATASET_YAML if name is None else dataset_yaml(name)
    data = _read_dataset_yaml(name)
    name = DATASET_NAME if name is None else name
    metadata = data.get("odp_meta", {}) or {}
    split_meta = metadata.get("split", {}) or {}
    names = data.get("names", {}) or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    classes = [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]

    raw_path = data.get("path")
    root = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
    if root is not None and not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    class_counts = Counter()
    samples: list[dict[str, str]] = []
    split_stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images_dir = root / "images" / split if root else None
        labels_dir = root / "labels" / split if root else None
        # 新增：展示本机真实图片，不把配置中的历史数量当作现有数据。
        image_files = (
            sorted(
                p
                for p in images_dir.glob("*.*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if images_dir and images_dir.is_dir()
            else []
        )
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir and labels_dir.is_dir() else []
        split_class_counts = Counter()
        box_count = 0
        for label_file in label_files:
            try:
                lines = label_file.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line in lines:
                # 新增：统一解析标签，避免非法结构及 NaN/Inf 进入概览统计。
                box, _kind = parse_label_line(line, metadata.get("task", "detect"))
                if box is None:
                    continue
                class_id = box[0]
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
            # 新增：文件修复或重建后刷新样本图片，避免沿用旧的坏图缓存。
            try:
                version = image_file.stat().st_mtime_ns
            except OSError:
                continue
            samples.append(
                {
                    "split": split,
                    "name": image_file.name,
                    "url": (
                        f"/api/data/sample/{split}/{quote(image_file.name)}?dataset={quote(name)}&v={version}"
                    ),
                }
            )

    return {
        "name": name,
        "classes": classes,
        "class_count": int(data.get("nc", len(classes)) or len(classes)),
        "splits": {
            split: {
                "images": split_stats[split]["images"],
                "boxes": split_stats[split]["boxes"],
            }
            for split in ("train", "val", "test")
        },
        "class_counts": dict(class_counts),
        "split_class_counts": {
            split: split_stats[split]["class_counts"] for split in ("train", "val", "test")
        },
        "total_images": sum(value["images"] for value in split_stats.values()),
        "total_boxes": sum(value["boxes"] for value in split_stats.values()),
        "strategy": split_meta.get("strategy"),
        "seed": split_meta.get("seed"),
        "validation": _latest_validation(name),
        "samples": samples,
        "yaml_path": str(yaml_path),
        "available": bool(root and root.is_dir()),
        "message": ""
        if root and root.is_dir()
        else "本机尚未找到配置指向的成品数据，请准备原始数据后运行流水线。",
    }


def dataset_sample_path(split: str, filename: str, name: str | None = None) -> Path | None:
    if split not in {"train", "val", "test"} or Path(filename).name != filename:
        return None
    yaml_path = DATASET_YAML if name is None else dataset_yaml(name)
    data = _read_dataset_yaml(name)
    raw_path = data.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    root = Path(raw_path)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    candidate = root / "images" / split / Path(filename).name
    resolved_root = (root / "images" / split).resolve()
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None
