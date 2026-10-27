"""只按数据集名称解析本仓库的数据，不接受浏览器传入任意路径。"""

import re
from pathlib import Path

import yaml

from od_platform.common import paths

DEFAULT_DATASET = "Rubbish_data_set"
NAME_RE = re.compile(r"[\w-]{1,80}", re.UNICODE)


def dataset_yaml(name: str) -> Path:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError("数据集名称只能包含文字、数字、下划线或短横线。")
    path = paths.dataset_yaml_path(name)
    path.resolve().relative_to(paths.DATASET_CONFIGS_DIR.resolve())
    raw = paths.raw_dataset_root(name)
    raw.resolve().relative_to(paths.RAW_DATA_DIR.resolve())
    return path


def read_yaml(name: str) -> dict:
    path = dataset_yaml(name)
    try:
        result = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return result if isinstance(result, dict) else {}


def class_names(data: dict) -> list[str]:
    names = data.get("names") or []
    if isinstance(names, dict):
        try:
            names = [value for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
        except (ValueError, TypeError):
            return []
    return [str(name) for name in names] if isinstance(names, list) else []


def data_root(name: str, data: dict) -> Path | None:
    value = data.get("path")
    if not isinstance(value, str) or not value:
        return None
    root = Path(value)
    return root.resolve() if root.is_absolute() else (dataset_yaml(name).parent / root).resolve()


def dataset_catalog() -> list[dict]:
    names = {path.stem for path in paths.DATASET_CONFIGS_DIR.glob("*.yaml")}
    if paths.RAW_DATA_DIR.exists():
        names.update(path.name for path in paths.RAW_DATA_DIR.iterdir() if path.is_dir())
    entries = []
    for name in sorted(names):
        try:
            config = dataset_yaml(name)
        except ValueError:
            continue
        data = read_yaml(name)
        meta = data.get("odp_meta") or {}
        split = meta.get("split") or {}
        raw = paths.raw_dataset_root(name)
        classes = class_names(data)
        if not classes:
            try:
                classes = [
                    line.strip()
                    for line in (raw / "classes.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeError):
                pass
        root = data_root(name, data)
        entries.append(
            {
                "name": name,
                "has_raw": (raw / "images").is_dir(),
                "has_config": config.is_file(),
                "has_processed": bool(root and root.is_dir()),
                "format": meta.get("source_format", "yolo"),
                "task": meta.get("task", "detect"),
                "classes": classes,
                "strategy": split.get("strategy", "random"),
                "seed": split.get("seed", 42),
                "rates": split.get("rates", {"train": 0.8, "val": 0.1}),
            }
        )
    return entries
