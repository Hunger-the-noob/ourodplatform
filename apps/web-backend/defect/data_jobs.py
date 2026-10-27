"""新增：D3/D4 后台任务。与现有训练管理器一样用于单进程 Web 服务。"""

from __future__ import annotations

import copy
import re
import threading
from datetime import datetime
from uuid import uuid4

import yaml

from od_platform.common import paths
from od_platform.common.constants import AnnotationFormat, SplitStrategy, Task
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.data_validation.history import report_history
from od_platform.data_validation.registry import list_check_names
from od_platform.data_validation.service import validate_dataset


def dataset_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]+", value) or value.startswith("."):
        raise ValueError("数据集名称只能包含文字、数字、下划线和连字符")
    return value


def engineering_catalog() -> dict:
    names = {p.stem for p in paths.DATASET_CONFIGS_DIR.glob("*.yaml")}
    names.update(p.name for p in paths.RAW_DATA_DIR.glob("*") if p.is_dir())
    datasets = []
    for name in sorted(names):
        try:
            dataset_name(name)
            document = (
                yaml.safe_load(paths.dataset_yaml_path(name).read_text(encoding="utf-8")) or {}
            )
        except (OSError, ValueError, yaml.YAMLError):
            document = {}
        classes = document.get("names", []) if isinstance(document, dict) else []
        if isinstance(classes, dict):
            classes = [v for _, v in sorted(classes.items(), key=lambda item: int(item[0]))]
        metadata = document.get("odp_meta", {}) if isinstance(document, dict) else {}
        datasets.append(
            {
                "name": name,
                "has_raw": paths.raw_dataset_root(name).is_dir(),
                "has_yaml": paths.dataset_yaml_path(name).is_file(),
                "classes": classes,
                "format": metadata.get("source_format", "yolo"),
                "task": metadata.get("task", "detect"),
            }
        )
    return {
        "datasets": datasets,
        "checks": list_check_names(),
        "formats": list(AnnotationFormat.all()),
        "strategies": list(SplitStrategy.all()),
    }


class DataJobManager:
    """新增：互斥执行转换/质检，轮询不会重读整个数据集或阻塞请求。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._state = {"status": "idle", "progress": {}, "result": None, "error": None}

    def status(self) -> dict:
        with self._lock:
            state = copy.deepcopy(self._state)
            if state.get("operation") == "validate" and state.get("result"):
                state["result"] = _public_report(state["result"])
            return state

    def start(self, operation: str, dataset: str, **options) -> dict:
        dataset_name(dataset)
        if operation not in {"transform", "validate"}:
            raise ValueError("operation 只能是 transform 或 validate")
        if operation == "validate" and not paths.dataset_yaml_path(dataset).is_file():
            raise ValueError("找不到数据集 YAML，请先运行 D3 转换")
        if operation == "transform":
            if not paths.raw_dataset_root(dataset).is_dir():
                raise ValueError("找不到原始数据目录 data/raw/<dataset>")
            if options.get("format", "yolo") not in AnnotationFormat.all():
                raise ValueError("不支持的标注格式")
            if options.get("task", "detect") not in Task.all():
                raise ValueError("不支持的任务类型")
            if options.get("split_strategy", "random") not in SplitStrategy.all():
                raise ValueError("不支持的划分策略")
            train, val = float(options.get("train_rate", 0.8)), float(options.get("val_rate", 0.1))
            if not (0 < train < 1 and 0 <= val < 1 and train + val < 1):
                raise ValueError("比例必须满足 0 < train < 1、val >= 0、train + val < 1")
            options["train_rate"], options["val_rate"] = train, val
            options["seed"] = int(options.get("seed", 42))
            classes = options.get("classes")
            if classes is not None and (
                not isinstance(classes, list)
                or not all(isinstance(c, str) and c.strip() for c in classes)
            ):
                raise ValueError("classes 应为非空类别名组成的数组")
            if options.get("format", "yolo") == "yolo" and not classes:
                raise ValueError("YOLO 转换必须填写类别顺序")
        with self._lock:
            if self._state["status"] == "running":
                raise RuntimeError("已有数据任务正在执行，请等待完成")
            self._state = {
                "id": uuid4().hex,
                "status": "running",
                "operation": operation,
                "dataset": dataset,
                "started_at": datetime.now().isoformat(),
                "progress": {"stage": "starting", "completed": 0, "total": 0, "percent": 0},
                "result": None,
                "error": None,
            }
            thread = threading.Thread(
                target=self._execute, args=(operation, dataset, options), daemon=True
            )
            thread.start()
            return self.status()

    def _progress(self, event: dict):
        with self._lock:
            self._state["progress"] = event

    def _execute(self, operation: str, dataset: str, options: dict):
        try:
            if operation == "transform":
                result = DatasetPipeline(
                    dataset,
                    options.get("format", "yolo"),
                    task=options.get("task", "detect"),
                    classes=options.get("classes"),
                    train_rate=options["train_rate"],
                    val_rate=options["val_rate"],
                    random_state=options["seed"],
                    split_strategy=options.get("split_strategy", "random"),
                    progress=self._progress,
                ).run()
            else:
                result = validate_dataset(
                    paths.dataset_yaml_path(dataset), progress=self._progress
                ).to_dict()
            with self._lock:
                # 质检 ERROR 是已完成的质检结论；执行异常才是任务失败。
                self._state.update(
                    status="completed", result=result, finished_at=datetime.now().isoformat()
                )
        except Exception as exc:  # 后台线程的边界护栏：把错误返回页面。
            with self._lock:
                self._state.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=datetime.now().isoformat(),
                )


def _public_report(report: dict) -> dict:
    """完整问题表留在磁盘，页面只显示有界预览及准确总数。"""
    report = copy.deepcopy(report)
    for result in report["results"]:
        result.pop("issues", None)
    comparison = report.get("comparison", {})
    for key in ("new", "resolved"):
        comparison[f"{key}_count"] = len(comparison.get(key, []))
        comparison[key] = comparison.get(key, [])[:100]
    return report


def dataset_history(name: str) -> dict:
    name = dataset_name(name)
    reports = report_history(paths.dataset_yaml_path(name))
    latest = _public_report(reports[0]) if reports else None
    return {
        "latest": latest,
        "reports": [
            {
                "run_id": r["run_id"],
                "severity": r["overall_severity"],
                "duration_seconds": r.get("duration_seconds"),
                "n_checks": len(r["results"]),
            }
            for r in reports[:50]
        ],
    }


data_jobs = DataJobManager()
