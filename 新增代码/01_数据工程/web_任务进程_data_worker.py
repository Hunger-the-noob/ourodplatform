"""独立执行 D3/D4，原子写入真实阶段进度；不对终端进度条做文本猜测。"""

import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from od_platform.common import paths, stamp
from od_platform.common.progress import progress_events
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.data_validation.service import validate_dataset


def write_state(directory: Path, state: dict) -> None:
    temporary = directory / "state.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    # Windows 轮询读取可能短暂占用目标文件，重试替换而不打断数据处理。
    for attempt in range(10):
        try:
            temporary.replace(directory / "state.json")
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.01 * (attempt + 1))


def configure_workspace(root: Path) -> None:
    """子进程复用启动方工作区，测试也能完全隔离真实数据。"""
    paths.ROOT_DIR = root
    paths.RAW_DATA_DIR = root / "data" / "raw"
    paths.PROCESSED_DATA_DIR = root / "data" / "processed"
    paths.DATASET_CONFIGS_DIR = root / "apps" / "platform" / "configs" / "datasets"
    paths.RUNS_DIR = root / "runs"
    paths.VALIDATION_RUNS_DIR = paths.RUNS_DIR / "data_validation"
    stamp.LINEAGE_DIR = paths.RUNS_DIR / "lineage"


def execute(directory: Path) -> int:
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    state = request["initial"]
    options = request["options"]
    configure_workspace(Path(request["project_root"]))

    def save():
        write_state(directory, state)

    def progress(event):
        state.update(stage=event["description"], progress=event["percent"], progress_event=event)
        save()

    save()
    try:
        with progress_events(progress):
            if state["action"] == "transform":
                result = DatasetPipeline(
                    options["dataset"],
                    options["format"],
                    task=options["task"],
                    train_rate=options["train_rate"],
                    val_rate=options["val_rate"],
                    classes=options.get("classes"),
                    random_state=options["seed"],
                    split_strategy=options["split_strategy"],
                ).run()
            else:
                report = validate_dataset(
                    paths.dataset_yaml_path(options["dataset"]),
                    run_id=state["id"],
                    compare_to=Path(options["compare_to"]) if options.get("compare_to") else None,
                    compare_previous=options.get("compare_previous", True),
                )
                result = {
                    "run_id": report.run_id,
                    "severity": report.overall_severity,
                    "n_checks": len(report.results),
                    "comparison": report.comparison,
                }
        state.update(status="completed", stage="completed", progress=100, result=result)
    except Exception as error:
        traceback.print_exc()
        state.update(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(execute(Path(sys.argv[1])))
