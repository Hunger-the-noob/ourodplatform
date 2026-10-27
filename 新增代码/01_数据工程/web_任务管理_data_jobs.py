"""本地 Web 数据任务：子进程执行，轮询结构化进度，质检失败仍返回完整报告。"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from uuid import uuid4

from django.conf import settings

from od_platform.common import paths

from .data_catalog import dataset_yaml
from .data_reports import get_report, report_path


class DataJobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._job = None
        self._process = None
        self._log = None

    def _status_locked(self) -> dict:
        if self._job is None:
            return {"status": "idle", "progress": 0, "logs": []}
        snapshot = dict(self._job["initial"])
        state_path = self._job["directory"] / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                snapshot.update(state)
        except (OSError, UnicodeError, ValueError):
            pass
        code = self._process.poll()
        if code is not None:
            self._log.close()
            if snapshot["status"] == "running":
                snapshot.update(
                    status="failed", error=f"数据任务意外退出（退出码 {code}），请查看日志。"
                )
        try:
            with (self._job["directory"] / "worker.log").open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 12000))
                snapshot["logs"] = (
                    stream.read().decode("utf-8", errors="replace").splitlines()[-20:]
                )
        except OSError:
            snapshot["logs"] = []
        return snapshot

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def start(self, action: str, options: dict) -> dict:
        with self._lock:
            if self._status_locked()["status"] == "running":
                raise RuntimeError("已有数据任务正在执行，请等待完成。")
            name = options["dataset"]
            config = dataset_yaml(name)
            if action == "validate" and not config.is_file():
                raise ValueError("尚无数据配置，请先运行数据流水线。")
            if action == "transform" and not (paths.raw_dataset_root(name) / "images").is_dir():
                raise ValueError("本机缺少原始图片目录，请先准备 data/raw 下的数据集。")
            baseline = options.get("baseline_run_id")
            if baseline:
                report = get_report(name, baseline)
                if not report["baseline_compatible"]:
                    raise ValueError("旧版报告缺少数据身份，不能用作对比基线。")
                options = dict(options, compare_to=str(report_path(baseline)))
            job_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + uuid4().hex[:6]
            directory = paths.RUNS_DIR / "web_data" / job_id
            directory.mkdir(parents=True, exist_ok=False)
            initial = {
                "id": job_id,
                "action": action,
                "dataset": name,
                "status": "running",
                "stage": "starting",
                "progress": 0,
                "progress_event": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
                "result": None,
            }
            payload = {"initial": initial, "options": options, "project_root": str(paths.ROOT_DIR)}
            (directory / "request.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    [str(settings.PLATFORM_SRC), str(settings.BASE_DIR), env.get("PYTHONPATH")],
                )
            )
            log = (directory / "worker.log").open("wb")
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "defect.data_worker", str(directory)],
                    cwd=paths.ROOT_DIR,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except OSError:
                log.close()
                raise
            self._job, self._process, self._log = (
                {"initial": initial, "directory": directory},
                process,
                log,
            )
            return dict(initial, logs=[])


data_jobs = DataJobManager()
