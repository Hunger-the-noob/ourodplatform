from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from django.conf import settings

PROJECT_ROOT = settings.BASE_DIR.parents[1]
WEB_TRAINING_RUNS = PROJECT_ROOT / "runs" / "web_training"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s+[^\r\n]*?(?:it/s|s/it)")
EPOCH_RE = re.compile(r"(?m)^\s*(\d+)/(\d+)\s+0G\s+")


@dataclass
class TrainingJob:
    id: str
    epochs: int
    imgsz: int
    batch: str
    device: str
    command: list[str]
    log_path: Path
    process: subprocess.Popen
    log_handle: Any
    status: str = "running"
    stage: str = "starting"
    progress: float = 0.0
    current_epoch: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    error: str | None = None


class TrainingJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: TrainingJob | None = None

    def start(
        self,
        *,
        epochs: int,
        imgsz: int,
        batch: str,
        device: str,
    ) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if self._job and self._job.status == "running":
                raise RuntimeError("已有训练任务正在运行。")
            if not 1 <= epochs <= 200:
                raise ValueError("训练轮数必须在 1 到 200 之间。")
            if not 128 <= imgsz <= 1536:
                raise ValueError("图像尺寸必须在 128 到 1536 之间。")
            if not re.fullmatch(r"[1-9]\d*|0?\.\d+", batch):
                raise ValueError("Batch 必须是正整数或 0 到 1 的显存比例。")
            if device not in {"cpu", "0", "0,1"}:
                raise ValueError("设备仅支持 cpu、0 或 0,1。")

            model_path = PROJECT_ROOT / "models" / "pretrained" / "yolov8n.pt"
            if not model_path.exists():
                raise FileNotFoundError("缺少 models/pretrained/yolov8n.pt。")
            WEB_TRAINING_RUNS.mkdir(parents=True, exist_ok=True)
            job_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            log_path = WEB_TRAINING_RUNS / f"{job_id}.log"
            command = [
                sys.executable,
                "-m",
                "od_platform.cli.train_model",
                "--config",
                "train",
                "--data",
                "Rubbish_data_set",
                "--model",
                str(model_path),
                "--epochs",
                str(epochs),
                "--batch",
                batch,
                "--imgsz",
                str(imgsz),
                "--device",
                device,
                "--experiment-name",
                "web_training",
            ]
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "KMP_DUPLICATE_LIB_OK": "TRUE",
                }
            )
            log_handle = log_path.open("w", encoding="utf-8", buffering=1)
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            )
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
            )
            self._job = TrainingJob(
                id=job_id,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                command=command,
                log_path=log_path,
                process=process,
                log_handle=log_handle,
            )
            return self._snapshot_locked(self._job)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if self._job is None:
                return {"status": "idle", "progress": 0, "logs": []}
            return self._snapshot_locked(self._job)

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if self._job is None or self._job.status != "running":
                if self._job is None:
                    return {"status": "idle", "progress": 0, "logs": []}
                return self._snapshot_locked(self._job)
            self._job.process.terminate()
            try:
                self._job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._job.process.kill()
            self._job.status = "cancelled"
            self._job.stage = "cancelled"
            self._job.finished_at = datetime.now().isoformat(timespec="seconds")
            self._job.log_handle.close()
            return self._snapshot_locked(self._job)

    def _refresh_locked(self) -> None:
        job = self._job
        if job is None or job.status != "running":
            return
        return_code = job.process.poll()
        if return_code is not None:
            job.finished_at = datetime.now().isoformat(timespec="seconds")
            if return_code == 0:
                job.status = "completed"
                job.stage = "completed"
                job.progress = 100.0
            else:
                job.status = "failed"
                job.stage = "failed"
                job.error = f"训练进程退出码: {return_code}"
            job.log_handle.close()

    def _snapshot_locked(self, job: TrainingJob) -> dict[str, Any]:
        text = ""
        if job.log_path.exists():
            try:
                text = job.log_path.read_text(encoding="utf-8", errors="replace")[-500_000:]
            except OSError:
                pass
        clean = ANSI_RE.sub("", text).replace("\r", "\n")
        log_lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if job.status == "running":
            epoch_matches = EPOCH_RE.findall(clean)
            if epoch_matches:
                job.current_epoch = int(epoch_matches[-1][0])
                total_epochs = max(int(epoch_matches[-1][1]), 1)
                job.epochs = total_epochs
            progress_matches = PROGRESS_RE.findall(clean)
            if progress_matches:
                done, total = map(int, progress_matches[-1])
                batch_ratio = min(done / max(total, 1), 1.0)
                epoch_ratio = (
                    (job.current_epoch - 1) + batch_ratio
                ) / max(job.epochs, 1)
                job.progress = round(8 + min(epoch_ratio, 1.0) * 82, 1)
                job.stage = "training"
            elif "Scanning " in clean or "Model summary:" in clean:
                job.stage = "preparing"
                job.progress = max(job.progress, 3.0)
            if "Starting training for" in clean:
                job.stage = "training"
                job.progress = max(job.progress, 5.0)
            if "Validating " in clean:
                job.stage = "evaluating"
                job.progress = max(job.progress, 92.0)
            if "训练完成 |" in clean:
                job.stage = "finalizing"
                job.progress = max(job.progress, 97.0)
        if job.status == "completed":
            job.progress = 100.0
        return {
            "id": job.id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs": job.epochs,
            "imgsz": job.imgsz,
            "batch": job.batch,
            "device": job.device,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "logs": log_lines[-24:],
        }


training_jobs = TrainingJobManager()
