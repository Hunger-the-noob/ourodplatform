"""视频分析调度：独立进程跑 YOLO，主进程负责进度、落库与大模型解释。"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from . import explain, llm, prompts
from .models import VideoAnalysisJob

logger = logging.getLogger(__name__)

EVENT_PREFIX = "@@ODP@@"
WORKER_PATH = Path(__file__).resolve().parent / "yolo_worker.py"

# 同一时刻只跑一个推理进程：CPU 上并发只会互相抢核，队列化反而更快。
_JOB_LOCK = threading.Lock()
_PYTHON_CACHE: dict[str, dict[str, Any]] = {}
_PROBE_TTL_SECONDS = 600


def video_settings() -> dict[str, Any]:
    return (getattr(settings, "ASSISTANT", {}) or {}).get("video", {}) or {}


def _repo_root() -> Path:
    # settings.BASE_DIR = apps/web-backend
    return Path(settings.BASE_DIR).parents[1]


# ============================================================================
# 权重与解释器探测
# ============================================================================

def resolve_model() -> dict[str, Any]:
    """按优先级挑一个本地权重：配置覆盖 → 自训练产物 → 预训练权重 → 让 ultralytics 自动下载。"""
    override = str(video_settings().get("model") or "").strip()
    if override:
        path = Path(override)
        if path.exists():
            return {"path": str(path), "source": "configured", "exists": True}
        if path.suffix == ".pt":
            return {"path": override, "source": "configured-autoload", "exists": False}

    root = _repo_root()
    candidates: list[tuple[Path, str]] = []
    for folder in (root / "models" / "trained",):
        if folder.exists():
            candidates += [(item, "trained") for item in folder.glob("*.pt")]
    training_runs = root / "runs" / "training"
    if training_runs.exists():
        candidates += [(item, "trained") for item in training_runs.glob("*/weights/best.pt")]
    pretrained_dir = root / "models" / "pretrained"
    if pretrained_dir.exists():
        candidates += [(item, "pretrained") for item in pretrained_dir.glob("*.pt")]

    if candidates:
        path, source = max(candidates, key=lambda item: item[0].stat().st_mtime)
        return {"path": str(path), "source": source, "exists": True}

    fallback = str(video_settings().get("fallback_model") or "yolo11n.pt")
    return {"path": fallback, "source": "pretrained-download", "exists": False}


def _python_candidates() -> list[Path]:
    """收集候选解释器：显式配置 → 当前进程 → conda 环境注册表 → 常见安装位置。"""
    candidates: list[Path] = []

    def add(value: str | os.PathLike[str] | None) -> None:
        if not value:
            return
        path = Path(str(value))
        if path.name.lower() != "python.exe" and path.is_dir():
            path = path / ("python.exe" if os.name == "nt" else "bin/python")
        if path.exists() and path not in candidates:
            candidates.append(path)

    add(video_settings().get("yolo_python"))
    add(sys.executable)

    env_file = Path.home() / ".conda" / "environments.txt"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                env_path = Path(line)
                add(env_path)
                add(env_path / "python.exe")
        except OSError:
            pass

    roots = [
        Path(os.environ.get("CONDA_PREFIX") or ""),
        Path.home() / "anaconda3",
        Path.home() / "miniconda3",
        Path.home() / "miniforge3",
        Path("C:/ProgramData/anaconda3"),
        Path("C:/ProgramData/miniconda3"),
        Path("D:/Anaconda"),
        Path("D:/Miniconda3"),
    ]
    for root in roots:
        if not root or not root.exists():
            continue
        add(root / "python.exe")
        envs_dir = root / "envs"
        if envs_dir.exists():
            try:
                for env_dir in envs_dir.iterdir():
                    add(env_dir / "python.exe")
            except OSError:
                pass
    return candidates


def _probe_python(python: Path, timeout: float = 90.0) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                "import ultralytics, cv2, torch;"
                "print(ultralytics.__version__, torch.__version__, cv2.__version__)",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = tail[-1][:220] if tail else f"退出码 {completed.returncode}"
        return False, detail
    return True, (completed.stdout or "").strip()


def _python_label(python: str) -> str:
    """给前端展示的环境名，例如 credit_risk / Anaconda。"""
    path = Path(python)
    return path.parent.name or path.stem or python


def resolve_yolo_python(force: bool = False) -> dict[str, Any]:
    """找到能 ``import ultralytics`` 的解释器，结果缓存 10 分钟。"""
    now = time.time()
    checked: list[dict[str, Any]] = []
    for python in _python_candidates():
        key = str(python)
        cached = _PYTHON_CACHE.get(key)
        if cached and not force and now - cached["checked_at"] < _PROBE_TTL_SECONDS:
            ok, detail = cached["ok"], cached["detail"]
        else:
            ok, detail = _probe_python(python)
            _PYTHON_CACHE[key] = {"ok": ok, "detail": detail, "checked_at": now}
        checked.append({"python": key, "ok": ok, "detail": detail})
        if ok:
            return {
                "available": True,
                "python": key,
                "label": _python_label(key),
                "detail": detail,
                "checked": checked,
            }
    return {
        "available": False,
        "python": "",
        "label": "",
        "detail": "未找到同时具备 ultralytics 与 opencv 的 Python 解释器。",
        "checked": checked,
    }


def cached_python() -> dict[str, Any] | None:
    """取最近一次探测成功/失败的解释器结果（不触发新的探测）。"""
    if not _PYTHON_CACHE:
        return None
    ok_items = [item for item in _PYTHON_CACHE.items() if item[1]["ok"]]
    if ok_items:
        python, info = ok_items[-1]
        return {
            "available": True,
            "python": python,
            "label": _python_label(python),
            "detail": info["detail"],
            "checked": [{"python": python, "ok": True, "detail": info["detail"]}],
        }
    python, info = list(_PYTHON_CACHE.items())[-1]
    return {
        "available": False,
        "python": "",
        "label": "",
        "detail": info["detail"],
        "checked": [{"python": python, "ok": False, "detail": info["detail"]}],
    }


def runtime_status(probe: bool = True, force: bool = False) -> dict[str, Any]:
    model = resolve_model()
    if probe:
        python_info = resolve_yolo_python(force=force)
    else:
        python_info = cached_python() or {
            "available": None,
            "python": "",
            "detail": "尚未探测（上传视频时会自动检测推理环境）",
            "checked": [],
        }
    return {
        "python": python_info,
        "model": model,
        "worker": str(WORKER_PATH),
        "limits": {
            "max_upload_mb": int(video_settings().get("max_upload_mb") or 200),
            "max_frames": int(video_settings().get("max_frames") or 24),
            "max_keyframes": int(video_settings().get("max_keyframes") or 6),
            "conf": float(video_settings().get("conf") or 0.25),
            "imgsz": int(video_settings().get("imgsz") or 640),
            "device": str(video_settings().get("device") or "cpu"),
        },
    }


def install_hint(runtime: dict[str, Any]) -> str:
    checked = "、".join(item["python"] for item in runtime.get("checked") or []) or "（无）"
    return (
        "未找到可用的 YOLO 推理环境。已尝试的解释器："
        f"{checked}。\n"
        "解决办法（任选其一）：\n"
        "1) 在当前 Web 环境安装推理依赖：python -m pip install ultralytics opencv-python\n"
        "2) 复用已有的 YOLO 环境：在 apps/web-backend/.env 里设置 "
        "ODP_YOLO_PYTHON=D:\\Anaconda\\envs\\credit_risk\\python.exe（换成你的环境）\n"
        f"当前应运行 worker 的脚本：{WORKER_PATH}"
    )


# ============================================================================
# 任务执行
# ============================================================================

def _update(job_id: int, **fields: Any) -> None:
    VideoAnalysisJob.objects.filter(pk=job_id).update(**fields)


def start_job(job_id: int) -> None:
    """后台线程启动分析，接口立即返回 job_id 供前端轮询。"""
    thread = threading.Thread(target=_run_safely, args=(job_id,), name=f"video-job-{job_id}", daemon=True)
    thread.start()


def _run_safely(job_id: int) -> None:
    close_old_connections()
    try:
        with _JOB_LOCK:
            _execute(job_id)
    except Exception as exc:  # noqa: BLE001 - 后台线程必须兜底，不能让任务卡在 running
        logger.exception("视频分析任务 #%s 失败", job_id)
        try:
            VideoAnalysisJob.objects.filter(pk=job_id).update(
                status=VideoAnalysisJob.Status.FAILED,
                stage="失败",
                message=str(exc)[:190],
                error=str(exc),
                finished_at=timezone.now(),
            )
        except Exception:  # noqa: BLE001
            logger.exception("标记任务 #%s 失败状态时再次出错", job_id)
    finally:
        close_old_connections()


def _execute(job_id: int) -> None:
    job = VideoAnalysisJob.objects.get(pk=job_id)
    _update(
        job_id,
        status=VideoAnalysisJob.Status.RUNNING,
        progress=2,
        stage="准备",
        message="检查推理环境与权重",
        error="",
    )

    runtime = resolve_yolo_python()
    if not runtime["available"]:
        raise RuntimeError(install_hint(runtime))

    model = resolve_model()
    video_path = Path(job.video.path)
    if not video_path.exists():
        raise RuntimeError(f"上传的视频文件不存在：{video_path}")

    workdir = Path(settings.MEDIA_ROOT) / "video_jobs" / str(job.id)
    workdir.mkdir(parents=True, exist_ok=True)

    options = job.options or {}
    config = {
        "video": str(video_path),
        "model": model["path"],
        "model_source": model["source"],
        "out_dir": str(workdir),
        "conf": float(options.get("conf") or video_settings().get("conf") or 0.25),
        "iou": float(video_settings().get("iou") or 0.7),
        "imgsz": int(video_settings().get("imgsz") or 640),
        "device": str(video_settings().get("device") or "cpu"),
        "max_frames": int(options.get("max_frames") or video_settings().get("max_frames") or 24),
        "max_keyframes": int(video_settings().get("max_keyframes") or 6),
        "time_budget_seconds": float(video_settings().get("time_budget_seconds") or 600),
    }
    config_path = workdir / "job.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [str(runtime["python"]), str(WORKER_PATH), "--config", str(config_path)]
    logger.info("启动视频分析进程: %s", command)

    _update(job_id, progress=5, stage="启动推理进程", message=Path(runtime["python"]).name)

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(workdir),
        creationflags=creation_flags,
    )

    result_data: dict[str, Any] | None = None
    worker_error = ""
    tail: deque[str] = deque(maxlen=40)
    last_progress = -1

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n").strip()
        if not line:
            continue
        if line.startswith(EVENT_PREFIX):
            try:
                event = json.loads(line[len(EVENT_PREFIX):])
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "progress":
                percent = int(event.get("percent") or 0)
                stage = str(event.get("stage") or "")
                message = str(event.get("message") or "")
                if percent != last_progress:
                    last_progress = percent
                    _update(job_id, progress=max(1, min(percent, 96)), stage=stage, message=message[:190])
            elif kind == "log":
                logger.info("[job %s] %s", job_id, event.get("message"))
            elif kind == "error":
                worker_error = str(event.get("message") or "推理进程返回错误")
                logger.error("[job %s] %s", job_id, worker_error)
            elif kind == "result":
                result_data = event.get("data") or {}
        else:
            tail.append(line)
            if len(tail) > 40:
                tail.popleft()

    return_code = process.wait()
    if result_data is None:
        detail = "\n".join(list(tail)[-8:])
        raise RuntimeError(
            (worker_error or f"推理进程异常退出（退出码 {return_code}）。")
            + (f"\n最近输出：\n{detail}" if detail else "")
        )

    # 关键帧路径转成相对 MEDIA_ROOT 的路径，前端才能用 /media/ 访问
    for item in result_data.get("keyframes") or []:
        file_name = item.get("file") or ""
        item["path"] = f"video_jobs/{job.id}/{file_name}" if file_name else ""
    poster = result_data.get("poster")
    result_data["poster_path"] = f"video_jobs/{job.id}/{poster}" if poster else ""
    result_data["runtime_python"] = runtime["python"]

    _update(
        job_id,
        progress=95,
        stage="大模型解读",
        message="把检测统计交给大模型解释",
        result=result_data,
    )

    llm_config = llm.LLMConfig.from_settings()
    payload = {
        "task": "video_object_detection",
        "video": result_data.get("video"),
        "model": {
            "path": (result_data.get("model") or {}).get("path"),
            "source": (result_data.get("model") or {}).get("source"),
            "class_count": (result_data.get("model") or {}).get("class_count"),
            "task": (result_data.get("model") or {}).get("task"),
        },
        "sampling": {
            "frames_analyzed": (result_data.get("summary") or {}).get("frames_analyzed"),
            "sampled_frames": (result_data.get("summary") or {}).get("sampled_frames"),
            "options": {
                "conf": config["conf"],
                "iou": config["iou"],
                "imgsz": config["imgsz"],
                "device": config["device"],
            },
        },
        "statistics": result_data.get("summary"),
        "warnings": result_data.get("warnings"),
    }

    explanation = ""
    engine = ""
    try:
        explanation = llm.chat(
            [
                {"role": "system", "content": prompts.VIDEO_SYSTEM},
                {"role": "user", "content": prompts.build_video_prompt(payload)},
            ],
            config=llm_config,
        )
        engine = llm_config.label
    except llm.LLMError as exc:
        explanation = explain.fallback_explanation(result_data, reason=str(exc))
        engine = "本地规则解释"

    _update(
        job_id,
        status=VideoAnalysisJob.Status.COMPLETED,
        progress=100,
        stage="已完成",
        message="分析完成",
        explanation=explanation,
        engine=engine,
        finished_at=timezone.now(),
    )
    logger.info("视频分析任务 #%s 完成，耗时 %.1fs", job_id, float(result_data.get("elapsed_seconds") or 0))
