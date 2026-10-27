#!/usr/bin/env python
# 阅读副本：独立进程执行本地 YOLO，Django 通过 JSON 事件流读取进度和结果。
"""本地 YOLO 视频推理 worker（独立进程运行）。

为什么单独一个进程？

- 大模型/Django 只需要轻量依赖，而 ultralytics + torch 体积大、可能装在另一个 Python 环境；
- 推理崩溃不会带崩 Web 服务，超时/显存问题也更好隔离；
- 用 JSON 事件流与 Django 通信，协议简单、可单独命令行调试。

调用方式（由 assistant/video.py 自动拼装）::

    python yolo_worker.py --config C:\\path\\job_config.json

标准输出中带 ``@@ODP@@`` 前缀的行是事件，其余行（ultralytics 日志等）会被忽略：

    @@ODP@@{"type": "progress", "percent": 42, "stage": "YOLO 推理", "message": "第 5/12 帧"}
    @@ODP@@{"type": "result", "data": {...}}
    @@ODP@@{"type": "error", "message": "..."}

单独调试示例::

    python yolo_worker.py --config job_config.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

EVENT_PREFIX = "@@ODP@@"


def emit(event_type: str, **payload: Any) -> None:
    print(EVENT_PREFIX + json.dumps({"type": event_type, **payload}, ensure_ascii=False), flush=True)


def _read_config(argv: list[str]) -> dict[str, Any]:
    if "--config" not in argv:
        raise SystemExit("用法: python yolo_worker.py --config <job_config.json>")
    path = Path(argv[argv.index("--config") + 1])
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_names(names: Any) -> dict[str, str]:
    if isinstance(names, dict):
        return {str(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {str(index): str(value) for index, value in enumerate(names)}
    return {}


def _frame_indices(total: int, max_frames: int) -> list[int]:
    """在整段视频上均匀取点，保证首尾都能覆盖到。"""
    if total <= 0:
        return []
    count = max(1, min(int(max_frames), total))
    if count == 1:
        return [0]
    step = (total - 1) / (count - 1)
    return sorted({int(round(index * step)) for index in range(count)})


def _encode_jpeg(frame: Any, quality: int = 85) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buffer.tobytes()


def _annotate(result: Any, time_seconds: float, count: int) -> Any:
    import cv2

    frame = result.plot()  # ultralytics 自带的框 + 标签绘制，返回 BGR ndarray
    banner = f"t={time_seconds:5.1f}s  objects={count}"
    cv2.rectangle(frame, (0, 0), (260, 26), (12, 36, 34), -1)
    cv2.putText(
        frame,
        banner,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 245, 236),
        1,
        cv2.LINE_AA,
    )
    return frame


def run(config: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import stats  # 同目录纯标准库模块，独立进程也能直接 import
    from ultralytics import YOLO

    video_path = Path(config["video"])
    model_path = str(config["model"])
    out_dir = Path(config["out_dir"])
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    conf = float(config.get("conf", 0.25))
    iou = float(config.get("iou", 0.7))
    imgsz = int(config.get("imgsz", 640))
    max_frames = int(config.get("max_frames", 24))
    max_keyframes = int(config.get("max_keyframes", 6))
    device = str(config.get("device") or "cpu")
    budget = float(config.get("time_budget_seconds") or 600)

    started = time.perf_counter()
    deadline = started + budget

    emit("progress", percent=8, stage="加载模型", message=f"加载本地权重 {Path(model_path).name}")
    model = YOLO(model_path)
    names = _normalize_names(getattr(model, "names", {}))
    task = str(getattr(model, "task", "detect") or "detect")
    emit(
        "log",
        message=f"模型就绪：{Path(model_path).name}（{len(names)} 个类别，task={task}）",
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total_frames > 0:
        indices = _frame_indices(total_frames, max_frames)
    else:  # 极少数编码拿不到总帧数时，顺序读前 N 帧
        indices = list(range(max_frames))

    emit(
        "progress",
        percent=12,
        stage="解析视频",
        message=f"{width}x{height} @ {fps:.1f}fps，计划抽样 {len(indices)} 帧",
    )

    detections: list[dict[str, Any]] = []
    keyframe_candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    analyzed = 0
    skipped = 0
    poster_bytes: bytes | None = None
    poster_frame_index: int | None = None
    poster_target = indices[len(indices) // 2] if indices else 0

    for position, frame_index in enumerate(indices):
        if time.perf_counter() > deadline:
            warnings.append(f"达到分析时间上限（{budget:.0f}s），剩余帧已跳过")
            break

        frame_index = int(frame_index)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            skipped += 1
            continue

        analyzed += 1
        frame_height, frame_width = frame.shape[:2]
        area_total = float(frame_width * frame_height) or 1.0
        time_seconds = frame_index / fps if fps else float(position)

        if poster_bytes is None or frame_index == poster_target:
            try:
                poster_bytes = _encode_jpeg(frame, quality=80)
                poster_frame_index = frame_index
            except RuntimeError:
                poster_bytes = None

        try:
            results = model.predict(
                frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device if device != "auto" else None,
                max_det=300,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - 单帧失败不应终止整个任务
            warnings.append(f"第 {frame_index} 帧推理失败：{exc}")
            continue

        result = results[0] if results else None
        boxes = getattr(result, "boxes", None) if result is not None else None
        count = int(len(boxes)) if boxes is not None else 0
        frame_conf_max = 0.0

        if count:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for order in range(count):
                class_id = int(classes[order])
                confidence = float(confs[order])
                x1, y1, x2, y2 = (float(value) for value in xyxy[order][:4])
                detections.append(
                    {
                        "frame_index": frame_index,
                        "time": round(time_seconds, 3),
                        "cls_id": class_id,
                        "cls_name": names.get(str(class_id), f"class_{class_id}"),
                        "conf": round(confidence, 4),
                        "area_ratio": round(
                            max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / area_total, 4
                        ),
                    }
                )
                frame_conf_max = max(frame_conf_max, confidence)

            try:
                annotated = _annotate(result, time_seconds, count)
                keyframe_candidates.append(
                    {
                        "frame_index": frame_index,
                        "time": round(time_seconds, 3),
                        "count": count,
                        "max_conf": round(frame_conf_max, 4),
                        "jpeg": _encode_jpeg(annotated),
                        "top_class": names.get(str(int(classes[0])), str(int(classes[0]))),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"第 {frame_index} 帧绘制标注失败：{exc}")

        percent = 12 + int(78 * (position + 1) / max(len(indices), 1))
        emit(
            "progress",
            percent=percent,
            stage="YOLO 推理",
            message=f"第 {position + 1}/{len(indices)} 帧，检出 {count} 个目标",
        )

    capture.release()

    if analyzed == 0:
        raise RuntimeError(
            "没有成功解码任何视频帧：请确认文件未损坏，且 OpenCV 支持该编码（建议 MP4/H.264）。"
        )

    keyframe_candidates.sort(key=lambda item: (-item["count"], -item["max_conf"]))
    keyframes: list[dict[str, Any]] = []
    for candidate in keyframe_candidates[:max_keyframes]:
        name = f"kf_{candidate['frame_index']:06d}.jpg"
        (frames_dir / name).write_bytes(candidate["jpeg"])
        keyframes.append(
            {
                "file": f"frames/{name}",
                "frame": candidate["frame_index"],
                "time": candidate["time"],
                "count": candidate["count"],
                "max_conf": candidate["max_conf"],
                "top_class": candidate["top_class"],
            }
        )
    keyframes.sort(key=lambda item: item["time"])

    poster_file = None
    if poster_bytes:
        (out_dir / "poster.jpg").write_bytes(poster_bytes)
        poster_file = "poster.jpg"

    duration = round(total_frames / fps, 3) if fps and total_frames else None
    summary = stats.summarize(
        detections,
        frames_analyzed=analyzed,
        total_frames=total_frames or None,
        fps=fps,
        extra_warnings=warnings,
    )
    summary["sampled_frames"] = len(indices)
    summary["skipped_frames"] = skipped

    emit("progress", percent=92, stage="汇总统计", message="正在汇总检测结果")
    return {
        "video": {
            "path": str(video_path),
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "duration_seconds": duration,
        },
        "model": {
            "path": model_path,
            "source": config.get("model_source") or "unknown",
            "names": names,
            "class_count": len(names),
            "task": task,
            "ultralytics": _ultralytics_version(),
            "device": device,
        },
        "config": {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_frames": max_frames,
            "max_keyframes": max_keyframes,
        },
        "summary": summary,
        "keyframes": keyframes,
        "poster": poster_file,
        "poster_frame": poster_frame_index,
        "warnings": warnings,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def _ultralytics_version() -> str:
    try:
        import ultralytics

        return str(getattr(ultralytics, "__version__", ""))
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    try:
        config = _read_config(args)
        data = run(config)
    except ModuleNotFoundError as exc:
        emit(
            "error",
            message=(
                f"当前 Python 缺少推理依赖（{exc.name}）。"
                "请在该解释器里安装：python -m pip install ultralytics opencv-python"
            ),
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        emit("error", message=f"视频分析失败：{exc}", traceback=traceback.format_exc()[-1500:])
        return 1
    emit("result", data=data)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
