"""把逐帧检测结果汇总成可读统计（纯标准库，方便单测与独立进程复用）。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def summarize(
    detections: Iterable[dict[str, Any]],
    *,
    frames_analyzed: int,
    total_frames: int | None = None,
    fps: float = 0.0,
    extra_warnings: Iterable[str] = (),
) -> dict[str, Any]:
    """汇总检测框。

    Args:
        detections: 每个元素形如
            {"frame_index": 12, "time": 0.4, "cls_id": 0, "cls_name": "person",
             "conf": 0.91, "area_ratio": 0.18}
        frames_analyzed: 实际完成推理的帧数（可能小于计划帧数）
        total_frames: 视频总帧数（未知时传 None）
        fps: 视频帧率，用于换算时间轴
        extra_warnings: 额外提示

    Returns:
        结构化统计，键名直接给大模型与前端使用。
    """
    frames_analyzed = max(int(frames_analyzed or 0), 0)
    per_class: dict[str, dict[str, Any]] = {}
    per_frame: dict[int, dict[str, Any]] = {}
    frames_with_detection: set[int] = set()
    conf_values: list[float] = []
    total_detections = 0

    for det in detections:
        try:
            name = str(det.get("cls_name") or det.get("cls_id"))
            conf = float(det.get("conf") or 0.0)
            frame_index = int(det.get("frame_index") or 0)
        except (TypeError, ValueError):
            continue

        total_detections += 1
        conf_values.append(conf)
        frames_with_detection.add(frame_index)

        bucket = per_class.setdefault(
            name,
            {
                "name": name,
                "class_id": det.get("cls_id"),
                "frames": set(),
                "detections": 0,
                "conf_sum": 0.0,
                "max_conf": 0.0,
                "area_sum": 0.0,
            },
        )
        bucket["frames"].add(frame_index)
        bucket["detections"] += 1
        bucket["conf_sum"] += conf
        bucket["max_conf"] = max(bucket["max_conf"], conf)
        bucket["area_sum"] += float(det.get("area_ratio") or 0.0)

        frame_bucket = per_frame.setdefault(
            frame_index,
            {
                "frame": frame_index,
                "time": round(float(det.get("time") or 0.0), 3),
                "total": 0,
                "classes": {},
            },
        )
        frame_bucket["total"] += 1
        frame_bucket["classes"][name] = frame_bucket["classes"].get(name, 0) + 1

    class_rows: list[dict[str, Any]] = []
    for bucket in per_class.values():
        bucket["detections"] = int(bucket["detections"])
        row = {
            "name": bucket["name"],
            "class_id": bucket["class_id"],
            "frames": len(bucket["frames"]),
            "frame_ratio": _round(len(bucket["frames"]) / frames_analyzed, 3)
            if frames_analyzed
            else 0.0,
            "detections": bucket["detections"],
            "avg_conf": _round(bucket["conf_sum"] / bucket["detections"], 4),
            "max_conf": _round(bucket["max_conf"], 4),
            "avg_area_ratio": _round(bucket["area_sum"] / bucket["detections"], 4),
        }
        class_rows.append(row)
    class_rows.sort(key=lambda item: (-item["detections"], -item["avg_conf"], item["name"]))

    timeline = [per_frame[key] for key in sorted(per_frame)]
    times = [item["time"] for item in timeline if item["time"]]
    duration = round((total_frames or 0) / fps, 2) if fps else None

    return {
        "frames_analyzed": frames_analyzed,
        "total_frames": total_frames,
        "fps": _round(fps, 2) if fps else 0.0,
        "duration_seconds": duration,
        "detected_frames": len(frames_with_detection),
        "detection_frame_ratio": _round(len(frames_with_detection) / frames_analyzed, 3)
        if frames_analyzed
        else 0.0,
        "total_detections": total_detections,
        "max_objects_in_frame": max((item["total"] for item in timeline), default=0),
        "avg_detections_per_frame": _round(total_detections / frames_analyzed, 3)
        if frames_analyzed
        else 0.0,
        "avg_confidence": _round(sum(conf_values) / len(conf_values), 4) if conf_values else None,
        "first_detection_time": min(times) if times else None,
        "last_detection_time": max(times) if times else None,
        "classes": class_rows,
        "timeline": timeline,
        "warnings": [str(item) for item in extra_warnings if item],
    }


def dominant_class(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """返回检出框最多的类别（无检出时为 None）。"""
    classes = (summary or {}).get("classes") or []
    return classes[0] if classes else None
