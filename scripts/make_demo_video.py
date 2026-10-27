#!/usr/bin/env python
"""生成一段用于「视频智能分析」演示/自测的短视频（需要 opencv）。

两种模式：

1. 有参考图片时做平移+缩放，适合验证「本地 YOLO 真的能检出目标」::

       python scripts/make_demo_video.py --source D:\\pics\\bus.jpg --output runs\\demo\\bus.mp4

2. 没有图片时画几个移动的矩形/圆（流水线仍可跑通，但通用 COCO 权重未必检出）::

       python scripts/make_demo_video.py --output runs\\demo\\shapes.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _pan_frames(image: np.ndarray, seconds: float, fps: int, size: tuple[int, int]):
    height, width = size[1], size[0]
    total = max(1, int(seconds * fps))
    for index in range(total):
        progress = index / max(total - 1, 1)
        scale = 1.0 + 0.25 * progress  # 缓慢推近，制造尺度变化
        crop_width = int(image.shape[1] / scale)
        crop_height = int(image.shape[0] / scale)
        max_x = max(image.shape[1] - crop_width, 0)
        max_y = max(image.shape[0] - crop_height, 0)
        left = int(max_x * progress)
        top = int(max_y * (0.5 + 0.5 * np.sin(progress * np.pi)))
        crop = image[top : top + crop_height, left : left + crop_width]
        yield cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def _shape_frames(seconds: float, fps: int, size: tuple[int, int]):
    width, height = size
    total = max(1, int(seconds * fps))
    colors = [(58, 148, 232), (74, 196, 120), (60, 90, 220), (200, 160, 40)]
    for index in range(total):
        canvas = np.full((height, width, 3), 34, dtype=np.uint8)
        cv2.line(canvas, (0, height - 46), (width, height - 46), (90, 110, 100), 2)
        for order, color in enumerate(colors):
            offset = (index * (3 + order)) % (width + 120) - 120
            x1 = offset
            y1 = int(height - 90 - order * 34)
            cv2.rectangle(canvas, (x1, y1), (x1 + 70, y1 + 46), color, -1)
            cv2.putText(
                canvas,
                f"bin-{order}",
                (x1 + 4, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
        yield canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="生成视频分析演示片段")
    parser.add_argument("--source", type=Path, help="参考图片（可选）")
    parser.add_argument("--output", type=Path, default=Path("runs/demo/demo.mp4"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    size = (args.width, args.height)
    if args.source:
        image = cv2.imread(str(args.source))
        if image is None:
            raise SystemExit(f"无法读取参考图片：{args.source}")
        frames = _pan_frames(image, args.seconds, args.fps, size)
    else:
        frames = _shape_frames(args.seconds, args.fps, size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        size,
    )
    if not writer.isOpened():
        raise SystemExit("无法创建视频，请确认 OpenCV 带 mp4v 编码支持")
    count = 0
    for frame in frames:
        writer.write(frame)
        count += 1
    writer.release()

    print(f"已生成 {args.output}（{count} 帧，{args.fps} fps，{size[0]}x{size[1]}）")
    print(f"大小：{args.output.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
