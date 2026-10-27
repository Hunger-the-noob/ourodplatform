from __future__ import annotations

import tempfile
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from .recognition import _MODEL_LOCK, _get_model, latest_trained_weight


def recognize_video_file(
    uploaded_file,
    *,
    confidence: float = 0.25,
    iou: float = 0.7,
    frame_stride: int = 1,
    max_frames: int = 300,
) -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image

    if not 0 <= confidence <= 1:
        raise ValueError("置信度阈值必须在 0 到 1 之间。")
    if not 0 <= iou <= 1:
        raise ValueError("IoU 阈值必须在 0 到 1 之间。")
    if not 1 <= frame_stride <= 30:
        raise ValueError("抽帧间隔必须在 1 到 30 之间。")

    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    with tempfile.TemporaryDirectory(prefix="odp_video_") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "annotated.mp4"
        with input_path.open("wb") as handle:
            for chunk in uploaded_file.chunks():
                handle.write(chunk)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("无法读取上传的视频文件。")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        original_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("视频尺寸无效。")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(fps / frame_stride, 1.0),
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("无法创建标注视频。")

        model = _get_model(latest_trained_weight())
        class_counts: Counter[str] = Counter()
        detections: list[dict[str, Any]] = []
        preview_frames: list[Image.Image] = []
        frame_index = 0
        processed_frames = 0
        started = time.perf_counter()
        try:
            while processed_frames < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_stride:
                    frame_index += 1
                    continue
                with _MODEL_LOCK:
                    result = model.predict(
                        frame,
                        conf=confidence,
                        iou=iou,
                        imgsz=640,
                        device="cpu",
                        verbose=False,
                    )[0]
                boxes = result.boxes
                if boxes is not None:
                    coordinates = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    classes = boxes.cls.int().cpu().tolist()
                    for bbox, score, class_id in zip(coordinates, scores, classes):
                        label = str(
                            (result.names or {}).get(
                                int(class_id),
                                f"class_{class_id}",
                            )
                        )
                        class_counts[label] += 1
                        detections.append(
                            {
                                "frame": frame_index,
                                "label": label,
                                "confidence": round(float(score), 4),
                                "bbox": [
                                    round(float(value), 1)
                                    for value in bbox
                                ],
                            }
                        )
                annotated = result.plot(labels=True, conf=True, line_width=2)
                writer.write(annotated)
                if len(preview_frames) < 60 and processed_frames % 3 == 0:
                    preview_rgb = annotated[:, :, ::-1].astype(np.uint8)
                    preview_image = Image.fromarray(preview_rgb)
                    preview_image.thumbnail((640, 640))
                    preview_frames.append(preview_image.resize((640, 640)))
                processed_frames += 1
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        if processed_frames == 0:
            raise ValueError("视频中没有可处理的画面。")
        inference_ms = (time.perf_counter() - started) * 1000
        output_bytes = output_path.read_bytes()

        preview_buffer = BytesIO()
        if preview_frames:
            preview_frames[0].save(
                preview_buffer,
                format="WEBP",
                save_all=True,
                append_images=preview_frames[1:],
                duration=max(int(1000 * frame_stride / max(fps, 1)), 60),
                loop=0,
                quality=72,
                method=4,
            )

    return {
        "annotated_bytes": output_bytes,
        "preview_bytes": preview_buffer.getvalue(),
        "original_frames": original_frames,
        "processed_frames": processed_frames,
        "object_count": len(detections),
        "model_name": latest_trained_weight().name,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "inference_ms": round(inference_ms, 2),
        "class_counts": dict(sorted(class_counts.items())),
        "detections": detections[-2000:],
        "truncated": processed_frames >= max_frames,
    }
