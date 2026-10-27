"""Gradio 演示界面 —— 答辩现场用: 上传图片/视频, 实时看检测结果。

为什么单独做一个而不是直接给 ultralytics 的 CLI:
    CLI 输出是个目录, 答辩时需要"拖一张图进去马上出结果"的交互;
    同时这里复用平台自己的 BeautifyVisualizer(带图例/配色/中文标签),
    证明 D7 的绘制层是能被外部直接调用的, 不是训练脚本的私有产物。

启动:
    python apps/demo/app.py
    # 或者指定权重
    python apps/demo/app.py --model models/trained/xxx-best.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from od_platform.common.paths import TRAINED_MODELS_DIR
from od_platform.visualization import BeautifyVisualizer, Detection, DrawStyle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def list_models() -> list[str]:
    """models/trained 下所有 best 权重, 新的排前面。"""
    if not TRAINED_MODELS_DIR.exists():
        return []
    return [
        str(p.relative_to(PROJECT_ROOT))
        for p in sorted(TRAINED_MODELS_DIR.glob("*.pt"), reverse=True)
    ]


class Detector:
    """包一层模型, 避免每次回调都重新加载。"""

    def __init__(self, model_path: str) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path if Path(model_path).is_absolute() else PROJECT_ROOT / model_path)
        names = self.model.names
        self.class_names = [names[i] for i in sorted(names)]
        # 按类别自动配色, 色相均匀分布, 保证相邻类别颜色差异明显
        self.visualizer = BeautifyVisualizer(
            labels=self.class_names,
            color_mapping=self._palette(len(self.class_names)),
        )

    @staticmethod
    def _palette(n: int) -> dict[str, tuple[int, int, int]]:
        import colorsys

        out: dict[str, tuple[int, int, int]] = {}
        for i in range(n):
            r, g, b = colorsys.hsv_to_rgb(i / max(1, n), 0.85, 0.95)
            out[f"class_{i}"] = (int(b * 255), int(g * 255), int(r * 255))
        return out

    def infer(self, image: np.ndarray, conf: float, iou: float) -> tuple[np.ndarray, str]:
        result = self.model.predict(image, conf=conf, iou=iou, verbose=False)[0]
        dets: list[Detection] = []
        labels: list[str] = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            labels.append(label)
            dets.append(
                Detection(
                    label=label,
                    confidence=float(box.conf.item()),
                    xyxy=tuple(box.xyxy[0].tolist()),  # type: ignore[arg-type]
                )
            )
        style = dataclasses.replace(
            DrawStyle.from_image_size(*image.shape[:2]), show_legend=bool(dets)
        )
        canvas = self.visualizer.draw(image, dets, style=style)
        return canvas, self._summary(dets)

    @staticmethod
    def _summary(dets: list[Detection]) -> str:
        if not dets:
            return "未检出任何目标。"
        from collections import Counter

        counts = Counter(d.label for d in dets)
        lines = [f"共检出 **{len(dets)}** 个目标, **{len(counts)}** 类:", ""]
        lines += [f"- `{name}` × {cnt}" for name, cnt in counts.most_common()]
        best = max(dets, key=lambda d: d.confidence)
        lines += ["", f"最高置信度: `{best.label}` {best.confidence:.3f}"]
        return "\n".join(lines)


_DETECTOR: Detector | None = None
_CURRENT_MODEL: str | None = None


def get_detector(model_path: str) -> Detector:
    global _DETECTOR, _CURRENT_MODEL
    if _DETECTOR is None or _CURRENT_MODEL != model_path:
        _DETECTOR = Detector(model_path)
        _CURRENT_MODEL = model_path
    return _DETECTOR


def on_image(image: np.ndarray | None, model_path: str, conf: float, iou: float):
    if image is None:
        return None, "请先上传一张图片。"
    if not model_path:
        return None, "没有可用权重, 请先训一个模型(models/trained/*.pt)。"
    t0 = time.perf_counter()
    canvas, summary = get_detector(model_path).infer(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), conf, iou
    )
    cost = (time.perf_counter() - t0) * 1000
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), f"{summary}\n\n耗时 {cost:.0f} ms"


def on_video(
    video_path: str | None, model_path: str, conf: float, iou: float, stride: int
):
    """逐帧抽检(每 stride 帧检一次), 结果写成新 mp4。"""
    if not video_path:
        return None, "请先上传一段视频。"
    if not model_path:
        return None, "没有可用权重, 请先训一个模型。"

    detector = get_detector(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = str(Path(video_path).with_name("detected_" + Path(video_path).name))
    if not out_path.lower().endswith(".mp4"):
        out_path += ".mp4"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = hits = 0
    t0 = time.perf_counter()
    last_canvas = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, stride) == 0:
            canvas, _ = detector.infer(frame, conf, iou)
            last_canvas = canvas
            hits += 1
        else:
            canvas = last_canvas if last_canvas is not None else frame
        writer.write(canvas)
        frame_idx += 1
    cap.release()
    writer.release()
    cost = time.perf_counter() - t0
    msg = (
        f"处理 {frame_idx} 帧(检测 {hits} 帧), 用时 {cost:.1f}s, "
        f"平均 {fps and frame_idx / max(cost, 1e-6):.0f} FPS\n\n输出: `{out_path}`"
    )
    return out_path, msg


def on_assistant(video_path: str | None, model_path: str, conf: float, stride: int):
    """视频 → 本地检测 → 大模型解读(没 Key 自动用规则)。"""
    if not video_path:
        return None, "请先上传一段视频。"
    if not model_path:
        return None, "没有可用权重。"

    from od_platform.assistant import LLMConfig, analyze_video, explain

    summary = analyze_video(video_path, PROJECT_ROOT / model_path, conf=conf, stride=stride)
    summary.source = Path(video_path).name
    text, source = explain(summary, LLMConfig.from_env())

    import pandas as pd

    rows = summary.to_rows() or [{"类别": "（无）", "出现帧数": 0, "框总数": 0,
                                  "最高置信度": 0.0, "首次出现(秒)": 0.0}]
    tag = "大模型" if source == "llm" else "规则模板(未配置 API Key)"
    return pd.DataFrame(rows), f"### 智能解读\n\n{text}\n\n---\n来源：{tag}"


def build_ui(default_model: str) -> gr.Blocks:
    models = list_models()
    with gr.Blocks(title="ODPlatform 演示") as demo:
        gr.Markdown(
            "# 目标检测平台 · 在线演示\n"
            "上传图片或视频, 直接在浏览器里看检测结果。"
            "绘制层用的是平台自己的 `BeautifyVisualizer`。"
        )
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=models or ["(无可用权重)"],
                value=default_model or (models[0] if models else "(无可用权重)"),
                label="选择权重",
                scale=2,
            )
            conf_sl = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
            iou_sl = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU 阈值")

        with gr.Tab("图片检测"):
            with gr.Row():
                img_in = gr.Image(label="上传图片", type="numpy")
                img_out = gr.Image(label="检测结果", type="numpy")
            img_msg = gr.Markdown()
            img_btn = gr.Button("开始检测", variant="primary")
            img_btn.click(
                on_image, [img_in, model_dd, conf_sl, iou_sl], [img_out, img_msg]
            )

        with gr.Tab("视频检测"):
            with gr.Row():
                vid_in = gr.Video(label="上传视频")
                vid_out = gr.Video(label="检测结果")
            stride_sl = gr.Slider(1, 10, value=3, step=1, label="每 N 帧检一次(越大越快)")
            vid_msg = gr.Markdown()
            vid_btn = gr.Button("开始处理", variant="primary")
            vid_btn.click(
                on_video,
                [vid_in, model_dd, conf_sl, iou_sl, stride_sl],
                [vid_out, vid_msg],
            )

        with gr.Tab("智能助手"):
            gr.Markdown(
                "上传视频 → **本地 YOLO 检测** → 把统计结果交给大模型, 输出一段人话解读。"
                "没配 `ODP_LLM_API_KEY` 时自动用规则模板, 不会演示失败。"
            )
            with gr.Row():
                ast_in = gr.Video(label="上传视频")
                ast_table = gr.Dataframe(label="检测摘要", interactive=False)
            ast_stride = gr.Slider(1, 20, value=5, step=1, label="每 N 帧检一次")
            ast_msg = gr.Markdown()
            ast_btn = gr.Button("开始分析", variant="primary")
            ast_btn.click(
                on_assistant,
                [ast_in, model_dd, conf_sl, ast_stride],
                [ast_table, ast_msg],
            )

        gr.Markdown(
            "> 提示: CPU 推理较慢, 视频建议把「每 N 帧检一次」调大; "
            "换权重后首次检测会重新加载模型(约 2 秒)。"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="ODPlatform Gradio 演示")
    parser.add_argument("--model", default=None, help="权重路径, 默认取 models/trained 里最新的")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网临时链接")
    args = parser.parse_args()

    default = args.model
    if default is None:
        models = list_models()
        default = models[0] if models else ""
    build_ui(default).launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
