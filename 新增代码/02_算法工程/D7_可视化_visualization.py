#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform
# @Function  : 推理结果美化绘制 —— 类别颜色/中文标签/自适应样式
"""推理结果美化绘制.

在 YOLO 原生 plot() 之外提供一套更精致的绘制:
  - 类别 → 颜色映射 (BGR), 没映射的用 default_color
  - 英文 → 中文标签映射 (label_mapping), 没映射的保持英文
  - 框线/字号/标签背景自适应图像尺寸 (DrawStyle.from_image_size)
  - 中文标签: 有 font_path 用 PIL 渲染; 没有则退回 cv2.putText (非中文可能乱码)

★ 与 D8 推理子系统的接口契约:
  - BeautifyVisualizer(labels, label_mapping=None, color_mapping=None,
                       default_color=(0,255,0), font_path=None)
  - draw(image, dets, *, style, use_label_mapping=True) -> annotated (BGR ndarray)
  - BeautifyVisualizer.from_yolo_results(boxes, confidences, labels)  # 静态
      从 ultralytics boxes 提取 Detection 列表 (boxes.xyxy / boxes.conf)
  - DrawStyle.from_image_size(h, w, **overrides)  自适应样式
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# BGR
_DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 0)


@dataclass
class Detection:
    """一个检测框 (纯数据, 可序列化)."""
    xyxy: Any                 # np.ndarray 或 list, 4 元素 (x1, y1, x2, y2)
    confidence: float = 0.0
    label: str = ""


@dataclass
class DrawStyle:
    """绘制样式. from_image_size 按图像尺寸给自适应默认值."""
    box_thickness: int = 2
    text_scale: float = 0.6
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_bg_alpha: float = 0.6
    # —— 显示内容开关: 演示时常用"只看框"或"只看类别" ——
    show_label: bool = True          # 是否画类别名
    show_confidence: bool = True     # 是否在类别名后带置信度
    show_legend: bool = False        # 是否在右上角画类别图例(色块+名称+数量)

    @classmethod
    def from_image_size(
        cls,
        h: int,
        w: int,
        *,
        box_thickness: Optional[int] = None,
        text_scale: Optional[float] = None,
        text_color: Optional[Tuple[int, int, int]] = None,
        text_bg_alpha: Optional[float] = None,
        show_label: Optional[bool] = None,
        show_confidence: Optional[bool] = None,
        show_legend: Optional[bool] = None,
    ) -> "DrawStyle":
        """按图像尺寸自适应默认值, 显式传的覆盖."""
        short = min(h, w) or 640
        style = cls(
            box_thickness=box_thickness or max(1, round(short / 400)),
            text_scale=text_scale or max(0.4, round(short / 900, 2)),
            text_color=text_color or (255, 255, 255),
            text_bg_alpha=text_bg_alpha if text_bg_alpha is not None else 0.6,
            show_label=True if show_label is None else show_label,
            show_confidence=True if show_confidence is None else show_confidence,
            show_legend=False if show_legend is None else show_legend,
        )
        return style


class BeautifyVisualizer:
    """推理结果美化绘制器.

    Args:
        labels:       模型全部类别名 (用于 label_mapping 之外保持英文)
        label_mapping: 英文 → 中文 (没映射的保持英文)
        color_mapping: 类别 → BGR 颜色 (没映射的用 default_color)
        default_color: 兜底框颜色 (BGR)
        font_path:     中文字体路径 (None → 中文可能乱码, 但不报错)
    """

    def __init__(
        self,
        labels: List[str],
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = _DEFAULT_COLOR,
        font_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.labels = list(labels)
        self.label_mapping: Dict[str, str] = dict(label_mapping or {})
        self.color_mapping: Dict[str, Tuple[int, int, int]] = dict(color_mapping or {})
        self.default_color = tuple(default_color)
        self.font_path = str(font_path) if font_path else None

        # PIL 字体 (lazy, 有 font_path 才加载)
        self._font = None
        self._pil_font_size = 24

    # ------------------------------------------------------------------
    # 检测列表提取
    # ------------------------------------------------------------------

    @staticmethod
    def from_yolo_results(boxes, confidences, labels) -> List[Detection]:
        """从 ultralytics boxes 提取 Detection 列表.

        Args:
            boxes:       boxes.xyxy (n,4) numpy
            confidences: boxes.conf (n,) numpy
            labels:      该帧的 label 字符串列表 (与框一一对应)
        """
        dets: List[Detection] = []
        n = 0 if boxes is None else len(boxes)
        if n == 0:
            return dets
        for i in range(n):
            row = boxes[i]
            dets.append(Detection(
                xyxy=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                confidence=float(confidences[i]) if confidences is not None else 0.0,
                label=str(labels[i]) if i < len(labels) else "",
            ))
        return dets

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def _display_label(self, label: str, use_label_mapping: bool) -> str:
        if use_label_mapping and label in self.label_mapping:
            return self.label_mapping[label]
        return label

    def _box_color(self, label: str) -> Tuple[int, int, int]:
        return self.color_mapping.get(label, self.default_color)

    def _ensure_font(self):
        """加载 PIL 字体 (有 font_path 且能 import PIL 时)."""
        if self._font is not None:
            return
        try:
            from PIL import ImageFont
        except Exception:
            self._font = False            # 无 PIL → 退 cv2.putText
            return
        if self.font_path and Path(self.font_path).exists():
            try:
                self._font = ImageFont.truetype(self.font_path, self._pil_font_size)
                return
            except Exception as e:
                logger.warning(f"字体加载失败, 退回默认: {e}")
        try:
            self._font = ImageFont.load_default()
        except Exception:
            self._font = False

    def draw(
        self,
        image,
        dets: List[Detection],
        *,
        style: DrawStyle,
        use_label_mapping: bool = True,
    ):
        """在 image 上画所有检测框 + 标签, 返回新图 (不改原图).

        Args:
            image:            BGR ndarray (原图)
            dets:             Detection 列表
            style:            DrawStyle
            use_label_mapping: False → 全显英文

        Returns:
            annotated: BGR ndarray
        """
        import cv2

        annotated = image.copy()
        h, w = image.shape[:2]
        self._ensure_font()

        for det in dets:
            x1, y1, x2, y2 = (float(v) for v in det.xyxy)
            color = self._box_color(det.label)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                          color, style.box_thickness)

            # 只画框不画字: style.show_label=False
            if not style.show_label:
                continue
            label = self._display_label(det.label, use_label_mapping)
            if style.show_confidence and det.confidence > 0:
                label = f"{label} {det.confidence:.2f}"

            # 标签文字: 有 PIL 字体 → PIL 画 (支持中文); 否则 cv2.putText
            if self._font:
                self._draw_label_pil(annotated, label, int(x1), int(y1), color, style)
            else:
                self._draw_label_cv2(annotated, label, int(x1), int(y1), color, style)

        if style.show_legend and dets:
            annotated = self.draw_legend(annotated, dets, style=style)
        return annotated

    def draw_legend(
        self,
        image,
        dets: list[Detection],
        *,
        style: DrawStyle,
        max_items: int = 10,
    ):
        """右上角画类别图例: 色块 + 名称 + 数量, 按数量降序.

        一张图检出十几个框时, 光看框分不清"总共几种、各几个"; 图例给的是汇总。
        """
        import cv2
        from collections import Counter

        counts = Counter(d.label for d in dets)
        items = counts.most_common(max_items)
        h, w = image.shape[:2]
        pad = 6
        (_, th), baseline = cv2.getTextSize(
            "Ag", cv2.FONT_HERSHEY_SIMPLEX, style.text_scale, 1
        )
        line_h = th + baseline + pad
        panel_w = 150
        panel_h = line_h * (len(items) + (1 if len(counts) > max_items else 0)) + pad * 2
        x0 = max(0, w - panel_w - 10)
        y0 = 10

        out = image.copy()
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        alpha = style.text_bg_alpha
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        for idx, (name, cnt) in enumerate(items):
            y = y0 + pad + idx * line_h
            color = self._box_color(name)
            cv2.rectangle(out, (x0 + pad, y + 4), (x0 + pad + 12, y + line_h - 4), color, -1)
            text = self._display_label(name, True)
            cv2.putText(out, f"{text} x{cnt}", (x0 + pad + 18, y + th),
                        cv2.FONT_HERSHEY_SIMPLEX, style.text_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
        if len(counts) > max_items:
            y = y0 + pad + len(items) * line_h
            cv2.putText(out, f"... {len(counts)} classes", (x0 + pad, y + th),
                        cv2.FONT_HERSHEY_SIMPLEX, style.text_scale,
                        (200, 200, 200), 1, cv2.LINE_AA)
        return out

    def _draw_label_cv2(self, annotated, label: str, x: int, y: int,
                        color: Tuple[int, int, int], style: DrawStyle) -> None:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = style.text_scale
        th = max(1, style.box_thickness)
        (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
        y0 = max(0, y - tht - 4)
        # 半透明标签背景
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x, y0), (x + tw + 4, y + 2), (0, 0, 0), -1)
        alpha = style.text_bg_alpha
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        cv2.putText(annotated, label, (x + 2, y0 + tht), font, fs,
                    style.text_color, th, cv2.LINE_AA)

    def _draw_label_pil(self, annotated, label: str, x: int, y: int,
                        color: Tuple[int, int, int], style: DrawStyle) -> None:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(annotated[:, :, ::-1])       # BGR → RGB
        drw = ImageDraw.Draw(pil_img, "RGBA")
        fs = max(12, int(style.text_scale * 32))
        try:
            font = ImageFont.truetype(self.font_path, fs) if self.font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        tw = drw.textlength(label, font=font)
        tht = int(fs * 1.2)
        y0 = max(0, y - tht - 4)
        bg = (0, 0, 0, int(255 * style.text_bg_alpha))
        drw.rectangle([x, y0, x + tw + 6, y + 2], fill=bg)
        drw.text((x + 3, y0), label, fill=(255, 255, 255, 255), font=font)
        annotated[:, :, ::-1] = np.asarray(pil_img)             # RGB → BGR