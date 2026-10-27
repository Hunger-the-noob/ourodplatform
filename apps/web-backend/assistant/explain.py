"""本地规则解释：大模型不可用时，仍然给出一份基于真实统计的中文报告。

这样「没配 Key / 断网」时演示链路依然完整，且报告里的数字全部来自 YOLO 结果。
"""

from __future__ import annotations

from typing import Any

from .stats import dominant_class


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return "—"


def _seconds(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}s"


def fallback_explanation(result: dict[str, Any], *, reason: str = "") -> str:
    """根据 result（video/model/config/summary）生成 Markdown 解释。"""
    summary = (result or {}).get("summary") or {}
    model = (result or {}).get("model") or {}
    classes = summary.get("classes") or []
    top = dominant_class(summary)
    frames = summary.get("frames_analyzed") or 0
    detected_frames = summary.get("detected_frames") or 0
    total = summary.get("total_detections") or 0
    source = model.get("source") or "unknown"

    lines: list[str] = ["## 一、结论摘要", ""]
    if not total:
        lines += [
            f"- 本次抽样分析了 **{frames}** 帧，**未检出**任何目标；"
            "可能是画面中没有模型可识别的类别，也可能需要降低置信度阈值。",
            "- 建议：确认权重与类别是否匹配，并在阈值 0.25 附近重新尝试。",
        ]
    else:
        lines += [
            f"- 本次抽样分析 **{frames}** 帧，其中 **{detected_frames}** 帧检出目标，"
            f"共 **{total}** 个检测框。",
            f"- 主导类别为 **{top['name']}**，平均置信度 {_pct(top.get('avg_conf'))}，"
            f"最高 {_pct(top.get('max_conf'))}；单帧最多 {summary.get('max_objects_in_frame')} 个目标。",
            f"- 首次检出约 {_seconds(summary.get('first_detection_time'))}，"
            f"最后一次约 {_seconds(summary.get('last_detection_time'))}。",
        ]

    lines += ["", "## 二、目标类别解读", ""]
    if classes:
        lines += ["| 类别 | 涉及帧数 | 检出框数 | 平均置信度 | 初步解读 |", "|---|---|---|---|---|"]
        for item in classes:
            ratio = item.get("frame_ratio") or 0
            if ratio >= 0.6:
                reading = "多帧持续出现，属于画面中的稳定目标，优先复核"
            elif ratio >= 0.2:
                reading = "间歇出现，注意是否随镜头移动进入画面"
            else:
                reading = "偶发出现，可能为边角噪声或短暂经过"
            lines.append(
                f"| {item['name']} | {item.get('frames')} | {item.get('detections')} | "
                f"{_pct(item.get('avg_conf'))} | {reading} |"
            )
    else:
        lines.append("本次没有可统计的类别。")

    lines += ["", "## 三、时间与分布观察", ""]
    timeline = [item for item in summary.get("timeline") or [] if item.get("total")]
    if timeline:
        first = timeline[0]
        last = timeline[-1]
        lines.append(
            f"- 有检出的抽样帧集中在 {_seconds(first.get('time'))} 到 {_seconds(last.get('time'))} 之间，"
            f"共 {len(timeline)} 个抽样点命中。"
        )
        peak = max(timeline, key=lambda item: item.get("total") or 0)
        peak_classes = "、".join(
            f"{name}×{count}" for name, count in (peak.get("classes") or {}).items()
        )
        lines.append(f"- 目标最密集的一帧约在 {_seconds(peak.get('time'))}：{peak_classes}。")
    else:
        lines.append("- 所有抽样帧均无检出，时间轴没有可分析的目标分布。")

    lines += ["", "## 四、处置建议与局限", "", "**立即处理**", ""]
    if total:
        lines.append(f"- 复核主导类别 **{top['name']}** 对应的关键帧，确认归类与数量是否与现场一致。")
        lines.append("- 若属于可回收/需清运垃圾，按现场流程登记并安排清理。")
    else:
        lines.append("- 目前无需处置；如现场确有垃圾，请检查镜头朝向、光照与权重类别。")
    lines += ["", "**后续跟进**", ""]
    lines.append("- 把有代表性的关键帧补入训练集，持续提升本场景精度。")
    if source != "trained":
        lines.append(
            "- **当前权重不是本仓库训练产物**（来源："
            f"{model.get('path') or '未知'}），类别可能不是垃圾类别，建议换用 `models/trained/*-best.pt`。"
        )
    else:
        lines.append(f"- 当前使用的是自训练权重（{model.get('path')}），可继续积累误检样本再训练。")
    lines += [
        "",
        f"> 本报告基于 {frames} 个抽样帧的统计，并非逐帧全量结论；"
        "置信度是模型自评分，最终判定仍需人工复核。",
    ]
    if reason:
        lines += ["", f"> 说明：本次解释由本地规则生成，没有调用大模型（{reason}）。"]
    else:
        lines += ["", "> 说明：本次解释由本地规则生成（未配置大模型 API Key）。"]
    return "\n".join(lines)
