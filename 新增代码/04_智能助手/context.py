"""把台账 / 模型指标 / 历史视频任务整理成大模型可读的上下文快照。"""

from __future__ import annotations

import logging
from typing import Any

from defect.model_info import get_model_overview
from defect.models import DefectRecord, DefectType
from django.db.models import Avg, Count
from django.utils import timezone

from .models import VideoAnalysisJob

logger = logging.getLogger(__name__)


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return "—"


def build_context(record_limit: int = 6, job_limit: int = 3) -> dict[str, Any]:
    """查询数据库，返回结构化上下文；单块失败不影响整体。"""
    context: dict[str, Any] = {
        "generated_at": timezone.localtime(timezone.now()).strftime("%Y-%m-%d %H:%M:%S"),
        "ledger": {},
        "records": [],
        "model": {},
        "video_jobs": [],
    }

    records = DefectRecord.objects.select_related("defect_type").all()
    try:
        aggregate = records.aggregate(total=Count("id"), avg=Avg("confidence"))
        today = timezone.localdate()
        context["ledger"] = {
            "total_records": int(aggregate["total"] or 0),
            "today_records": records.filter(detect_time__date=today).count(),
            "high_confidence_records": records.filter(confidence__gte=0.8).count(),
            "average_confidence": round(float(aggregate["avg"] or 0.0), 4),
            "type_count": DefectType.objects.count(),
            "types": [
                {"name": row["type_name"], "records": int(row["count"] or 0)}
                for row in DefectType.objects.annotate(count=Count("records"))
                .filter(count__gt=0)
                .order_by("-count", "type_name")[:8]
                .values("type_name", "count")
            ],
        }
        context["records"] = [
            {
                "id": record.id,
                "type": record.defect_type.type_name,
                "person": record.detect_person,
                "device": record.device_id,
                "confidence": round(float(record.confidence), 3),
                "time": timezone.localtime(record.detect_time).strftime("%Y-%m-%d %H:%M"),
                "remark": (record.remark or "")[:60],
            }
            for record in records.order_by("-detect_time", "-id")[:record_limit]
        ]
    except Exception as exc:  # noqa: BLE001 - 上下文不应导致对话整体失败
        logger.warning("读取台账上下文失败: %s", exc)
        context["ledger"] = {"error": "台账数据读取失败"}

    try:
        overview = get_model_overview()
        dataset = overview.get("dataset") or {}
        metrics = overview.get("metrics") or {}
        weights = overview.get("weights") or {}
        context["model"] = {
            "status": overview.get("status"),
            "run_id": overview.get("run_id"),
            "dataset": dataset.get("name"),
            "classes": dataset.get("classes") or [],
            "class_count": dataset.get("class_count"),
            "metrics": {
                "map50": metrics.get("map50"),
                "map50_95": metrics.get("map50_95"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
            },
            "weights": weights.get("archived") or weights.get("run_best"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取模型上下文失败: %s", exc)
        context["model"] = {"status": "unknown"}

    try:
        jobs = VideoAnalysisJob.objects.order_by("-created_at")[:job_limit]
        for job in jobs:
            result = job.result or {}
            summary = result.get("summary") or {}
            classes = summary.get("classes") or []
            context["video_jobs"].append(
                {
                    "id": job.id,
                    "status": job.get_status_display(),
                    "created_at": timezone.localtime(job.created_at).strftime("%m-%d %H:%M"),
                    "frames_analyzed": summary.get("frames_analyzed"),
                    "total_detections": summary.get("total_detections"),
                    "top_classes": [
                        {"name": item["name"], "conf": item.get("avg_conf")}
                        for item in classes[:3]
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取视频任务上下文失败: %s", exc)

    return context


def render_context_markdown(context: dict[str, Any]) -> str:
    """把上下文压成紧凑 Markdown，作为 system 消息的一部分。"""
    ledger = context.get("ledger") or {}
    model = context.get("model") or {}
    lines: list[str] = [f"数据时间：{context.get('generated_at', '—')}"]

    if ledger.get("error"):
        lines.append(f"台账：{ledger['error']}")
    else:
        lines.append(
            "台账概览：记录总数 {total} 条，今日 {today} 条，高置信度（≥0.80）{high} 条，"
            "平均置信度 {avg}。".format(
                total=ledger.get("total_records", 0),
                today=ledger.get("today_records", 0),
                high=ledger.get("high_confidence_records", 0),
                avg=_pct(ledger.get("average_confidence")),
            )
        )
        types = ledger.get("types") or []
        if types:
            lines.append(
                "类型分布：" + "、".join(f"{item['name']}({item['records']})" for item in types[:8])
            )
        records = context.get("records") or []
        if records:
            lines.append("最近检测记录：")
            for record in records:
                lines.append(
                    "- #{id} {type}｜{person}｜{device}｜置信度 {conf}｜{time}"
                    "{remark}".format(
                        id=str(record["id"]).zfill(4),
                        type=record["type"],
                        person=record["person"],
                        device=record["device"],
                        conf=_pct(record["confidence"]),
                        time=record["time"],
                        remark=f"｜备注：{record['remark']}" if record.get("remark") else "",
                    )
                )
        else:
            lines.append("最近检测记录：暂无（台账为空，可以引导用户先录入记录）。")

    if model.get("status") == "completed":
        metrics = model.get("metrics") or {}
        lines.append(
            "模型训练：运行 {run}，数据集 {dataset}（{count} 类），mAP@50 {map50}，"
            "mAP@50-95 {map50_95}，Precision {precision}，Recall {recall}，权重 {weights}。".format(
                run=model.get("run_id"),
                dataset=model.get("dataset"),
                count=model.get("class_count"),
                map50=_pct(metrics.get("map50")),
                map50_95=_pct(metrics.get("map50_95")),
                precision=_pct(metrics.get("precision")),
                recall=_pct(metrics.get("recall")),
                weights=model.get("weights") or "—",
            )
        )
        classes = model.get("classes") or []
        if classes:
            lines.append("模型类别：" + "、".join(str(item) for item in classes[:12]))
    elif model.get("status") == "not_found":
        lines.append("模型训练：暂无训练记录（台账里可以查看模型卡片，提示用户先执行 odp-train）。")

    jobs = context.get("video_jobs") or []
    if jobs:
        lines.append("最近视频分析任务：")
        for job in jobs:
            tops = "、".join(
                f"{item['name']}（{_pct(item.get('conf'))}）" for item in job.get("top_classes") or []
            )
            lines.append(
                "- #{id} {status}｜{time}｜抽样 {frames} 帧｜检出 {total} 个框"
                "{tops}".format(
                    id=job["id"],
                    status=job["status"],
                    time=job["created_at"],
                    frames=job.get("frames_analyzed") or 0,
                    total=job.get("total_detections") or 0,
                    tops=f"｜主要类别：{tops}" if tops else "｜未检出目标",
                )
            )
    else:
        lines.append("最近视频分析任务：暂无。")

    return "\n".join(lines)
