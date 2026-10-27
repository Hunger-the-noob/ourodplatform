"""智能助手与视频分析的 HTTP 接口（纯 Django 视图，不依赖 DRF，便于独立演进）。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from defect.models import DefectRecord, DefectType
from django.conf import settings
from django.core.files.base import ContentFile
from django.http import HttpRequest, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .context import build_context, render_context_markdown
from .llm import LLMConfig, LLMError, chat
from .models import VideoAnalysisJob
from .prompts import build_assistant_messages, offline_assistant_reply
from .video import runtime_status, start_job, video_settings

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".mpeg", ".mpg"}
MAX_HISTORY = 12
MAX_MESSAGE_CHARS = 4000


def _json_error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"detail": message}, status=status)


def _read_json(request: HttpRequest) -> dict[str, Any]:
    try:
        payload = json.loads((request.body or b"{}").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("请求体不是合法 JSON。") from exc
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象。")
    return payload


def _media_url(request: HttpRequest, relative_path: str) -> str:
    if not relative_path:
        return ""
    base = settings.MEDIA_URL.rstrip("/")
    url = f"{base}/{relative_path.lstrip('/')}"
    return request.build_absolute_uri(url)


def _job_payload(job: VideoAnalysisJob, request: HttpRequest) -> dict[str, Any]:
    result = job.result or {}
    summary = result.get("summary") or {}
    keyframes = []
    for item in result.get("keyframes") or []:
        keyframes.append({**item, "url": _media_url(request, item.get("path") or "")})
    return {
        "id": job.id,
        "status": job.status,
        "status_label": job.get_status_display(),
        "progress": job.progress,
        "stage": job.stage,
        "message": job.message,
        "original_name": job.original_name,
        "size_bytes": job.size_bytes,
        "created_at": timezone.localtime(job.created_at).strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": timezone.localtime(job.finished_at).strftime("%Y-%m-%d %H:%M:%S")
        if job.finished_at
        else None,
        "video_url": request.build_absolute_uri(job.video.url) if job.video else "",
        "options": job.options or {},
        "engine": job.engine,
        "error": job.error,
        "explanation": job.explanation,
        "record_id": job.defect_record_id,
        "result": {
            "video": result.get("video") or {},
            "model": result.get("model") or {},
            "config": result.get("config") or {},
            "summary": summary,
            "keyframes": keyframes,
            "poster_url": _media_url(request, result.get("poster_path") or ""),
            "warnings": result.get("warnings") or [],
            "elapsed_seconds": result.get("elapsed_seconds"),
        },
    }


@require_http_methods(["GET"])
def assistant_meta(request: HttpRequest) -> JsonResponse:
    """给前端的一次性配置探测：大模型是否就绪、本地 YOLO 环境与权重。"""
    probe = str(request.GET.get("probe") or "") in {"1", "true", "yes"}
    cfg = LLMConfig.from_settings()
    return JsonResponse(
        {
            "llm": cfg.public(),
            "video": runtime_status(probe=probe),
            "chat": {"max_history": MAX_HISTORY, "max_message_chars": MAX_MESSAGE_CHARS},
            "video_accept": sorted(VIDEO_EXTENSIONS),
        }
    )


@csrf_exempt
@require_http_methods(["GET", "POST"])
def assistant_chat(request: HttpRequest) -> JsonResponse:
    """POST {messages:[{role,content}], ...} → {reply, engine, llm_configured}"""
    cfg = LLMConfig.from_settings()
    if request.method == "GET":
        return JsonResponse({"llm": cfg.public()})

    try:
        payload = _read_json(request)
    except ValueError as exc:
        return _json_error(str(exc))

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return _json_error("messages 必须是非空数组。")

    history: list[dict[str, str]] = []
    for item in raw_messages[-MAX_HISTORY:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    if not history:
        return _json_error("没有有效的对话消息。")

    context = build_context()
    context_markdown = render_context_markdown(context)
    started = time.perf_counter()
    error_message = ""

    if not cfg.configured:
        reply = offline_assistant_reply(context_markdown)
        engine = "本地规则解释"
        error_message = "未配置 LLM_API_KEY"
    else:
        try:
            reply = chat(build_assistant_messages(history, context_markdown), config=cfg)
            engine = cfg.label
        except LLMError as exc:
            logger.warning("智能助手调用大模型失败: %s", exc)
            reply = offline_assistant_reply(context_markdown, str(exc))
            engine = "本地规则解释"
            error_message = str(exc)

    return JsonResponse(
        {
            "reply": reply,
            "engine": engine,
            "llm_configured": cfg.configured,
            "error": error_message,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "context_digest": {
                "total_records": (context.get("ledger") or {}).get("total_records", 0),
                "today_records": (context.get("ledger") or {}).get("today_records", 0),
            },
        }
    )


@csrf_exempt
@require_http_methods(["GET", "POST"])
def video_job_list(request: HttpRequest) -> JsonResponse:
    if request.method == "GET":
        jobs = VideoAnalysisJob.objects.select_related("defect_record").order_by("-created_at")[:8]
        return JsonResponse({"jobs": [_job_payload(job, request) for job in jobs]})

    upload = request.FILES.get("video")
    if upload is None:
        return _json_error("请选择要上传的视频文件（字段名 video）。")

    limit_mb = int(video_settings().get("max_upload_mb") or 200)
    if upload.size > limit_mb * 1024 * 1024:
        return _json_error(f"视频大小 {upload.size / 1024 / 1024:.1f} MB 超过上限 {limit_mb} MB。")

    suffix = Path(upload.name or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return _json_error(
            f"不支持的文件类型 {suffix or '（未知）'}，请上传：{'、'.join(sorted(VIDEO_EXTENSIONS))}。"
        )

    def _float(name: str, default: float) -> float:
        try:
            return float(request.POST.get(name) or default)
        except (TypeError, ValueError):
            return default

    def _int(name: str, default: int, low: int, high: int) -> int:
        try:
            value = int(float(request.POST.get(name) or default))
        except (TypeError, ValueError):
            value = default
        return max(low, min(high, value))

    options = {
        "conf": max(0.01, min(0.99, _float("conf", float(video_settings().get("conf") or 0.25)))),
        "max_frames": _int(
            "max_frames",
            int(video_settings().get("max_frames") or 24),
            4,
            int(video_settings().get("max_analysis_frames") or 60),
        ),
        "person": str(request.POST.get("person") or "").strip()[:50],
        "device": str(request.POST.get("device") or "").strip()[:50],
    }

    job = VideoAnalysisJob.objects.create(
        video=upload,
        original_name=(upload.name or "")[:200],
        size_bytes=upload.size,
        options=options,
        status=VideoAnalysisJob.Status.PENDING,
        stage="排队中",
        message="等待推理进程启动",
    )
    start_job(job.id)
    logger.info("创建视频分析任务 #%s：%s（%.1f MB）", job.id, job.original_name, upload.size / 1024 / 1024)
    return JsonResponse({"job": _job_payload(job, request)}, status=201)


@require_http_methods(["GET"])
def video_job_detail(request: HttpRequest, job_id: int) -> JsonResponse:
    try:
        job = VideoAnalysisJob.objects.select_related("defect_record").get(pk=job_id)
    except VideoAnalysisJob.DoesNotExist:
        return _json_error("视频分析任务不存在。", status=404)
    return JsonResponse({"job": _job_payload(job, request)})


@csrf_exempt
@require_http_methods(["POST"])
def video_job_create_record(request: HttpRequest, job_id: int) -> JsonResponse:
    """把一次视频分析归档成台账里的检测记录（关键帧 + 主导类别 + 置信度）。"""
    try:
        job = VideoAnalysisJob.objects.get(pk=job_id)
    except VideoAnalysisJob.DoesNotExist:
        return _json_error("视频分析任务不存在。", status=404)

    if job.status != VideoAnalysisJob.Status.COMPLETED:
        return _json_error("分析尚未完成，暂时不能归档。", status=409)
    if job.defect_record_id:
        return _json_error("该任务已经归档过了。", status=409)

    result = job.result or {}
    summary = result.get("summary") or {}
    classes = summary.get("classes") or []
    top = classes[0] if classes else None

    type_name = (top["name"] if top else "其他")[:50]
    defect_type, _ = DefectType.objects.get_or_create(
        type_name=type_name,
        defaults={"description": "由视频智能分析任务自动创建"},
    )

    media_root = Path(settings.MEDIA_ROOT)
    poster_path = result.get("poster_path") or ""
    keyframes = result.get("keyframes") or []
    best_frame = keyframes[0] if keyframes else None

    def _read(relative: str) -> bytes | None:
        if not relative:
            return None
        path = media_root / relative
        try:
            return path.read_bytes() if path.exists() else None
        except OSError:
            return None

    original_bytes = _read(poster_path) or _read((best_frame or {}).get("path") or "")
    if original_bytes is None:
        return _json_error("找不到分析产生的关键帧图片，无法归档。", status=409)
    defect_bytes = _read((best_frame or {}).get("path") or "")

    frames = summary.get("frames_analyzed") or 0
    total = summary.get("total_detections") or 0
    confidence = float(top.get("avg_conf") or 0.0) if top else 0.0
    confidence = max(0.0, min(1.0, round(confidence, 4)))
    person = (job.options or {}).get("person") or "AI 辅助"
    device = (job.options or {}).get("device") or "VIDEO-CAM"
    remark = (
        f"视频分析 #{job.id}：抽样 {frames} 帧，检出 {total} 个目标"
        + (f"，主导类别 {top['name']}（平均置信度 {confidence:.0%}）" if top else "，未检出目标")
        + f"；源文件 {job.original_name or '视频'}"
    )

    record = DefectRecord(
        defect_type=defect_type,
        detect_person=person,
        device_id=device,
        confidence=confidence,
        remark=remark[:500],
    )
    record.original_img.save(
        f"video-{job.id}-poster.jpg",
        ContentFile(original_bytes),
        save=False,
    )
    if defect_bytes is not None:
        record.defect_img.save(
            f"video-{job.id}-detect.jpg",
            ContentFile(defect_bytes),
            save=False,
        )
    record.save()

    job.defect_record = record
    job.save(update_fields=["defect_record"])
    return JsonResponse(
        {
            "record": {
                "id": record.id,
                "defect_type_name": defect_type.type_name,
                "confidence": record.confidence,
                "detect_person": record.detect_person,
                "device_id": record.device_id,
                "remark": record.remark,
                "original_img_url": request.build_absolute_uri(record.original_img.url),
                "defect_img_url": request.build_absolute_uri(record.defect_img.url)
                if record.defect_img
                else "",
            },
            "job": _job_payload(job, request),
        },
        status=201,
    )
