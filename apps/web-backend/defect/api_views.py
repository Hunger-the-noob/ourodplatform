from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path
from uuid import uuid4

from django.core.files.base import ContentFile
from django.db.models import Count, Q
from django.http import FileResponse
from django.utils import timezone
from rest_framework import filters, permissions, viewsets
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .data_understanding import (
    DATASET_YAML,
    dataset_sample_path,
    get_dataset_understanding,
)
from .evaluation import evaluation_artifact, latest_evaluation, run_test_evaluation
from .models import (
    DefectRecord,
    DefectType,
    RecognitionRecord,
    VideoRecognitionRecord,
)
from .model_info import get_model_overview, latest_training_curve
from .recognition import recognize_image_file
from .serializers import (
    DefectRecordSerializer,
    DefectTypeSerializer,
    RecognitionRecordSerializer,
    VideoRecognitionRecordSerializer,
)
from .training_jobs import training_jobs
from .video_recognition import recognize_video_file


class DefectTypeViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = DefectType.objects.annotate(record_count=Count("records"))
    serializer_class = DefectTypeSerializer
    authentication_classes = []
    permission_classes = [permissions.AllowAny]


class DefectRecordViewSet(viewsets.ModelViewSet):
    queryset = DefectRecord.objects.select_related("defect_type").all()
    serializer_class = DefectRecordSerializer
    authentication_classes = []
    permission_classes = [permissions.AllowAny]
    filter_backends = [filters.SearchFilter]
    search_fields = ["defect_type__type_name", "detect_person", "device_id", "remark"]

    def get_queryset(self):
        queryset = super().get_queryset()
        defect_type = self.request.query_params.get("defect_type")
        if defect_type:
            queryset = queryset.filter(defect_type_id=defect_type)
        return queryset


def _defect_stats_payload() -> dict:
    records = DefectRecord.objects.all()
    today = timezone.localdate()
    type_counts = (
        DefectType.objects.annotate(
            count=Count("records"),
        )
        .values("type_name", "count")
        .order_by("-count", "type_name")
    )
    return {
        "total_records": records.count(),
        "today_records": records.filter(detect_time__date=today).count(),
        "high_confidence": records.filter(confidence__gte=0.8).count(),
        "total_types": DefectType.objects.count(),
        "average_confidence": round(
            (
                sum(records.values_list("confidence", flat=True))
                / max(records.count(), 1)
            ),
            3,
        ),
        "type_counts": list(type_counts),
    }


@api_view(["GET"])
def dashboard_stats(request):
    return Response(_defect_stats_payload())


@api_view(["GET"])
def model_overview(request):
    return Response(get_model_overview())


@api_view(["GET"])
def training_curve(request):
    curve = latest_training_curve()
    if curve is None:
        return Response({"detail": "暂无训练曲线图。"}, status=404)
    return FileResponse(curve.open("rb"), content_type="image/png")


@api_view(["GET"])
def dataset_overview(request):
    return Response(get_dataset_understanding())


@api_view(["GET"])
def dataset_sample(request, split: str, filename: str):
    sample = dataset_sample_path(split, filename)
    if sample is None:
        return Response({"detail": "样本图片不存在。"}, status=404)
    content_type = "image/jpeg"
    if sample.suffix.lower() == ".png":
        content_type = "image/png"
    return FileResponse(sample.open("rb"), content_type=content_type)


@api_view(["POST"])
def validate_dataset(request):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "od_platform.cli.validate_data",
            str(DATASET_YAML),
        ],
        cwd=str(DATASET_YAML.parents[4]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    details = get_dataset_understanding()
    return Response(
        {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "validation": details.get("validation", {}),
            "summary": (
                "数据验证通过"
                if result.returncode == 0
                else "数据验证未通过，请查看命令行日志"
            ),
            "output_tail": "\n".join(result.stdout.splitlines()[-20:]),
        },
        status=200 if result.returncode == 0 else 422,
    )


@api_view(["GET", "POST"])
def training_control(request):
    if request.method == "GET":
        return Response(training_jobs.status())
    current = training_jobs.status()
    if current.get("status") == "running":
        return Response({"detail": "已有训练任务正在运行。"}, status=409)
    try:
        snapshot = training_jobs.start(
            epochs=int(request.data.get("epochs", 1)),
            imgsz=int(request.data.get("imgsz", 416)),
            batch=str(request.data.get("batch", "16")),
            device=str(request.data.get("device", "cpu")),
        )
    except RuntimeError as exc:
        return Response({"detail": str(exc)}, status=409)
    except (ValueError, FileNotFoundError) as exc:
        return Response({"detail": str(exc)}, status=400)
    return Response(snapshot, status=202)


@api_view(["POST"])
def training_cancel(request):
    return Response(training_jobs.cancel())


@api_view(["GET", "POST"])
def model_evaluation(request):
    if request.method == "GET":
        return Response(latest_evaluation() or {"status": "not_found"})
    if training_jobs.status().get("status") == "running":
        return Response(
            {"detail": "训练运行期间不能执行测试集评估。"},
            status=409,
        )
    try:
        evaluation = run_test_evaluation(
            imgsz=int(request.data.get("imgsz", 416)),
            device=str(request.data.get("device", "cpu")),
        )
    except (ValueError, FileNotFoundError) as exc:
        return Response({"detail": str(exc)}, status=400)
    except Exception as exc:
        return Response({"detail": f"测试集评估失败: {exc}"}, status=500)
    return Response(evaluation)


@api_view(["GET"])
def model_evaluation_artifact(request, name: str):
    artifact = evaluation_artifact(name)
    if artifact is None:
        return Response({"detail": "评估图表不存在。"}, status=404)
    return FileResponse(artifact.open("rb"), content_type="image/png")


@api_view(["GET"])
def dashboard_overview(request):
    defect_stats = _defect_stats_payload()
    recognition_records = RecognitionRecord.objects.all()
    recognition_class_counts: Counter[str] = Counter()
    for counts in recognition_records.values_list("class_counts", flat=True):
        if isinstance(counts, dict):
            recognition_class_counts.update(
                {
                    str(label): int(count)
                    for label, count in counts.items()
                    if isinstance(count, (int, float))
                }
            )
    today = timezone.localdate()
    model = get_model_overview()
    dataset = get_dataset_understanding()
    return Response(
        {
            "defects": defect_stats,
            "recognition": {
                "total": recognition_records.count(),
                "today": recognition_records.filter(
                    created_at__date=today,
                ).count(),
                "objects": sum(
                    recognition_records.values_list(
                        "object_count",
                        flat=True,
                    )
                ),
                "videos": VideoRecognitionRecord.objects.count(),
                "class_counts": dict(recognition_class_counts),
                "recent": RecognitionRecordSerializer(
                    recognition_records[:5],
                    many=True,
                    context={"request": request},
                ).data,
            },
            "model": {
                "status": model.get("status"),
                "run_id": model.get("run_id"),
                "metrics": model.get("metrics", {}),
                "training_job": training_jobs.status(),
            },
            "dataset": {
                "name": dataset.get("name"),
                "total_images": dataset.get("total_images"),
                "total_boxes": dataset.get("total_boxes"),
                "class_count": dataset.get("class_count"),
            },
            "recent_defects": DefectRecordSerializer(
                DefectRecord.objects.select_related("defect_type")[:5],
                many=True,
                context={"request": request},
            ).data,
        }
    )


@api_view(["GET", "POST"])
@parser_classes([MultiPartParser, FormParser])
def recognition_records(request):
    if request.method == "GET":
        records = RecognitionRecord.objects.all()[:30]
        serializer = RecognitionRecordSerializer(
            records,
            many=True,
            context={"request": request},
        )
        return Response(serializer.data)

    image = request.FILES.get("image")
    if image is None:
        return Response({"detail": "请上传图片。"}, status=400)
    if image.size > 10 * 1024 * 1024:
        return Response({"detail": "图片不能超过 10 MB。"}, status=400)
    try:
        confidence = float(request.data.get("confidence", 0.25))
        iou = float(request.data.get("iou", 0.7))
        result = recognize_image_file(image, confidence=confidence, iou=iou)
    except (ValueError, FileNotFoundError) as exc:
        return Response({"detail": str(exc)}, status=400)
    except Exception as exc:
        return Response(
            {"detail": f"模型识别失败: {exc}"},
            status=500,
        )

    token = uuid4().hex
    record = RecognitionRecord.objects.create(
        original_img=ContentFile(
            result["original_bytes"],
            name=f"{token}.jpg",
        ),
        annotated_img=ContentFile(
            result["annotated_bytes"],
            name=f"{token}.jpg",
        ),
        object_count=result["object_count"],
        model_name=result["model_name"],
        confidence_threshold=result["confidence_threshold"],
        iou_threshold=result["iou_threshold"],
        inference_ms=result["inference_ms"],
        detections=result["detections"],
        class_counts=result["class_counts"],
    )
    serializer = RecognitionRecordSerializer(
        record,
        context={"request": request},
    )
    return Response(serializer.data, status=201)


@api_view(["DELETE"])
def recognition_record_detail(request, record_id: int):
    try:
        record = RecognitionRecord.objects.get(id=record_id)
    except RecognitionRecord.DoesNotExist:
        return Response({"detail": "识别记录不存在。"}, status=404)
    record.original_img.delete(save=False)
    record.annotated_img.delete(save=False)
    record.delete()
    return Response(status=204)


@api_view(["GET", "POST"])
@parser_classes([MultiPartParser, FormParser])
def video_recognition_records(request):
    if request.method == "GET":
        records = VideoRecognitionRecord.objects.all()[:20]
        serializer = VideoRecognitionRecordSerializer(
            records,
            many=True,
            context={"request": request},
        )
        return Response(serializer.data)

    video = request.FILES.get("video")
    if video is None:
        return Response({"detail": "请上传视频。"}, status=400)
    if video.size > 100 * 1024 * 1024:
        return Response({"detail": "视频不能超过 100 MB。"}, status=400)
    try:
        confidence = float(request.data.get("confidence", 0.25))
        iou = float(request.data.get("iou", 0.7))
        frame_stride = int(request.data.get("frame_stride", 2))
        result = recognize_video_file(
            video,
            confidence=confidence,
            iou=iou,
            frame_stride=frame_stride,
        )
        video.seek(0)
        original_bytes = video.read()
    except (ValueError, FileNotFoundError) as exc:
        return Response({"detail": str(exc)}, status=400)
    except Exception as exc:
        return Response({"detail": f"视频识别失败: {exc}"}, status=500)

    token = uuid4().hex
    suffix = "".join(Path(video.name).suffix.lower().split()) or ".mp4"
    record = VideoRecognitionRecord.objects.create(
        original_video=ContentFile(original_bytes, name=f"{token}{suffix}"),
        annotated_video=ContentFile(result["annotated_bytes"], name=f"{token}.mp4"),
        annotated_preview=ContentFile(result["preview_bytes"], name=f"{token}.webp"),
        original_frames=result["original_frames"],
        processed_frames=result["processed_frames"],
        object_count=result["object_count"],
        model_name=result["model_name"],
        confidence_threshold=result["confidence_threshold"],
        iou_threshold=result["iou_threshold"],
        inference_ms=result["inference_ms"],
        class_counts=result["class_counts"],
        detections=result["detections"],
        truncated=result["truncated"],
    )
    serializer = VideoRecognitionRecordSerializer(
        record,
        context={"request": request},
    )
    return Response(serializer.data, status=201)


@api_view(["DELETE"])
def video_recognition_detail(request, record_id: int):
    try:
        record = VideoRecognitionRecord.objects.get(id=record_id)
    except VideoRecognitionRecord.DoesNotExist:
        return Response({"detail": "视频识别记录不存在。"}, status=404)
    record.original_video.delete(save=False)
    record.annotated_video.delete(save=False)
    record.annotated_preview.delete(save=False)
    record.delete()
    return Response(status=204)
