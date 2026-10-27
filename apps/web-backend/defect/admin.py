from __future__ import annotations

from django.contrib import admin

from .models import (
    DefectRecord,
    DefectType,
    RecognitionRecord,
    VideoRecognitionRecord,
)


@admin.register(DefectType)
class DefectTypeAdmin(admin.ModelAdmin):
    list_display = ["id", "type_name", "description"]
    search_fields = ["type_name"]


@admin.register(DefectRecord)
class DefectRecordAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "defect_type",
        "detect_person",
        "device_id",
        "confidence",
        "detect_time",
    ]
    list_filter = ["defect_type", "detect_person"]
    search_fields = ["device_id", "detect_person"]
    readonly_fields = ["detect_time"]


@admin.register(RecognitionRecord)
class RecognitionRecordAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "object_count",
        "model_name",
        "confidence_threshold",
        "inference_ms",
        "created_at",
    ]
    readonly_fields = ["created_at"]


@admin.register(VideoRecognitionRecord)
class VideoRecognitionRecordAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "processed_frames",
        "object_count",
        "model_name",
        "inference_ms",
        "created_at",
    ]
    readonly_fields = ["created_at"]
