from __future__ import annotations

from django.contrib import admin

from .models import VideoAnalysisJob


@admin.register(VideoAnalysisJob)
class VideoAnalysisJobAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "original_name",
        "status",
        "progress",
        "stage",
        "engine",
        "created_at",
        "finished_at",
    )
    list_filter = ("status", "engine")
    search_fields = ("original_name", "stage", "message")
    readonly_fields = ("result", "explanation", "error", "created_at", "finished_at")
    ordering = ("-created_at",)
