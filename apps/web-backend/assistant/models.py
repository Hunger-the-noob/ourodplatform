from __future__ import annotations

from typing import Any

from django.db import models


class VideoAnalysisJob(models.Model):
    """一次「上传视频 → 本地 YOLO 推理 → 大模型解读」的任务记录。"""

    class Status(models.TextChoices):
        PENDING = "pending", "排队中"
        RUNNING = "running", "分析中"
        COMPLETED = "completed", "已完成"
        FAILED = "failed", "失败"

    video = models.FileField(upload_to="videos/%Y%m%d/", verbose_name="原始视频")
    original_name = models.CharField(max_length=200, blank=True, verbose_name="原始文件名")
    size_bytes = models.BigIntegerField(default=0, verbose_name="文件大小(字节)")

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        verbose_name="状态",
    )
    progress = models.PositiveSmallIntegerField(default=0, verbose_name="进度(0-100)")
    stage = models.CharField(max_length=80, blank=True, verbose_name="当前阶段")
    message = models.CharField(max_length=200, blank=True, verbose_name="阶段信息")

    options = models.JSONField(default=dict, blank=True, verbose_name="分析参数")
    result = models.JSONField(default=dict, blank=True, verbose_name="结构化结果")
    explanation = models.TextField(blank=True, verbose_name="大模型解读")
    engine = models.CharField(max_length=120, blank=True, verbose_name="解释引擎")
    error = models.TextField(blank=True, verbose_name="失败原因")

    defect_record = models.ForeignKey(
        "defect.DefectRecord",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="video_jobs",
        verbose_name="归档的检测记录",
    )

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    finished_at = models.DateTimeField(null=True, blank=True, verbose_name="完成时间")

    class Meta:
        ordering = ["-created_at", "-id"]
        verbose_name = "视频分析任务"
        verbose_name_plural = "视频分析任务"

    def __str__(self) -> str:
        return f"#{self.id} {self.original_name or self.video.name} ({self.get_status_display()})"

    @property
    def is_active(self) -> bool:
        return self.status in (self.Status.PENDING, self.Status.RUNNING)

    @property
    def summary(self) -> dict[str, Any]:
        return (self.result or {}).get("summary") or {}
