from __future__ import annotations

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class DefectType(models.Model):
    type_name = models.CharField(
        max_length=50,
        unique=True,
        verbose_name="缺陷类型名称",
    )
    description = models.TextField(blank=True, verbose_name="类型说明")

    class Meta:
        ordering = ["id"]
        verbose_name = "缺陷类型"
        verbose_name_plural = "缺陷类型"

    def __str__(self) -> str:
        return self.type_name


class DefectRecord(models.Model):
    defect_type = models.ForeignKey(
        DefectType,
        on_delete=models.PROTECT,
        related_name="records",
        verbose_name="缺陷类型",
    )
    detect_person = models.CharField(max_length=50, verbose_name="检测人员")
    device_id = models.CharField(max_length=50, verbose_name="检测设备编号")
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name="检测置信度",
        help_text="0~1 之间",
    )
    original_img = models.ImageField(upload_to="original/", verbose_name="原图")
    defect_img = models.ImageField(
        upload_to="defect_img/",
        blank=True,
        verbose_name="缺陷标注图",
    )
    detect_time = models.DateTimeField(auto_now_add=True, verbose_name="检测时间")
    remark = models.TextField(blank=True, verbose_name="备注")

    class Meta:
        ordering = ["-detect_time", "-id"]
        verbose_name = "缺陷检测记录"
        verbose_name_plural = "缺陷检测记录"

    def __str__(self) -> str:
        return f"{self.defect_type.type_name} - {self.detect_person}"


class RecognitionRecord(models.Model):
    original_img = models.ImageField(
        upload_to="recognition/original/",
        verbose_name="识别原图",
    )
    annotated_img = models.ImageField(
        upload_to="recognition/annotated/",
        verbose_name="识别结果图",
    )
    object_count = models.PositiveIntegerField(default=0, verbose_name="目标数量")
    model_name = models.CharField(max_length=255, verbose_name="模型名称")
    confidence_threshold = models.FloatField(
        default=0.25,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name="置信度阈值",
    )
    iou_threshold = models.FloatField(
        default=0.7,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name="IoU 阈值",
    )
    inference_ms = models.FloatField(default=0.0, verbose_name="推理耗时(ms)")
    detections = models.JSONField(default=list, verbose_name="目标明细")
    class_counts = models.JSONField(default=dict, verbose_name="类别统计")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="识别时间")

    class Meta:
        ordering = ["-created_at", "-id"]
        verbose_name = "图片识别记录"
        verbose_name_plural = "图片识别记录"

    def __str__(self) -> str:
        return f"识别 #{self.id} - {self.object_count} 个目标"


class VideoRecognitionRecord(models.Model):
    original_video = models.FileField(
        upload_to="recognition/videos/original/",
        verbose_name="原始视频",
    )
    annotated_video = models.FileField(
        upload_to="recognition/videos/annotated/",
        verbose_name="标注视频",
    )
    annotated_preview = models.FileField(
        upload_to="recognition/videos/preview/",
        blank=True,
        verbose_name="标注动画预览",
    )
    original_frames = models.PositiveIntegerField(default=0, verbose_name="原视频帧数")
    processed_frames = models.PositiveIntegerField(default=0, verbose_name="处理帧数")
    object_count = models.PositiveIntegerField(default=0, verbose_name="目标累计数")
    model_name = models.CharField(max_length=255, verbose_name="模型名称")
    confidence_threshold = models.FloatField(default=0.25, verbose_name="置信度阈值")
    iou_threshold = models.FloatField(default=0.7, verbose_name="IoU阈值")
    inference_ms = models.FloatField(default=0.0, verbose_name="处理耗时(ms)")
    class_counts = models.JSONField(default=dict, verbose_name="类别统计")
    detections = models.JSONField(default=list, verbose_name="目标明细")
    truncated = models.BooleanField(default=False, verbose_name="是否截断")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="识别时间")

    class Meta:
        ordering = ["-created_at", "-id"]
        verbose_name = "视频识别记录"
        verbose_name_plural = "视频识别记录"

    def __str__(self) -> str:
        return f"视频识别 #{self.id} - {self.object_count} 个目标"
