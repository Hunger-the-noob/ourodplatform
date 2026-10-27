"""Web 数据任务与质检报告接口。"""

import math

from django.http import FileResponse
from rest_framework import serializers
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .data_catalog import DEFAULT_DATASET, dataset_catalog, dataset_yaml
from .data_jobs import data_jobs
from .data_reports import get_report, list_reports, report_path
from .training_jobs import training_jobs


class ValidationOptions(serializers.Serializer):
    dataset = serializers.CharField(default=DEFAULT_DATASET, max_length=80)
    compare_previous = serializers.BooleanField(default=True)
    baseline_run_id = serializers.RegexField(
        r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", required=False, allow_null=True
    )

    def validate_dataset(self, value):
        try:
            dataset_yaml(value)
        except ValueError as error:
            raise serializers.ValidationError(str(error)) from error
        return value

    def validate(self, attrs):
        if attrs.get("baseline_run_id") and not attrs["compare_previous"]:
            raise serializers.ValidationError("指定历史基线时需要开启历史对比。")
        return attrs


class TransformOptions(serializers.Serializer):
    dataset = serializers.CharField(max_length=80)
    format = serializers.ChoiceField(choices=("pascal_voc", "coco", "yolo"), default="yolo")
    task = serializers.ChoiceField(choices=("detect", "segment"), default="detect")
    split_strategy = serializers.ChoiceField(
        choices=("random", "stratified_multilabel"), default="random"
    )
    seed = serializers.IntegerField(min_value=0, max_value=2**32 - 1, default=42)
    train_rate = serializers.FloatField(min_value=0.01, max_value=1, default=0.8)
    val_rate = serializers.FloatField(min_value=0, max_value=0.99, default=0.1)
    classes = serializers.ListField(
        child=serializers.CharField(max_length=200), allow_null=True, required=False
    )
    validate_dataset = ValidationOptions.validate_dataset

    def validate(self, attrs):
        rates = (attrs["train_rate"], attrs["val_rate"])
        if not all(math.isfinite(value) for value in rates) or sum(rates) > 1:
            raise serializers.ValidationError(
                "划分比例必须是有限数值，训练集与验证集比例之和不能大于 1。"
            )
        if attrs["format"] == "pascal_voc" and attrs["task"] == "segment":
            raise serializers.ValidationError("VOC 格式不支持分割任务。")
        classes = attrs.get("classes")
        if attrs["format"] == "yolo" and not classes:
            raise serializers.ValidationError("YOLO 格式需要按 class_id 顺序提供类别名称。")
        if classes is not None and len(classes) != len(set(classes)):
            raise serializers.ValidationError("类别名称不能重复。")
        return attrs


def _name(request):
    name = request.query_params.get("dataset", DEFAULT_DATASET)
    dataset_yaml(name)
    return name


@api_view(["GET"])
def catalog(request):
    return Response({"datasets": dataset_catalog()})


# 兼容入口：原有接口的实际定义保留在 api_views.py，本文件只实现新增接口。
def dataset_overview(request):
    from .api_views import dataset_overview as original_view

    return original_view(getattr(request, "_request", request))


def dataset_sample(request, split, filename):
    from .api_views import dataset_sample as original_view

    return original_view(getattr(request, "_request", request), split=split, filename=filename)


def _start(action, serializer, request):
    options = serializer(data=request.data)
    if not options.is_valid():
        return Response({"detail": "请检查数据任务参数。", "errors": options.errors}, status=400)
    if action == "transform" and training_jobs.status().get("status") == "running":
        return Response({"detail": "训练正在使用数据，结束训练后才能重新构建数据集。"}, status=409)
    try:
        return Response(data_jobs.start(action, options.validated_data), status=202)
    except RuntimeError as error:
        return Response({"detail": str(error)}, status=409)
    except (ValueError, FileNotFoundError) as error:
        return Response({"detail": str(error)}, status=400)
    except OSError as error:
        return Response({"detail": f"无法启动数据任务: {error}"}, status=500)


@api_view(["POST"])
def transform_dataset(request):
    return _start("transform", TransformOptions, request)


def validate_dataset(request):
    from .api_views import validate_dataset as original_view

    return original_view(getattr(request, "_request", request))


@api_view(["GET"])
def job_status(request):
    return Response(data_jobs.status())


@api_view(["GET"])
def reports(request):
    try:
        return Response({"reports": list_reports(_name(request))})
    except ValueError as error:
        return Response({"detail": str(error)}, status=400)


@api_view(["GET"])
def report_detail(request, run_id):
    try:
        return Response(get_report(_name(request), run_id))
    except ValueError as error:
        return Response({"detail": str(error)}, status=400)
    except FileNotFoundError as error:
        return Response({"detail": str(error)}, status=404)


@api_view(["GET"])
def report_download(request, run_id, filename):
    try:
        get_report(_name(request), run_id)
        path = report_path(run_id, filename)
    except ValueError as error:
        return Response({"detail": str(error)}, status=400)
    except FileNotFoundError as error:
        return Response({"detail": str(error)}, status=404)
    if not path.is_file():
        return Response({"detail": "该报告尚未生成此文件。"}, status=404)
    return FileResponse(
        path.open("rb"),
        as_attachment=True,
        filename=f"{run_id}-{filename}",
        content_type="application/json"
        if filename.endswith(".json")
        else "text/csv; charset=utf-8",
    )
