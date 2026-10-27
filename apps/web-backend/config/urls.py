from __future__ import annotations

from defect.api_views import (
    DefectRecordViewSet,
    DefectTypeViewSet,
    dashboard_overview,
    dashboard_stats,
    data_engineering,
    data_job_control,
    data_validation_history,
    dataset_overview,
    dataset_sample,
    model_evaluation,
    model_evaluation_artifact,
    model_overview,
    recognition_record_detail,
    recognition_records,
    training_cancel,
    training_control,
    training_curve,
    validate_dataset,
    video_recognition_detail,
    video_recognition_records,
)
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.static import serve
from rest_framework.routers import DefaultRouter

from config.views import frontend_index

router = DefaultRouter()
router.register("defect-type", DefectTypeViewSet, basename="defect-type")
router.register("defect-record", DefectRecordViewSet, basename="defect-record")

urlpatterns = [
    path("", frontend_index, name="frontend_home"),
    path("admin/", admin.site.urls),
    path("api/stats/", dashboard_stats, name="dashboard-stats"),
    path("api/dashboard/", dashboard_overview, name="dashboard-overview"),
    path("api/data/", dataset_overview, name="dataset-overview"),
    path("api/data/validate/", validate_dataset, name="dataset-validate"),
    path("api/data/jobs/", data_job_control, name="data-jobs"),
    path("api/data/engineering/", data_engineering, name="data-engineering"),
    path("api/data/history/", data_validation_history, name="data-history"),
    path(
        "api/data/sample/<str:split>/<str:filename>",
        dataset_sample,
        name="dataset-sample",
    ),
    path("api/model/", model_overview, name="model-overview"),
    path("api/model/training-curve/", training_curve, name="training-curve"),
    path("api/model/evaluate/", model_evaluation, name="model-evaluate"),
    path(
        "api/model/evaluation-artifact/<str:name>/",
        model_evaluation_artifact,
        name="model-evaluation-artifact",
    ),
    path("api/training/", training_control, name="training-control"),
    path("api/training/cancel/", training_cancel, name="training-cancel"),
    path("api/recognition/", recognition_records, name="recognition-records"),
    path(
        "api/recognition/<int:record_id>/",
        recognition_record_detail,
        name="recognition-record-detail",
    ),
    path(
        "api/video-recognition/",
        video_recognition_records,
        name="video-recognition-records",
    ),
    path(
        "api/video-recognition/<int:record_id>/",
        video_recognition_detail,
        name="video-recognition-detail",
    ),
    path("api/", include(router.urls)),
    path(
        "assets/<path:path>",
        serve,
        {"document_root": settings.BASE_DIR / "frontend" / "static" / "frontend"},
        name="frontend-assets",
    ),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
