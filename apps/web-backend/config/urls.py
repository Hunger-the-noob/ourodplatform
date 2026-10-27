from __future__ import annotations

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.static import serve
from rest_framework.routers import DefaultRouter

from config.views import frontend_index
from defect.api_views import (
    DefectRecordViewSet,
    DefectTypeViewSet,
    dashboard_stats,
    model_overview,
    training_curve,
)

router = DefaultRouter()
router.register("defect-type", DefectTypeViewSet, basename="defect-type")
router.register("defect-record", DefectRecordViewSet, basename="defect-record")

urlpatterns = [
    path("", frontend_index, name="frontend_home"),
    path("admin/", admin.site.urls),
    path("defect/", include("defect.urls")),
    path("api/assistant/", include("assistant.urls")),
    path("api/stats/", dashboard_stats, name="dashboard-stats"),
    path("api/model/", model_overview, name="model-overview"),
    path("api/model/training-curve/", training_curve, name="training-curve"),
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
