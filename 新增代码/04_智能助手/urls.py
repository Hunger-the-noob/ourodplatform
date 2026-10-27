from __future__ import annotations

from django.urls import path

from . import api_views

app_name = "assistant"

urlpatterns = [
    path("meta/", api_views.assistant_meta, name="meta"),
    path("chat/", api_views.assistant_chat, name="chat"),
    path("video/", api_views.video_job_list, name="video-list"),
    path("video/<int:job_id>/", api_views.video_job_detail, name="video-detail"),
    path(
        "video/<int:job_id>/record/",
        api_views.video_job_create_record,
        name="video-record",
    ),
]
