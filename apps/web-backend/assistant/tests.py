"""assistant 应用测试：统计聚合、LLM 客户端、接口、归档逻辑。

运行：``python manage.py test assistant``
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest import mock

from defect.models import DefectRecord
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image

from . import explain, llm, stats
from .models import VideoAnalysisJob


def _jpeg_bytes(color: tuple[int, int, int] = (30, 90, 140)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="JPEG")
    return buffer.getvalue()


class StatsTests(TestCase):
    def test_summarize_groups_by_class_and_frame(self):
        detections = [
            {"frame_index": 0, "time": 0.0, "cls_id": 0, "cls_name": "bottle", "conf": 0.9,
             "area_ratio": 0.2},
            {"frame_index": 0, "time": 0.0, "cls_id": 0, "cls_name": "bottle", "conf": 0.7,
             "area_ratio": 0.1},
            {"frame_index": 10, "time": 1.0, "cls_id": 1, "cls_name": "cup", "conf": 0.6,
             "area_ratio": 0.05},
        ]
        summary = stats.summarize(detections, frames_analyzed=4, total_frames=40, fps=10)

        self.assertEqual(summary["frames_analyzed"], 4)
        self.assertEqual(summary["total_detections"], 3)
        self.assertEqual(summary["detected_frames"], 2)
        self.assertEqual(summary["detection_frame_ratio"], 0.5)
        self.assertEqual(summary["max_objects_in_frame"], 2)
        self.assertEqual(summary["classes"][0]["name"], "bottle")
        self.assertEqual(summary["classes"][0]["frames"], 1)
        self.assertEqual(summary["classes"][0]["detections"], 2)
        self.assertEqual(summary["classes"][0]["avg_conf"], 0.8)
        self.assertEqual(summary["classes"][1]["name"], "cup")
        self.assertEqual([item["frame"] for item in summary["timeline"]], [0, 10])
        self.assertEqual(summary["timeline"][0]["classes"], {"bottle": 2})
        self.assertAlmostEqual(summary["duration_seconds"], 4.0)

    def test_summarize_without_detections(self):
        summary = stats.summarize([], frames_analyzed=5, total_frames=50, fps=25)
        self.assertEqual(summary["total_detections"], 0)
        self.assertIsNone(summary["avg_confidence"])
        self.assertEqual(summary["classes"], [])
        self.assertIsNone(stats.dominant_class(summary))


class LLMClientTests(TestCase):
    def _settings(self, **overrides):
        video = dict(self.settings_dict["video"])
        llm_settings = dict(self.settings_dict["llm"])
        llm_settings.update(overrides)
        return {**self.settings_dict, "llm": llm_settings, "video": video}

    def setUp(self):
        from django.conf import settings as django_settings

        self.settings_dict = {
            "llm": dict(django_settings.ASSISTANT["llm"]),
            "video": dict(django_settings.ASSISTANT["video"]),
        }

    def test_not_configured_raises(self):
        with override_settings(ASSISTANT=self._settings(api_key="")):
            with self.assertRaises(llm.LLMError):
                llm.chat([{"role": "user", "content": "hi"}])

    @mock.patch("assistant.llm.requests.post")
    def test_chat_success(self, post):
        post.return_value = mock.Mock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "  这是模型回复  "}}]},
        )
        with override_settings(ASSISTANT=self._settings(api_key="sk-test", model="deepseek-chat")):
            reply = llm.chat([{"role": "user", "content": "你好"}])

        self.assertEqual(reply, "这是模型回复")
        args, kwargs = post.call_args
        self.assertTrue(args[0].endswith("/chat/completions"))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(kwargs["json"]["model"], "deepseek-chat")
        self.assertFalse(kwargs["json"]["stream"])

    @mock.patch("assistant.llm.requests.post")
    def test_chat_auth_error_is_friendly(self, post):
        post.return_value = mock.Mock(
            status_code=401,
            json=lambda: {"error": {"message": "invalid api key"}},
        )
        with override_settings(ASSISTANT=self._settings(api_key="bad")):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.chat([{"role": "user", "content": "hi"}])
        self.assertIn("鉴权失败", str(ctx.exception))
        self.assertIn("invalid api key", str(ctx.exception))


class ChatApiTests(TestCase):
    def test_chat_without_key_returns_local_context(self):
        from django.conf import settings as django_settings

        # .env 里可能已配置真实 Key，这里显式清空以隔离「无 Key」场景
        assistant_settings = {
            **django_settings.ASSISTANT,
            "llm": {**django_settings.ASSISTANT["llm"], "api_key": ""},
        }
        with override_settings(ASSISTANT=assistant_settings):
            response = self.client.post(
                "/api/assistant/chat/",
                data=json.dumps({"messages": [{"role": "user", "content": "现在有多少条检测记录？"}]}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["llm_configured"])
        self.assertEqual(payload["engine"], "本地规则解释")
        self.assertIn("台账概览", payload["reply"])

    @mock.patch("assistant.api_views.chat")
    def test_chat_proxies_to_llm_when_configured(self, fake_chat):
        from django.conf import settings as django_settings

        fake_chat.return_value = "台账里目前有 0 条记录。"
        assistant_settings = {
            **django_settings.ASSISTANT,
            "llm": {**django_settings.ASSISTANT["llm"], "api_key": "sk-test"},
        }
        with override_settings(ASSISTANT=assistant_settings):
            response = self.client.post(
                "/api/assistant/chat/",
                data=json.dumps({"messages": [{"role": "user", "content": "有多少记录"}]}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reply"], "台账里目前有 0 条记录。")
        self.assertTrue(payload["llm_configured"])
        self.assertEqual(payload["engine"], "DeepSeek · " + django_settings.ASSISTANT["llm"]["model"])
        sent_messages = fake_chat.call_args[0][0]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertIn("系统上下文", sent_messages[0]["content"])

    def test_chat_rejects_bad_body(self):
        response = self.client.post(
            "/api/assistant/chat/",
            data="not-json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON", response.json()["detail"])


class MetaApiTests(TestCase):
    def test_meta_does_not_probe_by_default(self):
        response = self.client.get("/api/assistant/meta/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("llm", payload)
        self.assertIn("model", payload["video"])
        self.assertNotIn("api_key", payload["llm"])
        self.assertIn(payload["video"]["python"]["available"], (None, True, False))


class VideoUploadTests(TestCase):
    def setUp(self):
        self.media_dir = tempfile.mkdtemp(prefix="odp-media-")
        self.override = override_settings(MEDIA_ROOT=self.media_dir)
        self.override.enable()

    def tearDown(self):
        self.override.disable()

    @mock.patch("assistant.api_views.start_job")
    def test_upload_creates_job(self, start_job):
        video = SimpleUploadedFile("demo.mp4", b"fake-video-bytes", content_type="video/mp4")
        response = self.client.post(
            "/api/assistant/video/",
            data={
                "video": video,
                "conf": "0.4",
                "max_frames": "10",
                "person": "张工",
                "device": "CAM-A01",
            },
        )
        self.assertEqual(response.status_code, 201, response.content)
        payload = response.json()["job"]
        self.assertEqual(payload["status"], VideoAnalysisJob.Status.PENDING)
        self.assertEqual(payload["options"]["conf"], 0.4)
        self.assertEqual(payload["options"]["max_frames"], 10)
        self.assertEqual(payload["options"]["person"], "张工")
        self.assertTrue(payload["video_url"].endswith(".mp4"))
        start_job.assert_called_once_with(payload["id"])

    @mock.patch("assistant.api_views.start_job")
    def test_upload_rejects_wrong_extension(self, start_job):
        video = SimpleUploadedFile("notes.txt", b"hello", content_type="text/plain")
        response = self.client.post("/api/assistant/video/", data={"video": video})
        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持的文件类型", response.json()["detail"])
        start_job.assert_not_called()

    def test_upload_requires_file(self):
        response = self.client.post("/api/assistant/video/", data={})
        self.assertEqual(response.status_code, 400)

    def test_job_detail_404(self):
        response = self.client.get("/api/assistant/video/9999/")
        self.assertEqual(response.status_code, 404)


class VideoJobRecordTests(TestCase):
    def setUp(self):
        self.media_dir = Path(tempfile.mkdtemp(prefix="odp-media-"))
        self.override = override_settings(MEDIA_ROOT=str(self.media_dir))
        self.override.enable()
        frame_dir = self.media_dir / "video_jobs" / "7"
        frame_dir.mkdir(parents=True)
        (frame_dir / "poster.jpg").write_bytes(_jpeg_bytes())
        (frame_dir / "frames").mkdir()
        (frame_dir / "frames" / "kf_000010.jpg").write_bytes(_jpeg_bytes((10, 120, 60)))

        self.job = VideoAnalysisJob.objects.create(
            video="videos/demo.mp4",
            original_name="demo.mp4",
            size_bytes=1024,
            status=VideoAnalysisJob.Status.COMPLETED,
            progress=100,
            options={"person": "李工", "device": "VID-02", "conf": 0.3},
            result={
                "summary": {
                    "frames_analyzed": 8,
                    "total_detections": 5,
                    "classes": [
                        {"name": "bottle", "frames": 4, "detections": 5, "avg_conf": 0.82},
                    ],
                },
                "poster_path": "video_jobs/7/poster.jpg",
                "keyframes": [
                    {"path": "video_jobs/7/frames/kf_000010.jpg", "time": 1.0, "count": 2},
                ],
            },
            explanation="## 一、结论摘要",
            engine="本地规则解释",
        )

    def tearDown(self):
        self.override.disable()

    def test_archive_creates_defect_record(self):
        response = self.client.post(f"/api/assistant/video/{self.job.id}/record/")
        self.assertEqual(response.status_code, 201, response.content)
        payload = response.json()["record"]
        self.assertEqual(payload["defect_type_name"], "bottle")
        self.assertEqual(payload["detect_person"], "李工")
        self.assertEqual(payload["confidence"], 0.82)

        record = DefectRecord.objects.get(pk=payload["id"])
        self.assertEqual(record.device_id, "VID-02")
        self.assertTrue(record.original_img.name.startswith("original/"))
        self.assertTrue(record.defect_img.name.startswith("defect_img/"))
        self.assertIn("视频分析", record.remark)

        self.job.refresh_from_db()
        self.assertEqual(self.job.defect_record_id, record.id)

    def test_archive_twice_is_rejected(self):
        self.assertEqual(self.client.post(f"/api/assistant/video/{self.job.id}/record/").status_code, 201)
        second = self.client.post(f"/api/assistant/video/{self.job.id}/record/")
        self.assertEqual(second.status_code, 409)

    def test_archive_requires_completed_job(self):
        self.job.status = VideoAnalysisJob.Status.RUNNING
        self.job.save(update_fields=["status"])
        response = self.client.post(f"/api/assistant/video/{self.job.id}/record/")
        self.assertEqual(response.status_code, 409)


class FallbackExplanationTests(TestCase):
    def test_fallback_contains_required_sections_and_real_numbers(self):
        result = {
            "video": {"duration_seconds": 4.0},
            "model": {"path": "yolo11n.pt", "source": "pretrained"},
            "summary": {
                "frames_analyzed": 6,
                "detected_frames": 3,
                "total_detections": 4,
                "max_objects_in_frame": 2,
                "first_detection_time": 0.5,
                "last_detection_time": 3.0,
                "classes": [
                    {"name": "bottle", "frames": 3, "frame_ratio": 0.5, "detections": 4,
                     "avg_conf": 0.71, "max_conf": 0.9},
                ],
                "timeline": [
                    {"frame": 5, "time": 0.5, "total": 2, "classes": {"bottle": 2}},
                    {"frame": 30, "time": 3.0, "total": 2, "classes": {"bottle": 2}},
                ],
            },
        }
        text = explain.fallback_explanation(result, reason="未配置 LLM_API_KEY")
        for heading in ("## 一、结论摘要", "## 二、目标类别解读", "## 三、时间与分布观察",
                        "## 四、处置建议与局限"):
            self.assertIn(heading, text)
        self.assertIn("bottle", text)
        self.assertIn("4", text)
        self.assertIn("未配置 LLM_API_KEY", text)
        self.assertIn("不是本仓库训练产物", text)


class ModelResolutionTests(TestCase):
    @mock.patch("assistant.video._repo_root")
    def test_falls_back_to_downloadable_model(self, repo_root):
        from django.conf import settings as django_settings

        repo_root.return_value = Path(tempfile.mkdtemp())
        video_settings = {**django_settings.ASSISTANT["video"], "model": "", "fallback_model": "yolo11n.pt"}
        with override_settings(ASSISTANT={**django_settings.ASSISTANT, "video": video_settings}):
            from assistant.video import resolve_model

            model = resolve_model()
        self.assertEqual(model["path"], "yolo11n.pt")
        self.assertEqual(model["source"], "pretrained-download")
        self.assertFalse(model["exists"])
