from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image
from rest_framework.test import APIClient

from .models import (
    DefectRecord,
    DefectType,
    RecognitionRecord,
    VideoRecognitionRecord,
)
from .training_jobs import TrainingJobManager


def make_test_image() -> SimpleUploadedFile:
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (80, 60), (32, 96, 160)).save(buffer, format="JPEG")
    return SimpleUploadedFile(
        "sample.jpg",
        buffer.getvalue(),
        content_type="image/jpeg",
    )


class DefectApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.defect_type, _ = DefectType.objects.get_or_create(
            type_name="测试划痕",
            defaults={"description": "自动化测试使用的类型"},
        )

    def test_record_crud(self):
        create_response = self.client.post(
            "/api/defect-record/",
            {
                "defect_type": self.defect_type.id,
                "detect_person": "张工",
                "device_id": "CAM-01",
                "confidence": 0.92,
                "original_img": make_test_image(),
                "remark": "自动创建测试",
            },
            format="multipart",
        )
        self.assertEqual(create_response.status_code, 201, create_response.data)
        record_id = create_response.data["id"]

        list_response = self.client.get("/api/defect-record/")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.data), 1)

        update_response = self.client.patch(
            f"/api/defect-record/{record_id}/",
            {"confidence": 0.88, "remark": "已复核"},
            format="json",
        )
        self.assertEqual(update_response.status_code, 200, update_response.data)
        self.assertEqual(update_response.data["confidence"], 0.88)

        detail_response = self.client.get(f"/api/defect-record/{record_id}/")
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.data["defect_type_name"], "测试划痕")

        delete_response = self.client.delete(f"/api/defect-record/{record_id}/")
        self.assertEqual(delete_response.status_code, 204)
        self.assertFalse(DefectRecord.objects.filter(id=record_id).exists())

    def test_stats_and_type_list(self):
        stats_response = self.client.get("/api/stats/")
        self.assertEqual(stats_response.status_code, 200)
        self.assertEqual(stats_response.data["total_records"], 0)
        self.assertEqual(
            stats_response.data["total_types"],
            DefectType.objects.count(),
        )

        type_response = self.client.get("/api/defect-type/")
        self.assertEqual(type_response.status_code, 200)
        type_names = {item["type_name"] for item in type_response.data}
        self.assertIn("测试划痕", type_names)

    def test_model_overview_endpoint(self):
        response = self.client.get("/api/model/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.data["status"],
            {"completed", "training", "not_found"},
        )

    @patch("defect.api_views.recognize_image_file")
    def test_recognition_upload_and_history(self, recognize_mock):
        uploaded = make_test_image()
        image_bytes = uploaded.read()
        uploaded.seek(0)
        recognize_mock.return_value = {
            "original_bytes": image_bytes,
            "annotated_bytes": image_bytes,
            "object_count": 2,
            "model_name": "test-best.pt",
            "confidence_threshold": 0.3,
            "iou_threshold": 0.7,
            "inference_ms": 42.5,
            "detections": [
                {
                    "class_id": 1,
                    "label": "class_1",
                    "confidence": 0.91,
                    "bbox": [1, 2, 30, 40],
                }
            ],
            "class_counts": {"class_1": 2},
            "width": 80,
            "height": 60,
        }

        response = self.client.post(
            "/api/recognition/",
            {
                "image": uploaded,
                "confidence": "0.3",
                "iou": "0.7",
            },
            format="multipart",
        )
        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["object_count"], 2)
        self.assertEqual(response.data["class_counts"], {"class_1": 2})
        self.assertTrue(response.data["annotated_img_url"].endswith(".jpg"))
        self.assertEqual(RecognitionRecord.objects.count(), 1)

        history = self.client.get("/api/recognition/")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.data), 1)
        recognize_mock.assert_called_once()

        record_id = response.data["id"]
        deleted = self.client.delete(f"/api/recognition/{record_id}/")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(RecognitionRecord.objects.count(), 0)

    def test_recognition_requires_image(self):
        response = self.client.post("/api/recognition/", {}, format="multipart")
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.data)

    def test_dataset_and_dashboard_endpoints(self):
        dataset = self.client.get("/api/data/")
        self.assertEqual(dataset.status_code, 200)
        self.assertEqual(dataset.data["class_count"], 10)
        self.assertGreater(dataset.data["total_images"], 0)

        dashboard = self.client.get("/api/dashboard/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("defects", dashboard.data)
        self.assertIn("recognition", dashboard.data)

    @patch("defect.api_views.recognize_video_file")
    def test_video_recognition_history_and_delete(self, recognize_mock):
        video = SimpleUploadedFile(
            "sample.mp4",
            b"fake video content",
            content_type="video/mp4",
        )
        recognize_mock.return_value = {
            "annotated_bytes": b"fake annotated video",
            "preview_bytes": b"fake preview",
            "original_frames": 20,
            "processed_frames": 10,
            "object_count": 3,
            "model_name": "test-best.pt",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "inference_ms": 520.0,
            "class_counts": {"class_2": 3},
            "detections": [
                {
                    "frame": 0,
                    "label": "class_2",
                    "confidence": 0.8,
                    "bbox": [1, 2, 3, 4],
                }
            ],
            "truncated": False,
        }
        response = self.client.post(
            "/api/video-recognition/",
            {"video": video, "confidence": "0.25", "frame_stride": "2"},
            format="multipart",
        )
        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["processed_frames"], 10)
        self.assertEqual(VideoRecognitionRecord.objects.count(), 1)

        history = self.client.get("/api/video-recognition/")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.data), 1)

        deleted = self.client.delete(
            f"/api/video-recognition/{response.data['id']}/",
        )
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(VideoRecognitionRecord.objects.count(), 0)

    @patch("defect.training_jobs.subprocess.Popen")
    def test_training_job_manager_lifecycle(self, popen_mock):
        process = MagicMock()
        process.poll.return_value = None
        popen_mock.return_value = process
        manager = TrainingJobManager()

        snapshot = manager.start(
            epochs=1,
            imgsz=416,
            batch="16",
            device="cpu",
        )
        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(snapshot["total_epochs"], 1)

        status = manager.status()
        self.assertEqual(status["status"], "running")

        cancelled = manager.cancel()
        self.assertEqual(cancelled["status"], "cancelled")
        process.terminate.assert_called_once()
