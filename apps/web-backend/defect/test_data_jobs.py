"""新增：D3/D4 Web API 与后台任务回归，无需模型和真实业务数据。"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase
from PIL import Image
from rest_framework.test import APIClient

from od_platform.common import paths, stamp

from .data_jobs import DataJobManager


class DataEngineeringApiTests(SimpleTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        values = {
            "ROOT_DIR": root,
            "RAW_DATA_DIR": root / "raw",
            "PROCESSED_DATA_DIR": root / "processed",
            "RUNS_DIR": root / "runs",
            "DATASET_CONFIGS_DIR": root / "configs",
            "VALIDATION_RUNS_DIR": root / "validation",
        }
        for key, value in values.items():
            replacement = patch.object(paths, key, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        replacement = patch.object(stamp, "LINEAGE_DIR", root / "lineage")
        replacement.start()
        self.addCleanup(replacement.stop)
        self.manager = DataJobManager()
        replacement = patch("defect.api_views.data_jobs", self.manager)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.client = APIClient()
        raw = paths.raw_dataset_root("tiny")
        (raw / "images").mkdir(parents=True)
        (raw / "labels").mkdir()
        for i in range(12):
            Image.new("RGB", (32, 32)).save(raw / "images" / f"{i}.jpg")
            (raw / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def wait_complete(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = self.client.get("/api/data/jobs/")
            if response.data["status"] != "running":
                self.assertEqual(response.data["status"], "completed", response.data)
                return response.data
            time.sleep(0.01)
        self.fail("数据任务未在 10 秒内完成")

    def transform(self):
        response = self.client.post(
            "/api/data/jobs/",
            {"operation": "transform", "dataset": "tiny", "format": "yolo", "classes": ["item"]},
            format="json",
        )
        self.assertEqual(response.status_code, 202, response.data)
        return self.wait_complete()

    def test_catalog_to_transform_to_validation_and_history(self):
        catalog = self.client.get("/api/data/engineering/")
        self.assertEqual(catalog.status_code, 200)
        self.assertIn("image_integrity", catalog.data["checks"])
        self.assertTrue(catalog.data["datasets"][0]["has_raw"])
        self.assertEqual(sum(self.transform()["result"]["counts"].values()), 12)
        response = self.client.post("/api/data/validate/", {"dataset": "tiny"}, format="json")
        self.assertEqual(response.status_code, 202, response.data)
        result = self.wait_complete()
        self.assertEqual(result["progress"]["stage"], "complete")
        self.assertIn(result["result"]["overall_severity"], {"PASS", "INFO"})
        history = self.client.get("/api/data/history/?dataset=tiny")
        self.assertEqual(len(history.data["reports"]), 1)
        self.assertEqual(len(history.data["latest"]["results"]), 16)
        self.assertNotIn("issues", history.data["latest"]["results"][0])

    def test_quality_errors_are_completed_results_and_fixes_are_visible(self):
        self.transform()
        image = next((paths.dataset_processed_dir("tiny") / "images" / "train").glob("*.jpg"))
        original = image.read_bytes()
        image.write_bytes(b"broken")
        self.client.post("/api/data/validate/", {"dataset": "tiny"}, format="json")
        self.assertEqual(self.wait_complete()["result"]["overall_severity"], "ERROR")
        image.write_bytes(original)
        self.client.post("/api/data/validate/", {"dataset": "tiny"}, format="json")
        self.wait_complete()
        comparison = self.client.get("/api/data/history/?dataset=tiny").data["latest"]["comparison"]
        self.assertEqual(comparison["resolved_count"], 1)
        self.assertEqual(comparison["resolved"][0]["check"], "image_integrity")

    def test_invalid_inputs_and_missing_sources(self):
        for data in [
            {"dataset": "../escape"},
            {"dataset": "missing"},
            {"dataset": "tiny", "format": "unknown"},
            {"dataset": "tiny", "format": "yolo", "classes": []},
            {"dataset": "tiny", "classes": ["x"], "train_rate": 0.9, "val_rate": 0.2},
            {"dataset": "tiny", "classes": ["x"], "train_rate": "NaN"},
            {"dataset": "tiny", "classes": "not-an-array"},
        ]:
            response = self.client.post("/api/data/jobs/", data, format="json")
            self.assertEqual(response.status_code, 400, response.data)
        self.assertEqual(
            self.client.post("/api/data/validate/", {"dataset": "tiny"}).status_code, 400
        )
        self.assertEqual(self.client.get("/api/data/history/?dataset=../escape").status_code, 400)

    def test_busy_tasks_reject_duplicate_submissions(self):
        self.manager._state["status"] = "running"
        response = self.client.post(
            "/api/data/jobs/", {"dataset": "tiny", "classes": ["x"]}, format="json"
        )
        self.assertEqual(response.status_code, 409)

    def test_worker_failure_is_returned_to_browser(self):
        with patch("defect.data_jobs.DatasetPipeline", side_effect=ValueError("bad source")):
            self.manager._state = {"status": "running"}
            self.manager._execute(
                "transform", "tiny", {"train_rate": 0.8, "val_rate": 0.1, "seed": 42}
            )
        status = self.manager.status()
        self.assertEqual(status["status"], "failed")
        self.assertIn("bad source", status["error"])
