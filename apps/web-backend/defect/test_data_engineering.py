"""Web 数据工程集成测试：临时真实数据、真实子进程、API 和报告隔离。"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from django.urls import resolve
from PIL import Image
from rest_framework.test import APIClient

from od_platform.common import paths, stamp
from od_platform.common.progress import iter_progress, progress_events
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.data_validation.service import validate_dataset

from .data_jobs import DataJobManager
from .data_worker import write_state


class WebDataTests(SimpleTestCase):
    def test_original_data_routes_keep_their_original_module(self):
        for url in ("/api/data/", "/api/data/validate/", "/api/data/sample/train/example.jpg"):
            with self.subTest(url=url):
                self.assertEqual(resolve(url).func.__module__, "defect.api_views")

    def test_original_yaml_reader_and_constants_remain_available(self):
        from . import api_views, data_understanding

        self.assertTrue(callable(data_understanding._read_dataset_yaml))
        self.assertEqual(api_views.DATASET_YAML, data_understanding.DATASET_YAML)
        self.assertTrue(hasattr(data_understanding, "PROJECT_ROOT"))
        with patch.object(data_understanding, "DATASET_YAML", paths.dataset_yaml_path("web_demo")):
            document = data_understanding._read_dataset_yaml()
            overview = data_understanding.get_dataset_understanding()
            self.assertEqual(document["nc"], 1)
            self.assertEqual(overview["total_images"], 12)

    def test_progress_write_retries_windows_sharing_violation(self):
        directory = self.root / "state-test"
        directory.mkdir()
        original = Path.replace
        calls = []

        def replace(path, target):
            calls.append(target)
            if len(calls) < 3:
                raise PermissionError("file temporarily busy")
            return original(path, target)

        with patch.object(Path, "replace", replace):
            write_state(directory, {"completed": 7})
        self.assertEqual(json.loads((directory / "state.json").read_text()), {"completed": 7})
        self.assertEqual(len(calls), 3)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="odp_web_data_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.names = ["web_demo", "other_dataset"]
        replacements = {
            "ROOT_DIR": self.root,
            "RAW_DATA_DIR": self.root / "data/raw",
            "PROCESSED_DATA_DIR": self.root / "data/processed",
            "DATASET_CONFIGS_DIR": self.root / "apps/platform/configs/datasets",
            "RUNS_DIR": self.root / "runs",
            "VALIDATION_RUNS_DIR": self.root / "runs/data_validation",
        }
        configuration = patch.multiple(paths, **replacements)
        configuration.start()
        self.addCleanup(configuration.stop)
        lineage = patch.object(stamp, "LINEAGE_DIR", self.root / "runs/lineage")
        lineage.start()
        self.addCleanup(lineage.stop)
        for name in self.names:
            raw = paths.raw_dataset_root(name)
            (raw / "images").mkdir(parents=True)
            (raw / "labels").mkdir()
            for number in range(12):
                Image.new("RGB", (16, 16), "white").save(raw / "images" / f"s{number}.jpg")
                (raw / "labels" / f"s{number}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            DatasetPipeline(name, "yolo", classes=["paper"]).run()
        self.client = APIClient()
        self.manager = DataJobManager()
        for target in ("defect.data_api.data_jobs", "defect.api_views.data_jobs"):
            substitution = patch(target, self.manager)
            substitution.start()
            self.addCleanup(substitution.stop)
        self.addCleanup(self.stop_worker)

    def stop_worker(self):
        if self.manager._process is not None:
            if self.manager._process.poll() is None:
                self.manager._process.terminate()
                self.manager._process.wait(timeout=10)
            self.manager._log.close()

    def baseline(self, name="web_demo", run_id="baseline"):
        return validate_dataset(
            paths.dataset_yaml_path(name), run_id=run_id, compare_previous=False
        )

    def wait_for_job(self):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            response = self.client.get("/api/data/jobs/")
            self.assertEqual(response.status_code, 200)
            if response.data["status"] != "running":
                return response.data
            time.sleep(0.05)
        self.fail("数据任务未在 30 秒内完成")

    def test_catalog_and_actual_dataset_counts(self):
        catalog = self.client.get("/api/data/catalog/")
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual({item["name"] for item in catalog.data["datasets"]}, set(self.names))
        overview = self.client.get("/api/data/?dataset=web_demo")
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.data["total_images"], 12)
        sample = self.client.get(overview.data["samples"][0]["url"])
        self.assertEqual(sample.status_code, 200)
        sample.close()
        image = next(paths.dataset_processed_dir("web_demo").glob("images/train/*.jpg"))
        image.unlink()
        overview = self.client.get("/api/data/?dataset=web_demo")
        self.assertEqual(overview.data["total_images"], 11)

    def test_unknown_dataset_is_explicitly_empty(self):
        response = self.client.get("/api/data/?dataset=not_ready")
        self.assertEqual(response.data["total_images"], 0)
        self.assertFalse(response.data["available"])
        self.assertTrue(response.data["message"])

    def test_async_validation_preserves_error_report_and_comparison(self):
        self.baseline()
        image = next(paths.dataset_processed_dir("web_demo").glob("images/train/*.jpg"))
        image.write_bytes(b"corrupt")
        response = self.client.post("/api/data/validate/", {"dataset": "web_demo"}, format="json")
        self.assertEqual(response.status_code, 202, response.data)
        result = self.wait_for_job()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["result"]["severity"], "ERROR")
        self.assertEqual(result["result"]["n_checks"], 16)
        self.assertGreater(result["progress_event"]["completed"], 0)
        run_id = result["result"]["run_id"]
        report = self.client.get(f"/api/data/reports/{run_id}/?dataset=web_demo")
        self.assertEqual(report.status_code, 200)
        self.assertIn(
            "image_integrity", {item["check"] for item in report.data["comparison"]["new_issues"]}
        )
        check = next(row for row in report.data["results"] if row["name"] == "image_integrity")
        self.assertEqual(check["details"]["n_bad"], 1)
        downloaded = self.client.get(report.data["downloads"]["comparison"])
        self.assertEqual(downloaded.status_code, 200)
        self.assertIn("attachment", downloaded["Content-Disposition"])
        self.assertTrue(b"".join(downloaded.streaming_content).startswith(b"\xef\xbb\xbf"))
        downloaded.close()

    def test_async_transform_uses_real_worker(self):
        payload = {
            "dataset": "web_demo",
            "format": "yolo",
            "classes": ["paper"],
            "split_strategy": "stratified_multilabel",
            "seed": 17,
            "train_rate": 0.7,
            "val_rate": 0.2,
        }
        response = self.client.post("/api/data/transform/", payload, format="json")
        self.assertEqual(response.status_code, 202, response.data)
        result = self.wait_for_job()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(sum(result["result"]["counts"].values()), 12)
        self.assertTrue(Path(result["result"]["manifest_path"]).is_file())
        self.assertTrue((paths.raw_dataset_root("web_demo") / "images/s0.jpg").is_file())

    def test_worker_failure_is_visible(self):
        label = next((paths.raw_dataset_root("web_demo") / "labels").glob("*.txt"))
        label.write_text("not a label", encoding="utf-8")
        response = self.client.post(
            "/api/data/transform/",
            {"dataset": "web_demo", "format": "yolo", "classes": ["paper"]},
            format="json",
        )
        self.assertEqual(response.status_code, 202)
        result = self.wait_for_job()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["error"])
        self.assertTrue(result["logs"])

    def test_reports_do_not_cross_datasets(self):
        self.baseline()
        self.baseline("other_dataset", "other-baseline")
        report_list = self.client.get("/api/data/reports/?dataset=web_demo")
        self.assertEqual([item["run_id"] for item in report_list.data["reports"]], ["baseline"])
        response = self.client.get("/api/data/reports/other-baseline/?dataset=web_demo")
        self.assertEqual(response.status_code, 404)
        response = self.client.post(
            "/api/data/validate/",
            {"dataset": "web_demo", "baseline_run_id": "other-baseline"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)

    def test_corrupt_reports_are_skipped(self):
        self.baseline()
        corrupt = paths.validation_run_dir("corrupt") / "report.json"
        corrupt.parent.mkdir()
        corrupt.write_text("{", encoding="utf-8")
        response = self.client.get("/api/data/reports/?dataset=web_demo")
        self.assertEqual(len(response.data["reports"]), 1)

    def test_explicit_baseline_and_downloads(self):
        self.baseline()
        response = self.client.post(
            "/api/data/validate/",
            {"dataset": "web_demo", "baseline_run_id": "baseline"},
            format="json",
        )
        self.assertEqual(response.status_code, 202)
        state = self.wait_for_job()
        self.assertEqual(state["result"]["comparison"]["baseline_run_id"], "baseline")
        for filename in ("report.json", "results.csv"):
            response = self.client.get(
                f"/api/data/reports/baseline/download/{filename}/?dataset=web_demo"
            )
            self.assertEqual(response.status_code, 200)
            response.close()
        self.assertEqual(
            self.client.get(
                "/api/data/reports/baseline/download/request.json/?dataset=web_demo"
            ).status_code,
            400,
        )

    def test_disabled_comparison(self):
        self.baseline()
        response = self.client.post(
            "/api/data/validate/", {"dataset": "web_demo", "compare_previous": False}, format="json"
        )
        self.assertEqual(response.status_code, 202)
        state = self.wait_for_job()
        self.assertIsNone(state["result"]["comparison"])

    def test_invalid_names_and_parameters_are_rejected(self):
        invalid = [
            {"dataset": "../escape"},
            {"dataset": "web_demo", "format": "unknown"},
            {"dataset": "web_demo", "classes": ["paper"], "train_rate": 0.9, "val_rate": 0.9},
            {"dataset": "web_demo", "format": "pascal_voc", "task": "segment"},
            {"dataset": "web_demo", "format": "yolo", "classes": []},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post("/api/data/transform/", payload, format="json")
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/data/?dataset=../escape").status_code, 400)
        response = self.client.post(
            "/api/data/validate/",
            {"dataset": "web_demo", "compare_previous": False, "baseline_run_id": "baseline"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)

    def test_duplicate_jobs_are_rejected(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch("defect.data_jobs.subprocess.Popen", return_value=process):
            first = self.client.post("/api/data/validate/", {"dataset": "web_demo"}, format="json")
            second = self.client.post("/api/data/validate/", {"dataset": "web_demo"}, format="json")
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)
        process.poll.return_value = 0

    def test_training_and_transform_do_not_mutate_data_concurrently(self):
        with patch("defect.data_api.training_jobs.status", return_value={"status": "running"}):
            response = self.client.post(
                "/api/data/transform/",
                {"dataset": "web_demo", "format": "yolo", "classes": ["paper"]},
                format="json",
            )
            self.assertEqual(response.status_code, 409)
        with patch.object(
            self.manager, "status", return_value={"status": "running", "action": "transform"}
        ):
            self.assertEqual(self.client.post("/api/training/", {}, format="json").status_code, 409)

    def test_progress_events_are_exact_and_context_is_restored(self):
        events = []
        with progress_events(events.append):
            self.assertEqual(list(iter_progress(range(3), "real samples")), [0, 1, 2])
        self.assertEqual(events[0]["completed"], 0)
        self.assertEqual(events[-1]["completed"], 3)
        self.assertEqual(events[-1]["total"], 3)
        self.assertEqual(events[-1]["percent"], 100)
        self.assertTrue(events[-1]["finished"])
        before = len(events)
        list(iter_progress(range(2), "quiet"))
        self.assertEqual(len(events), before)

    def test_web_assets_include_data_controls(self):
        response = self.client.get("/?module=understanding")
        html = b"".join(response.streaming_content).decode("utf-8")
        response.close()
        self.assertIn("/assets/data-engineering.js", html)
        self.assertIn("构建数据集", html)
        self.assertIn("质检明细与历史对比", html)
        self.assertIn('role="progressbar"', html)
