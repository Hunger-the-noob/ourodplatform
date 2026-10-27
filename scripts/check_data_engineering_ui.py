"""新增：使用真实 Django API 的浏览器验收；仅在临时目录生成数据和数据库。

可选依赖：pip install playwright；playwright install chromium。
运行：python scripts/check_data_engineering_ui.py --screenshot <输出图片路径>
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import threading
from pathlib import Path
from socketserver import ThreadingMixIn
from unittest.mock import patch
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "web-backend"))


def main():
    import os

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django
    from django.conf import settings

    django.setup()
    from defect import data_understanding
    from django.core.management import call_command
    from django.core.wsgi import get_wsgi_application
    from PIL import Image
    from playwright.sync_api import expect, sync_playwright

    from od_platform.common import paths, stamp
    from od_platform.data_pipeline.orchestrator import DatasetPipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--browser-channel", default=None, help="可使用已安装的 msedge 或 chrome")
    args = parser.parse_args()

    class QuietHandler(WSGIRequestHandler):
        def log_message(self, format, *args):
            pass

    class ThreadedServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    with (
        tempfile.TemporaryDirectory(prefix="odp_ui_") as directory,
        patch.multiple(
            paths,
            ROOT_DIR=Path(directory),
            RAW_DATA_DIR=Path(directory) / "raw",
            PROCESSED_DATA_DIR=Path(directory) / "processed",
            RUNS_DIR=Path(directory) / "runs",
            DATASET_CONFIGS_DIR=Path(directory) / "configs",
            VALIDATION_RUNS_DIR=Path(directory) / "validation",
        ),
        patch.object(stamp, "LINEAGE_DIR", Path(directory) / "lineage"),
    ):
        temp_root = Path(directory)
        settings.DATABASES["default"]["NAME"] = str(temp_root / "web.sqlite3")
        settings.ALLOWED_HOSTS = [*settings.ALLOWED_HOSTS, "127.0.0.1"]
        settings.MEDIA_ROOT = temp_root / "media"
        call_command("migrate", interactive=False, verbosity=0)
        raw = paths.raw_dataset_root("ui_demo")
        (raw / "images").mkdir(parents=True)
        (raw / "labels").mkdir()
        for index in range(20):
            Image.new("RGB", (64, 64), (40, 80, 120)).save(raw / "images" / f"{index}.jpg")
            (raw / "labels" / f"{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        DatasetPipeline("ui_demo", "yolo", classes=["item"]).run()
        with (
            patch.object(data_understanding, "DATASET_YAML", paths.dataset_yaml_path("ui_demo")),
            patch.object(data_understanding, "DATASET_NAME", "ui_demo"),
        ):
            server = make_server(
                "127.0.0.1", 0, get_wsgi_application(), ThreadedServer, QuietHandler
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(channel=args.browser_channel)
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(f"http://127.0.0.1:{server.server_port}/?module=understanding")
                    card = page.locator(".pipeline-card")
                    expect(card.get_by_role("button", name="运行 D4 质量检查")).to_be_enabled()

                    def validate():
                        card.get_by_role("button", name="运行 D4 质量检查").click()
                        expect(card.get_by_role("button", name="运行 D4 质量检查")).to_be_enabled(
                            timeout=15000
                        )
                        expect(card.locator(".data-report")).to_be_visible()

                    validate()
                    expect(card.locator(".data-table")).to_contain_text("image_integrity")
                    expect(card.locator(".data-report")).to_contain_text("首次质检")
                    image = next(
                        (paths.dataset_processed_dir("ui_demo") / "images" / "train").glob("*.jpg")
                    )
                    original = image.read_bytes()
                    image.write_bytes(b"corrupt")
                    validate()
                    expect(card.locator(".data-report")).to_contain_text("新增 1")
                    image.write_bytes(original)
                    validate()
                    expect(card.locator(".data-report")).to_contain_text("已修复 1")
                    card.get_by_role("button", name="运行 D3 转换与划分").click()
                    expect(card.get_by_role("button", name="运行 D3 转换与划分")).to_be_enabled(
                        timeout=15000
                    )
                    expect(card.locator(".data-progress")).to_contain_text("产出：")
                    card.scroll_into_view_if_needed()
                    if args.screenshot:
                        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                        card.screenshot(path=str(args.screenshot))
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    )
                    assert not errors, errors
                    browser.close()
                    print(
                        "Browser checks passed: D3, D4, progress, new/fixed issues, mobile layout."
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                from django.db import connections

                connections.close_all()


if __name__ == "__main__":
    main()
