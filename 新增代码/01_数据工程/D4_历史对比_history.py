"""同一数据配置、根目录及任务的报告对比，以质检项为粒度。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from od_platform.data_validation.registry import CheckSeverity

logger = logging.getLogger(__name__)


def dataset_identity(snapshot) -> dict[str, Any]:
    return {
        "yaml_path": str(snapshot.yaml_path.resolve()),
        "data_root": str(snapshot.data_root.resolve()) if snapshot.data_root else None,
        "task": snapshot.task_type,
    }


def read_report(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or not isinstance(doc.get("run_id"), str):
            raise ValueError("缺少 run_id")
        if not isinstance(doc.get("dataset"), dict) or not isinstance(doc.get("results"), list):
            raise ValueError("缺少 dataset/results，旧版报告请重新生成")
        seen = set()
        for result in doc["results"]:
            if not isinstance(result, dict) or not isinstance(result.get("name"), str):
                raise ValueError("质检项缺少名称")
            if result["name"] in seen:
                raise ValueError("质检项名称重复")
            seen.add(result["name"])
            if result.get("severity") not in CheckSeverity._ORDER:
                raise ValueError("质检项严重度无效")
            if not isinstance(result.get("details"), dict) or not isinstance(
                result.get("summary"), str
            ):
                raise ValueError("质检项缺少 summary/details")
        return doc
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise ValueError(f"无法读取历史报告 {path}: {error}") from error


def find_previous_report(root: Path, identity: dict[str, Any]) -> Path | None:
    candidates = []
    for path in root.glob("*/report.json"):
        try:
            candidates.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    for _, path in sorted(candidates, reverse=True):
        try:
            if read_report(path)["dataset"] == identity:
                return path
        except ValueError as error:
            logger.warning("跳过不可用历史报告: %s", error)
    return None


def compare_reports(previous: dict[str, Any], current: dict[str, Any], *, baseline: Path) -> dict:
    if previous["dataset"] != current["dataset"]:
        raise ValueError("历史报告与本次的 yaml 路径、数据根目录或 task 不一致，无法对比")
    old = {result["name"]: result for result in previous["results"]}
    new = {result["name"]: result for result in current["results"]}
    comparison = {
        "baseline_run_id": previous["run_id"],
        "baseline_report": str(baseline.resolve()),
        "granularity": "check",
        "new_issues": [],
        "resolved_issues": [],
        "changed_issues": [],
        "unchanged_issues": [],
        "added_checks": sorted(new.keys() - old.keys()),
        "removed_checks": sorted(old.keys() - new.keys()),
    }
    for name, result in sorted(new.items()):
        before = old.get(name)
        was_issue = before is not None and before["severity"] != CheckSeverity.PASS
        is_issue = result["severity"] != CheckSeverity.PASS
        item = {"check": name, "previous": before, "current": result}
        if is_issue and not was_issue:
            comparison["new_issues"].append(item)
        elif was_issue and not is_issue:
            comparison["resolved_issues"].append(item)
        elif was_issue and is_issue:
            if before == result:
                comparison["unchanged_issues"].append(item)
            else:
                difference = CheckSeverity.rank(result["severity"]) - CheckSeverity.rank(
                    before["severity"]
                )
                item["direction"] = (
                    "worsened" if difference > 0 else "improved" if difference < 0 else "changed"
                )
                comparison["changed_issues"].append(item)
    return comparison
