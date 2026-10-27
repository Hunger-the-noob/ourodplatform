"""新增：同一 YAML、同一任务的历史对比，精确问题不受预览条数限制。"""

from __future__ import annotations

import json
from pathlib import Path

from od_platform.common import paths


def report_history(yaml_path: Path, task_type: str | None = None) -> list[dict]:
    reports = []
    for report_path in paths.VALIDATION_RUNS_DIR.glob("*/report.json"):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("run_id"), str)
                or not isinstance(payload.get("results"), list)
                or any(
                    not isinstance(r, dict) or not {"name", "severity", "summary"} <= r.keys()
                    for r in payload["results"]
                )
            ):
                continue
            if Path(payload.get("yaml_path", "")).resolve() != Path(yaml_path).resolve():
                continue
            if task_type and payload.get("task_type", "detect") != task_type:
                continue
            reports.append((report_path.stat().st_mtime_ns, report_path.parent.name, payload))
        except (OSError, ValueError, TypeError):
            continue  # 中断写入或损坏的旧报告不影响本次质检。
    return [item[2] for item in sorted(reports, key=lambda item: item[:2], reverse=True)]


def _issues(result: dict) -> dict[str, dict]:
    check = result["name"]
    if result.get("issues") is not None:
        values = result["issues"]
    elif result["severity"] in {"WARNING", "ERROR"}:
        # 旧检查没有逐样本问题表时，按检查项跟踪，并在页面明确标识粒度。
        values = [{"kind": "check_failed", "message": result["summary"], "scope": "check"}]
    else:
        values = []
    return {
        json.dumps(
            [check, issue.get("split"), issue.get("file"), issue.get("line_no"), issue.get("kind")],
            ensure_ascii=False,
        ): {**issue, "check": check}
        for issue in values
    }


def compare_reports(current: dict, previous: dict | None) -> dict:
    """新增/修复仅比较两次都真正执行的检查；跳过或崩溃不能冒充修复。"""
    if previous is None:
        return {
            "baseline_run_id": None,
            "new": [],
            "resolved": [],
            "persistent_count": 0,
            "severity_changes": [],
            "message": "首次质检，已建立历史基线",
        }
    old = {r["name"]: r for r in previous["results"]}
    # 本次读不出的标签不能据此宣布之前的行问题已修复（包括几何检查）。
    unreadable = {
        (issue.get("split"), issue.get("file"))
        for result in current["results"]
        for issue in result.get("issues") or []
        if issue.get("kind") == "read_error"
    }
    new, resolved, severity_changes = [], [], []
    persistent = 0
    for result in current["results"]:
        before = old.get(result["name"])
        if before is None:
            new.extend(_issues(result).values())
            continue
        if any(
            r.get("details", {}).get("exception") or r.get("details", {}).get("reason")
            for r in (result, before)
        ):
            continue
        # 新旧版本若采用不同粒度，重新建立该检查的基线，避免假报新增/修复。
        if (result.get("issues") is None) != (before.get("issues") is None):
            if result["severity"] != before["severity"]:
                severity_changes.append(
                    {
                        "check": result["name"],
                        "before": before["severity"],
                        "after": result["severity"],
                    }
                )
            continue
        earlier, now = _issues(before), _issues(result)
        new.extend(now[k] for k in now.keys() - earlier.keys())
        resolved.extend(
            earlier[k]
            for k in earlier.keys() - now.keys()
            if (earlier[k].get("split"), earlier[k].get("file")) not in unreadable
        )
        persistent += len(now.keys() & earlier.keys())
        if result["severity"] != before["severity"]:
            severity_changes.append(
                {"check": result["name"], "before": before["severity"], "after": result["severity"]}
            )

    def order(issue):
        return json.dumps(issue, sort_keys=True, ensure_ascii=False)

    return {
        "baseline_run_id": previous["run_id"],
        "new": sorted(new, key=order),
        "resolved": sorted(resolved, key=order),
        "persistent_count": persistent,
        "severity_changes": severity_changes,
        "message": "与上一次同数据集质检对比",
    }
