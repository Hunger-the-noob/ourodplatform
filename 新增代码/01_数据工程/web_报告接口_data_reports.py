"""质检报告查询与下载；隔离不同数据配置、根目录及任务。"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from od_platform.common import paths

from .data_catalog import data_root, dataset_yaml, read_yaml

RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def report_path(run_id: str, filename: str = "report.json") -> Path:
    if not RUN_RE.fullmatch(run_id) or filename not in {
        "report.json",
        "results.csv",
        "comparison.csv",
    }:
        raise ValueError("无效的报告编号或文件名。")
    path = paths.VALIDATION_RUNS_DIR / run_id / filename
    path.resolve().relative_to(paths.VALIDATION_RUNS_DIR.resolve())
    return path


def get_report(name: str, run_id: str) -> dict:
    config = dataset_yaml(name)
    path = report_path(run_id)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise FileNotFoundError("质检报告不存在或无法读取。") from error
    if not isinstance(doc, dict) or not isinstance(doc.get("results"), list):
        raise FileNotFoundError("质检报告格式无效。")
    identity = doc.get("dataset") or {}
    if not isinstance(identity, dict):
        raise FileNotFoundError("质检报告的数据身份格式无效。")
    for result in doc["results"]:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("name"), str)
            or result.get("severity") not in ("PASS", "INFO", "WARNING", "ERROR")
            or not isinstance(result.get("details"), dict)
            or not isinstance(result.get("summary"), str)
        ):
            raise FileNotFoundError("质检报告的检查结果格式无效。")
    yaml_ref = identity.get("yaml_path") or doc.get("yaml_path")
    if not isinstance(yaml_ref, str) or Path(yaml_ref).resolve() != config.resolve():
        raise FileNotFoundError("该报告不属于当前数据集。")
    data = read_yaml(name)
    if identity:
        root = data_root(name, data)
        expected_root = str(root) if root else None
        actual_root = (
            str(Path(identity["data_root"]).resolve()) if identity.get("data_root") else None
        )
        if actual_root != expected_root or identity.get("task") != (data.get("odp_meta") or {}).get(
            "task", "detect"
        ):
            raise FileNotFoundError("该报告的数据根目录或任务与当前配置不一致。")
    doc["run_id"] = run_id
    doc["created_at"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    doc["severity"] = doc.get("overall_severity", "ERROR")
    doc["baseline_compatible"] = bool(identity)
    doc["downloads"] = {
        key: f"/api/data/reports/{run_id}/download/{filename}/?dataset={name}"
        for key, filename in (
            ("json", "report.json"),
            ("csv", "results.csv"),
            ("comparison", "comparison.csv"),
        )
        if report_path(run_id, filename).is_file()
    }
    return doc


def list_reports(name: str, limit: int = 30) -> list[dict]:
    dataset_yaml(name)
    candidates = []
    for path in paths.VALIDATION_RUNS_DIR.glob("*/report.json"):
        try:
            candidates.append((path.stat().st_mtime_ns, path.parent.name))
        except OSError:
            continue
    result = []
    for _, run_id in sorted(candidates, reverse=True):
        try:
            doc = get_report(name, run_id)
        except (FileNotFoundError, ValueError, TypeError):
            continue
        result.append(
            {
                key: doc.get(key)
                for key in (
                    "run_id",
                    "created_at",
                    "severity",
                    "duration_seconds",
                    "baseline_compatible",
                    "downloads",
                )
            }
            | {"n_checks": len(doc["results"])}
        )
        if len(result) >= limit:
            break
    return result


def latest_validation(name: str) -> dict:
    reports = list_reports(name, limit=1)
    return reports[0] if reports else {}
