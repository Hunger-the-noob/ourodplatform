#!/usr/bin/env python
# 阅读副本：校验标签列数、类别 ID 范围、归一化坐标和 NaN/Inf。
# @FileName  : label_format.py
# @Author    : ODPlatform Team
# @Project   : ODPlatform
# @Function  : check: label_format
"""check: label_format —— 每行标签的结构与取值合法吗。任何一行坏 → ERROR。"""

from __future__ import annotations

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT
from od_platform.common.progress import iter_progress
from od_platform.data_validation.labels import ParsedBox, parse_label_line
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)

_NAME = "label_format"
KIND_CLASS_ID_OUT_OF_RANGE = "class_id_out_of_range"
KIND_COORD_OUT_OF_RANGE = "coord_out_of_range"


def _validate_one_line(line: str, task_type: str, nc: int) -> tuple[ParsedBox | None, str | None]:
    """纯函数:(box, kind);kind=None 即合法。"""
    box, kind = parse_label_line(line, task_type)
    if box is None:
        return None, kind
    class_id, coords = box
    if not (0 <= class_id < nc):
        return None, KIND_CLASS_ID_OUT_OF_RANGE
    if any(v < 0 or v > 1 for v in coords):
        return None, KIND_COORD_OUT_OF_RANGE
    return box, None


@check(_NAME)
def validate_label_format(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    nc = snap.nc
    if not isinstance(nc, int):
        return CheckResult(
            _NAME,
            CheckSeverity.PASS,
            "nc 不可用,跳过(见 yaml_schema)",
            {"reason": "nc_unavailable"},
        )
    bad_lines: list[dict] = []
    error_kinds: dict[str, int] = {}
    n_bad = n_total = n_read_errors = 0
    for split in snap.splits:
        for lbl in iter_progress(snap.label_files_per_split[split], f"D4 label format {split}"):
            try:
                text = lbl.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                n_read_errors += 1
                error_kinds["label_read_error"] = error_kinds.get("label_read_error", 0) + 1
                if len(bad_lines) < DETAILS_PREVIEW_LIMIT:
                    bad_lines.append(
                        {
                            "split": split,
                            "label": str(lbl),
                            "kind": "label_read_error",
                            "error": str(error),
                        }
                    )
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                _box, kind = _validate_one_line(line, snap.task_type, nc)
                if kind is not None:
                    n_bad += 1
                    error_kinds[kind] = error_kinds.get(kind, 0) + 1
                    if len(bad_lines) < DETAILS_PREVIEW_LIMIT:
                        bad_lines.append(
                            {
                                "split": split,
                                "label": str(lbl),
                                "line_no": line_no,
                                "kind": kind,
                                "line": line,
                            }
                        )
    if n_bad == 0 and n_read_errors == 0:
        return CheckResult(
            _NAME,
            CheckSeverity.PASS,
            f"全部 {n_total} 行标签结构与取值合法",
            {"total_lines": n_total},
        )
    return CheckResult(
        _NAME,
        CheckSeverity.ERROR,
        f"{n_bad}/{n_total} 行标签非法，{n_read_errors} 个文件无法读取({error_kinds})",
        {
            "total_lines": n_total,
            "n_bad": n_bad,
            "n_read_errors": n_read_errors,
            "error_kinds": error_kinds,
            "bad_lines_preview": bad_lines,
        },
    )
