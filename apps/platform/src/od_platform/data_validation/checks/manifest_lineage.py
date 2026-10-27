#!/usr/bin/env python
# @FileName  : manifest_lineage.py
# @Author    : ODPlatform Team
# @Project   : ODPlatform
# @Function  : check: manifest_lineage
"""check: manifest_lineage —— 逐样本核验:在契约说的 split 里、内容 sha256 没变。
changed(内容变了)/ moved(split 挪了)/ missing(丢了)分开计数。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from od_platform.common.constants import LINEAGE_MAX_DETAIL
from od_platform.common.progress import track
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)

_NAME = "manifest_lineage"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@check(_NAME)
def validate_manifest_lineage(ctx: CheckContext) -> CheckResult:
    manifest, err, _ = load_manifest_from_yaml(ctx.snapshot.yaml_data)
    if err:
        return CheckResult(_NAME, CheckSeverity.ERROR, f"拿不到 D3 划分契约: {err}", {})
    snap = ctx.snapshot

    # 磁盘实况:(split, stem) → 标签路径;stem → 实际出现的 split 集合
    disk: dict[tuple[str, str], Path] = {}
    disk_splits_of_stem: dict[str, set] = {}
    for split in snap.splits:
        for lbl in snap.label_files_per_split[split]:
            disk[(split, lbl.stem)] = lbl
            disk_splits_of_stem.setdefault(lbl.stem, set()).add(split)

    changed: list[str] = []
    moved: list[dict] = []
    missing: list[str] = []
    issues = []  # 新增：记录每个契约异常，历史比较不会被 preview 截断。
    for s in track(manifest.samples, "manifest_lineage"):
        lbl = disk.get((s.split, s.stem))
        if lbl is None:
            elsewhere = disk_splits_of_stem.get(s.stem)
            if elsewhere:
                moved.append({"stem": s.stem, "expect": s.split, "actual": sorted(elsewhere)})
                issues.append(
                    {
                        "split": s.split,
                        "file": s.stem,
                        "kind": "moved",
                        "message": f"实际划分: {sorted(elsewhere)}",
                    }
                )
            else:
                missing.append(s.stem)
                issues.append({"split": s.split, "file": s.stem, "kind": "missing"})
        elif _sha256(lbl) != s.sha256:
            changed.append(s.stem)
            issues.append({"split": s.split, "file": s.stem, "kind": "changed"})

    details = {
        "n_samples": len(manifest.samples),
        "n_changed": len(changed),
        "n_moved": len(moved),
        "n_missing": len(missing),
        "changed_preview": changed[:LINEAGE_MAX_DETAIL],
        "moved_preview": moved[:LINEAGE_MAX_DETAIL],
        "missing_preview": missing[:LINEAGE_MAX_DETAIL],
    }
    if changed or moved or missing:
        return CheckResult(
            _NAME,
            CheckSeverity.ERROR,
            f"契约核验失败: changed={len(changed)} moved={len(moved)} missing={len(missing)}"
            f" —— 与 D3 产出不一致,重跑 odp-transform 或查谁动了 processed",
            details,
            issues=issues,
        )
    return CheckResult(
        _NAME,
        CheckSeverity.PASS,
        f"{len(manifest.samples)} 个样本逐条核验通过",
        details,
        issues=issues,
    )
