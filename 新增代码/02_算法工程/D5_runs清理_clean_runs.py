# 阅读副本：对应 D5“runs 长期累积占用空间”，默认只预览，显式确认后才删除。
"""odp-clean —— runs/ 目录清理。

为什么需要: 每跑一次 transform / validate / train / infer 都会在 runs/<子系统>/
下留一个新目录, 长期累积能吃掉好几个 G(一台机器上单个实验就见过 1.2 GB)。
这里给一个显式、可预览、有围栏的清理入口。

安全规则(比功能重要):
  1. 默认 --dry-run 只列不删, 想真删必须显式 --yes
  2. 只动 runs/<子系统>/<一层子目录>, 绝不递归到更深, 也绝不动 runs 本身
  3. 符号链接一律跳过(防止顺藤摸瓜删到仓库外)
  4. runs/lineage 默认不碰 —— 那是数据身份的盖章, 删了就没法对账

用法:
    odp-clean --keep 5 --dry-run        # 每个子系统保留最近 5 个, 先看要删什么
    odp-clean --keep 5 --yes            # 真删
    odp-clean --older-than 30 --yes     # 删 30 天前的
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from od_platform.common import paths

# 这些子系统里的东西是"证据", 清理命令默认绕开
_PROTECTED = {"lineage"}


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _run_dirs(subsystem: Path) -> list[Path]:
    """一层子目录, 跳过符号链接。按名字降序(run_id 是时间戳, 越大越新)。"""
    dirs = [
        d
        for d in subsystem.iterdir()
        if d.is_dir() and not d.is_symlink() and not d.name.startswith("__")
    ]
    return sorted(dirs, key=lambda d: d.name, reverse=True)


def _select(
    runs: list[Path], keep: int | None, older_than: int | None
) -> list[Path]:
    """挑出该删的。keep 与 older_than 同时给时取并集(任一条件命中就删)。"""
    doomed: list[Path] = []
    if keep is not None:
        doomed.extend(runs[keep:])
    if older_than is not None:
        cutoff = datetime.now() - timedelta(days=older_than)
        for run in runs:
            try:
                mtime = datetime.fromtimestamp(run.stat().st_mtime)
            except OSError:
                continue
            if mtime < cutoff and run not in doomed:
                doomed.append(run)
    return doomed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="odp-clean", description="清理 runs/ 下的历史运行目录(默认只预览)"
    )
    parser.add_argument(
        "--keep", type=int, default=None, help="每个子系统保留最近 N 个(其余可删)"
    )
    parser.add_argument(
        "--older-than", type=int, default=None, dest="older_than", help="删除 N 天前的"
    )
    parser.add_argument(
        "--only", nargs="+", default=None, help="只处理这些子系统(默认全部, lineage 除外)"
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出, 不删除")
    parser.add_argument("--yes", action="store_true", help="确认执行删除")
    parser.add_argument(
        "--include-protected", action="store_true", help="连 lineage 一起清(危险)"
    )
    args = parser.parse_args(argv)

    if args.keep is None and args.older_than is None:
        parser.print_help()
        print("\n至少要给 --keep 或 --older-than 其中之一。", file=sys.stderr)
        return 2
    if args.keep is not None and args.keep < 0:
        print("--keep 不能是负数", file=sys.stderr)
        return 2

    root = paths.RUNS_DIR
    if not root.exists():
        print(f"runs 目录不存在: {root}")
        return 0

    protected = set() if args.include_protected else _PROTECTED
    total_files = total_bytes = 0
    plan: list[tuple[Path, int]] = []

    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("__"):
            continue
        if args.only and sub.name not in args.only:
            continue
        if sub.name in protected:
            print(f"[跳过] {sub.name}/ —— 受保护(数据身份盖章)")
            continue
        runs = _run_dirs(sub)
        doomed = _select(runs, args.keep, args.older_than)
        if not doomed:
            continue
        print(f"\n{sub.name}/  共 {len(runs)} 个, 计划清理 {len(doomed)} 个:")
        for run in doomed:
            size = _dir_size(run)
            total_bytes += size
            total_files += 1
            print(f"  - {run.name}  ({_human_size(size)})")
            plan.append((run, size))

    if not plan:
        print("\n没有需要清理的目录。")
        return 0

    print(f"\n合计: {total_files} 个目录, 可释放 {_human_size(total_bytes)}")

    if not args.yes:
        print("\n[dry-run] 什么都没删。确认无误后加 --yes 重跑。")
        return 0

    removed = 0
    for run, _size in plan:
        # 围栏: 必须严格在 runs/<子系统>/ 下一层, 且不是符号链接
        try:
            run.resolve().relative_to(root.resolve())
        except ValueError:
            print(f"  [拒绝] {run} 不在 runs/ 内", file=sys.stderr)
            continue
        if run.is_symlink():
            print(f"  [拒绝] {run.name} 是符号链接", file=sys.stderr)
            continue
        try:
            shutil.rmtree(run)
            removed += 1
        except OSError as exc:
            print(f"  [失败] {run.name}: {exc}", file=sys.stderr)
    print(f"\n已删除 {removed} 个目录, 释放约 {_human_size(total_bytes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
