"""新增：D3/D4 共用进度事件；CLI 和 Web 使用同一份真实计数。"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

ProgressCallback = Callable[[dict], None]
_callback: ContextVar[ProgressCallback | None] = ContextVar("odp_progress", default=None)
T = TypeVar("T")


@contextmanager
def progress_scope(callback: ProgressCallback | None = None):
    """每次运行隔离回调，嵌套调用保留外层回调，异常时也恢复上下文。"""
    token = _callback.set(callback or _callback.get())
    try:
        yield
    finally:
        _callback.reset(token)


def emit_progress(stage: str, completed: int = 0, total: int = 0) -> None:
    callback = _callback.get()
    if callback is not None:
        callback(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "percent": round(completed / total * 100, 1) if total else 0.0,
            }
        )


def track(items: Iterable[T], stage: str, total: int | None = None) -> Iterator[T]:
    """完成一次工作后再递增；限制回调频率，避免大数据集被 UI 更新拖慢。"""
    if total is None:
        total = len(items) if hasattr(items, "__len__") else 0
    emit_progress(stage, 0, total)
    last = time.monotonic()
    completed = 0
    for item in items:
        yield item
        completed += 1
        now = time.monotonic()
        if now - last >= 0.1:
            emit_progress(stage, completed, total)
            last = now
    emit_progress(stage, completed, total)


class ConsoleProgress:
    """新增：无第三方依赖的终端进度条；重定向输出时使用逐行日志。"""

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.stage = None

    def __call__(self, event: dict) -> None:
        stage, completed, total = event["stage"], event["completed"], event["total"]
        tty = self.stream.isatty()
        if not tty and stage == self.stage and total and completed < total:
            return
        if tty and self.stage is not None and stage != self.stage:
            self.stream.write("\n")
        self.stage = stage
        filled = min(20, int(event["percent"] / 5))
        bar = "#" * filled + "-" * (20 - filled)
        count = f"{completed}/{total} {event['percent']:.1f}%" if total else "处理中"
        prefix = "\r" if tty else ""
        self.stream.write(
            f"{prefix}[{bar}] {stage}: {count}" + ("\n" if not tty or stage == "complete" else "")
        )
        self.stream.flush()
