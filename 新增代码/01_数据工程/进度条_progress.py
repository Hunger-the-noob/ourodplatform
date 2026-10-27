# 阅读副本：对应 D3/D4“大数据集执行时用户无感知”，展示统一进度条与异常恢复。
"""按运行隔离的进度开关；库调用默认安静，CLI 显式开启。"""

from collections.abc import Callable, Iterable, Iterator, Sized
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import TypeVar

from tqdm import tqdm

_ENABLED = ContextVar("odp_progress_enabled", default=False)
_LISTENER: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "odp_progress_listener", default=None
)
_T = TypeVar("_T")


@contextmanager
def progress_reporting(enabled: bool) -> Iterator[None]:
    token = _ENABLED.set(enabled)
    try:
        yield
    finally:
        _ENABLED.reset(token)


@contextmanager
def progress_events(listener: Callable[[dict], None]) -> Iterator[None]:
    """向 Web 等调用方报告真实阶段进度，不依赖解析终端文本。"""
    token = _LISTENER.set(listener)
    try:
        yield
    finally:
        _LISTENER.reset(token)


def iter_progress(
    items: Iterable[_T], description: str, *, total: int | None = None
) -> Iterator[_T]:
    listener = _LISTENER.get()
    if not _ENABLED.get() and listener is None:
        yield from items
        return
    if total is None and isinstance(items, Sized):
        total = len(items)
    started = last_update = monotonic()
    completed = 0

    def emit(finished: bool = False) -> None:
        elapsed = max(monotonic() - started, 0.0)
        rate = completed / elapsed if elapsed > 0 else 0.0
        listener(
            {
                "description": description,
                "completed": completed,
                "total": total,
                "percent": round(completed / total * 100, 1)
                if total
                else (100.0 if finished else None),
                "rate": round(rate, 2),
                "elapsed_seconds": round(elapsed, 2),
                "eta_seconds": round(max(total - completed, 0) / rate, 1)
                if total and rate
                else None,
                "finished": finished,
            }
        )

    if listener is not None:
        emit()
    # 上下文确保异常/中断时也会关闭终端进度条。tqdm 默认写 stderr。
    with tqdm(
        items,
        desc=description,
        total=total,
        unit="item",
        dynamic_ncols=True,
        disable=not _ENABLED.get(),
    ) as bar:
        for item in bar:
            yield item
            completed += 1
            if listener is not None and monotonic() - last_update >= 0.25:
                emit()
                last_update = monotonic()
    if listener is not None:
        emit(finished=True)
