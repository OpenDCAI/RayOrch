"""RunDriver：在共享 StageExecutor 上协调有界 multi-Arena overlap。

Driver 只调用 ArenaEngine 公共接口并转发 ExecutionEvent，不读取 GrainTable、ItemTable、
ReduceAccumulator 或 ValueTable。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .api import CompiledPipeline
from .contracts import ExecutionError
from .arena import (
    ArenaEngine,
    ArenaLimits,
)
from .execution import ExecutionEvent, ExecutionPool
from .protocol import ArenaResult, BlockSlice, DispatchCompletion


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """按 run 输入顺序切分、最终 admission 为一个 Arena 的 source slice。"""
    index: int
    values: tuple[tuple[Any, ...], ...]
    position_starts: tuple[int, ...]


def _merge_metrics(parts: tuple[ArenaResult, ...]) -> dict[str, float]:
    """按 grain/RPC 权重合并多个 ArenaResult 的核心指标。"""

    rpc_count = sum(part.metrics.get("rpc_count", 0.0) for part in parts)
    grains = sum(
        part.metrics.get("rpc_count", 0.0)
        * part.metrics.get("grains_per_rpc", 0.0)
        for part in parts
    )
    capacity = sum(
        (
            part.metrics.get("rpc_count", 0.0)
            * part.metrics.get("grains_per_rpc", 0.0)
            / part.metrics.get("batch_fill_ratio", 1.0)
        )
        if part.metrics.get("batch_fill_ratio", 0.0) > 0
        else 0.0
        for part in parts
    )
    tail_calls = sum(
        part.metrics.get("rpc_count", 0.0)
        * part.metrics.get("tail_or_recovery_rpc_fraction", 0.0)
        for part in parts
    )
    return {
        "rpc_count": rpc_count,
        "grains_per_rpc": grains / rpc_count if rpc_count else 0.0,
        "batch_fill_ratio": grains / capacity if capacity else 0.0,
        "tail_or_recovery_rpc_fraction": (
            tail_calls / rpc_count if rpc_count else 0.0
        ),
        "live_blocks_at_delivery": max(
            (part.metrics.get("live_blocks_at_delivery", 0.0) for part in parts),
            default=0.0,
        ),
        "reduce_slots": max(
            (part.metrics.get("reduce_slots", 0.0) for part in parts),
            default=0.0,
        ),
    }


class RunDriver:
    """在不读取 Arena 内部 tables 的前提下协调 active Arenas。"""

    def __init__(
        self,
        compiled: CompiledPipeline,
        execution: ExecutionPool,
        chunks: tuple[SourceChunk, ...],
        *,
        run_salt: bytes,
        arena_id_start: int,
        max_inflight_arenas: int,
        limits: ArenaLimits,
    ) -> None:
        """初始化待 admission chunks、共享执行池和 in-flight 边界。"""

        self.compiled = compiled
        self.execution = execution
        self.pending = deque(chunks)
        self.run_salt = run_salt
        self.next_arena_id = arena_id_start
        self.max_inflight = max_inflight_arenas
        self.limits = limits
        self.active: dict[int, ArenaEngine] = {}
        self.completed: dict[int, ArenaResult] = {}
        self.active_high_watermark = 0
        self.live_blocks_high_watermark = 0

    def _admit(self) -> bool:
        """在 max_inflight 限制内创建 Arena，并用 coarse source blocks admission。"""

        progress = False
        while self.pending and len(self.active) < self.max_inflight:
            chunk = self.pending.popleft()
            arena = ArenaEngine(
                self.next_arena_id,
                self.compiled.dag,
                self.run_salt,
                limits=self.limits,
            )
            self.next_arena_id += 1
            import ray

            arena.admit_sources(
                chunk.values,
                position_starts=chunk.position_starts,
                block_factory=lambda values: ray.put(values),
            )
            self.active[chunk.index] = arena
            progress = True
        self.active_high_watermark = max(
            self.active_high_watermark, len(self.active)
        )
        return progress

    def _submit(self) -> bool:
        """按 Arena 顺序推进，并把 ready Grain 提交给有容量的 StageExecutor。"""

        progress = False
        now = time.monotonic()
        for chunk_index in sorted(self.active):
            arena = self.active[chunk_index]
            arena.advance(now)
            for stage in self.compiled.dag.stages:
                if stage.execution is None:
                    continue
                while self.execution.can_submit(stage.id):
                    intent = arena.reserve_dispatch(stage.id, now)
                    if intent is None:
                        break
                    if not self.execution.submit(intent):
                        raise ExecutionError(
                            "StageExecutor capacity changed during submit"
                        )
                    progress = True
        self.live_blocks_high_watermark = max(
            self.live_blocks_high_watermark,
            sum(arena.live_block_count for arena in self.active.values()),
        )
        return progress

    def _nearest_deadline(self) -> float | None:
        """返回所有 active Arena 中最近的 batch timeout 剩余时间。"""

        deadlines = [
            deadline
            for arena in self.active.values()
            for deadline in (arena.next_deadline(),)
            if deadline is not None
        ]
        return min(deadlines) if deadlines else None

    def _apply_event(self, event: ExecutionEvent) -> None:
        """把 completion/failure 只路由给拥有该 arena_id 的 ArenaEngine。"""

        arena = next(
            (
                arena
                for arena in self.active.values()
                if arena.id == event.arena_id
            ),
            None,
        )
        if arena is None:
            return
        if isinstance(event.result, DispatchCompletion):
            arena.commit(event.result)
        else:
            arena.handle_failure(event.result)

    def _finish_completed(self) -> bool:
        """完成、detach 并回收所有达到 completion 条件的 Arena。"""

        progress = False
        for chunk_index, arena in tuple(self.active.items()):
            if (
                not arena.is_complete()
                or self.execution.has_outstanding(arena.id)
            ):
                continue
            self.completed[chunk_index] = arena.finish()
            self.active.pop(chunk_index)
            progress = True
        return progress

    def _cancel_all(self) -> None:
        """run 失败时 best-effort 取消所有 active Arena 的 pending RPC。"""

        for arena in self.active.values():
            self.execution.cancel_arena(arena.id)
        self.active.clear()

    def run(self) -> tuple[ArenaResult, int, int, int]:
        """运行 single-writer event loop，直到所有 chunks 和 RPC 完成。"""

        try:
            self._admit()
            while self.pending or self.active or self.execution.pending_count:
                progress = self._submit()
                progress |= self._finish_completed()
                progress |= self._admit()

                if self.execution.pending_count:
                    delay = self._nearest_deadline()
                    timeout = 1.0 if delay is None else min(1.0, delay)
                    event = self.execution.poll(timeout=timeout)
                    if event is not None:
                        self._apply_event(event)
                        progress = True
                    continue
                if not progress and (self.pending or self.active):
                    delay = self._nearest_deadline()
                    if delay is not None:
                        time.sleep(delay)
                        continue
                    raise ExecutionError("RunDriver reached a non-terminal deadlock")
        except Exception:
            self._cancel_all()
            raise

        parts = tuple(self.completed[index] for index in sorted(self.completed))
        return (
            ArenaResult(
                outputs=tuple(value for part in parts for value in part.outputs),
                failures=tuple(value for part in parts for value in part.failures),
                suppressions=tuple(
                    value for part in parts for value in part.suppressions
                ),
                sources=tuple(value for part in parts for value in part.sources),
                metrics=_merge_metrics(parts),
                timeline=tuple(
                    value for part in parts for value in part.timeline
                ),
            ),
            self.next_arena_id,
            self.active_high_watermark,
            self.live_blocks_high_watermark,
        )


def resolve_outputs(outputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """按 coarse block 合并 ray.get，并解析最终 BlockSlice。"""

    slices = [value for value in outputs if isinstance(value, BlockSlice)]
    if not slices:
        return outputs
    import ray

    blocks: dict[Any, tuple[Any, ...]] = {}
    resolved = []
    for value in outputs:
        if not isinstance(value, BlockSlice):
            resolved.append(value)
            continue
        if value.block not in blocks:
            blocks[value.block] = ray.get(value.block)
        resolved.append(blocks[value.block][value.row])
    return tuple(resolved)
