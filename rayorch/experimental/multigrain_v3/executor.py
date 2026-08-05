"""Multigrain V3 的 public Executor facade。

本模块负责切 source microbatches、组装 ExecutionPool/RunDriver，并生成 detached
RunResult；不实现 Arena 语义或 Ray actor 选择。
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .api import CompiledPipeline, Pipeline
from .contracts import ExecutionError
from .arena import (
    ArenaLimits,
)
from .driver import RunDriver, SourceChunk, resolve_outputs
from .execution import ExecutionPool
from .protocol import (
    DispatchTimeline,
    FailureSnapshot,
    SourceSnapshot,
    SuppressionSnapshot,
)


@dataclass(frozen=True, slots=True)
class RunResult:
    """run 完成后的 detached outputs、失败血缘、指标和 dispatch timeline。"""
    outputs: tuple[Any, ...] = ()
    failures: tuple[FailureSnapshot, ...] = ()
    suppressions: tuple[SuppressionSnapshot, ...] = ()
    sources: tuple[SourceSnapshot, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    timeline: tuple[DispatchTimeline, ...] = ()

    def get(self) -> tuple[Any, ...]:
        """解析最终 BlockSlice；同一 coarse block 只执行一次 ray.get。"""

        return resolve_outputs(self.outputs)


class Executor:
    """编译 Pipeline，并在 Ray 上执行有限、等长、按 position 对齐的 sources。"""

    def __init__(
        self,
        pipeline: Pipeline | CompiledPipeline,
        *,
        microbatch_size: int | None = None,
        max_inflight_arenas: int = 1,
        arena_limits: ArenaLimits = ArenaLimits(),
        max_outstanding_per_actor: int | None = None,
        actor_max_concurrency: int = 1,
        max_pending_per_actor: int | None = None,
    ) -> None:
        """保存 compiled pipeline、microbatch/in-flight 边界和 Arena limits。"""

        self.compiled = (
            pipeline.compile() if isinstance(pipeline, Pipeline) else pipeline
        )
        if microbatch_size is not None and microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if max_inflight_arenas <= 0:
            raise ValueError("max_inflight_arenas must be positive")
        if (
            max_outstanding_per_actor is not None
            and max_pending_per_actor is not None
        ):
            raise ValueError(
                "use only max_outstanding_per_actor; "
                "max_pending_per_actor is a compatibility alias"
            )
        outstanding = (
            max_outstanding_per_actor
            if max_outstanding_per_actor is not None
            else max_pending_per_actor
        )
        if outstanding is None:
            outstanding = 1
        if outstanding <= 0:
            raise ValueError("max_outstanding_per_actor must be positive")
        if actor_max_concurrency <= 0:
            raise ValueError("actor_max_concurrency must be positive")
        if outstanding < actor_max_concurrency:
            raise ValueError(
                "max_outstanding_per_actor must be greater than or equal to "
                "actor_max_concurrency"
            )
        self.microbatch_size = microbatch_size
        self.max_inflight_arenas = max_inflight_arenas
        self.arena_limits = arena_limits
        self.max_outstanding_per_actor = outstanding
        self.actor_max_concurrency = actor_max_concurrency
        self.next_arena_id = 0

    @property
    def max_pending_per_actor(self) -> int:
        """兼容旧实验代码，返回规范 outstanding window 配置。"""

        return self.max_outstanding_per_actor

    def run(self, *sources: Any) -> RunResult:
        """启动 persistent Stage actors，运行多 Arena event loop 并返回 detached result。"""

        if not sources or any(
            not isinstance(source, (list, tuple)) for source in sources
        ):
            raise ExecutionError("sources must be finite list/tuple sequences")
        lengths = {len(source) for source in sources}
        if len(lengths) != 1:
            raise ExecutionError("all source sequences must have equal length")
        total = next(iter(lengths))
        chunk_size = self.microbatch_size or max(total, 1)
        chunks = tuple(
            SourceChunk(
                index=index,
                values=tuple(
                    tuple(source[start : start + chunk_size])
                    for source in sources
                ),
                position_starts=tuple(start for _ in sources),
            )
            for index, start in enumerate(range(0, total, chunk_size))
        )
        if not chunks:
            chunks = (
                SourceChunk(
                    0,
                    tuple(() for _ in sources),
                    tuple(0 for _ in sources),
                ),
            )

        started = time.monotonic()
        execution = ExecutionPool(
            self.compiled.dag,
            max_outstanding_per_actor=self.max_outstanding_per_actor,
            actor_max_concurrency=self.actor_max_concurrency,
        )
        try:
            execution.ready()
            measured = time.monotonic()
            driver = RunDriver(
                self.compiled,
                execution,
                chunks,
                run_salt=secrets.token_bytes(16),
                arena_id_start=self.next_arena_id,
                max_inflight_arenas=self.max_inflight_arenas,
                limits=self.arena_limits,
            )
            arena_result, self.next_arena_id, active_peak, block_peak = driver.run()
            finished = time.monotonic()
            metrics = dict(arena_result.metrics)
            metrics.update(
                {
                    "startup_time_s": measured - started,
                    "measured_wall_time_s": finished - measured,
                    "end_to_end_wall_time_s": finished - started,
                    "active_arenas_high_watermark": float(active_peak),
                    "live_blocks_across_arenas_high_watermark": float(block_peak),
                }
            )
            for stage, stats in execution.actor_stats().items():
                metrics[f"actor_count_stage_{stage}"] = float(len(stats))
                metrics[f"actor_calls_stage_{stage}"] = float(
                    sum(item["calls"] for item in stats)
                )
                audit_totals: dict[str, float] = {}
                for item in stats:
                    for key, value in item.get("audit", {}).items():
                        if isinstance(value, (int, float)):
                            audit_totals[key] = audit_totals.get(key, 0.0) + value
                for key, value in audit_totals.items():
                    metrics[f"actor_audit_stage_{stage}_{key}"] = value
            return RunResult(
                outputs=arena_result.outputs,
                failures=arena_result.failures,
                suppressions=arena_result.suppressions,
                sources=arena_result.sources,
                metrics=metrics,
                timeline=arena_result.timeline,
            )
        finally:
            execution.shutdown()
