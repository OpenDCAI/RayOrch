"""persistent per-Stage Ray actor pools 与 coarse-block RPC transport。

本模块只拥有 Actor、ObjectRef 和 PendingRPC，不读取 Arena semantic tables，也不决定
lineage/recovery 语义。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .contracts import ExecutionError
from .protocol import (
    BatchReport,
    DispatchCompletion,
    DispatchFailure,
    DispatchIntent,
    FailureKind,
)
from .dag import CompiledDAG, Primitive, StageSpec
from .worker import get_ray_worker_class


@dataclass(frozen=True, slots=True)
class PendingRPC:
    """Transport 对一个已提交 DispatchIntent 持有的 Ray handles。"""
    intent: DispatchIntent
    stage: int
    worker_slot: int
    report_ref: Any
    output_refs: tuple[Any, ...]
    submitted_at: float


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """completion/failure 与所属 Arena routing key 的组合。"""
    arena_id: int
    call: Any
    result: DispatchCompletion | DispatchFailure


class StageExecutor:
    """一个 Stage 对应的 persistent actor pool 与 per-actor backpressure。"""

    def __init__(
        self,
        stage: StageSpec,
        *,
        max_pending_per_actor: int = 1,
    ) -> None:
        """按 StageSpec 创建固定 replicas，并初始化 pending 计数。"""

        import ray

        if stage.kind is Primitive.SOURCE:
            raise ValueError("Source has no StageExecutor")
        if not ray.is_initialized():
            raise ExecutionError("Ray must be initialized before StageExecutor")
        if max_pending_per_actor <= 0:
            raise ValueError("max_pending_per_actor must be positive")
        assert stage.execution is not None and stage.udf is not None
        self.stage = stage
        self.max_pending_per_actor = max_pending_per_actor
        self._worker_class = get_ray_worker_class()
        self._actor_spec = (
            stage.udf.target,
            stage.udf.init_args,
            dict(stage.udf.init_kwargs),
            dict(stage.execution.ray_options),
        )
        self.actors = [
            self._spawn() for _ in range(stage.execution.replicas)
        ]
        self.pending_by_slot = [0] * len(self.actors)
        self.round_robin = 0
        self.pending: dict[Any, PendingRPC] = {}

    def _spawn(self):
        """根据 UdfSpec 和 Ray options 创建一个 persistent actor。"""

        target, init_args, init_kwargs, options = self._actor_spec
        return self._worker_class.options(
            max_task_retries=0,
            **options,
        ).remote(self.stage, target, init_args, init_kwargs)

    def can_submit(self) -> bool:
        """判断是否至少有一个 actor 未达到 pending 上限。"""

        return any(
            pending < self.max_pending_per_actor
            for pending in self.pending_by_slot
        )

    def _choose(self, intent: DispatchIntent) -> int | None:
        """按 round-robin 和 actor policy 选择可用 worker slot。"""

        count = len(self.actors)
        choices = range(count)
        if intent.actor_policy == "fresh":
            choices = tuple(
                index
                for index in range(count)
                if index != intent.avoid_worker_slot
            ) or range(count)
        for offset in range(count):
            index = (self.round_robin + offset) % count
            if index not in choices:
                continue
            if self.pending_by_slot[index] < self.max_pending_per_actor:
                self.round_robin = (index + 1) % count
                return index
        return None

    def submit(self, intent: DispatchIntent) -> bool:
        """提交一个 coarse RPC，并登记 report/output ObjectRefs。"""

        slot = self._choose(intent)
        if slot is None:
            return False
        output_count = (
            0 if self.stage.kind is Primitive.FILTER else self.stage.output_count
        )
        refs = self.actors[slot].run.options(
            num_returns=1 + output_count,
            max_task_retries=0,
        ).remote(intent.call, *intent.input_blocks)
        refs_tuple = (
            (refs,)
            if output_count == 0
            else tuple(refs)
        )
        pending = PendingRPC(
            intent,
            self.stage.id,
            slot,
            refs_tuple[0],
            refs_tuple[1:],
            time.monotonic(),
        )
        self.pending[pending.report_ref] = pending
        self.pending_by_slot[slot] += 1
        return True

    def pop(self, report_ref: Any) -> PendingRPC:
        """移除已完成 PendingRPC，并释放对应 actor pending credit。"""

        pending = self.pending.pop(report_ref)
        self.pending_by_slot[pending.worker_slot] -= 1
        return pending

    def replace(self, slot: int) -> None:
        """杀死并重建指定 actor slot，用于基础设施失败恢复。"""

        import ray

        try:
            ray.kill(self.actors[slot])
        except Exception:
            pass
        self.actors[slot] = self._spawn()

    def cancel_arena(self, arena_id: int) -> None:
        """best-effort 取消属于指定 Arena 的 pending reports。"""

        import ray

        for report_ref, pending in tuple(self.pending.items()):
            if pending.intent.arena_id != arena_id:
                continue
            self.pending.pop(report_ref)
            self.pending_by_slot[pending.worker_slot] -= 1
            try:
                ray.cancel(report_ref, force=True)
            except Exception:
                pass

    def shutdown(self) -> None:
        """终止本 Stage 的全部 actors 并清空 transport 状态。"""

        import ray

        for actor in self.actors:
            ray.kill(actor)
        self.actors.clear()
        self.pending.clear()


class ExecutionPool:
    """一次 run 内由所有 in-flight Arena 共享的 StageExecutor 集合。"""
    """All StageExecutors shared by every in-flight Arena in one run."""

    def __init__(
        self,
        dag: CompiledDAG,
        *,
        max_pending_per_actor: int = 1,
    ) -> None:
        """为所有非 Source Stage 创建独立 persistent actor pool。"""

        import ray

        self.dag = dag
        self.executors = {
            stage.id: StageExecutor(
                stage,
                max_pending_per_actor=max_pending_per_actor,
            )
            for stage in dag.stages
            if stage.kind is not Primitive.SOURCE
        }
        self.ray = ray

    @property
    def pending_count(self) -> int:
        """返回当前 run 的 pending RPC 总数。"""

        return sum(len(executor.pending) for executor in self.executors.values())

    def pending_for_arena(self, arena_id: int) -> int:
        """返回指定 Arena 尚未完成的 RPC 数。"""

        return sum(
            pending.intent.arena_id == arena_id
            for executor in self.executors.values()
            for pending in executor.pending.values()
        )

    def can_submit(self, stage: int) -> bool:
        """查询指定 Stage actor pool 是否仍有提交容量。"""

        return self.executors[stage].can_submit()

    def submit(self, intent: DispatchIntent) -> bool:
        """把 DispatchIntent 转交给对应 StageExecutor。"""

        return self.executors[intent.call.stage].submit(intent)

    def ready(self) -> None:
        """等待所有 persistent actors 完成构造和 UDF 初始化。"""

        refs = [
            actor.stats.remote()
            for executor in self.executors.values()
            for actor in executor.actors
        ]
        if refs:
            self.ray.get(refs)

    def poll(
        self,
        *,
        timeout: float | None = None,
    ) -> ExecutionEvent | None:
        """等待一个 report，并转换为带 arena_id 的 completion/failure event。"""

        refs = [
            report_ref
            for executor in self.executors.values()
            for report_ref in executor.pending
        ]
        if not refs:
            return None
        ready, _ = self.ray.wait(refs, num_returns=1, timeout=timeout)
        if not ready:
            return None
        report_ref = ready[0]
        executor = next(
            executor
            for executor in self.executors.values()
            if report_ref in executor.pending
        )
        pending = executor.pop(report_ref)
        try:
            report = self.ray.get(report_ref)
        except Exception as error:
            executor.replace(pending.worker_slot)
            return ExecutionEvent(
                pending.intent.arena_id,
                pending.intent.call,
                DispatchFailure(
                    pending.intent.call.dispatch,
                    FailureKind.INFRA_FAILURE,
                    f"{type(error).__name__}: {error}",
                    worker_slot=pending.worker_slot,
                ),
            )
        if isinstance(report, DispatchFailure):
            return ExecutionEvent(
                pending.intent.arena_id,
                pending.intent.call,
                DispatchFailure(
                    report.dispatch,
                    report.kind,
                    report.message,
                    bad_token=report.bad_token,
                    worker_slot=pending.worker_slot,
                ),
            )
        if not isinstance(report, BatchReport):
            return ExecutionEvent(
                pending.intent.arena_id,
                pending.intent.call,
                DispatchFailure(
                    pending.intent.call.dispatch,
                    FailureKind.CONTRACT_ABORT,
                    "worker returned unknown report",
                    worker_slot=pending.worker_slot,
                ),
            )
        return ExecutionEvent(
            pending.intent.arena_id,
            pending.intent.call,
            DispatchCompletion(
                pending.intent.arena_id,
                pending.intent.call,
                report,
                pending.output_refs,
                pending.worker_slot,
                pending.submitted_at,
                time.monotonic(),
            ),
        )

    def cancel_arena(self, arena_id: int) -> None:
        """通知所有 StageExecutor 取消指定 Arena 的 pending RPC。"""

        for executor in self.executors.values():
            executor.cancel_arena(arena_id)

    def actor_stats(self) -> dict[int, tuple[dict[str, int], ...]]:
        """收集每个 Stage actor 的调用次数、PID 和 RSS。"""

        return {
            stage: tuple(
                self.ray.get([actor.stats.remote() for actor in executor.actors])
            )
            for stage, executor in self.executors.items()
        }

    def shutdown(self) -> None:
        """关闭所有 StageExecutor。"""
        for executor in self.executors.values():
            executor.shutdown()
