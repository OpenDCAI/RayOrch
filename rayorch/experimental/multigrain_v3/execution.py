"""persistent per-Stage Ray actor pools 与 coarse-block RPC transport。

本模块只拥有 Actor、ObjectRef 和 PendingRPC，不读取 Arena semantic tables，也不决定
lineage/recovery 语义。
"""

from __future__ import annotations

import time
from collections import deque
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
    worker_generation: int
    report_ref: Any
    output_refs: tuple[Any, ...]
    submitted_at: float


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """completion/failure 与所属 Arena routing key 的组合。"""
    arena_id: int
    call: Any
    result: DispatchCompletion | DispatchFailure


@dataclass(slots=True)
class ActorCreditWindow:
    """维护每个 actor slot 的有界 outstanding RPC credit。"""

    max_outstanding_per_actor: int
    outstanding_by_slot: list[int]

    def __init__(self, actor_count: int, max_outstanding_per_actor: int) -> None:
        """按固定 actor 数量创建 credit ledger，并校验正容量。"""

        if actor_count <= 0:
            raise ValueError("actor_count must be positive")
        if max_outstanding_per_actor <= 0:
            raise ValueError("max_outstanding_per_actor must be positive")
        self.max_outstanding_per_actor = max_outstanding_per_actor
        self.outstanding_by_slot = [0] * actor_count

    def has_capacity(self) -> bool:
        """返回是否至少有一个 actor slot 仍可接纳 RPC。"""

        return any(
            value < self.max_outstanding_per_actor
            for value in self.outstanding_by_slot
        )

    def try_acquire(
        self,
        *,
        start: int,
        candidates: tuple[int, ...],
    ) -> int | None:
        """从 start 起轮询候选 slot，并原子占用一个可用 credit。"""

        candidate_set = set(candidates)
        count = len(self.outstanding_by_slot)
        for offset in range(count):
            slot = (start + offset) % count
            if slot not in candidate_set:
                continue
            if (
                self.outstanding_by_slot[slot]
                >= self.max_outstanding_per_actor
            ):
                continue
            self.outstanding_by_slot[slot] += 1
            return slot
        return None

    def release(self, slot: int) -> None:
        """终止一个 RPC 时释放 credit，并拒绝重复释放。"""

        if self.outstanding_by_slot[slot] <= 0:
            raise ExecutionError(f"actor slot {slot} credit released twice")
        self.outstanding_by_slot[slot] -= 1

    def clear(self) -> None:
        """关闭执行池时清空全部 slot credit。"""

        for slot in range(len(self.outstanding_by_slot)):
            self.outstanding_by_slot[slot] = 0


class StageExecutor:
    """一个 Stage 对应的 persistent actor pool 与 per-actor backpressure。"""

    def __init__(
        self,
        stage: StageSpec,
        *,
        max_outstanding_per_actor: int = 1,
        actor_max_concurrency: int = 1,
    ) -> None:
        """按 StageSpec 创建固定 replicas，并初始化 pending 计数。"""

        import ray

        if stage.kind is Primitive.SOURCE:
            raise ValueError("Source has no StageExecutor")
        if not ray.is_initialized():
            raise ExecutionError("Ray must be initialized before StageExecutor")
        if max_outstanding_per_actor <= 0:
            raise ValueError("max_outstanding_per_actor must be positive")
        if actor_max_concurrency <= 0:
            raise ValueError("actor_max_concurrency must be positive")
        assert stage.execution is not None and stage.udf is not None
        self.stage = stage
        self._worker_class = get_ray_worker_class()
        self._actor_spec = (
            stage.udf.target,
            stage.udf.init_args,
            dict(stage.udf.init_kwargs),
            dict(stage.execution.ray_options),
        )
        configured_concurrency = int(
            self._actor_spec[3].get("max_concurrency", actor_max_concurrency)
        )
        configured_outstanding = (
            stage.execution.max_outstanding_per_actor
            if stage.execution.max_outstanding_per_actor is not None
            else max_outstanding_per_actor
        )
        if configured_concurrency <= 0:
            raise ValueError("actor max_concurrency must be positive")
        if configured_outstanding < configured_concurrency:
            raise ValueError(
                "max_outstanding_per_actor must be greater than or equal to "
                "actor max_concurrency"
            )
        self.max_outstanding_per_actor = configured_outstanding
        self.actor_max_concurrency = configured_concurrency
        self.actors = [
            self._spawn() for _ in range(stage.execution.replicas)
        ]
        self.worker_generations = [0] * len(self.actors)
        self.credits = ActorCreditWindow(
            len(self.actors), self.max_outstanding_per_actor
        )
        self.round_robin = 0
        self.pending: dict[Any, PendingRPC] = {}

    @property
    def pending_by_slot(self) -> list[int]:
        """兼容旧诊断代码，返回规范 outstanding credit 计数。"""

        return self.credits.outstanding_by_slot

    def _spawn(self):
        """根据 UdfSpec 和 Ray options 创建一个 persistent actor。"""

        target, init_args, init_kwargs, options = self._actor_spec
        options = dict(options)
        options.pop("max_concurrency", None)
        return self._worker_class.options(
            max_task_retries=0,
            max_concurrency=self.actor_max_concurrency,
            **options,
        ).remote(self.stage, target, init_args, init_kwargs)

    def can_submit(self) -> bool:
        """判断是否至少有一个 actor 未达到 outstanding 上限。"""

        return self.credits.has_capacity()

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
        index = self.credits.try_acquire(
            start=self.round_robin,
            candidates=tuple(choices),
        )
        if index is not None:
            self.round_robin = (index + 1) % count
        return index

    def submit(self, intent: DispatchIntent) -> bool:
        """提交一个 coarse RPC，并登记 report/output ObjectRefs。"""

        slot = self._choose(intent)
        if slot is None:
            return False
        output_count = (
            0 if self.stage.kind is Primitive.FILTER else self.stage.output_count
        )
        try:
            refs = self.actors[slot].run.options(
                num_returns=1 + output_count,
                max_task_retries=0,
            ).remote(intent.call, *intent.input_blocks)
        except Exception:
            self.credits.release(slot)
            raise
        refs_tuple = (
            (refs,)
            if output_count == 0
            else tuple(refs)
        )
        pending = PendingRPC(
            intent,
            self.stage.id,
            slot,
            self.worker_generations[slot],
            refs_tuple[0],
            refs_tuple[1:],
            time.monotonic(),
        )
        self.pending[pending.report_ref] = pending
        return True

    def pop(self, report_ref: Any) -> PendingRPC:
        """移除已完成 PendingRPC，并释放对应 actor outstanding credit。"""

        pending = self.pending.pop(report_ref)
        self.credits.release(pending.worker_slot)
        return pending

    def replace(self, slot: int) -> None:
        """杀死并重建指定 actor slot，用于基础设施失败恢复。"""

        import ray

        try:
            ray.kill(self.actors[slot])
        except Exception:
            pass
        self.worker_generations[slot] += 1
        self.actors[slot] = self._spawn()

    def fail_generation(
        self,
        slot: int,
        generation: int,
    ) -> tuple[PendingRPC, ...]:
        """摘除故障 slot generation 的全部 RPC，释放 credit 并替换 actor。"""

        failed = tuple(
            pending
            for pending in self.pending.values()
            if pending.worker_slot == slot
            and pending.worker_generation == generation
        )
        for pending in failed:
            self.pending.pop(pending.report_ref)
            self.credits.release(slot)
        self.replace(slot)
        return failed

    def cancel_arena(self, arena_id: int) -> None:
        """best-effort 取消属于指定 Arena 的 pending reports。"""

        import ray

        for report_ref, pending in tuple(self.pending.items()):
            if pending.intent.arena_id != arena_id:
                continue
            self.pending.pop(report_ref)
            self.credits.release(pending.worker_slot)
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
        self.credits.clear()


class ExecutionPool:
    """一次 run 内由所有 in-flight Arena 共享的 StageExecutor 集合。"""

    def __init__(
        self,
        dag: CompiledDAG,
        *,
        max_outstanding_per_actor: int = 1,
        actor_max_concurrency: int = 1,
    ) -> None:
        """为所有非 Source Stage 创建独立 persistent actor pool。"""

        import ray

        self.dag = dag
        self.executors = {
            stage.id: StageExecutor(
                stage,
                max_outstanding_per_actor=max_outstanding_per_actor,
                actor_max_concurrency=actor_max_concurrency,
            )
            for stage in dag.stages
            if stage.kind is not Primitive.SOURCE
        }
        self.ray = ray
        self.buffered_events: deque[ExecutionEvent] = deque()

    @property
    def pending_count(self) -> int:
        """返回当前 run 的 pending RPC 总数。"""

        return len(self.buffered_events) + sum(
            len(executor.pending) for executor in self.executors.values()
        )

    def has_outstanding(self, arena_id: int) -> bool:
        """返回 transport 是否仍持有指定 Arena 的 RPC 或待路由事件。"""

        return any(
            event.arena_id == arena_id
            for event in self.buffered_events
        ) or any(
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

        if self.buffered_events:
            return self.buffered_events.popleft()
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
        pending = executor.pending[report_ref]
        try:
            report = self.ray.get(report_ref)
        except Exception as error:
            failed = executor.fail_generation(
                pending.worker_slot,
                pending.worker_generation,
            )
            events = tuple(
                ExecutionEvent(
                    item.intent.arena_id,
                    item.intent.call,
                    DispatchFailure(
                        item.intent.call.dispatch,
                        FailureKind.INFRA_FAILURE,
                        f"{type(error).__name__}: {error}",
                        worker_slot=item.worker_slot,
                    ),
                )
                for item in failed
            )
            self.buffered_events.extend(events[1:])
            return events[0]
        pending = executor.pop(report_ref)
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

        self.buffered_events = deque(
            event
            for event in self.buffered_events
            if event.arena_id != arena_id
        )
        for executor in self.executors.values():
            executor.cancel_arena(arena_id)

    def actor_stats(self) -> dict[int, tuple[dict[str, Any], ...]]:
        """收集每个 Stage actor 的调用次数、PID 和 RSS。"""

        return {
            stage: tuple(
                self.ray.get([actor.stats.remote() for actor in executor.actors])
            )
            for stage, executor in self.executors.items()
        }

    def shutdown(self) -> None:
        """关闭所有 StageExecutor。"""
        self.buffered_events.clear()
        for executor in self.executors.values():
            executor.shutdown()
