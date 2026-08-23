"""基于 Ray actor pools 的 v3.6 执行器与多 microbatch driver。

逻辑 Program、MicrobatchEngine 状态机和 Worker ABI 均不依赖 Ray；本模块是唯一持有
actor handle 与 pending RPC ObjectRef 的 driver。不同 microbatch 共享 actor capacity，但一个 RPC
不会静默混合多个 microbatch 的 Grain，从而保持与设计文档一致的实验口径。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ..api import Pipeline
from ..runtime.materialize import materialize_tree
from ..model import CallRef, ExecutionError
from ..program.plan import ActorPoolSpec, CompiledProgram
from ..protocol import (
    DispatchFailure,
    DispatchFailureKind,
    RowBinding,
)
from ..recovery import RecoveryAction
from ..runtime import DispatchBatch, MicrobatchEngine
from .ray_backend import _RayBlockStore, _RayWorkerActor
from .result import CallMetrics, MicrobatchMetrics, RunResult
from .worker import WorkerSnapshot


# ── Driver-local mutable counters and physical ownership records ─────────────


@dataclass(slots=True)
class _CallCounters:
    """Executor.run 内部唯一的可变 Call 计数器。"""

    actor_instances: int = 0
    rpcs: int = 0
    grains: int = 0
    retries: int = 0
    batch_sizes: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _ActorSlot:
    """driver 内部的 actor capacity token。"""

    call: CallRef
    handle: Any
    busy: bool = False


@dataclass(slots=True)
class _MicrobatchSlot:
    """一个 source microbatch 及其唯一语义状态机。"""

    index: int
    engine: MicrobatchEngine


@dataclass(frozen=True, slots=True)
class _DispatchLease:
    """pending ObjectRef 对应的 generation-fenced dispatch lease。"""

    microbatch_index: int
    actor: _ActorSlot
    batch: DispatchBatch


class Executor:
    """Call-only actor pools 与多 microbatch overlap 的参考执行器。

    Executor 独占 actor capacity、pending RPC lease 和 run-local counters；
    primitive 传播、Entity lineage 与 Grain phase 仍分别归 Engine/DispatchState。
    """

    def __init__(
        self,
        pipeline: Pipeline | CompiledProgram,
        *,
        address: str | None = None,
        ray_init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """编译 Pipeline、连接 Ray，并按 Call 创建持久 actor pools。"""

        import ray  # pyright: ignore[reportMissingImports]

        self.ray = ray
        self.compiled = (
            pipeline if isinstance(pipeline, CompiledProgram) else pipeline.compile()
        )
        self.plan = self.compiled.plan
        self._owns_ray = not ray.is_initialized()
        if self._owns_ray:
            init_kwargs = dict(ray_init_kwargs or {})
            if address is not None:
                init_kwargs["address"] = address
            ray.init(**init_kwargs)

        self.store = _RayBlockStore(ray)
        self._actors: dict[CallRef, list[_ActorSlot]] = {}
        self._counters: dict[CallRef, _CallCounters] = {}
        self._closed = False
        try:
            self._actor_class = ray.remote(_RayWorkerActor)
            # 先登记可清理容器再逐 actor append；即使第 N 个构造同步失败，
            # 前 N-1 个 handle 仍属于统一 close 路径。
            for call in self.plan.calls:
                self._actors[call] = []
                self._create_pool(call)
            # 统一 ready 屏障让 run() 计时不混入 UDF/模型初始化。
            startup_refs = [
                actor.handle.ready.remote()
                for actors in self._actors.values()
                for actor in actors
            ]
            ray.get(startup_refs)
        except Exception:
            # 构造失败时对象不会交给调用方，必须在此回收已创建的 actors。
            self.close()
            raise

    # ── Public run/close lifecycle ───────────────────────────────────────

    def run(
        self,
        *source_columns: Sequence[Any],
        microbatch_size: int | None = None,
        max_active_microbatches: int = 1,
    ) -> RunResult:
        """执行行对齐的有限 Sequences，并限制同时活跃的 microbatch 数。"""

        if self._closed:
            raise RuntimeError("Executor is closed")

        columns = self._normalize_sources(source_columns)
        row_count = len(columns[0])
        if max_active_microbatches <= 0:
            raise ValueError("max_active_microbatches must be positive")
        if microbatch_size is None:
            microbatch_size = max(1, row_count)
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")

        slices = [
            tuple(column[start : start + microbatch_size] for column in columns)
            for start in range(0, row_count, microbatch_size)
        ]
        if not slices:
            slices = [tuple(() for _ in columns)]

        # actor pool 跨 run 持久化，但可变计数器严格按 run 隔离。
        self.store.clear_cache()
        self._counters = {
            call: _CallCounters(actor_instances=len(self._actors[call]))
            for call in self.plan.calls
        }
        metrics_by_microbatch: list[MicrobatchMetrics | None] = [None] * len(slices)
        active: dict[int, _MicrobatchSlot] = {}
        completed: dict[int, object] = {}
        pending: dict[Any, _DispatchLease] = {}
        next_microbatch = 0
        execution_started = False
        high_watermark = 0
        started = time.perf_counter()

        # Event-loop invariants:
        # 1. active[index] uniquely owns that microbatch's Engine;
        # 2. every pending ObjectRef maps to exactly one fenced lease;
        # 3. a busy actor has one such lease and is released in finally;
        # 4. materialization requires both no pending lease and Engine complete.
        try:
            while len(completed) < len(slices):
                while (
                    next_microbatch < len(slices)
                    and len(active) < max_active_microbatches
                ):
                    execution_started = True
                    engine = self._admit_microbatch(slices[next_microbatch])
                    active[next_microbatch] = _MicrobatchSlot(
                        next_microbatch,
                        engine,
                    )
                    next_microbatch += 1
                    high_watermark = max(high_watermark, len(active))

                self._dispatch_ready(active, pending)

                # 只有没有 pending lease 的 microbatch 才能离开 active 集合。
                pending_microbatches = {
                    lease.microbatch_index for lease in pending.values()
                }
                for index, slot in tuple(active.items()):
                    if (
                        index not in pending_microbatches
                        and slot.engine.is_complete()
                    ):
                        completed[index] = materialize_tree(
                            self.plan,
                            slot.engine,
                            self.store,
                        )
                        # materialize 已把最终业务对象复制到 driver output；cache
                        # 只用于一次粗块去重，不能把所有 microbatch 的 payload 留到 run 结束。
                        self.store.clear_cache()
                        # 最终业务值已经复制到 driver 输出；清空 ValueTable 会释放
                        # page-image 等 ObjectRef；结果只保留不可变计数快照。
                        released = slot.engine.release_values()
                        metrics_by_microbatch[index] = MicrobatchMetrics(
                            index=index,
                            entity_count=slot.engine.entity_count,
                            item_count=slot.engine.item_count,
                            expansion_count=slot.engine.expansion_count,
                            grain_count=slot.engine.grain_count,
                            released_values=released,
                        )
                        del active[index]

                if len(completed) == len(slices):
                    break
                if not pending and next_microbatch < len(slices):
                    # 当前 active microbatch 已完成，但仍有尚未 admission 的 source
                    # slice；下一 turn 会填充空出的 credit，这不是 deadlock。
                    continue
                if not pending:
                    summaries = ", ".join(
                        f"microbatch[{index}] {slot.engine.progress_summary()}"
                        for index, slot in sorted(active.items())
                    )
                    raise RuntimeError(f"v3.6 Ray runtime deadlocked: {summaries}")

                ready, _ = self.ray.wait(list(pending), num_returns=1)
                result_ref = ready[0]
                lease = pending.pop(result_ref)
                engine = active[lease.microbatch_index].engine
                try:
                    result = self.ray.get(result_ref)
                except Exception as error:  # Ray 对用户异常和 actor 异常统一在 get 抛出
                    self._handle_infrastructure_failure(engine, lease, error)
                else:
                    if isinstance(result, DispatchFailure):
                        self._handle_dispatch_failure(engine, lease, result)
                    else:
                        for report in result:
                            engine.commit_report(report)
                finally:
                    lease.actor.busy = False

            elapsed_s = time.perf_counter() - started
            worker_snapshots = self._observe_workers()
            calls = self._freeze_call_metrics(worker_snapshots)
            if any(metrics is None for metrics in metrics_by_microbatch):
                raise AssertionError("completed run lost a microbatch metrics snapshot")
            return RunResult(
                self._merge_outputs([completed[index] for index in range(len(slices))]),
                elapsed_s,
                calls,
                cast(tuple[MicrobatchMetrics, ...], tuple(metrics_by_microbatch)),
                high_watermark,
            )
        except BaseException:
            # Ray Data uses the same execution-level fail-stop contract: once an
            # exception escapes an active execution, pending actor work is not
            # treated as a reusable clean queue. Preserve the original exception.
            if execution_started:
                try:
                    self.close()
                except BaseException:
                    pass
            raise

    def close(self) -> None:
        """释放 actors；仅关闭由本 Executor 初始化的 Ray runtime。"""

        if self._closed:
            return
        self._closed = True

        for actors in self._actors.values():
            for actor in actors:
                try:
                    self.ray.kill(actor.handle, no_restart=True)
                except Exception:
                    pass
        self._actors.clear()
        self.store.clear_cache()
        if self._owns_ray and self.ray.is_initialized():
            self.ray.shutdown()

    # ── Observation and immutable result snapshots ──────────────────────

    def _observe_workers(self) -> dict[CallRef, tuple[WorkerSnapshot, ...]]:
        """Best-effort physical diagnostics must not invalidate business output."""

        result: dict[CallRef, list[WorkerSnapshot | None]] = {
            call: [None] * len(actors)
            for call, actors in self._actors.items()
        }
        pending = {}
        for call, actors in self._actors.items():
            for index, actor in enumerate(actors):
                try:
                    reference = actor.handle.observe.remote()
                except Exception as error:
                    result[call][index] = WorkerSnapshot(
                        lifetime_calls=0,
                        pid=0,
                        rss_bytes=0,
                        error=repr(error),
                    )
                else:
                    pending[reference] = (call, index)
        while pending:
            ready, _ = self.ray.wait(list(pending), num_returns=1)
            reference = ready[0]
            call, index = pending.pop(reference)
            try:
                result[call][index] = self.ray.get(reference)
            except Exception as error:
                result[call][index] = WorkerSnapshot(
                    lifetime_calls=0,
                    pid=0,
                    rss_bytes=0,
                    error=repr(error),
                )
        if any(
            observation is None
            for observations in result.values()
            for observation in observations
        ):
            raise AssertionError("worker observation collection lost an actor")
        return {
            call: cast(tuple[WorkerSnapshot, ...], tuple(observations))
            for call, observations in result.items()
        }

    def _freeze_call_metrics(
        self,
        workers: dict[CallRef, tuple[WorkerSnapshot, ...]],
    ) -> tuple[CallMetrics, ...]:
        """按 CallRef 顺序冻结内部计数，并移除公开 Ref-keyed 映射。"""

        snapshots = []
        for call in sorted(self.plan.calls, key=lambda ref: ref.value):
            counters = self._counters[call]
            target = self.plan.call(call).udf.target
            snapshots.append(
                CallMetrics(
                    call_index=call.value,
                    udf_name=self._udf_name(target),
                    actor_instances=counters.actor_instances,
                    rpcs=counters.rpcs,
                    grains=counters.grains,
                    retries=counters.retries,
                    batch_sizes=tuple(counters.batch_sizes),
                    worker_snapshots=workers[call],
                )
            )
        return tuple(snapshots)

    def __enter__(self):
        """支持用 context manager 约束 ExecutionPool 生命周期。"""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """离开 context 时释放 actors。"""

        self.close()

    # ── Source admission and work-conserving dispatch ───────────────────

    def _normalize_sources(
        self,
        source_columns: tuple[Sequence[Any], ...],
    ) -> tuple[tuple[Any, ...], ...]:
        """Eagerly snapshot finite sources and validate row alignment."""

        if len(source_columns) != len(self.plan.source_ports):
            raise ValueError("source column count does not match Pipeline.forward")
        columns = tuple(tuple(column) for column in source_columns)
        if len({len(column) for column in columns}) != 1:
            raise ValueError("source columns must be row-aligned")
        return columns

    def _admit_microbatch(
        self,
        columns: tuple[tuple[Any, ...], ...],
    ) -> MicrobatchEngine:
        """把一个 source slice 原子接纳为独立 microbatch 状态机。"""

        engine = MicrobatchEngine(self.plan)
        bindings = {}
        controls = {}
        for port, values in zip(self.plan.source_ports, columns):
            block = self.store.put(values)
            bindings[port] = tuple(
                RowBinding(block, row) for row in range(len(values))
            )
            if port in self.plan.control_ports:
                controls[port] = values
        engine.admit_sources(bindings, controls=controls)
        engine.close_admission()
        return engine

    def _dispatch_ready(
        self,
        active: dict[int, _MicrobatchSlot],
        pending: dict[Any, _DispatchLease],
    ) -> None:
        """把 READY Grain 分配给空闲 actor；每批严格属于一个 microbatch。"""

        for call, actors in self._actors.items():
            for actor in actors:
                if actor.busy:
                    continue
                candidates = [
                    (priority, index, slot)
                    for index, slot in active.items()
                    if (priority := slot.engine.dispatch_priority(call)) is not None
                ]
                if not candidates:
                    break
                _, _, candidate = min(candidates, key=lambda item: item[:2])
                pool = self._pool(call)
                # 参考调度器采用即时、work-conserving 聚批：只要 actor
                # 空闲就发送当前可见 Grain，不伪装支持定时等待窗口。
                batch = candidate.engine.reserve_dispatch(
                    call,
                    max_size=pool.batch_size,
                    parent_bound=pool.batch_scope == "parent_bound",
                )
                grain_plans = tuple(
                    candidate.engine.grain_plan(grain)
                    for grain in batch.grains
                )
                layouts = self.plan.output_layouts_by_call[call]
                result_ref = actor.handle.execute.remote(grain_plans, layouts)
                actor.busy = True
                pending[result_ref] = _DispatchLease(
                    candidate.index,
                    actor,
                    batch,
                )
                counters = self._counters[call]
                counters.rpcs += 1
                counters.grains += len(batch.grains)
                counters.batch_sizes.append(len(batch.grains))

    # ── Typed failure classification and recovery handoff ───────────────

    def _handle_dispatch_failure(
        self,
        engine: MicrobatchEngine,
        lease: _DispatchLease,
        failure: DispatchFailure,
    ) -> None:
        """Map one typed Worker failure to an exhaustive recovery action."""

        if failure.kind is DispatchFailureKind.CONTRACT_ERROR:
            raise self._execution_error(engine, lease, failure)
        if failure.kind is not DispatchFailureKind.UDF_ERROR:
            raise AssertionError(
                f"unsupported DispatchFailureKind: {failure.kind!r}"
            )
        policy = self._pool(lease.actor.call).recovery
        action = policy.decide_udf(
            completed_retries=lease.batch.udf_retries,
            grain_count=len(lease.batch.grains),
        )
        if action is RecoveryAction.ABORT:
            raise self._execution_error(engine, lease, failure)
        self._counters[lease.actor.call].retries += engine.apply_udf_recovery(
            lease.batch,
            action,
            failure,
        )

    def _handle_infrastructure_failure(
        self,
        engine: MicrobatchEngine,
        lease: _DispatchLease,
        error: Exception,
    ) -> None:
        """Replace an untrusted actor and retry without data-failure fiction."""

        policy = self._pool(lease.actor.call).recovery
        accepted = engine.retry_infrastructure_dispatch(
            lease.batch,
            policy,
        )
        if not accepted:
            raise self._execution_error(engine, lease, error) from error
        self._replace_actor(lease.actor)
        self._counters[lease.actor.call].retries += len(lease.batch.grains)

    def _replace_actor(self, actor: _ActorSlot) -> None:
        """Discard one untrusted handle and install a fresh actor instance."""

        try:
            self.ray.kill(actor.handle, no_restart=True)
        except Exception:
            pass
        actor.handle = self._create_actor(actor.call)
        self._counters[actor.call].actor_instances += 1

    def _execution_error(
        self,
        engine: MicrobatchEngine,
        lease: _DispatchLease,
        failure: DispatchFailure | Exception,
    ) -> ExecutionError:
        """Join wire details with Call and generation context owned by driver."""

        call = lease.actor.call
        target = self.plan.call(call).udf.target
        name = self._udf_name(target)
        grains = ", ".join(
            f"{grain!r}@generation={engine.grain_snapshot(grain).generation}"
            for grain in lease.batch.grains
        )
        if isinstance(failure, DispatchFailure):
            detail = (
                f"{failure.kind.name}: {failure.error_type}: {failure.message}\n"
                f"worker traceback:\n{failure.traceback}"
            )
        else:
            detail = f"INFRA_FAILURE: {type(failure).__name__}: {failure}"
        return ExecutionError(
            f"Call {call.value} ({name}) dispatch failed for [{grains}]\n{detail}"
        )

    # ── Actor-pool lifecycle and small pure helpers ─────────────────────

    def _create_pool(self, call: CallRef) -> None:
        """只为 Call 创建 pool；Port 结构关系没有对应入口。"""

        replicas = self._pool(call).replicas
        for _ in range(replicas):
            self._actors[call].append(
                _ActorSlot(call, self._create_actor(call))
            )

    def _create_actor(self, call: CallRef) -> Any:
        """创建或替换一个持久 actor handle。"""

        spec = self.plan.call(call)
        actor_options = dict(self._pool(call).ray_options)
        actor_class = self._actor_class.options(**actor_options)
        handle = actor_class.remote(
            spec.udf.target,
            spec.udf.init_args,
            spec.udf.init_kwargs,
            self.plan.input_layouts_by_call[call],
        )
        return handle

    @staticmethod
    def _udf_name(target: Any) -> str:
        """返回稳定的人类可读 UDF 名称。"""

        return getattr(
            target,
            "__qualname__",
            getattr(target, "__name__", repr(target)),
        )

    def _pool(self, call: CallRef) -> ActorPoolSpec:
        """Return the one typed physical execution contract for a Call."""

        return self.plan.pool(call)

    @classmethod
    def _merge_outputs(cls, outputs: list[object]) -> object:
        """按 microbatch admission 顺序拼接同构输出树。"""

        first = outputs[0]
        if isinstance(first, list):
            list_outputs = cast(list[list[object]], outputs)
            return [item for output in list_outputs for item in output]
        if isinstance(first, tuple):
            tuple_outputs = cast(list[tuple[object, ...]], outputs)
            return tuple(
                cls._merge_outputs(
                    [output[index] for output in tuple_outputs]
                )
                for index in range(len(first))
            )
        raise RuntimeError("invalid materialized output tree")


__all__ = [
    "Executor",
]
