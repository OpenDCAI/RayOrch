"""Ray actor-pool executor with overlapping input batch admission.

The logical program, semantic engine, and Worker ABI are Ray-free. This module
alone owns actor handles and pending RPC ObjectRefs. Actor capacity is shared
across input batches and may be shared by several logical Calls, but one RPC
never mixes Calls or input batches.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ..api import Pipeline
from .._runtime.materialize import materialize_tree
from .._model import CallRef, PoolRef
from .._program.plan import CallDispatchSpec, CompiledProgram
from .._protocol import (
    DispatchFailure,
    DispatchFailureKind,
    RowBinding,
)
from ..errors import ExecutionError
from .._runtime import ExecutionMicrobatch, InputBatchEngine
from .ray_backend import _RayBlockStore, _RayWorkerActor
from ..result import CallMetrics, InputBatchMetrics, RunResult


# ── Driver-local mutable counters and physical ownership records ─────────────


@dataclass(slots=True)
class _CallCounters:
    """Mutable per-Call counters owned by one ``Executor.run``."""

    rpcs: int = 0
    grain_dispatches: int = 0
    grain_requeues: int = 0
    batch_sizes: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _ActorSlot:
    """A driver-owned actor-capacity token."""

    pool: PoolRef
    handle: Any
    busy: bool = False


@dataclass(slots=True)
class _InputBatchSlot:
    """One source input batch and its sole semantic state machine."""

    index: int
    engine: InputBatchEngine


@dataclass(frozen=True, slots=True)
class _PendingRpc:
    """One pending worker RPC and the state needed to finalize it exactly once."""

    input_batch_index: int
    call: CallRef
    actor: _ActorSlot
    execution_microbatch: ExecutionMicrobatch


class Executor:
    """Drive persistent actor pools across overlapping input batches.

    The executor owns actor capacity, pending RPCs, and run-local counters.
    Logical propagation, entity lineage, and Grain lifecycle state remain in
    :class:`InputBatchEngine` and :class:`DispatchState`.
    """

    def __init__(
        self,
        pipeline: Pipeline | CompiledProgram,
        *,
        address: str | None = None,
        ray_init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Compile the pipeline, connect to Ray, and create persistent actors."""

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
        self._actors: dict[PoolRef, list[_ActorSlot]] = {}
        self._calls_by_pool = dict(self.plan.calls_by_pool)
        self._pool_cursor: dict[PoolRef, int] = {
            pool: 0 for pool in self._calls_by_pool
        }
        self._counters: dict[CallRef, _CallCounters] = {}
        self._closed = False
        try:
            self._actor_class = ray.remote(_RayWorkerActor)
            # Register cleanup ownership before actor creation. If construction
            # fails partway through a pool, close() still sees every prior handle.
            for pool in self._calls_by_pool:
                self._actors[pool] = []
                self._create_pool(pool)
            # Exclude UDF and model initialization from run-time measurements.
            startup_refs = [
                actor.handle.ready.remote()
                for actors in self._actors.values()
                for actor in actors
            ]
            ray.get(startup_refs)
        except Exception:
            # Construction failures never hand the object to the caller.
            self.close()
            raise

    # ── Public run/close lifecycle ───────────────────────────────────────

    def run(
        self,
        *source_columns: Sequence[Any],
        input_batch_size: int | None = None,
        max_active_input_batches: int = 1,
    ) -> RunResult:
        """Execute finite row-aligned sequences with bounded input batch overlap.

        ``input_batch_size`` counts source rows (None uses the full input).
        ``max_active_input_batches`` limits overlapping input batch lifecycles.
        Each Call separately limits its execution microbatches via ``batch_size``.
        """

        if self._closed:
            raise RuntimeError("Executor is closed")

        columns = self._normalize_sources(source_columns)
        row_count = len(columns[0])
        if max_active_input_batches <= 0:
            raise ValueError("max_active_input_batches must be positive")
        if input_batch_size is None:
            input_batch_size = max(1, row_count)
        if input_batch_size <= 0:
            raise ValueError("input_batch_size must be positive")

        slices = [
            tuple(column[start : start + input_batch_size] for column in columns)
            for start in range(0, row_count, input_batch_size)
        ]
        if not slices:
            slices = [tuple(() for _ in columns)]

        # Actor pools persist across runs; mutable counters do not.
        self.store.clear_cache()
        self._pool_cursor = {pool: 0 for pool in self._calls_by_pool}
        self._counters = {call: _CallCounters() for call in self.plan.calls}
        self._actor_instances_by_pool = {
            pool: len(actors)
            for pool, actors in self._actors.items()
        }
        metrics_by_input_batch: list[InputBatchMetrics | None] = [None] * len(slices)
        active: dict[int, _InputBatchSlot] = {}
        completed: dict[int, object] = {}
        pending_rpcs: dict[Any, _PendingRpc] = {}
        next_input_batch = 0
        execution_started = False
        high_watermark = 0
        started = time.perf_counter()

        # Event-loop invariants:
        # 1. active[index] uniquely owns that input batch's Engine;
        # 2. every pending ObjectRef maps to exactly one _PendingRpc;
        # 3. a busy actor has one such RPC and is released in finally;
        # 4. materialization requires both no pending RPC and Engine complete.
        try:
            while len(completed) < len(slices):
                while (
                    next_input_batch < len(slices)
                    and len(active) < max_active_input_batches
                ):
                    execution_started = True
                    engine = self._admit_input_batch(slices[next_input_batch])
                    active[next_input_batch] = _InputBatchSlot(
                        next_input_batch,
                        engine,
                    )
                    next_input_batch += 1
                    high_watermark = max(high_watermark, len(active))

                made_progress = self._dispatch_ready(active, pending_rpcs)

                # An input batch cannot retire while one of its RPCs is pending.
                pending_input_batches = {
                    rpc.input_batch_index for rpc in pending_rpcs.values()
                }
                for index, slot in tuple(active.items()):
                    if (
                        index not in pending_input_batches
                        and slot.engine.is_complete()
                    ):
                        completed[index] = materialize_tree(
                            self.plan,
                            slot.engine,
                            self.store,
                        )
                        # Materialization copied final values into driver output.
                        # The cache only deduplicates block reads within one turn.
                        self.store.clear_cache()
                        # Drop bindings such as page-image ObjectRefs after output
                        # materialization; results retain immutable metrics only.
                        released = slot.engine.release_values()
                        metrics_by_input_batch[index] = InputBatchMetrics(
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
                if (
                    not pending_rpcs
                    and next_input_batch < len(slices)
                    and len(active) < max_active_input_batches
                ):
                    # Retirement freed admission credit for a remaining source
                    # slice, so the following turn can make progress.
                    continue
                if not pending_rpcs and made_progress:
                    # A cleanup-only reservation can publish SUPPRESSED facts
                    # and expose work for a Call whose actor loop already ran.
                    continue
                if not pending_rpcs:
                    summaries = ", ".join(
                        f"input_batch[{index}] {slot.engine.progress_summary()}"
                        for index, slot in sorted(active.items())
                    )
                    raise RuntimeError(f"RayOrch runtime deadlocked: {summaries}")

                ready, _ = self.ray.wait(list(pending_rpcs), num_returns=1)
                result_ref = ready[0]
                pending_rpc = pending_rpcs.pop(result_ref)
                engine = active[pending_rpc.input_batch_index].engine
                try:
                    result = self.ray.get(result_ref)
                except Exception as error:  # Ray surfaces actor failures at get().
                    self._handle_infrastructure_failure(engine, pending_rpc, error)
                else:
                    if isinstance(result, DispatchFailure):
                        self._handle_dispatch_failure(engine, pending_rpc, result)
                    else:
                        engine.commit_reports(pending_rpc.execution_microbatch, result)
                finally:
                    pending_rpc.actor.busy = False

            elapsed_s = time.perf_counter() - started
            calls = self._freeze_call_metrics()
            if any(metrics is None for metrics in metrics_by_input_batch):
                raise AssertionError("completed run lost an input batch metrics snapshot")
            return RunResult(
                self._merge_outputs([completed[index] for index in range(len(slices))]),
                elapsed_s,
                calls,
                cast(tuple[InputBatchMetrics, ...], tuple(metrics_by_input_batch)),
                high_watermark,
                actor_count=sum(self._actor_instances_by_pool.values()),
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
        """Release actors and shut down only a Ray runtime owned by this executor."""

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

    # ── Immutable result snapshots ──────────────────────

    def _freeze_call_metrics(self) -> tuple[CallMetrics, ...]:
        """Freeze counters in stable CallRef order for the public result."""

        snapshots = []
        for call in sorted(self.plan.calls, key=lambda ref: ref.value):
            counters = self._counters[call]
            target = self.plan.call(call).udf.target
            pool = self._dispatch(call).pool
            snapshots.append(
                CallMetrics(
                    call_index=call.value,
                    udf_name=self._udf_name(target),
                    actor_instances=self._actor_instances_by_pool[pool],
                    rpcs=counters.rpcs,
                    grain_dispatches=counters.grain_dispatches,
                    grain_requeues=counters.grain_requeues,
                    batch_sizes=tuple(counters.batch_sizes),
                )
            )
        return tuple(snapshots)

    def __enter__(self):
        """Enter a context-managed executor lifetime."""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Release actors when leaving the context."""

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

    def _admit_input_batch(
        self,
        columns: tuple[tuple[Any, ...], ...],
    ) -> InputBatchEngine:
        """Admit one source slice into an independent input batch engine."""

        engine = InputBatchEngine(self.plan)
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
        active: dict[int, _InputBatchSlot],
        pending_rpcs: dict[Any, _PendingRpc],
    ) -> bool:
        """Assign READY Grains to idle actors without mixing input batches."""

        made_progress = False
        for pool, actors in self._actors.items():
            for actor in actors:
                if actor.busy:
                    continue
                selected = self._select_pool_work(pool, active)
                if selected is None:
                    break
                call, candidate = selected
                dispatch = self._dispatch(call)
                # Dispatch visible work immediately when an actor is idle. This
                # completion-driven path is what lets a downstream stage start
                # before the upstream stage or input domain has fully drained.
                batch = candidate.engine.reserve_dispatch(
                    call,
                    max_size=dispatch.batch_size,
                )
                if batch is None:
                    made_progress = True
                    continue
                invocations = tuple(
                    candidate.engine.grain_invocation(grain)
                    for grain in batch.grains
                )
                layouts = self.plan.output_layouts_by_call[call]
                result_ref = actor.handle.execute.remote(
                    invocations,
                    layouts,
                    self.plan.input_layouts_by_call[call],
                )
                actor.busy = True
                pending_rpcs[result_ref] = _PendingRpc(
                    candidate.index,
                    call,
                    actor,
                    batch,
                )
                counters = self._counters[call]
                counters.rpcs += 1
                counters.grain_dispatches += len(batch.grains)
                counters.batch_sizes.append(len(batch.grains))
                made_progress = True
        return made_progress

    def _select_pool_work(
        self,
        pool: PoolRef,
        active: dict[int, _InputBatchSlot],
    ) -> tuple[CallRef, _InputBatchSlot] | None:
        """Choose one Call fairly while retaining queue/recovery priority.

        Calls sharing a pool are considered in rotating order. Queue priority
        remains immediate retry, fresh READY work, then deferred recovery.
        Input batches of the selected Call retain admission order.
        """

        calls = self._calls_by_pool[pool]
        cursor = self._pool_cursor[pool]
        for offset in range(len(calls)):
            call = calls[(cursor + offset) % len(calls)]
            candidates: list[tuple[int, int, _InputBatchSlot]] = []
            for index, slot in active.items():
                priority = slot.engine.dispatch_priority(call)
                if priority is not None:
                    candidates.append((priority, index, slot))
            if candidates:
                _, _, slot = min(
                    candidates,
                    key=lambda candidate: candidate[:2],
                )
                self._pool_cursor[pool] = (cursor + offset + 1) % len(calls)
                return call, slot
        return None

    # ── Typed failure classification and recovery handoff ───────────────

    def _handle_dispatch_failure(
        self,
        engine: InputBatchEngine,
        pending_rpc: _PendingRpc,
        failure: DispatchFailure,
    ) -> None:
        """Classify a Worker failure and delegate UDF recovery to its Engine."""

        if failure.kind is DispatchFailureKind.CONTRACT_ERROR:
            raise self._execution_error(engine, pending_rpc, failure)
        if failure.kind is not DispatchFailureKind.UDF_ERROR:
            raise AssertionError(
                f"unsupported DispatchFailureKind: {failure.kind!r}"
            )
        policy = self._dispatch(pending_rpc.call).recovery
        requeued = engine.apply_udf_recovery(
            pending_rpc.execution_microbatch,
            policy,
            failure,
        )
        if requeued is None:
            raise self._execution_error(engine, pending_rpc, failure)
        self._counters[pending_rpc.call].grain_requeues += requeued

    def _handle_infrastructure_failure(
        self,
        engine: InputBatchEngine,
        pending_rpc: _PendingRpc,
        error: Exception,
    ) -> None:
        """Replace an untrusted actor and retry without data-failure fiction."""

        policy = self._dispatch(pending_rpc.call).recovery
        retried = engine.retry_infrastructure_dispatch(
            pending_rpc.execution_microbatch,
            policy,
        )
        if retried is None:
            raise self._execution_error(engine, pending_rpc, error) from error
        self._replace_actor(pending_rpc.actor)
        self._counters[pending_rpc.call].grain_requeues += retried

    def _replace_actor(self, actor: _ActorSlot) -> None:
        """Discard one untrusted handle and install a fresh actor instance."""

        try:
            self.ray.kill(actor.handle, no_restart=True)
        except Exception:
            pass
        replacement = self._create_actor(actor.pool)
        try:
            self.ray.get(replacement.ready.remote())
        except BaseException:
            try:
                self.ray.kill(replacement, no_restart=True)
            except Exception:
                pass
            raise
        actor.handle = replacement
        self._actor_instances_by_pool[actor.pool] += 1

    def _execution_error(
        self,
        engine: InputBatchEngine,
        pending_rpc: _PendingRpc,
        failure: DispatchFailure | Exception,
    ) -> ExecutionError:
        """Join wire details with Call and generation context owned by driver."""

        call = pending_rpc.call
        target = self.plan.call(call).udf.target
        name = self._udf_name(target)
        grains = ", ".join(
            f"{grain!r}@generation={engine.grain_snapshot(grain).generation}"
            for grain in pending_rpc.execution_microbatch.grains
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

    def _create_pool(self, pool: PoolRef) -> None:
        """Create one actor set for a physical pool."""

        for _ in range(self.plan.actor_pools[pool].replicas):
            self._actors[pool].append(
                _ActorSlot(pool, self._create_actor(pool))
            )

    def _create_actor(self, pool: PoolRef) -> Any:
        """Create one persistent actor replica for a physical pool."""

        spec = self.plan.actor_pools[pool]
        actor_options = dict(spec.ray_options)
        actor_class = self._actor_class.options(**actor_options)
        handle = actor_class.remote(
            spec.udf.target,
            spec.udf.init_args,
            spec.udf.init_kwargs,
        )
        return handle

    @staticmethod
    def _udf_name(target: Any) -> str:
        """Return a stable human-readable UDF name."""

        return getattr(
            target,
            "__qualname__",
            getattr(target, "__name__", repr(target)),
        )

    def _dispatch(self, call: CallRef) -> CallDispatchSpec:
        """Return the logical scheduling contract for a Call."""

        return self.plan.dispatch(call)

    @classmethod
    def _merge_outputs(cls, outputs: list[object]) -> object:
        """Merge homogeneous output trees in input batch admission order."""

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
