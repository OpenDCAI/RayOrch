"""Ray actor-pool executor with overlapping microbatch admission.

The logical program, semantic engine, and Worker ABI are Ray-free. This module
alone owns actor handles and pending RPC ObjectRefs. Actor capacity is shared
across microbatches, but a single RPC never mixes Grains from different
microbatches.
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
    """Mutable per-Call counters owned by one ``Executor.run``."""

    actor_instances: int = 0
    rpcs: int = 0
    grains: int = 0
    retries: int = 0
    batch_sizes: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _ActorSlot:
    """A driver-owned actor-capacity token."""

    call: CallRef
    handle: Any
    busy: bool = False


@dataclass(slots=True)
class _MicrobatchSlot:
    """One source microbatch and its sole semantic state machine."""

    index: int
    engine: MicrobatchEngine


@dataclass(frozen=True, slots=True)
class _PendingRpc:
    """One pending worker RPC and the state needed to finalize it exactly once."""

    microbatch_index: int
    actor: _ActorSlot
    dispatch_batch: DispatchBatch


class Executor:
    """Drive persistent per-Call actor pools across overlapping microbatches.

    The executor owns actor capacity, pending RPCs, and run-local counters.
    Logical propagation, entity lineage, and Grain lifecycle state remain in
    :class:`MicrobatchEngine` and :class:`DispatchState`.
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
        self._actors: dict[CallRef, list[_ActorSlot]] = {}
        self._counters: dict[CallRef, _CallCounters] = {}
        self._closed = False
        try:
            self._actor_class = ray.remote(_RayWorkerActor)
            # Register cleanup ownership before actor creation. If construction
            # fails partway through a pool, close() still sees every prior handle.
            for call in self.plan.calls:
                self._actors[call] = []
                self._create_pool(call)
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
        microbatch_size: int | None = None,
        max_active_microbatches: int = 1,
    ) -> RunResult:
        """Execute finite row-aligned sequences with bounded microbatch overlap."""

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

        # Actor pools persist across runs; mutable counters do not.
        self.store.clear_cache()
        self._counters = {
            call: _CallCounters(actor_instances=len(self._actors[call]))
            for call in self.plan.calls
        }
        metrics_by_microbatch: list[MicrobatchMetrics | None] = [None] * len(slices)
        active: dict[int, _MicrobatchSlot] = {}
        completed: dict[int, object] = {}
        pending_rpcs: dict[Any, _PendingRpc] = {}
        next_microbatch = 0
        execution_started = False
        high_watermark = 0
        started = time.perf_counter()

        # Event-loop invariants:
        # 1. active[index] uniquely owns that microbatch's Engine;
        # 2. every pending ObjectRef maps to exactly one _PendingRpc;
        # 3. a busy actor has one such RPC and is released in finally;
        # 4. materialization requires both no pending RPC and Engine complete.
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

                made_progress = self._dispatch_ready(active, pending_rpcs)

                # A microbatch cannot retire while one of its RPCs is pending.
                pending_microbatches = {
                    rpc.microbatch_index for rpc in pending_rpcs.values()
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
                        # Materialization copied final values into driver output.
                        # The cache only deduplicates block reads within one turn.
                        self.store.clear_cache()
                        # Drop bindings such as page-image ObjectRefs after output
                        # materialization; results retain immutable metrics only.
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
                if not pending_rpcs and next_microbatch < len(slices):
                    # Completed microbatches freed admission credit for the next
                    # source slice; the following turn can make progress.
                    continue
                if not pending_rpcs and made_progress:
                    # A cleanup-only reservation can publish SUPPRESSED facts
                    # and expose work for a Call whose actor loop already ran.
                    continue
                if not pending_rpcs:
                    summaries = ", ".join(
                        f"microbatch[{index}] {slot.engine.progress_summary()}"
                        for index, slot in sorted(active.items())
                    )
                    raise RuntimeError(f"RayOrch multigrain runtime deadlocked: {summaries}")

                ready, _ = self.ray.wait(list(pending_rpcs), num_returns=1)
                result_ref = ready[0]
                pending_rpc = pending_rpcs.pop(result_ref)
                engine = active[pending_rpc.microbatch_index].engine
                try:
                    result = self.ray.get(result_ref)
                except Exception as error:  # Ray surfaces actor failures at get().
                    self._handle_infrastructure_failure(engine, pending_rpc, error)
                else:
                    if isinstance(result, DispatchFailure):
                        self._handle_dispatch_failure(engine, pending_rpc, result)
                    else:
                        engine.commit_reports(pending_rpc.dispatch_batch, result)
                finally:
                    pending_rpc.actor.busy = False

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

    # ── Observation and immutable result snapshots ──────────────────────

    def _observe_workers(self) -> dict[CallRef, tuple[WorkerSnapshot, ...]]:
        """Best-effort physical diagnostics must not invalidate business output."""

        result: dict[CallRef, list[WorkerSnapshot | None]] = {
            call: [None] * len(actors)
            for call, actors in self._actors.items()
        }
        pending_observations = {}
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
                    pending_observations[reference] = (call, index)
        while pending_observations:
            ready, _ = self.ray.wait(list(pending_observations), num_returns=1)
            reference = ready[0]
            call, index = pending_observations.pop(reference)
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
        """Freeze counters in stable CallRef order for the public result."""

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

    def _admit_microbatch(
        self,
        columns: tuple[tuple[Any, ...], ...],
    ) -> MicrobatchEngine:
        """Admit one source slice into an independent microbatch engine."""

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
        pending_rpcs: dict[Any, _PendingRpc],
    ) -> bool:
        """Assign READY Grains to idle actors without mixing microbatches."""

        made_progress = False
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
                # Dispatch visible work immediately when an actor is idle. This
                # completion-driven path is what lets a downstream stage start
                # before the upstream stage or input domain has fully drained.
                batch = candidate.engine.reserve_dispatch(
                    call,
                    max_size=pool.batch_size,
                )
                if batch is None:
                    made_progress = True
                    continue
                invocations = tuple(
                    candidate.engine.grain_invocation(grain)
                    for grain in batch.grains
                )
                layouts = self.plan.output_layouts_by_call[call]
                result_ref = actor.handle.execute.remote(invocations, layouts)
                actor.busy = True
                pending_rpcs[result_ref] = _PendingRpc(
                    candidate.index,
                    actor,
                    batch,
                )
                counters = self._counters[call]
                counters.rpcs += 1
                counters.grains += len(batch.grains)
                counters.batch_sizes.append(len(batch.grains))
                made_progress = True
        return made_progress

    # ── Typed failure classification and recovery handoff ───────────────

    def _handle_dispatch_failure(
        self,
        engine: MicrobatchEngine,
        pending_rpc: _PendingRpc,
        failure: DispatchFailure,
    ) -> None:
        """Map one typed Worker failure to an exhaustive recovery action."""

        if failure.kind is DispatchFailureKind.CONTRACT_ERROR:
            raise self._execution_error(engine, pending_rpc, failure)
        if failure.kind is not DispatchFailureKind.UDF_ERROR:
            raise AssertionError(
                f"unsupported DispatchFailureKind: {failure.kind!r}"
            )
        policy = self._pool(pending_rpc.actor.call).recovery
        live = engine.live_recovery_batch(pending_rpc.dispatch_batch)
        if live is None:
            engine.suppress_barriered_batch(pending_rpc.dispatch_batch)
            return
        action = policy.decide_udf(
            completed_retries=live.udf_retries,
            grain_count=len(live.grains),
        )
        if action is RecoveryAction.ABORT:
            raise self._execution_error(engine, pending_rpc, failure)
        self._counters[pending_rpc.actor.call].retries += engine.apply_udf_recovery(
            pending_rpc.dispatch_batch,
            action,
            failure,
        )

    def _handle_infrastructure_failure(
        self,
        engine: MicrobatchEngine,
        pending_rpc: _PendingRpc,
        error: Exception,
    ) -> None:
        """Replace an untrusted actor and retry without data-failure fiction."""

        policy = self._pool(pending_rpc.actor.call).recovery
        retried = engine.retry_infrastructure_dispatch(
            pending_rpc.dispatch_batch,
            policy,
        )
        if retried is None:
            raise self._execution_error(engine, pending_rpc, error) from error
        self._replace_actor(pending_rpc.actor)
        self._counters[pending_rpc.actor.call].retries += retried

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
        pending_rpc: _PendingRpc,
        failure: DispatchFailure | Exception,
    ) -> ExecutionError:
        """Join wire details with Call and generation context owned by driver."""

        call = pending_rpc.actor.call
        target = self.plan.call(call).udf.target
        name = self._udf_name(target)
        grains = ", ".join(
            f"{grain!r}@generation={engine.grain_snapshot(grain).generation}"
            for grain in pending_rpc.dispatch_batch.grains
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
        """Create actors only for Calls; structural Ports have no workers."""

        replicas = self._pool(call).replicas
        for _ in range(replicas):
            self._actors[call].append(
                _ActorSlot(call, self._create_actor(call))
            )

    def _create_actor(self, call: CallRef) -> Any:
        """Create a persistent actor handle for one Call."""

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
        """Return a stable human-readable UDF name."""

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
        """Merge homogeneous output trees in microbatch admission order."""

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
