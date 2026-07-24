"""Single-process Phase 2 arena, dispatch, commit, and retry semantics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping

from .api import CompiledPipeline, ExecutionError, Pipeline, Port
from .grain import (
    AttemptToken,
    ConsumerIndex,
    Emission,
    ExpandOrigin,
    ExpandOriginIndex,
    Failed,
    FiberState,
    GrainFailure,
    GrainId,
    GrainPhase,
    GrainRecord,
    GrainTable,
    ItemRef,
    PortId,
    PortIndex,
    ProducerIndex,
    Success,
    Suppressed,
    ValueIndex,
    expand_entity,
    expand_grain_id,
    RoleItems,
)
from .graph import (
    BindingReceipt,
    CompiledGraph,
    PlanAction,
    PlanDecision,
    PlannerContractError,
    Primitive,
    ReceiptState,
    admit_source,
    ensure_decision,
    failed_outcome,
    plan_expand,
    plan_filter,
    plan_map,
    plan_reduce,
    plan_relate_bounded,
)
from .metrics import DispatchTimeline, percentile
from .worker import (
    BatchManifest,
    DispatchEntry,
    DispatchErrorReport,
    DispatchPlan,
    NormalizedBatchOutput,
    RowTake,
    WorkerContractError,
    build_batch_manifest,
    get_ray_worker_class,
)


class ArenaAbort(ExecutionError):
    """A run-control failure atomically aborts one microbatch arena."""


class ArenaState(Enum):
    RUNNING = "running"
    DELIVERED = "delivered"
    ABORTED = "aborted"
    RECLAIMED = "reclaimed"


class CommitStatus(Enum):
    ACCEPTED = "accepted"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class ArenaLimits:
    max_grains: int = 10_000
    max_pending_dispatches: int = 64
    max_fanout_per_grain: int = 10_000
    max_relation_cardinality: int = 10_000
    max_infra_retries: int = 1

    def __post_init__(self) -> None:
        if (
            self.max_grains <= 0
            or self.max_pending_dispatches <= 0
            or self.max_fanout_per_grain < 0
            or self.max_relation_cardinality < 0
            or self.max_infra_retries < 0
        ):
            raise ValueError("arena limits must be positive/non-negative")


@dataclass(frozen=True, slots=True)
class LocalValue:
    block: int
    row: int


@dataclass(frozen=True, slots=True)
class BlockSlice:
    block: Any
    row: int


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    grain: GrainId
    item: ItemRef
    outcome: Success | Failed


@dataclass(frozen=True, slots=True)
class FailureSnapshot:
    grain: GrainId
    failure: GrainFailure


@dataclass(frozen=True, slots=True)
class IsolationContext:
    root_dispatch: int
    depth: int


@dataclass(slots=True)
class DispatchRuntime:
    plan: DispatchPlan
    input_blocks: tuple[int, ...]
    pending_handle: Any = None
    output_blocks: tuple[int, ...] = ()
    isolation: IsolationContext | None = None
    flush_reason: str = "full"


@dataclass(frozen=True, slots=True)
class RayPending:
    arena: "Arena"
    plan: DispatchPlan
    node: int
    actor_index: int
    manifest_ref: Any
    output_refs: tuple[Any, ...]
    submitted_at: float


@dataclass(frozen=True, slots=True)
class RunResult:
    outputs: tuple[Any, ...] = ()
    failures: tuple[FailureSnapshot, ...] = ()
    sources: tuple[SourceSnapshot, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    timeline: tuple[DispatchTimeline, ...] = ()

    def get(self) -> tuple[Any, ...]:
        """Resolve final Ray BlockSlices; local values pass through unchanged."""

        slices = [
            value for value in self.outputs if isinstance(value, BlockSlice)
        ]
        if not slices:
            return self.outputs
        import ray

        blocks: dict[Any, tuple[Any, ...]] = {}
        resolved = []
        for value in self.outputs:
            if not isinstance(value, BlockSlice):
                resolved.append(value)
                continue
            if value.block not in blocks:
                blocks[value.block] = ray.get(value.block)
            resolved.append(blocks[value.block][value.row])
        return tuple(resolved)


@dataclass(frozen=True, slots=True)
class CommitDelta:
    blocks: tuple[tuple[int, Any], ...]
    values: tuple[tuple[ItemRef, LocalValue], ...]
    outcomes: tuple[tuple[GrainRecord, AttemptToken, Success], ...]
    origins: tuple[tuple[Any, ExpandOrigin], ...]


class SourcePositionAllocator:
    """Run-scoped per-source-port logical ordinal allocator."""

    def __init__(self) -> None:
        self._next: dict[PortId, int] = {}

    def allocate(self, source_port: PortId) -> int:
        position = self._next.get(source_port, 0)
        self._next[source_port] = position + 1
        return position

    def peek(self, source_port: PortId) -> int:
        return self._next.get(source_port, 0)


class Arena:
    """Single-threaded semantic runtime for one bounded microbatch."""

    def __init__(
        self,
        arena_id: int,
        graph: CompiledGraph,
        run_salt: bytes,
        *,
        limits: ArenaLimits = ArenaLimits(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.id = arena_id
        self.graph = graph
        self.run_salt = run_salt
        self.limits = limits
        self._clock = clock
        self.state = ArenaState.RUNNING
        self.abort_reason: str | None = None

        self.grains = GrainTable()
        self.producers = ProducerIndex()
        self.consumers = ConsumerIndex()
        self.ports = PortIndex()
        self.values = ValueIndex()
        self.expand_origins = ExpandOriginIndex()

        self._blocks: dict[int, Any] = {}
        self._next_block = 0
        self._blocks_high_watermark = 0
        self._next_dispatch = 0
        self._dispatches: dict[int, DispatchRuntime] = {}
        self._ready: dict[int, list[GrainId]] = {}
        self._ready_set: dict[int, set[GrainId]] = {}
        self._tail_wait_started_at: dict[int, float | None] = {}
        self._forced: dict[
            int,
            list[tuple[tuple[GrainId, ...], IsolationContext]],
        ] = {}
        self._sources: list[SourceSnapshot] = []
        self._source_admitted_at: dict[GrainId, float] = {}
        self._grain_completed_at: dict[GrainId, float] = {}
        self._dispatch_count = 0
        self._dispatched_grains = 0
        self._tail_or_isolation_dispatches = 0
        self._dispatch_capacity = 0
        self._ready_queue_high_watermark = 0
        self._flush_reason_counts = {
            "full": 0,
            "timeout": 0,
            "port_sealed": 0,
            "arena_drain": 0,
            "isolation": 0,
        }

    def _require_running(self) -> None:
        if self.state is not ArenaState.RUNNING:
            raise ArenaAbort(f"arena is not running: {self.state.value}")

    def _abort(self, reason: str) -> None:
        self.abort_reason = reason
        self.state = ArenaState.ABORTED
        self._reclaim()
        raise ArenaAbort(reason)

    def _reclaim(self) -> None:
        self.grains = GrainTable()
        self.producers = ProducerIndex()
        self.consumers = ConsumerIndex()
        self.ports = PortIndex()
        self.values = ValueIndex()
        self.expand_origins = ExpandOriginIndex()
        self._blocks.clear()
        self._dispatches.clear()
        self._ready.clear()
        self._ready_set.clear()
        self._tail_wait_started_at.clear()
        self._forced.clear()
        self._sources.clear()
        self._source_admitted_at.clear()
        self._grain_completed_at.clear()
        self.state = ArenaState.RECLAIMED

    def _check_record_indexes(self, record: GrainRecord) -> None:
        for item in record.output_slots:
            existing = self.producers.get(item)
            if existing is not None and existing != record.id:
                self._abort("expected output slot has a conflicting producer")

    def ensure_plans(
        self,
        decisions: tuple[PlanDecision, ...],
    ) -> tuple[GrainRecord, ...]:
        """Preflight all insertions, then publish them as one semantic step."""

        self._require_running()
        preview = GrainTable(self.grains.values())
        try:
            for decision in decisions:
                ensure_decision(preview, decision)
        except PlannerContractError as error:
            self._abort(str(error))
        if len(preview) > self.limits.max_grains:
            self._abort("max_grains_per_arena exceeded")

        for decision in decisions:
            if decision.grain is not None:
                self._check_record_indexes(decision.grain)

        records: list[GrainRecord] = []
        try:
            for decision in decisions:
                record = ensure_decision(self.grains, decision)
                if record is None:
                    continue
                self.producers.register(record)
                self.consumers.register(record)
                records.append(record)
                if record.phase is GrainPhase.SEALED:
                    self._grain_completed_at.setdefault(
                        record.id,
                        self._clock(),
                    )
                if (
                    decision.action is PlanAction.ENSURE_EXECUTABLE
                    and record.phase is GrainPhase.READY
                ):
                    self.enqueue_ready(record)
        except Exception as error:
            self._abort(f"grain insertion failed: {error}")
        return tuple(records)

    def admit_source(
        self,
        record: GrainRecord,
        *,
        value: Any = None,
    ) -> None:
        self._require_running()
        if len(self.grains) + (self.grains.get(record.id) is None) > (
            self.limits.max_grains
        ):
            self._abort("max_grains_per_arena exceeded during source admission")
        self._check_record_indexes(record)
        try:
            inserted = self.grains.add_terminal(record)
            self.producers.register(inserted)
            self.consumers.register(inserted)
            self.ports.register(inserted.outcome)
            if isinstance(inserted.outcome, Success):
                emission = inserted.outcome.emissions_by_port[0][0]
                block = self._allocate_block((value,))
                self.values.put(emission.item, LocalValue(block, 0))
            self._sources.append(
                SourceSnapshot(
                    inserted.id,
                    inserted.output_slots[0],
                    inserted.outcome,
                )
            )
            now = self._clock()
            self._source_admitted_at[inserted.id] = now
            self._grain_completed_at[inserted.id] = now
        except Exception as error:
            self._abort(f"source admission failed: {error}")

    def admit_source_batch(
        self,
        records: tuple[GrainRecord, ...],
        block_handle: Any,
    ) -> None:
        """Admit a coarse external source block without reading its values."""

        self._require_running()
        new_count = sum(
            self.grains.get(record.id) is None for record in records
        )
        if len(self.grains) + new_count > self.limits.max_grains:
            self._abort("max_grains_per_arena exceeded during source admission")
        for record in records:
            self._check_record_indexes(record)
            if not isinstance(record.outcome, Success):
                self._abort("external source block requires successful sources")
        block = self._allocate_block(block_handle)
        try:
            for row, record in enumerate(records):
                inserted = self.grains.add_terminal(record)
                self.producers.register(inserted)
                self.consumers.register(inserted)
                self.ports.register(inserted.outcome)
                emission = inserted.outcome.emissions_by_port[0][0]
                self.values.put(emission.item, LocalValue(block, row))
                self._sources.append(
                    SourceSnapshot(
                        inserted.id,
                        inserted.output_slots[0],
                        inserted.outcome,
                    )
                )
                now = self._clock()
                self._source_admitted_at[inserted.id] = now
                self._grain_completed_at[inserted.id] = now
        except Exception as error:
            self._abort(f"source batch admission failed: {error}")

    def _allocate_block(self, values: Any) -> int:
        block = self._next_block
        self._next_block += 1
        self._blocks[block] = values
        self._blocks_high_watermark = max(
            self._blocks_high_watermark,
            len(self._blocks),
        )
        return block

    def enqueue_ready(self, record: GrainRecord) -> None:
        if record.phase is not GrainPhase.READY:
            return
        ready_set = self._ready_set.setdefault(record.node, set())
        if record.id in ready_set:
            return
        queue = self._ready.setdefault(record.node, [])
        was_empty = not queue
        ready_set.add(record.id)
        queue.append(record.id)
        node = self.graph.node(record.node)
        assert node.execution is not None
        if len(queue) >= node.execution.batch_size:
            self._tail_wait_started_at[record.node] = None
        elif was_empty:
            self._tail_wait_started_at[record.node] = self._clock()
        self._ready_queue_high_watermark = max(
            self._ready_queue_high_watermark,
            len(queue),
        )

    def _force_group(
        self,
        node: int,
        grains: tuple[GrainId, ...],
        context: IsolationContext,
    ) -> None:
        self._forced.setdefault(node, []).append((grains, context))

    def _normal_batch_candidates(
        self,
        node: Any,
        queue: list[GrainId],
    ) -> tuple[GrainId, ...]:
        if node.execution.batch_scope == "elastic":
            return tuple(queue)
        groups: dict[Any, list[GrainId]] = {}
        order: list[Any] = []
        for grain_id in queue:
            record = self.grains.get(grain_id)
            if record is None or not record.inputs or not record.inputs[0].items:
                continue
            driving = record.inputs[0].items[0]
            origin = self.expand_origins.get(driving.entity)
            group = origin.anchor if origin is not None else driving.entity
            if group not in groups:
                groups[group] = []
                order.append(group)
            groups[group].append(grain_id)
        for group in order:
            if len(groups[group]) >= node.execution.batch_size:
                return tuple(groups[group])
        return tuple(groups[order[0]]) if order else ()

    def reserve_dispatch(
        self,
        node_id: int,
        *,
        admission_closed: bool = True,
        draining: bool = False,
        now: float | None = None,
    ) -> DispatchPlan | None:
        self._require_running()
        if len(self._dispatches) >= self.limits.max_pending_dispatches:
            return None
        now = self._clock() if now is None else now
        node = self.graph.node(node_id)
        if node.kind is Primitive.SOURCE:
            self._abort(f"{node.kind.value} is not dispatchable in Phase 2")
        assert node.execution is not None

        isolation: IsolationContext | None = None
        flush_reason: str
        selected: list[GrainRecord] = []
        forced = self._forced.get(node_id)
        if forced:
            flush_reason = "isolation"
            grain_ids, isolation = forced.pop(0)
            for grain_id in grain_ids:
                record = self.grains.get(grain_id)
                if record is not None and record.phase is GrainPhase.READY:
                    selected.append(record)
            if len(selected) != len(grain_ids):
                self._abort("isolation group is not entirely READY")
        else:
            queue = self._ready.setdefault(node_id, [])
            ready_set = self._ready_set.setdefault(node_id, set())
            if not queue:
                return None
            candidates = self._normal_batch_candidates(node, queue)
            candidate_set = set(candidates)
            if len(candidates) >= node.execution.batch_size:
                flush_reason = "full"
            elif admission_closed:
                flush_reason = "port_sealed"
            elif draining:
                flush_reason = "arena_drain"
            else:
                wait_started = self._tail_wait_started_at.get(node_id)
                if wait_started is None:
                    wait_started = now
                    self._tail_wait_started_at[node_id] = wait_started
                wait_seconds = node.execution.max_batch_wait_ms / 1000.0
                if wait_seconds > 0 and now - wait_started < wait_seconds:
                    return None
                flush_reason = "timeout"
            remaining: list[GrainId] = []
            for grain_id in queue:
                if (
                    grain_id not in candidate_set
                    or len(selected) >= node.execution.batch_size
                ):
                    remaining.append(grain_id)
                    continue
                ready_set.discard(grain_id)
                record = self.grains.get(grain_id)
                if record is not None and record.phase is GrainPhase.READY:
                    selected.append(record)
            queue[:] = remaining
            if not queue:
                self._tail_wait_started_at[node_id] = None
            elif len(queue) >= node.execution.batch_size:
                self._tail_wait_started_at[node_id] = None
            else:
                self._tail_wait_started_at[node_id] = now
        if not selected:
            return None

        input_blocks: list[int] = []
        block_slots: dict[int, int] = {}
        role_takes_by_record: list[tuple[tuple[RowTake, ...], ...]] = []
        for record in selected:
            role_takes: list[tuple[RowTake, ...]] = []
            for role in record.inputs:
                takes: list[RowTake] = []
                for item in role.items:
                    if not self.values.contains(item):
                        self._abort(
                            f"READY grain input has no value location: {item}"
                        )
                    location = self.values.get(item)
                    if not isinstance(location, LocalValue):
                        self._abort("Phase 2 requires LocalValue locations")
                    if location.block not in block_slots:
                        block_slots[location.block] = len(input_blocks)
                        input_blocks.append(location.block)
                    takes.append(
                        RowTake(block_slots[location.block], location.row)
                    )
                role_takes.append(tuple(takes))
            role_takes_by_record.append(tuple(role_takes))

        dispatch_id = self._next_dispatch
        self._next_dispatch += 1
        entries: list[DispatchEntry] = []
        for record, role_takes in zip(selected, role_takes_by_record):
            token = record.reserve(self.id, dispatch_id)
            entries.append(DispatchEntry(token, role_takes))
        plan = DispatchPlan(dispatch_id, node_id, tuple(entries))
        self._dispatches[dispatch_id] = DispatchRuntime(
            plan=plan,
            input_blocks=tuple(input_blocks),
            isolation=isolation,
            flush_reason=flush_reason,
        )
        self._dispatch_count += 1
        self._dispatched_grains += len(entries)
        self._dispatch_capacity += node.execution.batch_size
        self._flush_reason_counts[flush_reason] += 1
        if len(entries) < node.execution.batch_size or isolation is not None:
            self._tail_or_isolation_dispatches += 1
        return plan

    def input_values(self, plan: DispatchPlan) -> tuple[tuple[Any, ...], ...]:
        runtime = self._dispatches.get(plan.id)
        if runtime is None or runtime.plan != plan:
            raise ArenaAbort("dispatch is not pending")
        return tuple(self._blocks[block] for block in runtime.input_blocks)

    def ready_count(self, node_id: int) -> int:
        return len(self._ready.get(node_id, ()))

    def batch_wait_remaining(
        self,
        node_id: int,
        *,
        now: float | None = None,
    ) -> float | None:
        if self._forced.get(node_id):
            return 0.0
        queue = self._ready.get(node_id, ())
        if not queue:
            return None
        node = self.graph.node(node_id)
        assert node.execution is not None
        candidates = self._normal_batch_candidates(node, list(queue))
        if len(candidates) >= node.execution.batch_size:
            return 0.0
        now = self._clock() if now is None else now
        started = self._tail_wait_started_at.get(node_id)
        if started is None:
            started = now
            self._tail_wait_started_at[node_id] = started
        deadline = started + node.execution.max_batch_wait_ms / 1000.0
        return max(0.0, deadline - now)

    def dispatch_flush_reason(self, plan: DispatchPlan) -> str:
        runtime = self._dispatches.get(plan.id)
        if runtime is None or runtime.plan != plan:
            raise ArenaAbort("dispatch is not pending")
        return runtime.flush_reason

    def input_block_handles(self, plan: DispatchPlan) -> tuple[Any, ...]:
        runtime = self._dispatches.get(plan.id)
        if runtime is None or runtime.plan != plan:
            raise ArenaAbort("dispatch is not pending")
        return tuple(self._blocks[block] for block in runtime.input_blocks)

    def _currentness(self, plan: DispatchPlan) -> CommitStatus:
        current: list[bool] = []
        for entry in plan.entries:
            record = self.grains.get(entry.token.grain)
            current.append(record is not None and record.active == entry.token)
        if not any(current):
            return CommitStatus.STALE
        if not all(current):
            self._abort("dispatch has mixed current/stale attempt tokens")
        return CommitStatus.ACCEPTED

    def _finish_dispatch(self, plan: DispatchPlan) -> None:
        runtime = self._dispatches.get(plan.id)
        if runtime is not None and runtime.plan == plan:
            self._dispatches.pop(plan.id, None)

    def commit_normalized(
        self,
        plan: DispatchPlan,
        outputs_by_port: NormalizedBatchOutput,
    ) -> CommitStatus:
        try:
            manifest, columns = build_batch_manifest(plan, outputs_by_port)
        except WorkerContractError as error:
            self._abort(str(error))
        return self.commit_manifest(plan, manifest, columns)

    def commit_manifest(
        self,
        plan: DispatchPlan,
        manifest: BatchManifest,
        columns: tuple[tuple[Any, ...], ...],
    ) -> CommitStatus:
        self._require_running()
        currentness = self._currentness(plan)
        if currentness is CommitStatus.STALE:
            self._finish_dispatch(plan)
            return currentness
        runtime = self._dispatches.get(plan.id)
        if runtime is None or runtime.plan != plan:
            self._abort("current dispatch has no runtime authority")
        try:
            cardinalities = self._validate_manifest(
                plan,
                manifest,
                tuple(len(column) for column in columns),
            )
            delta = self._prepare_commit_delta(
                plan,
                manifest,
                columns,
                cardinalities,
            )
        except (ValueError, RuntimeError) as error:
            self._abort(f"manifest validation failed: {error}")
        self._apply_commit_delta(plan, runtime, delta)
        return CommitStatus.ACCEPTED

    def commit_external_manifest(
        self,
        plan: DispatchPlan,
        manifest: BatchManifest,
        output_blocks: tuple[Any, ...],
    ) -> CommitStatus:
        """Commit opaque Ray-style output blocks after reading only a manifest."""

        self._require_running()
        currentness = self._currentness(plan)
        if currentness is CommitStatus.STALE:
            self._finish_dispatch(plan)
            return currentness
        runtime = self._dispatches.get(plan.id)
        if runtime is None or runtime.plan != plan:
            self._abort("current dispatch has no runtime authority")
        if len(output_blocks) != len(manifest.column_lengths):
            self._abort("output block arity does not match manifest")
        try:
            cardinalities = self._validate_manifest(
                plan,
                manifest,
                manifest.column_lengths,
            )
            delta = self._prepare_commit_delta(
                plan,
                manifest,
                output_blocks,
                cardinalities,
            )
        except (ValueError, RuntimeError) as error:
            self._abort(f"manifest validation failed: {error}")
        self._apply_commit_delta(plan, runtime, delta)
        return CommitStatus.ACCEPTED

    def _validate_manifest(
        self,
        plan: DispatchPlan,
        manifest: BatchManifest,
        column_lengths: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        node = self.graph.node(plan.node)
        output_arity = len(node.output_ports)
        if manifest.dispatch != plan.id:
            raise ValueError("manifest dispatch id mismatch")
        if len(manifest.acks) != len(plan.entries):
            raise ValueError("manifest must ack every dispatch entry")
        if len(column_lengths) != output_arity:
            raise ValueError("output column arity mismatch")
        if manifest.column_lengths != column_lengths:
            raise ValueError("manifest column lengths mismatch")
        if any(len(ack.spans_by_port) != output_arity for ack in manifest.acks):
            raise ValueError("ack output arity mismatch")
        if tuple(ack.token for ack in manifest.acks) != tuple(
            entry.token for entry in plan.entries
        ):
            raise ValueError("manifest tokens do not match DispatchPlan")

        for port_index, column_length in enumerate(column_lengths):
            cursor = 0
            for ack in manifest.acks:
                span = ack.spans_by_port[port_index]
                if span.start != cursor or span.stop < span.start:
                    raise ValueError("output spans must form a contiguous partition")
                cursor = span.stop
            if cursor != column_length:
                raise ValueError("output spans do not cover their column")

        cardinalities: list[tuple[int, ...]] = []
        for ack in manifest.acks:
            counts = tuple(
                span.stop - span.start for span in ack.spans_by_port
            )
            if node.kind in {
                Primitive.MAP,
                Primitive.REDUCE,
                Primitive.RELATE,
            }:
                if any(count != 1 for count in counts):
                    raise ValueError(f"{node.kind.value} must emit one row per port")
            elif node.kind is Primitive.FILTER:
                if len(set(counts)) != 1 or counts[0] not in {0, 1}:
                    raise ValueError("Filter ports must all emit zero or one row")
            elif node.kind is Primitive.EXPAND:
                if len(set(counts)) != 1:
                    raise ValueError("Expand ports must share one cardinality")
                if counts[0] > self.limits.max_fanout_per_grain:
                    raise ValueError("max_fanout_per_grain exceeded")
            else:
                raise ValueError(f"{node.kind.value} cannot commit in Phase 2")
            cardinalities.append(counts)
        return tuple(cardinalities)

    def _prepare_commit_delta(
        self,
        plan: DispatchPlan,
        manifest: BatchManifest,
        block_payloads: tuple[Any, ...],
        cardinalities: tuple[tuple[int, ...], ...],
    ) -> CommitDelta:
        node = self.graph.node(plan.node)
        block_ids = tuple(
            self._next_block + index for index in range(len(block_payloads))
        )
        blocks = tuple(zip(block_ids, block_payloads))
        values: list[tuple[ItemRef, LocalValue]] = []
        outcomes: list[tuple[GrainRecord, AttemptToken, Success]] = []
        origins: list[tuple[Any, ExpandOrigin]] = []
        pending_items: set[ItemRef] = set()

        for entry_index, (entry, counts) in enumerate(
            zip(plan.entries, cardinalities)
        ):
            record = self.grains.get(entry.token.grain)
            if record is None or record.active != entry.token:
                raise ValueError("attempt became stale during validation")
            emissions_by_port: list[tuple[Emission, ...]] = []
            if node.kind is Primitive.EXPAND:
                parent = _one_role_item(record, "parent")
                entities = tuple(
                    expand_entity(
                        self.run_salt,
                        node.id,
                        parent.entity,
                        ordinal,
                    )
                    for ordinal in range(counts[0])
                )
            else:
                entities = ()

            for port_index, port in enumerate(node.output_ports):
                span = manifest.acks[entry_index].spans_by_port[port_index]
                emissions: list[Emission] = []
                for offset in range(counts[port_index]):
                    if node.kind is Primitive.EXPAND:
                        item = ItemRef(port, entities[offset])
                        ordinal = offset
                    else:
                        item = record.output_slots[port_index]
                        ordinal = 0
                    if item in pending_items or self.values.contains(item):
                        raise ValueError("logical output value would be published twice")
                    existing_port_item = self.ports.get(item.port, item.entity)
                    if existing_port_item is not None:
                        raise ValueError("logical output already exists on its port")
                    producer = self.producers.get(item)
                    if producer is not None and producer != record.id:
                        raise ValueError("logical output has a conflicting producer")
                    pending_items.add(item)
                    emissions.append(Emission(item, ordinal))
                    values.append(
                        (
                            item,
                            LocalValue(block_ids[port_index], span.start + offset),
                        )
                    )
                    if node.kind is Primitive.EXPAND:
                        origin = ExpandOrigin(parent, ordinal, record.id)
                        existing_origin = self.expand_origins.get(item.entity)
                        if existing_origin is not None and existing_origin != origin:
                            raise ValueError("Expand entity origin conflict")
                        origins.append((item.entity, origin))
                emissions_by_port.append(tuple(emissions))
            outcomes.append(
                (record, entry.token, Success(tuple(emissions_by_port)))
            )
        return CommitDelta(
            blocks=blocks,
            values=tuple(values),
            outcomes=tuple(outcomes),
            origins=tuple(origins),
        )

    def _apply_commit_delta(
        self,
        plan: DispatchPlan,
        runtime: DispatchRuntime,
        delta: CommitDelta,
    ) -> None:
        try:
            for block, values in delta.blocks:
                self._blocks[block] = values
            self._blocks_high_watermark = max(
                self._blocks_high_watermark,
                len(self._blocks),
            )
            for item, location in delta.values:
                self.values.put(item, location)
            for record, token, outcome in delta.outcomes:
                self.producers.register(
                    GrainRecord.sealed(
                        id=record.id,
                        node=record.node,
                        inputs=record.inputs,
                        output_slots=record.output_slots,
                        outcome=outcome,
                    )
                )
                self.ports.register(outcome)
                record.seal(outcome, token=token)
                self._grain_completed_at[record.id] = self._clock()
            for entity, origin in delta.origins:
                self.expand_origins.add(entity, origin)
            self._next_block += len(delta.blocks)
            runtime.output_blocks = tuple(block for block, _ in delta.blocks)
            self._finish_dispatch(plan)
        except Exception as error:
            self._abort(f"commit apply failed: {error}")

    def handle_infrastructure_failure(
        self,
        plan: DispatchPlan,
    ) -> CommitStatus:
        self._require_running()
        currentness = self._currentness(plan)
        if currentness is CommitStatus.STALE:
            self._finish_dispatch(plan)
            return currentness
        records = [self.grains.get(entry.token.grain) for entry in plan.entries]
        if any(record is None for record in records):
            self._abort("dispatch references an unknown grain")
        if any(
            record.infra_failures + 1 > self.limits.max_infra_retries
            for record in records
            if record is not None
        ):
            self._abort("infrastructure retry budget exhausted")
        for record, entry in zip(records, plan.entries):
            assert record is not None
            record.retry_infrastructure_failure(entry.token)
            self.enqueue_ready(record)
        self._finish_dispatch(plan)
        return CommitStatus.ACCEPTED

    def handle_error(
        self,
        plan: DispatchPlan,
        report: DispatchErrorReport,
    ) -> CommitStatus:
        self._require_running()
        currentness = self._currentness(plan)
        if currentness is CommitStatus.STALE:
            self._finish_dispatch(plan)
            return currentness
        if report.dispatch != plan.id:
            self._abort("error report dispatch id mismatch")
        node = self.graph.node(plan.node)
        assert node.execution is not None

        if report.kind == "bad_record":
            if report.bad_token is None:
                self._abort("bad_record report has no bad token")
            matching = [
                entry
                for entry in plan.entries
                if entry.token == report.bad_token
            ]
            if len(matching) != 1:
                self._abort("bad_record token is not in the DispatchPlan")
            for entry in plan.entries:
                record = self.grains.get(entry.token.grain)
                assert record is not None
                if entry.token == report.bad_token:
                    record.seal(
                        failed_outcome(
                            record,
                            kind="bad_record",
                            message=report.message,
                        ),
                        token=entry.token,
                    )
                    self._grain_completed_at[record.id] = self._clock()
                else:
                    record.release_for_reexecution(entry.token)
                    self.enqueue_ready(record)
            self._finish_dispatch(plan)
            return CommitStatus.ACCEPTED

        if report.kind != "generic_udf":
            self._abort(f"unknown dispatch error kind: {report.kind}")
        if node.execution.error_policy == "raise":
            self._abort(report.message)

        records = [
            self.grains.get(entry.token.grain) for entry in plan.entries
        ]
        assert all(record is not None for record in records)
        if len(plan.entries) == 1:
            record = records[0]
            assert record is not None
            record.seal(
                failed_outcome(
                    record,
                    kind="generic_udf",
                    message=report.message,
                ),
                token=plan.entries[0].token,
            )
            self._grain_completed_at[record.id] = self._clock()
        else:
            grain_ids: list[GrainId] = []
            for record, entry in zip(records, plan.entries):
                assert record is not None
                record.release_for_reexecution(entry.token)
                grain_ids.append(record.id)
            midpoint = len(grain_ids) // 2
            context = IsolationContext(
                root_dispatch=(
                    plan.id
                    if self._dispatches[plan.id].isolation is None
                    else self._dispatches[plan.id].isolation.root_dispatch
                ),
                depth=(
                    1
                    if self._dispatches[plan.id].isolation is None
                    else self._dispatches[plan.id].isolation.depth + 1
                ),
            )
            self._force_group(node.id, tuple(grain_ids[:midpoint]), context)
            self._force_group(node.id, tuple(grain_ids[midpoint:]), context)
        self._finish_dispatch(plan)
        return CommitStatus.ACCEPTED

    def resolve(self, item: ItemRef) -> Any:
        location = self.values.get(item)
        if not isinstance(location, LocalValue):
            raise ArenaAbort("Phase 2 requires LocalValue locations")
        return self._blocks[location.block][location.row]

    def slice(self, item: ItemRef) -> BlockSlice:
        location = self.values.get(item)
        if not isinstance(location, LocalValue):
            raise ArenaAbort("value location is not row-addressable")
        return BlockSlice(self._blocks[location.block], location.row)

    def deliver_slices(self, outputs: tuple[ItemRef, ...]) -> RunResult:
        return self._deliver(tuple(self.slice(item) for item in outputs))

    def deliver(self, outputs: tuple[ItemRef, ...]) -> RunResult:
        return self._deliver(tuple(self.resolve(item) for item in outputs))

    def metrics_snapshot(self) -> dict[str, float]:
        parent_latencies = []
        for record in self.grains.values():
            if self.graph.node(record.node).kind is not Primitive.REDUCE:
                continue
            completed = self._grain_completed_at.get(record.id)
            if completed is None:
                continue
            anchor_roles = [
                role for role in record.inputs if role.role == "anchor"
            ]
            if len(anchor_roles) != 1 or len(anchor_roles[0].items) != 1:
                continue
            admitted = self._source_time_for_item(anchor_roles[0].items[0])
            if admitted is not None:
                parent_latencies.append(completed - admitted)
        metrics = {
            "grains_per_rpc": (
                self._dispatched_grains / self._dispatch_count
                if self._dispatch_count
                else 0.0
            ),
            "rpc_count": float(self._dispatch_count),
            "pending_dispatches": float(len(self._dispatches)),
            "tail_or_isolation_rpc_fraction": (
                self._tail_or_isolation_dispatches / self._dispatch_count
                if self._dispatch_count
                else 0.0
            ),
            "batch_fill_ratio": (
                self._dispatched_grains / self._dispatch_capacity
                if self._dispatch_capacity
                else 0.0
            ),
            "ready_queue_high_watermark": float(
                self._ready_queue_high_watermark
            ),
            "live_blocks_at_delivery": float(len(self._blocks)),
            "live_blocks_high_watermark": float(
                self._blocks_high_watermark
            ),
            "parent_completion_count": float(len(parent_latencies)),
            "parent_completion_p50_s": percentile(parent_latencies, 0.50),
            "parent_completion_p95_s": percentile(parent_latencies, 0.95),
            "parent_completion_p99_s": percentile(parent_latencies, 0.99),
        }
        metrics.update(
            {
                f"flush_{reason}": float(count)
                for reason, count in self._flush_reason_counts.items()
            }
        )
        return metrics

    def _source_time_for_item(
        self,
        item: ItemRef,
        seen: set[GrainId] | None = None,
    ) -> float | None:
        producer = self.producers.get(item)
        if producer is None:
            return None
        admitted = self._source_admitted_at.get(producer)
        if admitted is not None:
            return admitted
        seen = set() if seen is None else seen
        if producer in seen:
            return None
        seen.add(producer)
        record = self.grains.get(producer)
        if record is None:
            return None
        times = [
            time_value
            for role in record.inputs
            for parent in role.items
            for time_value in (self._source_time_for_item(parent, seen),)
            if time_value is not None
        ]
        return min(times) if times else None

    def _deliver(self, values: tuple[Any, ...]) -> RunResult:
        self._require_running()
        if self._dispatches:
            self._abort("cannot deliver with pending dispatches")
        if any(
            record.phase is not GrainPhase.SEALED
            for record in self.grains.values()
        ):
            self._abort("cannot deliver while logical grains are non-terminal")
        failures = tuple(
            FailureSnapshot(record.id, record.outcome.failure)
            for record in self.grains.values()
            if isinstance(record.outcome, Failed)
        )
        metrics = self.metrics_snapshot()
        metrics["pending_dispatches"] = 0.0
        result = RunResult(
            outputs=values,
            failures=failures,
            sources=tuple(self._sources),
            metrics=metrics,
        )
        self.state = ArenaState.DELIVERED
        self._reclaim()
        return result


@dataclass(slots=True)
class _PortDomain:
    receipts: dict[Any, BindingReceipt] = field(default_factory=dict)
    order: list[Any] = field(default_factory=list)
    sealed: bool = False

    def publish(self, receipt: BindingReceipt) -> bool:
        if receipt.item is None:
            raise ExecutionError("published port receipt needs an ItemRef")
        entity = receipt.item.entity
        existing = self.receipts.get(entity)
        if existing is not None:
            if (
                existing.state != receipt.state
                or existing.item != receipt.item
                or existing.cause != receipt.cause
            ):
                raise ExecutionError("port entity received conflicting receipts")
            return False
        self.receipts[entity] = receipt
        self.order.append(entity)
        return True


class _PipelineDriver:
    """Small event-loop driver for traced bounded DAG integration."""

    def __init__(
        self,
        compiled: CompiledPipeline,
        arena: Arena,
        transport: "RayTransport",
    ) -> None:
        self.compiled = compiled
        self.graph = compiled.graph
        self.arena = arena
        self.transport = transport
        self.domains = {
            port: _PortDomain()
            for node in self.graph.nodes
            for port in node.output_ports
        }
        self.processed: set[GrainId] = set()
        self.planned: set[tuple[int, Any]] = set()
        self.barriers: dict[tuple[int, Any], Any] = {}
        self.relate_planned: set[int] = set()

    def admit_sources(self, batches: tuple[Any, ...]) -> None:
        import ray

        if len(batches) != len(self.compiled.source_ports):
            raise ExecutionError("source argument count does not match Pipeline.forward")
        for public_port, values in zip(self.compiled.source_ports, batches):
            if not isinstance(values, (list, tuple)):
                raise ExecutionError("each source argument must be a finite sequence")
            node = self.graph.producer(public_port.id)
            records = tuple(
                admit_source(node, self.arena.run_salt, position)
                for position in range(len(values))
            )
            self.arena.admit_source_batch(records, ray.put(tuple(values)))
            domain = self.domains[public_port.id]
            for record in records:
                domain.publish(
                    BindingReceipt.present("source", record.output_slots[0])
                )
                self.processed.add(record.id)
            domain.sealed = True

    def _domain_receipt(
        self,
        binding: Any,
        entity: Any,
    ) -> BindingReceipt:
        domain = self.domains[binding.port]
        stored = domain.receipts.get(entity)
        if stored is not None:
            return BindingReceipt(
                binding.role,
                stored.state,
                stored.item,
                stored.cause,
            )
        expected = ItemRef(binding.port, entity)
        if domain.sealed:
            return BindingReceipt.absent(binding.role, expected)
        return BindingReceipt.pending(binding.role, expected)

    def _publish_terminal_records(self) -> bool:
        progress = False
        for record in self.arena.grains.values():
            if record.id in self.processed or record.phase is not GrainPhase.SEALED:
                continue
            node = self.graph.node(record.node)
            if isinstance(record.outcome, Success):
                emitted: set[ItemRef] = set()
                for port_index, emissions in enumerate(
                    record.outcome.emissions_by_port
                ):
                    for emission in emissions:
                        emitted.add(emission.item)
                        progress |= self.domains[
                            node.output_ports[port_index]
                        ].publish(
                            BindingReceipt.present(
                                "output",
                                emission.item,
                            )
                        )
                for slot in record.output_slots:
                    if slot not in emitted:
                        progress |= self.domains[slot.port].publish(
                            BindingReceipt.absent("output", slot)
                        )
            elif isinstance(record.outcome, Failed):
                for slot in record.output_slots:
                    progress |= self.domains[slot.port].publish(
                        BindingReceipt.failed(
                            "output",
                            slot,
                            record.id,
                        )
                    )
            elif isinstance(record.outcome, Suppressed):
                for slot in record.output_slots:
                    progress |= self.domains[slot.port].publish(
                        BindingReceipt.suppressed(
                            "output",
                            slot,
                            record.id,
                        )
                    )
            self.processed.add(record.id)
        return progress

    @staticmethod
    def _driving_role(node: Any) -> str:
        return {
            Primitive.MAP: "primary",
            Primitive.FILTER: "target",
            Primitive.EXPAND: "parent",
        }[node.kind]

    def _plan_unary(self, node: Any) -> bool:
        driving_role = self._driving_role(node)
        driving_binding = next(
            binding for binding in node.inputs if binding.role == driving_role
        )
        progress = False
        for entity in tuple(self.domains[driving_binding.port].order):
            key = (node.id, entity)
            if key in self.planned:
                continue
            receipts = tuple(
                self._domain_receipt(binding, entity)
                for binding in node.inputs
            )
            planner = {
                Primitive.MAP: plan_map,
                Primitive.FILTER: plan_filter,
                Primitive.EXPAND: plan_expand,
            }[node.kind]
            decision = planner(node, self.arena.run_salt, receipts)
            if decision.action is PlanAction.WAIT:
                continue
            if decision.action is PlanAction.NORMAL_ABSENCE:
                if node.kind is not Primitive.EXPAND:
                    for port in node.output_ports:
                        self.domains[port].publish(
                            BindingReceipt.absent(
                                "output",
                                ItemRef(port, entity),
                            )
                        )
            else:
                self.arena.ensure_plans((decision,))
            self.planned.add(key)
            progress = True
        return progress

    def _reduce_origin(self, node: Any) -> tuple[Any, PortId]:
        current = node.reduce_members
        assert current is not None
        while True:
            producer = self.graph.producer(current)
            if producer.kind is Primitive.MAP:
                current = next(
                    binding.port
                    for binding in producer.inputs
                    if binding.role == "primary"
                )
                continue
            if producer.kind is Primitive.FILTER:
                current = next(
                    binding.port
                    for binding in producer.inputs
                    if binding.role == "target"
                )
                continue
            if producer.kind is not Primitive.EXPAND:
                raise ExecutionError("Reduce path lost its origin Expand")
            return producer, current

    def _plan_reduce(self, node: Any) -> bool:
        anchor_binding = next(
            binding for binding in node.inputs if binding.role == "anchor"
        )
        members_port = node.reduce_members
        assert members_port is not None
        origin_node, origin_port = self._reduce_origin(node)
        progress = False
        for entity in tuple(self.domains[anchor_binding.port].order):
            key = (node.id, entity)
            if key in self.planned:
                continue
            anchor = self._domain_receipt(anchor_binding, entity)
            if anchor.state is ReceiptState.PENDING or anchor.item is None:
                continue
            origin_inputs = (RoleItems("parent", (anchor.item,)),)
            origin_id = expand_grain_id(
                self.arena.run_salt,
                origin_node.id,
                origin_inputs,
            )
            origin_record = self.arena.grains.get(origin_id)
            if origin_record is None or origin_record.outcome is None:
                continue
            barrier_key = (node.id, entity)
            barrier = self.barriers.get(barrier_key)
            if barrier is None:
                from .grain import FiberBarrier, FiberId

                barrier = FiberBarrier(
                    FiberId(node.id, anchor.item),
                    origin_record.id,
                )
                self.barriers[barrier_key] = barrier
            if isinstance(origin_record.outcome, Success):
                emissions = origin_record.outcome.emissions_by_port[
                    origin_port.slot
                ]
                barrier.set_expected(len(emissions))
                member_domain = self.domains[members_port]
                for emission in emissions:
                    receipt = member_domain.receipts.get(emission.item.entity)
                    if receipt is None:
                        if member_domain.sealed:
                            raise ExecutionError(
                                "sealed members port lost a child occurrence"
                            )
                        continue
                    if receipt.state is ReceiptState.PRESENT:
                        assert receipt.item is not None
                        barrier.settle_present(emission.ordinal, receipt.item)
                    elif receipt.state is ReceiptState.NORMAL_ABSENCE:
                        barrier.settle_dropped(emission.ordinal)
                    elif receipt.state in {
                        ReceiptState.FAILED,
                        ReceiptState.SUPPRESSED,
                    }:
                        assert receipt.item is not None and receipt.cause is not None
                        barrier.settle_failed(
                            emission.ordinal,
                            receipt.item,
                            receipt.cause,
                        )
            else:
                barrier.block_origin(origin_record.id)
            aligned_roles: list[RoleItems] = []
            aligned_causes: list[GrainId] = []
            aligned_pending = False
            if barrier.state is FiberState.READY:
                member_items = barrier.present_members()
                for binding in node.inputs:
                    if binding.role in {"anchor", "members"}:
                        continue
                    domain = self.domains[binding.port]
                    aligned_items: list[ItemRef] = []
                    for member in member_items:
                        receipt = domain.receipts.get(member.entity)
                        if receipt is None:
                            if domain.sealed:
                                raise ExecutionError(
                                    f"sealed aligned Reduce role "
                                    f"{binding.role!r} lost entity"
                                )
                            aligned_pending = True
                            break
                        if receipt.state is ReceiptState.NORMAL_ABSENCE:
                            raise ExecutionError(
                                f"aligned Reduce role {binding.role!r} "
                                "is normally absent"
                            )
                        assert receipt.item is not None
                        aligned_items.append(receipt.item)
                        if receipt.state in {
                            ReceiptState.FAILED,
                            ReceiptState.SUPPRESSED,
                        }:
                            assert receipt.cause is not None
                            aligned_causes.append(receipt.cause)
                    if aligned_pending:
                        break
                    aligned_roles.append(
                        RoleItems(binding.role, tuple(aligned_items))
                    )
            if aligned_pending:
                continue
            decision = plan_reduce(
                node,
                self.arena.run_salt,
                anchor,
                barrier,
                aligned_roles=tuple(aligned_roles),
                aligned_causes=tuple(aligned_causes),
            )
            if decision.action is PlanAction.WAIT:
                continue
            if decision.action is not PlanAction.NORMAL_ABSENCE:
                self.arena.ensure_plans((decision,))
            self.planned.add(key)
            progress = True
        return progress

    def _key_value(self, key_spec: Any, item: ItemRef) -> int:
        if not isinstance(key_spec.by, Port):
            raise ExecutionError(
                "automatic Relate requires keyed(data, by=key_port)"
            )
        key_domain = self.domains[key_spec.by.id]
        receipt = key_domain.receipts.get(item.entity)
        if receipt is None or receipt.state is not ReceiptState.PRESENT:
            raise ExecutionError("Relate key port is missing an aligned key")
        assert receipt.item is not None
        value = self.arena.slice(receipt.item)
        import ray

        key = ray.get(value.block)[value.row]
        if type(key) is not int:
            raise ExecutionError("bounded automatic Relate accepts int keys")
        return key

    def _plan_relate(self, node: Any) -> bool:
        if node.id in self.relate_planned:
            return False
        key_ports = tuple(
            key.by.id
            for key in node.relate_keys
            if isinstance(key.by, Port)
        )
        if len(key_ports) != len(node.relate_keys):
            return False
        if not all(
            self.domains[binding.port].sealed for binding in node.inputs
        ) or not all(self.domains[port].sealed for port in key_ports):
            return False
        roles = []
        for binding, key_spec in zip(node.inputs, node.relate_keys):
            rows = tuple(
                (
                    receipt.item,
                    self._key_value(key_spec, receipt.item),
                )
                for receipt in self.domains[binding.port].receipts.values()
                if receipt.state is ReceiptState.PRESENT
                and receipt.item is not None
            )
            roles.append((binding.role, rows))
        result = plan_relate_bounded(
            node,
            self.arena.run_salt,
            tuple(roles),
            sealed_roles=frozenset(binding.role for binding in node.inputs),
            max_cardinality=self.arena.limits.max_relation_cardinality,
        )
        self.arena.ensure_plans(result.decisions)
        self.relate_planned.add(node.id)
        return True

    def _node_terminal(self, node_id: int) -> bool:
        return all(
            record.phase is GrainPhase.SEALED
            for record in self.arena.grains.values()
            if record.node == node_id
        )

    def _seal_ports(self) -> bool:
        progress = False
        for node in self.graph.nodes:
            if node.kind is Primitive.SOURCE:
                continue
            if all(self.domains[port].sealed for port in node.output_ports):
                continue
            if node.kind in {
                Primitive.MAP,
                Primitive.FILTER,
                Primitive.EXPAND,
            }:
                driving = next(
                    binding
                    for binding in node.inputs
                    if binding.role == self._driving_role(node)
                )
                ready = (
                    all(self.domains[binding.port].sealed for binding in node.inputs)
                    and all(
                        (node.id, entity) in self.planned
                        for entity in self.domains[driving.port].order
                    )
                    and self._node_terminal(node.id)
                )
            elif node.kind is Primitive.REDUCE:
                anchor = next(
                    binding
                    for binding in node.inputs
                    if binding.role == "anchor"
                )
                ready = (
                    self.domains[anchor.port].sealed
                    and all(
                        (node.id, entity) in self.planned
                        for entity in self.domains[anchor.port].order
                    )
                    and self._node_terminal(node.id)
                )
            else:
                ready = (
                    node.id in self.relate_planned
                    and self._node_terminal(node.id)
                )
            if ready:
                for port in node.output_ports:
                    self.domains[port].sealed = True
                progress = True
        return progress

    def _plan(self) -> bool:
        progress = False
        for node in self.graph.nodes:
            if node.kind is Primitive.SOURCE:
                continue
            if node.kind in {
                Primitive.MAP,
                Primitive.FILTER,
                Primitive.EXPAND,
            }:
                progress |= self._plan_unary(node)
            elif node.kind is Primitive.REDUCE:
                progress |= self._plan_reduce(node)
            elif node.kind is Primitive.RELATE:
                progress |= self._plan_relate(node)
        return progress

    def _admission_closed(self, node: Any) -> bool:
        if node.kind in {
            Primitive.MAP,
            Primitive.FILTER,
            Primitive.EXPAND,
        }:
            driving = next(
                binding
                for binding in node.inputs
                if binding.role == self._driving_role(node)
            )
            return (
                all(self.domains[binding.port].sealed for binding in node.inputs)
                and all(
                    (node.id, entity) in self.planned
                    for entity in self.domains[driving.port].order
                )
            )
        if node.kind is Primitive.REDUCE:
            anchor = next(
                binding
                for binding in node.inputs
                if binding.role == "anchor"
            )
            return (
                self.domains[anchor.port].sealed
                and all(
                    (node.id, entity) in self.planned
                    for entity in self.domains[anchor.port].order
                )
            )
        if node.kind is Primitive.RELATE:
            return node.id in self.relate_planned
        return True

    def _dispatch(self) -> bool:
        progress = False
        for node in self.graph.nodes:
            if node.kind is Primitive.SOURCE:
                continue
            while self.transport.can_submit(node.id):
                plan = self.arena.reserve_dispatch(
                    node.id,
                    admission_closed=self._admission_closed(node),
                )
                if plan is None:
                    break
                if not self.transport.submit(self.arena, plan):
                    raise ExecutionError("actor capacity changed during submit")
                progress = True
        return progress

    def _next_batch_delay(self) -> float | None:
        delays = []
        for node in self.graph.nodes:
            if node.kind is Primitive.SOURCE:
                continue
            if not self.transport.can_submit(node.id):
                continue
            if self._admission_closed(node):
                if self.arena.ready_count(node.id):
                    return 0.0
                continue
            delay = self.arena.batch_wait_remaining(node.id)
            if delay is not None:
                delays.append(delay)
        return min(delays) if delays else None

    def run(self) -> RunResult:
        final_ports = tuple(port.id for port in self.compiled.outputs)
        while True:
            progress = self._publish_terminal_records()
            progress |= self._plan()
            progress |= self._seal_ports()
            progress |= self._dispatch()
            if self.transport.pending_dispatches:
                delay = self._next_batch_delay()
                timeout = 1.0 if delay is None else min(1.0, delay)
                result = self.transport.poll_one(timeout=timeout)
                progress |= result is not None
                continue
            if all(self.domains[port].sealed for port in final_ports):
                output_items = tuple(
                    receipt.item
                    for port in final_ports
                    for entity in self.domains[port].order
                    for receipt in (self.domains[port].receipts[entity],)
                    if receipt.state is ReceiptState.PRESENT
                    and receipt.item is not None
                )
                return self.arena.deliver_slices(output_items)
            if not progress:
                delay = self._next_batch_delay()
                if delay is not None:
                    time.sleep(delay)
                    continue
                raise ExecutionError("pipeline reached a non-terminal deadlock")


class Executor:
    """Factory for explicit arenas and traced Pipeline execution."""

    def __init__(self, graph: CompiledGraph | Pipeline | CompiledPipeline) -> None:
        if isinstance(graph, Pipeline):
            self.compiled_pipeline: CompiledPipeline | None = graph.compile()
            self.graph = self.compiled_pipeline.graph
        elif isinstance(graph, CompiledPipeline):
            self.compiled_pipeline = graph
            self.graph = graph.graph
        else:
            self.compiled_pipeline = None
            self.graph = graph
        self._next_arena = 0

    def new_arena(
        self,
        arena_id: int,
        run_salt: bytes,
        *,
        limits: ArenaLimits = ArenaLimits(),
    ) -> Arena:
        return Arena(arena_id, self.graph, run_salt, limits=limits)

    def ray_transport(
        self,
        *,
        max_pending_per_actor: int = 1,
    ) -> "RayTransport":
        return RayTransport(
            self.graph,
            max_pending_per_actor=max_pending_per_actor,
        )

    def run(self, *sources: Any) -> RunResult:
        if self.compiled_pipeline is None:
            raise ExecutionError(
                "Executor.run requires a traced Pipeline, not bare CompiledGraph"
            )
        import secrets

        arena = self.new_arena(
            self._next_arena,
            secrets.token_bytes(16),
        )
        self._next_arena += 1
        end_to_end_started_at = time.monotonic()
        transport = self.ray_transport()
        try:
            transport.ready()
            measured_started_at = time.monotonic()
            driver = _PipelineDriver(
                self.compiled_pipeline,
                arena,
                transport,
            )
            driver.admit_sources(tuple(sources))
            result = driver.run()
            measured_finished_at = time.monotonic()
            timeline = transport.timeline()
            metrics = dict(result.metrics)
            metrics.update(
                {
                    "startup_time_s": (
                        measured_started_at - end_to_end_started_at
                    ),
                    "measured_wall_time_s": (
                        measured_finished_at - measured_started_at
                    ),
                    "end_to_end_wall_time_s": (
                        measured_finished_at - end_to_end_started_at
                    ),
                }
            )
            metrics.update(
                {
                    f"actor_count_node_{node}": float(count)
                    for node, count in transport.actor_counts().items()
                }
            )
            worker_rss = [
                event.worker_rss_bytes
                for event in timeline
                if event.worker_rss_bytes is not None
            ]
            metrics["worker_rss_peak_bytes"] = float(
                max(worker_rss, default=0)
            )
            return replace(
                result,
                metrics=metrics,
                timeline=timeline,
            )
        finally:
            transport.shutdown()


class RayTransport:
    """Persistent-actor coarse-block transport for Phase 3 integration."""

    def __init__(
        self,
        graph: CompiledGraph,
        *,
        max_pending_per_actor: int = 1,
    ) -> None:
        if max_pending_per_actor <= 0:
            raise ValueError("max_pending_per_actor must be positive")
        import ray

        if not ray.is_initialized():
            raise ExecutionError("Ray must be initialized before RayTransport")
        self.graph = graph
        self.max_pending_per_actor = max_pending_per_actor
        self._worker_class = get_ray_worker_class()
        self._actors: dict[int, list[Any]] = {}
        self._actor_specs: dict[
            int,
            tuple[Any, tuple[Any, ...], dict[str, Any], dict[str, Any]],
        ] = {}
        self._pending_by_actor: dict[tuple[int, int], int] = {}
        self._round_robin: dict[int, int] = {}
        self._pending: dict[Any, RayPending] = {}
        self._timeline: list[DispatchTimeline] = []

        for node in graph.nodes:
            if node.kind is Primitive.SOURCE:
                continue
            assert node.execution is not None
            assert node.udf_recipe is not None
            options = dict(node.execution.options)
            spec = (
                node.udf_recipe.target,
                node.udf_recipe.init_args,
                dict(node.udf_recipe.init_kwargs),
                options,
            )
            self._actor_specs[node.id] = spec
            actors = [
                self._spawn_actor(node.id)
                for _ in range(node.execution.replicas)
            ]
            self._actors[node.id] = actors
            for index in range(len(actors)):
                self._pending_by_actor[(node.id, index)] = 0

    @property
    def pending_dispatches(self) -> int:
        return len(self._pending)

    def can_submit(self, node: int) -> bool:
        return any(
            self._pending_by_actor[(node, index)]
            < self.max_pending_per_actor
            for index in range(len(self._actors[node]))
        )

    def _choose_actor(self, node: int) -> tuple[int, Any] | None:
        actors = self._actors[node]
        start = self._round_robin.get(node, 0)
        for offset in range(len(actors)):
            index = (start + offset) % len(actors)
            if (
                self._pending_by_actor[(node, index)]
                < self.max_pending_per_actor
            ):
                self._round_robin[node] = (index + 1) % len(actors)
                return index, actors[index]
        return None

    def _spawn_actor(self, node: int):
        target, init_args, init_kwargs, options = self._actor_specs[node]
        return self._worker_class.options(
            max_task_retries=0,
            **options,
        ).remote(target, init_args, init_kwargs)

    def _replace_actor(self, node: int, actor_index: int) -> None:
        import ray

        old = self._actors[node][actor_index]
        try:
            ray.kill(old)
        except Exception:
            pass
        self._actors[node][actor_index] = self._spawn_actor(node)

    def submit(self, arena: Arena, plan: DispatchPlan) -> bool:
        import ray

        choice = self._choose_actor(plan.node)
        if choice is None:
            return False
        actor_index, actor = choice
        node = self.graph.node(plan.node)
        input_refs = arena.input_block_handles(plan)
        if not all(isinstance(ref, ray.ObjectRef) for ref in input_refs):
            raise ExecutionError(
                "Ray dispatch inputs must be coarse ObjectRef blocks"
            )
        refs = actor.run.options(
            num_returns=1 + len(node.output_ports),
            max_task_retries=0,
        ).remote(
            node.kind.value,
            len(node.output_ports),
            tuple(binding.role for binding in node.inputs),
            plan,
            *input_refs,
        )
        refs_tuple = tuple(refs) if isinstance(refs, list) else tuple(refs)
        pending = RayPending(
            arena=arena,
            plan=plan,
            node=plan.node,
            actor_index=actor_index,
            manifest_ref=refs_tuple[0],
            output_refs=refs_tuple[1:],
            submitted_at=time.monotonic(),
        )
        self._pending[pending.manifest_ref] = pending
        self._pending_by_actor[(plan.node, actor_index)] += 1
        return True

    def poll_one(self, *, timeout: float | None = None) -> CommitStatus | None:
        import ray

        if not self._pending:
            return None
        ready, _ = ray.wait(
            list(self._pending),
            num_returns=1,
            timeout=timeout,
        )
        if not ready:
            return None
        manifest_ref = ready[0]
        pending = self._pending.pop(manifest_ref)
        manifest_received_at = time.monotonic()
        flush_reason = pending.arena.dispatch_flush_reason(pending.plan)
        self._pending_by_actor[
            (pending.node, pending.actor_index)
        ] -= 1
        try:
            manifest = ray.get(manifest_ref)
        except Exception:
            self._replace_actor(pending.node, pending.actor_index)
            result = pending.arena.handle_infrastructure_failure(pending.plan)
            self._record_timeline(
                pending,
                manifest_received_at,
                None,
                "infrastructure_failure",
                flush_reason,
            )
            return result
        if isinstance(manifest, BatchManifest):
            result = pending.arena.commit_external_manifest(
                pending.plan,
                manifest,
                pending.output_refs,
            )
            self._record_timeline(
                pending,
                manifest_received_at,
                manifest,
                result.value,
                flush_reason,
            )
            return result
        if isinstance(manifest, DispatchErrorReport):
            result = pending.arena.handle_error(pending.plan, manifest)
            self._record_timeline(
                pending,
                manifest_received_at,
                None,
                manifest.kind,
                flush_reason,
            )
            return result
        pending.arena._abort("worker returned an unknown manifest type")
        raise AssertionError("unreachable")

    def drain(self) -> tuple[CommitStatus, ...]:
        results: list[CommitStatus] = []
        while self._pending:
            result = self.poll_one()
            if result is not None:
                results.append(result)
        return tuple(results)

    def actor_stats(self, node: int) -> tuple[dict[str, int], ...]:
        import ray

        return tuple(
            ray.get([actor.stats.remote() for actor in self._actors[node]])
        )

    def ready(self) -> None:
        """Block until every persistent actor has finished construction."""

        import ray

        refs = [
            actor.stats.remote()
            for actors in self._actors.values()
            for actor in actors
        ]
        if refs:
            ray.get(refs)

    def actor_counts(self) -> dict[int, int]:
        return {
            node: len(actors) for node, actors in self._actors.items()
        }

    def timeline(self) -> tuple[DispatchTimeline, ...]:
        return tuple(self._timeline)

    def _record_timeline(
        self,
        pending: RayPending,
        manifest_received_at: float,
        manifest: BatchManifest | None,
        status: str,
        flush_reason: str,
    ) -> None:
        self._timeline.append(
            DispatchTimeline(
                arena=pending.arena.id,
                node=pending.node,
                dispatch=pending.plan.id,
                actor_index=pending.actor_index,
                grains=len(pending.plan.entries),
                flush_reason=flush_reason,
                submitted_at=pending.submitted_at,
                manifest_received_at=manifest_received_at,
                committed_at=time.monotonic(),
                worker_started_at=(
                    manifest.worker_started_at
                    if manifest is not None
                    else None
                ),
                worker_finished_at=(
                    manifest.worker_finished_at
                    if manifest is not None
                    else None
                ),
                worker_rss_bytes=(
                    manifest.worker_rss_bytes
                    if manifest is not None
                    else None
                ),
                status=status,
            )
        )

    def shutdown(self) -> None:
        import ray

        for actors in self._actors.values():
            for actor in actors:
                ray.kill(actor)
        self._actors.clear()
        self._pending.clear()


def _one_role_item(record: GrainRecord, role: str) -> ItemRef:
    matches = [binding for binding in record.inputs if binding.role == role]
    if len(matches) != 1 or len(matches[0].items) != 1:
        raise ValueError(f"grain needs one item for role {role!r}")
    return matches[0].items[0]
