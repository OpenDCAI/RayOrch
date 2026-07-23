"""Single-process Phase 2 arena, dispatch, commit, and retry semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .api import ExecutionError
from .grain import (
    AttemptToken,
    ConsumerIndex,
    Emission,
    ExpandOrigin,
    ExpandOriginIndex,
    Failed,
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
    ValueIndex,
    expand_entity,
)
from .graph import (
    CompiledGraph,
    PlanAction,
    PlanDecision,
    PlannerContractError,
    Primitive,
    ensure_decision,
    failed_outcome,
)
from .worker import (
    BatchManifest,
    DispatchEntry,
    DispatchErrorReport,
    DispatchPlan,
    NormalizedBatchOutput,
    RowTake,
    WorkerContractError,
    build_batch_manifest,
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
    max_infra_retries: int = 1

    def __post_init__(self) -> None:
        if (
            self.max_grains <= 0
            or self.max_pending_dispatches <= 0
            or self.max_fanout_per_grain < 0
            or self.max_infra_retries < 0
        ):
            raise ValueError("arena limits must be positive/non-negative")


@dataclass(frozen=True, slots=True)
class LocalValue:
    block: int
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


@dataclass(frozen=True, slots=True)
class RunResult:
    outputs: tuple[Any, ...] = ()
    failures: tuple[FailureSnapshot, ...] = ()
    sources: tuple[SourceSnapshot, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CommitDelta:
    blocks: tuple[tuple[int, tuple[Any, ...]], ...]
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
    ) -> None:
        self.id = arena_id
        self.graph = graph
        self.run_salt = run_salt
        self.limits = limits
        self.state = ArenaState.RUNNING
        self.abort_reason: str | None = None

        self.grains = GrainTable()
        self.producers = ProducerIndex()
        self.consumers = ConsumerIndex()
        self.ports = PortIndex()
        self.values = ValueIndex()
        self.expand_origins = ExpandOriginIndex()

        self._blocks: dict[int, tuple[Any, ...]] = {}
        self._next_block = 0
        self._next_dispatch = 0
        self._dispatches: dict[int, DispatchRuntime] = {}
        self._ready: dict[int, list[GrainId]] = {}
        self._ready_set: dict[int, set[GrainId]] = {}
        self._forced: dict[
            int,
            list[tuple[tuple[GrainId, ...], IsolationContext]],
        ] = {}
        self._sources: list[SourceSnapshot] = []
        self._dispatch_count = 0
        self._dispatched_grains = 0
        self._tail_or_isolation_dispatches = 0

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
        self._forced.clear()
        self._sources.clear()
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
        except Exception as error:
            self._abort(f"source admission failed: {error}")

    def _allocate_block(self, values: tuple[Any, ...]) -> int:
        block = self._next_block
        self._next_block += 1
        self._blocks[block] = values
        return block

    def enqueue_ready(self, record: GrainRecord) -> None:
        if record.phase is not GrainPhase.READY:
            return
        ready_set = self._ready_set.setdefault(record.node, set())
        if record.id in ready_set:
            return
        ready_set.add(record.id)
        self._ready.setdefault(record.node, []).append(record.id)

    def _force_group(
        self,
        node: int,
        grains: tuple[GrainId, ...],
        context: IsolationContext,
    ) -> None:
        self._forced.setdefault(node, []).append((grains, context))

    def reserve_dispatch(self, node_id: int) -> DispatchPlan | None:
        self._require_running()
        if len(self._dispatches) >= self.limits.max_pending_dispatches:
            return None
        node = self.graph.node(node_id)
        if node.kind in {Primitive.SOURCE, Primitive.RELATE}:
            self._abort(f"{node.kind.value} is not dispatchable in Phase 2")
        assert node.execution is not None

        isolation: IsolationContext | None = None
        selected: list[GrainRecord] = []
        forced = self._forced.get(node_id)
        if forced:
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
            while queue and len(selected) < node.execution.batch_size:
                grain_id = queue.pop(0)
                ready_set.discard(grain_id)
                record = self.grains.get(grain_id)
                if record is not None and record.phase is GrainPhase.READY:
                    selected.append(record)
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
        )
        self._dispatch_count += 1
        self._dispatched_grains += len(entries)
        if len(entries) < node.execution.batch_size or isolation is not None:
            self._tail_or_isolation_dispatches += 1
        return plan

    def input_values(self, plan: DispatchPlan) -> tuple[tuple[Any, ...], ...]:
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
                columns,
            )
            delta = self._prepare_commit_delta(
                plan,
                manifest,
                columns,
                cardinalities,
            )
        except (ValueError, RuntimeError) as error:
            self._abort(f"manifest validation failed: {error}")

        try:
            for block, values in delta.blocks:
                self._blocks[block] = values
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
            for entity, origin in delta.origins:
                self.expand_origins.add(entity, origin)
            self._next_block += len(delta.blocks)
            runtime.output_blocks = tuple(block for block, _ in delta.blocks)
            self._finish_dispatch(plan)
        except Exception as error:
            self._abort(f"commit apply failed: {error}")
        return CommitStatus.ACCEPTED

    def _validate_manifest(
        self,
        plan: DispatchPlan,
        manifest: BatchManifest,
        columns: tuple[tuple[Any, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        node = self.graph.node(plan.node)
        output_arity = len(node.output_ports)
        if manifest.dispatch != plan.id:
            raise ValueError("manifest dispatch id mismatch")
        if len(manifest.acks) != len(plan.entries):
            raise ValueError("manifest must ack every dispatch entry")
        if len(columns) != output_arity:
            raise ValueError("output column arity mismatch")
        if manifest.column_lengths != tuple(len(column) for column in columns):
            raise ValueError("manifest column lengths mismatch")
        if any(len(ack.spans_by_port) != output_arity for ack in manifest.acks):
            raise ValueError("ack output arity mismatch")
        if tuple(ack.token for ack in manifest.acks) != tuple(
            entry.token for entry in plan.entries
        ):
            raise ValueError("manifest tokens do not match DispatchPlan")

        for port_index, column in enumerate(columns):
            cursor = 0
            for ack in manifest.acks:
                span = ack.spans_by_port[port_index]
                if span.start != cursor or span.stop < span.start:
                    raise ValueError("output spans must form a contiguous partition")
                cursor = span.stop
            if cursor != len(column):
                raise ValueError("output spans do not cover their column")

        cardinalities: list[tuple[int, ...]] = []
        for ack in manifest.acks:
            counts = tuple(
                span.stop - span.start for span in ack.spans_by_port
            )
            if node.kind in {Primitive.MAP, Primitive.REDUCE}:
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
        columns: tuple[tuple[Any, ...], ...],
        cardinalities: tuple[tuple[int, ...], ...],
    ) -> CommitDelta:
        node = self.graph.node(plan.node)
        block_ids = tuple(
            self._next_block + index for index in range(len(columns))
        )
        blocks = tuple(zip(block_ids, columns))
        values: list[tuple[ItemRef, LocalValue]] = []
        outcomes: list[tuple[GrainRecord, Any, Success]] = []
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

    def deliver(self, outputs: tuple[ItemRef, ...]) -> RunResult:
        self._require_running()
        if self._dispatches:
            self._abort("cannot deliver with pending dispatches")
        if any(
            record.phase is not GrainPhase.SEALED
            for record in self.grains.values()
        ):
            self._abort("cannot deliver while logical grains are non-terminal")
        values = tuple(self.resolve(item) for item in outputs)
        failures = tuple(
            FailureSnapshot(record.id, record.outcome.failure)
            for record in self.grains.values()
            if isinstance(record.outcome, Failed)
        )
        metrics = {
            "grains_per_rpc": (
                self._dispatched_grains / self._dispatch_count
                if self._dispatch_count
                else 0.0
            ),
            "pending_dispatches": 0.0,
            "tail_or_isolation_rpc_fraction": (
                self._tail_or_isolation_dispatches / self._dispatch_count
                if self._dispatch_count
                else 0.0
            ),
        }
        result = RunResult(
            outputs=values,
            failures=failures,
            sources=tuple(self._sources),
            metrics=metrics,
        )
        self.state = ArenaState.DELIVERED
        self._reclaim()
        return result


class Executor:
    """Factory for Phase 2 single-process arenas."""

    def __init__(self, graph: CompiledGraph) -> None:
        self.graph = graph

    def new_arena(
        self,
        arena_id: int,
        run_salt: bytes,
        *,
        limits: ArenaLimits = ArenaLimits(),
    ) -> Arena:
        return Arena(arena_id, self.graph, run_salt, limits=limits)

    def run(self, *sources: Any) -> RunResult:
        del sources
        raise ExecutionError(
            "Phase 2 exposes explicit single-process arenas; "
            "automatic graph driving is not implemented yet"
        )


def _one_role_item(record: GrainRecord, role: str) -> ItemRef:
    matches = [binding for binding in record.inputs if binding.role == role]
    if len(matches) != 1 or len(matches[0].items) != 1:
        raise ValueError(f"grain needs one item for role {role!r}")
    return matches[0].items[0]
