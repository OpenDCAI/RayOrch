"""ArenaEngine facade for one microbatch's event-driven state machine.

The engine owns the complete lifetime of one bounded Arena, but delegates its
small record types to :mod:`.state`.  It never imports Ray or StageExecutor.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

from ..api import MISSING
from ..dag import (
    CompiledDAG,
    InputMode,
    Primitive,
    RecoveryPreset,
    StageSpec,
)
from ..model import (
    AttemptToken,
    BlockRow,
    Emission,
    EntityId,
    EntityOrigin,
    Failed,
    GrainFailure,
    GrainId,
    GrainPhase,
    GrainRecord,
    InputBinding,
    InvariantError,
    ItemRecord,
    ItemRef,
    ItemTerminal,
    PortId,
    Success,
    Suppressed,
    expand_entity,
    source_entity,
    source_grain_id,
    stage_grain_id,
)
from ..protocol import (
    ArenaResult,
    BatchCall,
    BatchReport,
    BlockSlice,
    DispatchCompletion,
    DispatchFailure,
    DispatchIntent,
    DispatchTimeline,
    FailureKind,
    FailureSnapshot,
    Invocation,
    MissingTake,
    RowTake,
    SourceSnapshot,
    SuppressionSnapshot,
    ValueTake,
)
from .state import (
    DROPPED,
    FAILED,
    PRESENT,
    SUPPRESSED,
    ArenaAbort,
    ArenaLimits,
    DispatchLease,
    ExpandInstance,
    GroupedSlots,
    PendingInvocation,
    RecoveryBudgetState,
    RecoveryTask,
    ReduceAccumulator,
    StageBatchQueue,
)


class ArenaEngine:
    """Single-writer state machine for one bounded source microbatch.

    Public methods are the only RunDriver integration surface.  Tables,
    queues, accumulators, leases, and value locations remain Arena-private.
    """

    def __init__(
        self,
        arena_id: int,
        dag: CompiledDAG,
        run_salt: bytes,
        *,
        limits: ArenaLimits = ArenaLimits(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.id = arena_id
        self.dag = dag
        self.run_salt = run_salt
        self.limits = limits
        self.clock = clock
        self.admission_closed = False

        self.grains: dict[GrainId, GrainRecord] = {}
        self.items: dict[ItemRef, ItemRecord] = {}
        self.entity_origins: dict[EntityId, EntityOrigin] = {}
        self.expand_instances: dict[tuple[int, EntityId], ExpandInstance] = {}
        self.pending_invocations: dict[tuple[int, EntityId], PendingInvocation] = {}
        self.reduce_accumulators: dict[tuple[int, EntityId], ReduceAccumulator] = {}

        self.values: dict[ItemRef, BlockRow] = {}
        self.blocks: dict[int, Any] = {}
        self.next_block = 0

        self.receipts: deque[ItemRef] = deque()
        self.queues: dict[int, StageBatchQueue] = {}
        self.leases: dict[int, DispatchLease] = {}
        self.next_dispatch = 0
        self.reduce_slot_count = 0
        self.sources: list[SourceSnapshot] = []
        self.failure_count = 0
        self.dispatch_count = 0
        self.dispatched_grains = 0
        self.dispatch_capacity = 0
        self.tail_or_recovery_dispatches = 0
        self.timeline: list[DispatchTimeline] = []

    def _abort(self, message: str) -> None:
        raise ArenaAbort(message)

    @property
    def live_block_count(self) -> int:
        """Return the current number of Arena-owned coarse block handles."""

        return len(self.blocks)

    def _stage_queue(self, stage: int) -> StageBatchQueue:
        return self.queues.setdefault(stage, StageBatchQueue())

    def _allocate_block(self, handle: Any) -> int:
        if len(self.blocks) + 1 > self.limits.max_blocks:
            self._abort("max_blocks_per_arena exceeded")
        block_id = self.next_block
        self.next_block += 1
        self.blocks[block_id] = handle
        return block_id

    def admit_sources(
        self,
        source_values: tuple[tuple[Any, ...], ...],
        *,
        position_starts: tuple[int, ...],
        block_factory: Callable[[tuple[Any, ...]], Any] = tuple,
    ) -> None:
        if len(source_values) != len(self.dag.source_ports):
            self._abort("source argument count does not match CompiledDAG")
        if len(position_starts) != len(source_values):
            self._abort("source position starts do not match sources")
        lengths = {len(values) for values in source_values}
        if len(lengths) > 1:
            self._abort("aligned source batches must have equal length")

        positions = next(iter(lengths), 0)
        source_blocks = [
            self._allocate_block(block_factory(tuple(values)))
            for values in source_values
        ]
        additions = positions * len(source_values)
        if len(self.grains) + additions > self.limits.max_grains:
            self._abort("max_grains_per_arena exceeded during source admission")

        for local_position in range(positions):
            entity = source_entity(
                self.run_salt,
                position_starts[0] + local_position,
            )
            for source_index, port in enumerate(self.dag.source_ports):
                position = position_starts[source_index] + local_position
                grain_id = source_grain_id(self.run_salt, port, position)
                item = ItemRef(port, entity)
                outcome = Success(((Emission(item, 0),),))
                record = GrainRecord.sealed(
                    id=grain_id,
                    stage=port.stage,
                    inputs=(),
                    output_ports=(port,),
                    outcome=outcome,
                )
                self.grains[grain_id] = record
                self.items[item] = ItemRecord(
                    item,
                    grain_id,
                    ItemTerminal.PRESENT,
                )
                self.values[item] = BlockRow(source_blocks[source_index], local_position)
                self.sources.append(SourceSnapshot(grain_id, item))
                self.receipts.append(item)
        self.admission_closed = True

    def advance(self, now: float | None = None) -> bool:
        del now
        progress = False
        while self.receipts:
            item = self.receipts.popleft()
            record = self.items[item]
            for edge in self.dag.consumers(item.port):
                stage = self.dag.stage(edge.stage)
                input_spec = stage.inputs[edge.input_index]
                if stage.kind is Primitive.REDUCE:
                    progress |= self._route_reduce_input(
                        stage,
                        edge.input_index,
                        record,
                    )
                else:
                    progress |= self._route_aligned(
                        stage, edge.input_index, record
                    )
        return progress

    def _route_reduce_input(
        self,
        stage: StageSpec,
        input_index: int,
        item_record: ItemRecord,
    ) -> bool:
        mode = stage.inputs[input_index].mode
        if mode is InputMode.ANCHOR:
            return self._route_anchor(stage, input_index, item_record)
        if mode is InputMode.GROUP:
            return self._route_group(stage, input_index, item_record)
        return self._route_reduce_scalar(stage, input_index, item_record)

    def _route_aligned(
        self,
        stage: StageSpec,
        input_index: int,
        item_record: ItemRecord,
    ) -> bool:
        key = (stage.id, item_record.ref.entity)
        pending = self.pending_invocations.get(key)
        if pending is None:
            pending = PendingInvocation(
                stage.id,
                item_record.ref.entity,
                [None] * len(stage.inputs),
            )
            self.pending_invocations[key] = pending
        existing = pending.inputs[input_index]
        if existing is not None and existing != item_record.ref:
            self._abort("aligned invocation input changed")
        pending.inputs[input_index] = item_record.ref
        if any(value is None for value in pending.inputs):
            return True
        self.pending_invocations.pop(key)
        self._classify_aligned(stage, tuple(pending.inputs))
        return True

    def _classify_aligned(
        self,
        stage: StageSpec,
        refs: tuple[ItemRef, ...],
    ) -> None:
        records = tuple(self.items[ref] for ref in refs)
        assert stage.driving_input is not None
        driving = records[stage.driving_input]

        if driving.terminal is ItemTerminal.DROPPED:
            self._propagate_dropped(stage, refs[stage.driving_input].entity, driving)
            return
        failure_records = tuple(
            record
            for record in records
            if record.terminal in {ItemTerminal.FAILED, ItemTerminal.SUPPRESSED}
        )
        required_drop = tuple(
            record
            for index, record in enumerate(records)
            if record.terminal is ItemTerminal.DROPPED
            and stage.inputs[index].mode is InputMode.ONE
        )
        inputs = tuple(
            InputBinding(spec.name, (ref,))
            for spec, ref in zip(stage.inputs, refs)
        )
        if failure_records or required_drop:
            causes = tuple(
                cause
                for record in (*failure_records, *required_drop)
                for cause in (record.cause or record.producer,)
                if cause is not None
            )
            self._ensure_suppressed(stage, inputs, driving.ref.entity, causes)
            return
        self._ensure_executable(stage, inputs, driving.ref.entity)

    def _propagate_dropped(
        self,
        stage: StageSpec,
        entity: EntityId,
        driving: ItemRecord,
    ) -> None:
        cause = driving.cause or driving.producer
        if stage.kind is Primitive.EXPAND:
            # No child coordinate exists when the parent occurrence is absent.
            return
        for port in stage.output_ports():
            item = ItemRef(port, entity)
            self._publish_item(
                ItemRecord(item, None, ItemTerminal.DROPPED, cause)
            )

    def _ensure_executable(
        self,
        stage: StageSpec,
        inputs: tuple[InputBinding, ...],
        entity: EntityId,
    ) -> GrainRecord:
        grain_id = stage_grain_id(self.run_salt, stage.id, inputs)
        existing = self.grains.get(grain_id)
        if existing is not None:
            return existing
        self._check_grain_limit(1)
        record = GrainRecord(
            grain_id,
            stage.id,
            inputs,
            stage.output_ports(),
        )
        self.grains[grain_id] = record
        self._enqueue(record)
        return record

    def _ensure_suppressed(
        self,
        stage: StageSpec,
        inputs: tuple[InputBinding, ...],
        entity: EntityId,
        causes: tuple[GrainId, ...],
    ) -> GrainRecord:
        grain_id = stage_grain_id(self.run_salt, stage.id, inputs)
        existing = self.grains.get(grain_id)
        if existing is not None:
            return existing
        self._check_grain_limit(1)
        record = GrainRecord.sealed(
            id=grain_id,
            stage=stage.id,
            inputs=inputs,
            output_ports=stage.output_ports(),
            outcome=Suppressed(causes),
        )
        self.grains[grain_id] = record
        if stage.kind is Primitive.EXPAND:
            assert stage.driving_input is not None
            parent = inputs[stage.driving_input].items[0]
            instance = ExpandInstance(
                grain_id,
                stage.id,
                parent,
                None,
                failed=True,
            )
            self.expand_instances[(stage.id, parent.entity)] = instance
            self._activate_reduce_placeholders(instance)
            return record
        for port in stage.output_ports():
            self._publish_item(
                ItemRecord(
                    ItemRef(port, entity),
                    grain_id,
                    ItemTerminal.SUPPRESSED,
                    grain_id,
                )
            )
        return record

    def _route_anchor(
        self,
        stage: StageSpec,
        input_index: int,
        item_record: ItemRecord,
    ) -> bool:
        del input_index
        if item_record.terminal is ItemTerminal.DROPPED:
            self._propagate_dropped(stage, item_record.ref.entity, item_record)
            return False
        if item_record.terminal in {
            ItemTerminal.FAILED,
            ItemTerminal.SUPPRESSED,
        }:
            cause = item_record.cause or item_record.producer
            self._ensure_suppressed(
                stage,
                (InputBinding(stage.inputs[input_index].name, (item_record.ref,)),),
                item_record.ref.entity,
                (cause,) if cause is not None else (),
            )
            return True
        self._ensure_reduce_accumulator(stage, item_record)
        return True

    def _route_reduce_scalar(
        self,
        stage: StageSpec,
        input_index: int,
        item_record: ItemRecord,
    ) -> bool:
        anchor_spec = next(
            spec for spec in stage.inputs if spec.mode is InputMode.ANCHOR
        )
        anchor_item = ItemRef(anchor_spec.port, item_record.ref.entity)
        anchor_record = self.items.get(anchor_item)
        if anchor_record is None:
            return True
        accumulator = self._ensure_reduce_accumulator(stage, anchor_record)
        accumulator.scalar_inputs[input_index] = item_record.ref
        if self._reduce_inputs_complete(stage, accumulator):
            self._finalize_reduce(accumulator)
        return True

    def _route_group(
        self,
        stage: StageSpec,
        input_index: int,
        item_record: ItemRecord,
    ) -> bool:
        assert stage.reduce is not None
        try:
            anchor_entity, ordinal = self._resolve_scope(
                item_record.ref.entity,
                stage.reduce.origin_expand,
            )
        except KeyError:
            self._abort("group item has no matching Expand ancestry")
        anchor_spec = next(
            spec for spec in stage.inputs if spec.mode is InputMode.ANCHOR
        )
        anchor_item = ItemRef(anchor_spec.port, anchor_entity)
        anchor_record = self.items.get(anchor_item)
        if anchor_record is None:
            # Source/anchor receipt will create the accumulator later.
            return True
        accumulator = self._ensure_reduce_accumulator(stage, anchor_record)
        group = accumulator.groups[input_index]
        try:
            changed = group.settle(ordinal, item_record)
        except InvariantError as error:
            self._abort(str(error))
        if changed and accumulator.complete:
            self._finalize_reduce(accumulator)
        return changed

    def _ensure_reduce_accumulator(
        self,
        stage: StageSpec,
        anchor_record: ItemRecord,
    ) -> ReduceAccumulator:
        key = (stage.id, anchor_record.ref.entity)
        existing = self.reduce_accumulators.get(key)
        if existing is not None:
            return existing
        assert stage.reduce is not None
        origin = self.expand_instances.get(
            (stage.reduce.origin_expand, anchor_record.ref.entity)
        )
        if origin is None:
            # The anchor can arrive before the Expand terminal result.
            return self._placeholder_reduce(stage, anchor_record)
        accumulator = ReduceAccumulator(
            stage.id,
            anchor_record.ref,
            origin,
            {},
            {},
        )
        self.reduce_accumulators[key] = accumulator
        self._activate_reduce_accumulator(stage, accumulator, origin)
        return accumulator

    @staticmethod
    def _reduce_inputs_complete(
        stage: StageSpec,
        accumulator: ReduceAccumulator,
    ) -> bool:
        assert stage.reduce is not None
        if (
            accumulator.origin.cardinality is None
            or stage.reduce.members_input not in accumulator.groups
        ):
            return False
        scalar_indexes = tuple(
            index
            for index, spec in enumerate(stage.inputs)
            if spec.mode in {InputMode.ONE, InputMode.OPTIONAL_ONE}
        )
        return accumulator.complete and all(
            index in accumulator.scalar_inputs for index in scalar_indexes
        )

    def _placeholder_reduce(
        self,
        stage: StageSpec,
        anchor_record: ItemRecord,
    ) -> ReduceAccumulator:
        key = (stage.id, anchor_record.ref.entity)
        placeholder = ReduceAccumulator(
            stage.id,
            anchor_record.ref,
            ExpandInstance(
                GrainId(b"\0" * 16),
                stage.reduce.origin_expand,
                anchor_record.ref,
                None,
            ),
            {},
            {},
        )
        self.reduce_accumulators[key] = placeholder
        return placeholder

    def _activate_reduce_placeholders(self, expand: ExpandInstance) -> None:
        for stage in self.dag.stages:
            if (
                stage.kind is Primitive.REDUCE
                and stage.reduce is not None
                and stage.reduce.origin_expand == expand.stage
            ):
                key = (stage.id, expand.anchor.entity)
                existing = self.reduce_accumulators.get(key)
                if existing is not None and existing.groups:
                    continue
                anchor_record = self.items.get(
                    ItemRef(
                        next(
                            spec.port
                            for spec in stage.inputs
                            if spec.mode is InputMode.ANCHOR
                        ),
                        expand.anchor.entity,
                    )
                )
                if anchor_record is not None:
                    if existing is None:
                        self._ensure_reduce_accumulator(stage, anchor_record)
                    else:
                        self._activate_reduce_accumulator(
                            stage,
                            existing,
                            expand,
                        )

    def _activate_reduce_accumulator(
        self,
        stage: StageSpec,
        accumulator: ReduceAccumulator,
        expand: ExpandInstance,
    ) -> None:
        if expand.failed:
            inputs = (InputBinding("anchor", (accumulator.anchor,)),)
            self._ensure_suppressed(
                stage,
                inputs,
                accumulator.anchor.entity,
                (expand.grain,),
            )
            self.reduce_accumulators.pop(
                (stage.id, accumulator.anchor.entity),
                None,
            )
            return
        assert expand.cardinality is not None
        group_indexes = tuple(
            index
            for index, spec in enumerate(stage.inputs)
            if spec.mode is InputMode.GROUP
        )
        requested_slots = expand.cardinality * len(group_indexes)
        if self.reduce_slot_count + requested_slots > self.limits.max_reduce_slots:
            self._abort("max_reduce_slots_per_arena exceeded")
        accumulator.origin = expand
        accumulator.groups = {
            index: GroupedSlots.create(expand.cardinality)
            for index in group_indexes
        }
        self.reduce_slot_count += requested_slots
        if self._reduce_inputs_complete(stage, accumulator):
            self._finalize_reduce(accumulator)

    def _finalize_reduce(self, accumulator: ReduceAccumulator) -> None:
        stage = self.dag.stage(accumulator.stage)
        assert stage.reduce is not None
        key = (stage.id, accumulator.anchor.entity)
        if key not in self.reduce_accumulators:
            return

        members = accumulator.groups[stage.reduce.members_input]
        surviving = tuple(
            ordinal
            for ordinal, state in enumerate(members.states)
            if state == PRESENT
        )
        causes: list[GrainId] = []
        bindings: list[InputBinding] = []
        for input_index, spec in enumerate(stage.inputs):
            if spec.mode is InputMode.ANCHOR:
                bindings.append(InputBinding(spec.name, (accumulator.anchor,)))
            elif spec.mode is InputMode.GROUP:
                group = accumulator.groups[input_index]
                if input_index == stage.reduce.members_input:
                    selected_ordinals = tuple(
                        ordinal
                        for ordinal, state in enumerate(group.states)
                        if state in {PRESENT, FAILED, SUPPRESSED}
                    )
                else:
                    selected_ordinals = surviving
                selected_items = tuple(
                    group.items[ordinal]
                    for ordinal in selected_ordinals
                    if group.items[ordinal] is not None
                )
                bindings.append(InputBinding(spec.name, selected_items))
                for ordinal in selected_ordinals:
                    state = group.states[ordinal]
                    if state in {FAILED, SUPPRESSED} or (
                        input_index != stage.reduce.members_input
                        and state != PRESENT
                    ):
                        cause = group.causes[ordinal]
                        if cause is not None:
                            causes.append(cause)
            else:
                item = accumulator.scalar_inputs.get(input_index)
                record = self.items.get(item)
                if record is None:
                    return
                bindings.append(InputBinding(spec.name, (item,)))
                if record.terminal in {
                    ItemTerminal.FAILED,
                    ItemTerminal.SUPPRESSED,
                } or (
                    spec.mode is InputMode.ONE
                    and record.terminal is ItemTerminal.DROPPED
                ):
                    cause = record.cause or record.producer
                    if cause is not None:
                        causes.append(cause)
        inputs = tuple(bindings)
        if causes:
            self._ensure_suppressed(
                stage,
                inputs,
                accumulator.anchor.entity,
                tuple(causes),
            )
        else:
            self._ensure_executable(
                stage,
                inputs,
                accumulator.anchor.entity,
            )
        self._drop_reduce_accumulator(key)

    def _drop_reduce_accumulator(self, key: tuple[int, EntityId]) -> None:
        accumulator = self.reduce_accumulators.pop(key, None)
        if accumulator is not None:
            self.reduce_slot_count -= sum(
                len(group.states) for group in accumulator.groups.values()
            )

    def _resolve_scope(
        self,
        entity: EntityId,
        expand_stage: int,
    ) -> tuple[EntityId, int]:
        current = entity
        while current in self.entity_origins:
            origin = self.entity_origins[current]
            if origin.expand_stage == expand_stage:
                return origin.parent_entity, origin.ordinal
            current = origin.parent_entity
        raise KeyError(entity)

    def _publish_item(self, record: ItemRecord, location: BlockRow | None = None) -> None:
        existing = self.items.get(record.ref)
        if existing is not None:
            if existing != record:
                self._abort("ItemRef received conflicting terminal records")
            return
        if record.terminal is ItemTerminal.PRESENT:
            if location is None:
                self._abort("PRESENT item has no physical location")
            self.values[record.ref] = location
        elif location is not None:
            self._abort("non-present item cannot have a physical location")
        self.items[record.ref] = record
        self.receipts.append(record.ref)

    def _enqueue(self, record: GrainRecord) -> None:
        queue = self._stage_queue(record.stage)
        if record.id in queue.normal_set:
            return
        if not queue.normal:
            queue.first_wait_at = self.clock()
        queue.normal.append(record.id)
        queue.normal_set.add(record.id)

    def _check_grain_limit(self, additions: int) -> None:
        if len(self.grains) + additions > self.limits.max_grains:
            self._abort("max_grains_per_arena exceeded")

    def _normal_candidates(self, stage: StageSpec) -> tuple[GrainId, ...]:
        queue = self._stage_queue(stage.id)
        if stage.execution.batch_scope == "elastic":
            return tuple(queue.normal)
        groups: dict[EntityId, list[GrainId]] = {}
        order: list[EntityId] = []
        for grain_id in queue.normal:
            record = self.grains[grain_id]
            driving = record.inputs[stage.driving_input].items[0]
            parent = self._nearest_parent_entity(driving.entity)
            if parent not in groups:
                groups[parent] = []
                order.append(parent)
            groups[parent].append(grain_id)
        for parent in order:
            if len(groups[parent]) >= stage.execution.batch_size:
                return tuple(groups[parent])
        return tuple(groups[order[0]]) if order else ()

    def _nearest_parent_entity(self, entity: EntityId) -> EntityId:
        origin = self.entity_origins.get(entity)
        return origin.parent_entity if origin is not None else entity

    def next_deadline(self, now: float | None = None) -> float | None:
        now = self.clock() if now is None else now
        deadlines = []
        for stage_id, queue in self.queues.items():
            if not queue.normal or queue.first_wait_at is None:
                continue
            stage = self.dag.stage(stage_id)
            deadlines.append(
                max(
                    0.0,
                    queue.first_wait_at
                    + stage.execution.max_batch_wait_ms / 1000.0
                    - now,
                )
            )
        return min(deadlines) if deadlines else None

    def reserve_dispatch(
        self,
        stage_id: int,
        now: float | None = None,
    ) -> DispatchIntent | None:
        if len(self.leases) >= self.limits.max_pending_dispatches:
            return None
        stage = self.dag.stage(stage_id)
        if stage.kind is Primitive.SOURCE:
            return None
        assert stage.execution is not None
        now = self.clock() if now is None else now
        queue = self._stage_queue(stage_id)
        task: RecoveryTask | None = None
        flush_reason = "full"
        grain_ids: tuple[GrainId, ...]

        if queue.immediate:
            task = queue.immediate.popleft()
            grain_ids = task.grains
            flush_reason = "recovery"
        elif queue.normal:
            candidates = self._normal_candidates(stage)
            if len(candidates) >= stage.execution.batch_size:
                grain_ids = candidates[: stage.execution.batch_size]
            else:
                deadline = (
                    queue.first_wait_at
                    + stage.execution.max_batch_wait_ms / 1000.0
                    if queue.first_wait_at is not None
                    else now
                )
                if now < deadline:
                    return None
                flush_reason = "timeout"
                grain_ids = candidates[: stage.execution.batch_size]
            selected = set(grain_ids)
            queue.normal = deque(
                grain_id
                for grain_id in queue.normal
                if grain_id not in selected
            )
            queue.normal_set.difference_update(selected)
            queue.first_wait_at = self.clock() if queue.normal else None
        elif queue.tail and self._normal_work_empty():
            task = queue.tail.popleft()
            grain_ids = task.grains
            flush_reason = "tail_recovery"
        else:
            return None

        records = [self.grains[grain_id] for grain_id in grain_ids]
        input_blocks: list[int] = []
        block_slots: dict[int, int] = {}
        input_takes: list[tuple[InputTake, ...]] = []
        dispatch = self.next_dispatch
        self.next_dispatch += 1

        for record in records:
            takes: list[InputTake] = []
            for input_index, (spec, binding) in enumerate(
                zip(stage.inputs, record.inputs)
            ):
                if spec.mode is InputMode.ANCHOR:
                    continue
                if (
                    spec.mode is InputMode.OPTIONAL_ONE
                    and self.items[binding.items[0]].terminal
                    is ItemTerminal.DROPPED
                ):
                    takes.append(MissingTake())
                    continue
                rows: list[RowTake] = []
                locations = [self.values.get(item) for item in binding.items]
                if any(location is None for location in locations):
                    self._abort("executable Grain input has no value")
                for item, location in zip(binding.items, locations):
                    assert location is not None
                    slot = block_slots.get(location.block)
                    if slot is None:
                        slot = len(input_blocks)
                        block_slots[location.block] = slot
                        input_blocks.append(location.block)
                    rows.append(RowTake(slot, location.row))
                takes.append(ValueTake(tuple(rows)))
            input_takes.append(tuple(takes))

        invocations = []
        for record, takes in zip(records, input_takes):
            token = record.reserve(self.id, dispatch)
            invocations.append(Invocation(token, takes))
        call = BatchCall(dispatch, stage_id, tuple(invocations))
        self.leases[dispatch] = DispatchLease(
            call,
            tuple(input_blocks),
            grain_ids,
            recovery=task,
            flush_reason=flush_reason,
        )
        self.dispatch_count += 1
        self.dispatched_grains += len(grain_ids)
        self.dispatch_capacity += stage.execution.batch_size
        if (
            len(grain_ids) < stage.execution.batch_size
            or task is not None
        ):
            self.tail_or_recovery_dispatches += 1
        return DispatchIntent(
            self.id,
            call,
            tuple(self.blocks[block] for block in input_blocks),
            actor_policy=task.actor_policy if task else "any",
            avoid_worker_slot=task.avoid_worker_slot if task else None,
            flush_reason=flush_reason,
        )

    def _normal_work_empty(self) -> bool:
        return all(not queue.normal and not queue.immediate for queue in self.queues.values())

    def commit(self, completion: DispatchCompletion) -> None:
        if completion.arena_id != self.id:
            self._abort("completion targets another Arena")
        lease = self.leases.get(completion.call.dispatch)
        if lease is None or lease.call != completion.call:
            return
        stage = self.dag.stage(completion.call.stage)
        report = completion.report
        if report.dispatch != completion.call.dispatch:
            self._abort("report dispatch mismatch")
        if tuple(ack.token for ack in report.acks) != tuple(
            invocation.token for invocation in completion.call.invocations
        ):
            self._abort("report AttemptTokens mismatch")
        if len(report.acks) != len(completion.call.invocations):
            self._abort("report ack count mismatch")
        current = tuple(
            self.grains[invocation.token.grain].active_attempt
            == invocation.token
            for invocation in completion.call.invocations
        )
        if not any(current):
            self.leases.pop(completion.call.dispatch, None)
            return
        if not all(current):
            self._abort("dispatch has mixed current/stale attempts")

        if stage.kind is Primitive.FILTER:
            if completion.output_blocks or report.column_lengths:
                self._abort("Filter must not return business output blocks")
            if any(ack.keep is None or ack.output_counts is not None for ack in report.acks):
                self._abort("Filter report shape is invalid")
            self._commit_filter(stage, lease, report)
        else:
            if len(completion.output_blocks) != stage.output_count:
                self._abort("output block arity mismatch")
            if len(report.column_lengths) != stage.output_count:
                self._abort("output column length arity mismatch")
            if any(ack.output_counts is None or ack.keep is not None for ack in report.acks):
                self._abort("value-producing report shape is invalid")
            self._commit_values(stage, lease, report, completion.output_blocks)
        self.timeline.append(
            DispatchTimeline(
                arena=self.id,
                stage=stage.id,
                dispatch=completion.call.dispatch,
                worker_slot=completion.worker_slot,
                grains=len(completion.call.invocations),
                flush_reason=lease.flush_reason,
                submitted_at=completion.submitted_at or 0.0,
                report_received_at=completion.report_received_at or self.clock(),
                committed_at=self.clock(),
                worker_started_at=report.worker_started_at,
                worker_finished_at=report.worker_finished_at,
                worker_rss_bytes=report.worker_rss_bytes,
                status="accepted",
            )
        )
        self.leases.pop(completion.call.dispatch, None)

    def _commit_filter(
        self,
        stage: StageSpec,
        lease: DispatchLease,
        report: BatchReport,
    ) -> None:
        for record, invocation, ack in zip(
            (self.grains[grain_id] for grain_id in lease.grain_ids),
            lease.call.invocations,
            report.acks,
        ):
            assert ack.keep is not None
            emissions = []
            for output_index, (port, binding) in enumerate(
                zip(stage.output_ports(), record.inputs)
            ):
                input_item = binding.items[0]
                output_item = ItemRef(port, input_item.entity)
                if ack.keep:
                    self._publish_item(
                        ItemRecord(
                            output_item,
                            record.id,
                            ItemTerminal.PRESENT,
                        ),
                        self.values[input_item],
                    )
                    emissions.append((Emission(output_item, 0),))
                else:
                    self._publish_item(
                        ItemRecord(
                            output_item,
                            record.id,
                            ItemTerminal.DROPPED,
                            record.id,
                        )
                    )
                    emissions.append(())
            if not record.seal(Success(tuple(emissions)), invocation.token):
                continue

    def _commit_values(
        self,
        stage: StageSpec,
        lease: DispatchLease,
        report: BatchReport,
        output_blocks: tuple[Any, ...],
    ) -> None:
        block_ids = tuple(self._allocate_block(block) for block in output_blocks)
        cursors = [0] * stage.output_count
        prepared: list[tuple[GrainRecord, AttemptToken, Success]] = []
        for grain_id, invocation, ack in zip(
            lease.grain_ids,
            lease.call.invocations,
            report.acks,
        ):
            record = self.grains[grain_id]
            counts = ack.output_counts
            assert counts is not None
            if len(counts) != stage.output_count:
                self._abort("output count arity mismatch")
            if stage.kind in {Primitive.MAP, Primitive.REDUCE} and any(
                count != 1 for count in counts
            ):
                self._abort(f"{stage.kind.value} must emit one row per port")
            if stage.kind is Primitive.EXPAND:
                if len(set(counts)) != 1:
                    self._abort("Expand output ports must share cardinality")
                if counts[0] > self.limits.max_fanout_per_grain:
                    self._abort("max_fanout_per_grain exceeded")

            if stage.kind is Primitive.EXPAND:
                parent = record.inputs[stage.driving_input].items[0]
                entities = tuple(
                    expand_entity(
                        self.run_salt,
                        stage.id,
                        parent.entity,
                        ordinal,
                    )
                    for ordinal in range(counts[0])
                )
            else:
                parent = None
                entities = ()

            emissions_by_port = []
            for output_index, (port, count) in enumerate(
                zip(stage.output_ports(), counts)
            ):
                emissions = []
                for ordinal in range(count):
                    entity = (
                        entities[ordinal]
                        if stage.kind is Primitive.EXPAND
                        else self._output_entity(record, stage)
                    )
                    item = ItemRef(port, entity)
                    location = BlockRow(
                        block_ids[output_index],
                        cursors[output_index] + ordinal,
                    )
                    self._publish_item(
                        ItemRecord(item, record.id, ItemTerminal.PRESENT),
                        location,
                    )
                    emissions.append(Emission(item, ordinal))
                    if stage.kind is Primitive.EXPAND:
                        assert parent is not None
                        origin = EntityOrigin(
                            parent.entity,
                            stage.id,
                            record.id,
                            ordinal,
                        )
                        existing = self.entity_origins.get(entity)
                        if existing is not None and existing != origin:
                            self._abort("child Entity has conflicting origin")
                        self.entity_origins[entity] = origin
                cursors[output_index] += count
                emissions_by_port.append(tuple(emissions))
            prepared.append(
                (
                    record,
                    invocation.token,
                    Success(tuple(emissions_by_port)),
                )
            )
            if stage.kind is Primitive.EXPAND:
                assert parent is not None
                instance = ExpandInstance(
                    record.id,
                    stage.id,
                    parent,
                    counts[0],
                )
                self.expand_instances[(stage.id, parent.entity)] = instance
                self._activate_reduce_placeholders(instance)
        if tuple(cursors) != report.column_lengths:
            self._abort("output counts do not cover output blocks")
        for record, token, outcome in prepared:
            record.seal(outcome, token)

    def _output_entity(self, record: GrainRecord, stage: StageSpec) -> EntityId:
        if stage.kind is Primitive.REDUCE:
            return next(
                binding.items[0].entity
                for spec, binding in zip(stage.inputs, record.inputs)
                if spec.mode is InputMode.ANCHOR
            )
        assert stage.driving_input is not None
        return record.inputs[stage.driving_input].items[0].entity

    def handle_failure(self, failure: DispatchFailure) -> None:
        lease = self.leases.get(failure.dispatch)
        if lease is None:
            return
        stage = self.dag.stage(lease.call.stage)
        if failure.kind is FailureKind.CONTRACT_ABORT:
            self._abort(failure.message)
        if failure.kind is FailureKind.BAD_RECORD:
            self._handle_bad_record(lease, failure)
        elif failure.kind is FailureKind.INFRA_FAILURE:
            self._handle_infra_failure(stage, lease, failure)
        else:
            self._handle_udf_error(stage, lease, failure)
        self.leases.pop(failure.dispatch, None)

    def _handle_bad_record(
        self,
        lease: DispatchLease,
        failure: DispatchFailure,
    ) -> None:
        if failure.bad_token is None:
            self._abort("BAD_RECORD failure has no token")
        matched = False
        for invocation in lease.call.invocations:
            record = self.grains[invocation.token.grain]
            if invocation.token == failure.bad_token:
                matched = True
                outcome = Failed(
                    GrainFailure("bad_record", failure.message, ())
                )
                record.seal(outcome, invocation.token)
                self._publish_failed_outputs(record, outcome.failure)
            else:
                record.release(invocation.token)
                self._enqueue(record)
        if not matched:
            self._abort("BAD_RECORD token is not in failed dispatch")

    def _handle_infra_failure(
        self,
        stage: StageSpec,
        lease: DispatchLease,
        failure: DispatchFailure,
    ) -> None:
        assert stage.execution is not None
        limit = stage.execution.recovery.limits.max_infra_retries
        for invocation in lease.call.invocations:
            record = self.grains[invocation.token.grain]
            if record.infra_failures + 1 > limit:
                self._abort("infrastructure retry budget exhausted")
            record.release(invocation.token, infrastructure=True)
        task = RecoveryTask(
            stage.id,
            lease.grain_ids,
            stage.execution.recovery.preset,
            actor_policy="fresh",
            avoid_worker_slot=failure.worker_slot,
        )
        self._stage_queue(stage.id).immediate.append(task)

    def _handle_udf_error(
        self,
        stage: StageSpec,
        lease: DispatchLease,
        failure: DispatchFailure,
    ) -> None:
        del failure
        assert stage.execution is not None
        preset = stage.execution.recovery.preset
        for invocation in lease.call.invocations:
            self.grains[invocation.token.grain].release(invocation.token)
        if preset is RecoveryPreset.RAISE:
            self._abort("UDF error")
        if preset is RecoveryPreset.FAIL_BATCH:
            for grain_id in lease.grain_ids:
                record = self.grains[grain_id]
                outcome = Failed(GrainFailure("udf_error", "batch failed", ()))
                record.seal(outcome)
                self._publish_failed_outputs(record, outcome.failure)
            return
        task = lease.recovery or RecoveryTask(
            stage.id,
            lease.grain_ids,
            preset,
        )
        task.attempts += 1
        task.budget.extra_rpcs += 1
        task.budget.reexecuted_grains += len(task.grains)
        limits = stage.execution.recovery.limits
        if (
            task.attempts > limits.max_recovery_attempts
            or task.budget.extra_rpcs > limits.max_extra_rpcs
            or task.budget.reexecuted_grains > limits.max_reexecuted_grains
        ):
            self._abort("UDF recovery budget exhausted")
        queue = self._stage_queue(stage.id)
        if preset is RecoveryPreset.RETRY_BATCH:
            queue.immediate.append(task)
        elif preset is RecoveryPreset.RETRY_TAIL:
            queue.tail.append(task)
        else:
            if lease.recovery is None:
                task.phase = "retry"
                queue.tail.append(task)
            else:
                self._schedule_split(stage, task)

    def _schedule_split(self, stage: StageSpec, task: RecoveryTask) -> None:
        limits = stage.execution.recovery.limits
        if len(task.grains) == 1:
            record = self.grains[task.grains[0]]
            outcome = Failed(GrainFailure("udf_error", "isolated record", ()))
            record.seal(outcome)
            self._publish_failed_outputs(record, outcome.failure)
            return
        if task.depth >= limits.max_split_depth:
            self._abort("isolation split depth exhausted")
        midpoint = len(task.grains) // 2
        queue = self._stage_queue(stage.id)
        for group in (task.grains[:midpoint], task.grains[midpoint:]):
            child = RecoveryTask(
                stage.id,
                group,
                task.preset,
                attempts=task.attempts,
                depth=task.depth + 1,
                budget=task.budget,
                phase="split",
            )
            queue.tail.append(child)

    def _publish_failed_outputs(
        self,
        record: GrainRecord,
        failure: GrainFailure,
    ) -> None:
        stage = self.dag.stage(record.stage)
        if stage.kind is Primitive.EXPAND:
            parent = record.inputs[stage.driving_input].items[0]
            instance = ExpandInstance(
                record.id,
                stage.id,
                parent,
                None,
                failed=True,
            )
            self.expand_instances[(stage.id, parent.entity)] = instance
            self._activate_reduce_placeholders(instance)
        else:
            entity = self._output_entity(record, stage)
            for port in stage.output_ports():
                self._publish_item(
                    ItemRecord(
                        ItemRef(port, entity),
                        record.id,
                        ItemTerminal.FAILED,
                        record.id,
                    )
                )
        self.failure_count += 1

    def is_complete(self, pending_rpc_count: int = 0) -> bool:
        return (
            self.admission_closed
            and not self.receipts
            and not self.pending_invocations
            and not self.reduce_accumulators
            and all(
                not queue.normal
                and not queue.immediate
                and not queue.tail
                for queue in self.queues.values()
            )
            and not self.leases
            and pending_rpc_count == 0
            and all(
                record.phase is GrainPhase.SEALED
                for record in self.grains.values()
            )
        )

    def finish(self) -> ArenaResult:
        if not self.is_complete():
            self._abort("cannot finish incomplete Arena")
        source_order = {
            snapshot.item.entity: index
            for index, snapshot in enumerate(self.sources)
            if snapshot.item.port == self.dag.source_ports[0]
        }
        output_items = [
            item
            for port in self.dag.output_ports
            for item, record in self.items.items()
            if item.port == port and record.terminal is ItemTerminal.PRESENT
        ]
        output_items.sort(
            key=lambda item: (
                self.dag.output_ports.index(item.port),
                *self._entity_order_key(item.entity, source_order),
            )
        )
        outputs = tuple(
            BlockSlice(self.blocks[self.values[item].block], self.values[item].row)
            for item in output_items
        )
        failures = tuple(
            FailureSnapshot(record.id, record.outcome.failure)
            for record in self.grains.values()
            if isinstance(record.outcome, Failed)
        )
        suppressions = tuple(
            SuppressionSnapshot(record.id, record.outcome.direct_causes)
            for record in self.grains.values()
            if isinstance(record.outcome, Suppressed)
        )
        result = ArenaResult(
            outputs,
            failures,
            suppressions,
            tuple(self.sources),
            {
                "rpc_count": float(self.dispatch_count),
                "grains_per_rpc": (
                    self.dispatched_grains / self.dispatch_count
                    if self.dispatch_count
                    else 0.0
                ),
                "batch_fill_ratio": (
                    self.dispatched_grains / self.dispatch_capacity
                    if self.dispatch_capacity
                    else 0.0
                ),
                "tail_or_recovery_rpc_fraction": (
                    self.tail_or_recovery_dispatches / self.dispatch_count
                    if self.dispatch_count
                    else 0.0
                ),
                "live_blocks_at_delivery": float(len(self.blocks)),
                "reduce_slots": float(self.reduce_slot_count),
            },
            tuple(self.timeline),
        )
        self._reclaim()
        return result

    def _entity_order_key(
        self,
        entity: EntityId,
        source_order: dict[EntityId, int],
    ) -> tuple[int, ...]:
        ordinals: list[int] = []
        current = entity
        while current in self.entity_origins:
            origin = self.entity_origins[current]
            ordinals.append(origin.ordinal)
            current = origin.parent_entity
        ordinals.reverse()
        return (source_order.get(current, len(source_order)), *ordinals)

    def _reclaim(self) -> None:
        self.grains.clear()
        self.items.clear()
        self.entity_origins.clear()
        self.expand_instances.clear()
        self.pending_invocations.clear()
        self.reduce_accumulators.clear()
        self.values.clear()
        self.blocks.clear()
        self.receipts.clear()
        self.queues.clear()
        self.leases.clear()
        self.timeline.clear()
