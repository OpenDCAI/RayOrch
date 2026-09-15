"""Microbatch-owned physical dispatch and Grain lifecycle state machine."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import AbstractSet, Mapping

from ..model import CallRef, EntityRef, GrainPhase, GrainRef
from ..recovery import RecoveryAction
from .transitions import GrainEvent, grain_transition
from .state import CommitError


@dataclass(slots=True)
class GrainRecord:
    """Mutable Grain execution state private to ``DispatchState``."""

    phase: GrainPhase
    generation: int = 0
    infra_failures: int = 0
    parent_anchor: EntityRef | None = None


@dataclass(frozen=True, slots=True)
class GrainSnapshot:
    """Immutable Grain state exposed for cross-component diagnostics."""

    phase: GrainPhase
    generation: int
    infra_failures: int


@dataclass(frozen=True, slots=True)
class DispatchBatch:
    """One exact READY or recovery batch reserved for a Worker RPC."""

    grains: tuple[GrainRef, ...]
    udf_retries: int = 0

    def __post_init__(self) -> None:
        if not self.grains:
            raise ValueError("DispatchBatch requires at least one Grain")
        if type(self.udf_retries) is not int or self.udf_retries < 0:
            raise ValueError("udf_retries must be a non-negative integer")
        if any(grain.call != self.grains[0].call for grain in self.grains):
            raise ValueError("one DispatchBatch cannot mix Calls")


class DispatchState:
    """Own runnable queues and every mutable Grain execution record.

    A Grain enters its Call's READY queue as soon as the engine observes all
    required inputs. Reserving directly from these queues makes dispatch
    completion-driven: downstream work need not wait for an upstream stage or
    input domain to finish globally.
    """

    def __init__(self) -> None:
        self._records: dict[GrainRef, GrainRecord] = {}
        # Per-Call queues avoid scanning unrelated fan-out while preserving the
        # order in which dependency completion made Grains runnable.
        self._ready_by_call: dict[CallRef, deque[GrainRef]] = defaultdict(deque)
        self._immediate_retry_queue: deque[DispatchBatch] = deque()
        self._deferred_recovery_queue: deque[DispatchBatch] = deque()

    @property
    def ready_count(self) -> int:
        return (
            sum(len(queue) for queue in self._ready_by_call.values())
            + sum(len(batch.grains) for batch in self._immediate_retry_queue)
            + sum(len(batch.grains) for batch in self._deferred_recovery_queue)
        )

    @property
    def grain_count(self) -> int:
        return len(self._records)

    @property
    def is_idle(self) -> bool:
        return not (
            any(self._ready_by_call.values())
            or self._immediate_retry_queue
            or self._deferred_recovery_queue
        )

    @property
    def all_sealed(self) -> bool:
        return all(
            record.phase is GrainPhase.SEALED
            for record in self._records.values()
        )

    def contains(self, grain: GrainRef) -> bool:
        """Return whether this physical state machine owns the Grain."""

        return grain in self._records

    def generation(self, grain: GrainRef) -> int:
        """Return the current fencing generation without exposing its record."""

        return self._record(grain).generation

    def parent_anchor(self, grain: GrainRef) -> EntityRef:
        """Return the direct parent, or the root Entity itself, frozen at READY."""

        anchor = self._record(grain).parent_anchor
        if anchor is None:
            raise CommitError("Grain has no parent anchor")
        return anchor

    def snapshot(self, grain: GrainRef) -> GrainSnapshot:
        """Copy one mutable record into an immutable diagnostic DTO."""

        record = self._record(grain)
        return GrainSnapshot(
            record.phase,
            record.generation,
            record.infra_failures,
        )

    def snapshots(self) -> Mapping[GrainRef, GrainSnapshot]:
        """Return an immutable point-in-time copy of every Grain record."""

        return MappingProxyType(
            {grain: self.snapshot(grain) for grain in self._records}
        )

    def inputs_ready(self, grain: GrainRef, parent_anchor: EntityRef) -> None:
        """Create one READY Grain and append it to its Call queue exactly once."""

        self._create(grain, GrainEvent.INPUTS_READY)
        self._record(grain).parent_anchor = parent_anchor
        self._ready_by_call[grain.call].append(grain)

    def inputs_terminal(self, grain: GrainRef) -> None:
        """Create one dependency-terminal Grain directly as SEALED."""

        self._create(grain, GrainEvent.INPUTS_TERMINAL)

    def _create(
        self,
        grain: GrainRef,
        event: GrainEvent,
    ) -> None:
        if grain in self._records:
            raise CommitError(f"Grain has already been created: {grain!r}")
        self._records[grain] = GrainRecord(grain_transition(None, event))

    def priority(self, call: CallRef) -> int | None:
        """Return immediate-retry/ready/deferred-recovery priority for one Call."""

        if any(
            batch.grains[0].call == call
            for batch in self._immediate_retry_queue
        ):
            return 0
        if self._ready_by_call.get(call):
            return 1
        if any(
            batch.grains[0].call == call
            for batch in self._deferred_recovery_queue
        ):
            return 2
        return None

    def reserve_with_barriers(
        self,
        call: CallRef,
        *,
        max_size: int,
        barriered_anchors: AbstractSet[EntityRef] = frozenset(),
    ) -> tuple[DispatchBatch | None, tuple[GrainRef, ...]]:
        """Reserve live work and atomically seal encountered barriered READY work."""

        suppressed: list[GrainRef] = []
        dispatch_batch, barriered = self._reserve_recovery(
            self._immediate_retry_queue,
            call,
            barriered_anchors,
        )
        suppressed.extend(barriered)
        if dispatch_batch is not None:
            return dispatch_batch, tuple(suppressed)
        grains, barriered = self._reserve_ready(
            call,
            max_size=max_size,
            barriered_anchors=barriered_anchors,
        )
        suppressed.extend(barriered)
        if grains:
            return DispatchBatch(grains), tuple(suppressed)
        dispatch_batch, barriered = self._reserve_recovery(
            self._deferred_recovery_queue,
            call,
            barriered_anchors,
        )
        suppressed.extend(barriered)
        if dispatch_batch is not None:
            return dispatch_batch, tuple(suppressed)
        return None, tuple(suppressed)

    def _reserve_ready(
        self,
        call: CallRef,
        *,
        max_size: int,
        barriered_anchors: AbstractSet[EntityRef],
    ) -> tuple[tuple[GrainRef, ...], tuple[GrainRef, ...]]:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        queue = self._ready_by_call.get(call)
        if not queue:
            return (), ()

        selected: list[GrainRef] = []
        suppressed: list[GrainRef] = []
        while queue and len(selected) < max_size:
            grain = queue.popleft()
            if not self._is_ready(grain):
                continue
            if self.parent_anchor(grain) in barriered_anchors:
                self._suppress_ready((grain,))
                suppressed.append(grain)
                continue
            self._reserve_exact((grain,))
            selected.append(grain)
        if not queue:
            self._ready_by_call.pop(call, None)
        return tuple(selected), tuple(suppressed)

    def _reserve_recovery(
        self,
        queue: deque[DispatchBatch],
        call: CallRef,
        barriered_anchors: AbstractSet[EntityRef],
    ) -> tuple[DispatchBatch | None, tuple[GrainRef, ...]]:
        suppressed: list[GrainRef] = []
        index = 0
        while index < len(queue):
            dispatch_batch = queue[index]
            if dispatch_batch.grains[0].call != call:
                index += 1
                continue
            del queue[index]
            live, barriered = self._partition_ready(dispatch_batch, barriered_anchors)
            if barriered:
                self._suppress_ready(barriered)
                suppressed.extend(barriered)
            if live is not None:
                self._reserve_exact(live.grains)
                return live, tuple(suppressed)
        return None, tuple(suppressed)

    def _partition_ready(
        self,
        dispatch_batch: DispatchBatch,
        barriered_anchors: AbstractSet[EntityRef],
    ) -> tuple[DispatchBatch | None, tuple[GrainRef, ...]]:
        return self._partition(
            dispatch_batch,
            barriered_anchors,
            expected=GrainPhase.READY,
        )

    def _reserve_exact(self, grains: tuple[GrainRef, ...]) -> None:
        self._transition(grains, GrainPhase.READY, GrainEvent.RESERVE)

    def _suppress_ready(self, grains: tuple[GrainRef, ...]) -> None:
        self._transition(grains, GrainPhase.READY, GrainEvent.SUPPRESS)

    def validate_in_flight(
        self,
        grain: GrainRef,
        generation: int | None,
    ) -> None:
        """Validate a generation-fenced report without mutating state."""

        record = self._records.get(grain)
        if record is None or record.phase is not GrainPhase.IN_FLIGHT:
            raise CommitError("report requires one IN_FLIGHT Grain")
        if generation is not None and generation != record.generation:
            raise CommitError("stale generation")

    def seal(self, grain: GrainRef, generation: int | None) -> None:
        self.validate_in_flight(grain, generation)
        record = self._record(grain)
        record.phase = grain_transition(record.phase, GrainEvent.REPORT)

    def seal_in_flight(self, grains: tuple[GrainRef, ...]) -> None:
        """Atomically seal an already-preflighted exact in-flight subset."""

        if grains:
            self._transition(grains, GrainPhase.IN_FLIGHT, GrainEvent.REPORT)

    def partition_in_flight(
        self,
        dispatch_batch: DispatchBatch,
        barriered_anchors: AbstractSet[EntityRef],
    ) -> tuple[DispatchBatch | None, tuple[GrainRef, ...]]:
        """Partition an in-flight batch by the current suppression barriers."""

        return self._partition(
            dispatch_batch,
            barriered_anchors,
            expected=GrainPhase.IN_FLIGHT,
        )

    def _partition(
        self,
        dispatch_batch: DispatchBatch,
        barriered_anchors: AbstractSet[EntityRef],
        *,
        expected: GrainPhase,
    ) -> tuple[DispatchBatch | None, tuple[GrainRef, ...]]:
        self._records_in_phase(dispatch_batch.grains, expected)
        live = tuple(
            grain
            for grain in dispatch_batch.grains
            if self.parent_anchor(grain) not in barriered_anchors
        )
        barriered = tuple(
            grain
            for grain in dispatch_batch.grains
            if self.parent_anchor(grain) in barriered_anchors
        )
        return (
            DispatchBatch(live, dispatch_batch.udf_retries) if live else None,
            barriered,
        )

    def recover_udf(
        self,
        dispatch_batch: DispatchBatch,
        action: RecoveryAction,
    ) -> int:
        """Apply one non-terminal UDF recovery decision; return retried Grains."""

        grains = dispatch_batch.grains
        match action:
            case RecoveryAction.RETRY_IMMEDIATE | RecoveryAction.RETRY_TAIL:
                self._release(grains, infrastructure=False)
                queue = (
                    self._immediate_retry_queue
                    if action is RecoveryAction.RETRY_IMMEDIATE
                    else self._deferred_recovery_queue
                )
                queue.append(DispatchBatch(grains, dispatch_batch.udf_retries + 1))
            case RecoveryAction.SPLIT_TAIL:
                if dispatch_batch.udf_retries == 0 or len(grains) <= 1:
                    raise CommitError("split requires one failed recovery batch")
                self._release(grains, infrastructure=False)
                midpoint = len(grains) // 2
                for partition in (grains[:midpoint], grains[midpoint:]):
                    self._deferred_recovery_queue.append(
                        DispatchBatch(partition, dispatch_batch.udf_retries)
                    )
            case RecoveryAction.ABORT | RecoveryAction.FAIL_SINGLETON:
                raise CommitError(
                    f"action is not a dispatch retry: {action!r}"
                )
        return len(grains)

    def recover_infrastructure(
        self,
        dispatch_batch: DispatchBatch,
    ) -> int:
        """Requeue an approved DispatchBatch and return its Grain count."""

        grains = dispatch_batch.grains
        self._release(grains, infrastructure=True)
        self._immediate_retry_queue.append(dispatch_batch)
        return len(grains)

    def infrastructure_failures(
        self,
        dispatch_batch: DispatchBatch,
    ) -> tuple[int, ...]:
        """Read the sole physical infra-attempt counters for policy reduction."""

        records = tuple(self._records.get(grain) for grain in dispatch_batch.grains)
        if any(record is None for record in records):
            raise CommitError("infrastructure retry references an unknown Grain")
        return tuple(
            record.infra_failures
            for record in records
            if record is not None
        )

    def _release(
        self,
        grains: tuple[GrainRef, ...],
        *,
        infrastructure: bool,
    ) -> None:
        records = self._transition(
            grains,
            GrainPhase.IN_FLIGHT,
            GrainEvent.RETRY,
        )
        for record in records:
            record.generation += 1
            if infrastructure:
                record.infra_failures += 1

    def _transition(
        self,
        grains: tuple[GrainRef, ...],
        expected: GrainPhase,
        event: GrainEvent,
    ) -> tuple[GrainRecord, ...]:
        """Preflight then transition every Grain in one exact DispatchBatch."""

        if not grains:
            raise CommitError("DispatchBatch must not be empty")
        records = self._records_in_phase(grains, expected)
        for record in records:
            record.phase = grain_transition(record.phase, event)
        return records

    def _records_in_phase(
        self,
        grains: tuple[GrainRef, ...],
        expected: GrainPhase,
    ) -> tuple[GrainRecord, ...]:
        if not grains:
            raise CommitError("DispatchBatch must not be empty")
        optional = tuple(self._records.get(grain) for grain in grains)
        if any(
            record is None or record.phase is not expected
            for record in optional
        ):
            raise CommitError(f"dispatch requires {expected.name} Grains")
        return tuple(record for record in optional if record is not None)

    def _is_ready(self, grain: GrainRef) -> bool:
        record = self._records.get(grain)
        return record is not None and record.phase is GrainPhase.READY

    def _record(self, grain: GrainRef) -> GrainRecord:
        try:
            return self._records[grain]
        except KeyError as error:
            raise CommitError(f"unknown Grain: {grain!r}") from error


__all__ = ["DispatchBatch", "DispatchState", "GrainSnapshot"]
