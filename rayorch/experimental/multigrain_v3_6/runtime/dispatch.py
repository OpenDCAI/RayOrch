"""Microbatch-owned physical dispatch and Grain lifecycle state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..model import CallRef, EntityRef, GrainPhase, GrainRef
from ..recovery import RecoveryAction
from ..transitions import GrainEvent, grain_transition
from .state import CommitError


@dataclass(slots=True)
class GrainRecord:
    """DispatchState 私有的可变 Grain 执行记录。"""

    phase: GrainPhase
    generation: int = 0
    infra_failures: int = 0


@dataclass(frozen=True, slots=True)
class GrainSnapshot:
    """跨组件诊断可见的不可变 Grain 状态副本。"""

    phase: GrainPhase
    generation: int
    infra_failures: int


@dataclass(frozen=True, slots=True)
class DispatchBatch:
    """One exact normal or recovery group reserved by a microbatch."""

    grains: tuple[GrainRef, ...]
    udf_retries: int = 0

    def __post_init__(self) -> None:
        if not self.grains:
            raise ValueError("DispatchBatch requires a non-empty Grain group")
        if type(self.udf_retries) is not int or self.udf_retries < 0:
            raise ValueError("udf_retries must be a non-negative integer")
        if any(grain.call != self.grains[0].call for grain in self.grains):
            raise ValueError("one DispatchBatch cannot mix Calls")


@dataclass(frozen=True, slots=True)
class _ReadyEntry:
    """Transient normal-queue entry with its precomputed packing key."""

    grain: GrainRef
    batch_key: EntityRef


class DispatchState:
    """The sole owner of runnable queues and mutable Grain execution state."""

    def __init__(self) -> None:
        self._records: dict[GrainRef, GrainRecord] = {}
        self._normal: deque[_ReadyEntry] = deque()
        self._immediate: deque[DispatchBatch] = deque()
        self._tail: deque[DispatchBatch] = deque()

    @property
    def ready_count(self) -> int:
        return (
            len(self._normal)
            + sum(len(selection.grains) for selection in self._immediate)
            + sum(len(selection.grains) for selection in self._tail)
        )

    @property
    def grain_count(self) -> int:
        return len(self._records)

    @property
    def is_idle(self) -> bool:
        return not (self._normal or self._immediate or self._tail)

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

    def inputs_ready(self, grain: GrainRef, batch_key: EntityRef) -> None:
        """Create one READY Grain and enqueue it exactly once."""

        self._create(grain, GrainEvent.INPUTS_READY)
        self._normal.append(_ReadyEntry(grain, batch_key))

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
        """Return immediate/normal/tail priority for one Call."""

        if any(item.grains[0].call == call for item in self._immediate):
            return 0
        if any(
            entry.grain.call == call and self._is_ready(entry.grain)
            for entry in self._normal
        ):
            return 1
        if any(item.grains[0].call == call for item in self._tail):
            return 2
        return None

    def reserve(
        self,
        call: CallRef,
        *,
        max_size: int,
        parent_bound: bool,
    ) -> DispatchBatch | None:
        """Reserve immediate recovery, normal work, then deferred recovery."""

        selection = self._reserve_recovery(self._immediate, call)
        if selection is not None:
            return selection
        grains = self._reserve_normal(
            call,
            max_size=max_size,
            parent_bound=parent_bound,
        )
        if grains:
            return DispatchBatch(grains)
        selection = self._reserve_recovery(self._tail, call)
        if selection is not None:
            return selection
        return None

    def _reserve_normal(
        self,
        call: CallRef,
        *,
        max_size: int,
        parent_bound: bool,
    ) -> tuple[GrainRef, ...]:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        selected: list[GrainRef] = []
        remaining: deque[_ReadyEntry] = deque()
        parent: EntityRef | None = None
        while self._normal:
            entry = self._normal.popleft()
            grain = entry.grain
            if not self._is_ready(grain):
                continue
            if grain.call != call or len(selected) >= max_size:
                remaining.append(entry)
                continue
            candidate_parent = entry.batch_key
            if parent_bound and parent is not None and candidate_parent != parent:
                remaining.append(entry)
                continue
            if parent is None:
                parent = candidate_parent
            self._reserve_exact((grain,))
            selected.append(grain)
        self._normal = remaining
        return tuple(selected)

    def _reserve_recovery(
        self,
        queue: deque[DispatchBatch],
        call: CallRef,
    ) -> DispatchBatch | None:
        for index, selection in enumerate(queue):
            if selection.grains[0].call == call:
                self._reserve_exact(selection.grains)
                del queue[index]
                return selection
        return None

    def _reserve_exact(self, grains: tuple[GrainRef, ...]) -> None:
        self._transition(grains, GrainPhase.READY, GrainEvent.RESERVE)

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

    def recover_udf(
        self,
        selection: DispatchBatch,
        action: RecoveryAction,
    ) -> int:
        """Apply one non-terminal UDF recovery decision; return retried Grains."""

        grains = selection.grains
        match action:
            case RecoveryAction.RETRY_IMMEDIATE | RecoveryAction.RETRY_TAIL:
                self._release(grains, infrastructure=False)
                queue = (
                    self._immediate
                    if action is RecoveryAction.RETRY_IMMEDIATE
                    else self._tail
                )
                queue.append(DispatchBatch(grains, selection.udf_retries + 1))
            case RecoveryAction.SPLIT_TAIL:
                if selection.udf_retries == 0 or len(grains) <= 1:
                    raise CommitError("split requires one failed recovery group")
                self._release(grains, infrastructure=False)
                midpoint = len(grains) // 2
                for group in (grains[:midpoint], grains[midpoint:]):
                    self._tail.append(
                        DispatchBatch(group, selection.udf_retries)
                    )
            case RecoveryAction.ABORT | RecoveryAction.FAIL_SINGLETON:
                raise CommitError(
                    f"action is not a dispatch retry: {action!r}"
                )
        return len(grains)

    def recover_infrastructure(
        self,
        selection: DispatchBatch,
    ) -> int:
        """Requeue an already-approved exact group and return its Grain count."""

        grains = selection.grains
        self._release(grains, infrastructure=True)
        self._immediate.append(selection)
        return len(grains)

    def infrastructure_failures(
        self,
        selection: DispatchBatch,
    ) -> tuple[int, ...]:
        """Read the sole physical infra-attempt counters for policy reduction."""

        records = tuple(self._records.get(grain) for grain in selection.grains)
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
        """Preflight then atomically transition one exact Grain group."""

        if not grains:
            raise CommitError("dispatch group must not be empty")
        optional = tuple(self._records.get(grain) for grain in grains)
        if any(
            record is None or record.phase is not expected
            for record in optional
        ):
            raise CommitError(f"dispatch requires {expected.name} Grains")
        records = tuple(record for record in optional if record is not None)
        for record in records:
            record.phase = grain_transition(record.phase, event)
        return records

    def _is_ready(self, grain: GrainRef) -> bool:
        record = self._records.get(grain)
        return record is not None and record.phase is GrainPhase.READY

    def _record(self, grain: GrainRef) -> GrainRecord:
        try:
            return self._records[grain]
        except KeyError as error:
            raise CommitError(f"unknown Grain: {grain!r}") from error


__all__ = ["DispatchBatch", "DispatchState", "GrainSnapshot"]
