"""Small Arena-local state records.

These records contain no orchestration logic and no Ray handles.  Keeping them
separate makes the large ArenaEngine reviewable without introducing mixins or
additional runtime authorities.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..api import ExecutionError
from ..dag import RecoveryPreset
from ..model import EntityId, GrainId, InvariantError, ItemRecord, ItemRef, ItemTerminal
from ..protocol import BatchCall


class ArenaAbort(ExecutionError):
    """A run-control or contract error aborts one bounded Arena."""


@dataclass(frozen=True, slots=True)
class ArenaLimits:
    """Hard bounds applied before Arena metadata publication."""

    max_grains: int = 100_000
    max_fanout_per_grain: int = 100_000
    max_reduce_slots: int = 1_000_000
    max_pending_dispatches: int = 256
    max_blocks: int = 100_000

    def __post_init__(self) -> None:
        if min(
            self.max_grains,
            self.max_fanout_per_grain,
            self.max_reduce_slots,
            self.max_pending_dispatches,
            self.max_blocks,
        ) <= 0:
            raise ValueError("Arena limits must be positive")


@dataclass(slots=True)
class PendingInvocation:
    """A short-lived aligned fan-in waiting for terminal input receipts."""

    stage: int
    entity: EntityId
    inputs: list[ItemRef | None]


@dataclass(frozen=True, slots=True)
class ExpandInstance:
    """The terminal cardinality fact for one parent/Expand occurrence."""

    grain: GrainId
    stage: int
    anchor: ItemRef
    cardinality: int | None
    failed: bool = False


WAITING = 0
PRESENT = 1
DROPPED = 2
FAILED = 3
SUPPRESSED = 4


@dataclass(slots=True)
class GroupedSlots:
    """Compact dense terminal state for one Reduce GROUP input."""

    states: bytearray
    items: list[ItemRef | None]
    causes: list[GrainId | None]
    remaining: int

    @classmethod
    def create(cls, cardinality: int) -> "GroupedSlots":
        return cls(
            states=bytearray(cardinality),
            items=[None] * cardinality,
            causes=[None] * cardinality,
            remaining=cardinality,
        )

    def settle(self, ordinal: int, record: ItemRecord) -> bool:
        if ordinal < 0 or ordinal >= len(self.states):
            raise InvariantError("Reduce ordinal outside cardinality")
        state = {
            ItemTerminal.PRESENT: PRESENT,
            ItemTerminal.DROPPED: DROPPED,
            ItemTerminal.FAILED: FAILED,
            ItemTerminal.SUPPRESSED: SUPPRESSED,
        }[record.terminal]
        existing = self.states[ordinal]
        if existing:
            if (
                existing != state
                or self.items[ordinal] != record.ref
                or self.causes[ordinal] != record.cause
            ):
                raise InvariantError("Reduce slot settled inconsistently")
            return False
        self.states[ordinal] = state
        self.items[ordinal] = record.ref
        self.causes[ordinal] = record.cause or record.producer
        self.remaining -= 1
        return True


@dataclass(slots=True)
class ReduceAccumulator:
    """Incremental grouped-input state for one Reduce anchor."""

    stage: int
    anchor: ItemRef
    origin: ExpandInstance
    groups: dict[int, GroupedSlots]
    scalar_inputs: dict[int, ItemRef]

    @property
    def complete(self) -> bool:
        return all(group.remaining == 0 for group in self.groups.values())


@dataclass(slots=True)
class StageBatchQueue:
    """Normal and recovery queues owned by one Arena/Stage pair."""

    normal: deque[GrainId] = field(default_factory=deque)
    normal_set: set[GrainId] = field(default_factory=set)
    immediate: deque["RecoveryTask"] = field(default_factory=deque)
    tail: deque["RecoveryTask"] = field(default_factory=deque)
    first_wait_at: float | None = None


@dataclass(slots=True)
class RecoveryBudgetState:
    """Counters shared by every child of one recovery split tree."""

    extra_rpcs: int = 0
    reexecuted_grains: int = 0


@dataclass(slots=True)
class RecoveryTask:
    """A deferred or immediate physical re-execution group."""

    stage: int
    grains: tuple[GrainId, ...]
    preset: RecoveryPreset
    attempts: int = 0
    depth: int = 0
    budget: RecoveryBudgetState = field(default_factory=RecoveryBudgetState)
    actor_policy: str = "any"
    avoid_worker_slot: int | None = None
    phase: str = "retry"


@dataclass(slots=True)
class DispatchLease:
    """Arena-side authority for one pending physical dispatch."""

    call: BatchCall
    block_ids: tuple[int, ...]
    grain_ids: tuple[GrainId, ...]
    recovery: RecoveryTask | None = None
    flush_reason: str = "full"
