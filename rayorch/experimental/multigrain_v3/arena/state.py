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
from ..model import EntityId, GrainId, ItemRef
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
