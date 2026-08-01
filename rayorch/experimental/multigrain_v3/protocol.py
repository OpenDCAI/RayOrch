"""Stable DTOs exchanged by ArenaEngine, StageExecutor, and workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .model import AttemptToken, GrainFailure, GrainId, GroupShape, ItemRef


@dataclass(frozen=True, slots=True)
class RowTake:
    """Select one row from one deduplicated RPC block argument."""
    ref_slot: int
    row: int


@dataclass(frozen=True, slots=True)
class ValueTake:
    """Ordered row selectors for one scalar or grouped UDF input."""
    rows: tuple[RowTake, ...]
    group_shape: GroupShape | None = None

    def __post_init__(self) -> None:
        if self.group_shape is not None and (
            self.group_shape.leaf_count != len(self.rows)
        ):
            raise ValueError("ValueTake shape does not match selected rows")


@dataclass(frozen=True, slots=True)
class MissingTake:
    """Explicit OPTIONAL_ONE absence; workers materialize MISSING."""
    pass


InputTake = ValueTake | MissingTake


@dataclass(frozen=True, slots=True)
class Invocation:
    """One Logical Grain projected into a physical BatchCall."""
    token: AttemptToken
    inputs: tuple[InputTake, ...]


@dataclass(frozen=True, slots=True)
class BatchCall:
    """Small control manifest sent to one persistent Stage actor."""
    dispatch: int
    stage: int
    invocations: tuple[Invocation, ...]


@dataclass(frozen=True, slots=True)
class InvocationAck:
    """Per-Grain output shape or Filter keep decision."""
    token: AttemptToken
    output_counts: tuple[int, ...] | None = None
    keep: bool | None = None

    def __post_init__(self) -> None:
        if (self.output_counts is None) == (self.keep is None):
            raise ValueError("ack needs exactly one of output_counts or keep")


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Bounded worker report; business values live in separate blocks."""
    dispatch: int
    acks: tuple[InvocationAck, ...]
    column_lengths: tuple[int, ...]
    worker_started_at: float | None = None
    worker_finished_at: float | None = None
    worker_rss_bytes: int | None = None


class FailureKind(Enum):
    """The intentionally small framework-level failure taxonomy."""
    BAD_RECORD = "bad_record"
    UDF_ERROR = "udf_error"
    CONTRACT_ABORT = "contract_abort"
    INFRA_FAILURE = "infra_failure"


@dataclass(frozen=True, slots=True)
class DispatchFailure:
    """Structured failure returned to the Arena through RunDriver."""
    dispatch: int
    kind: FailureKind
    message: str
    bad_token: AttemptToken | None = None
    worker_slot: int | None = None


@dataclass(frozen=True, slots=True)
class DispatchIntent:
    """Arena-produced physical work request for a StageExecutor."""
    arena_id: int
    call: BatchCall
    input_blocks: tuple[Any, ...]
    actor_policy: str = "any"
    avoid_worker_slot: int | None = None
    flush_reason: str = "full"


@dataclass(frozen=True, slots=True)
class DispatchCompletion:
    """StageExecutor result routed back to the owning Arena."""
    arena_id: int
    call: BatchCall
    report: BatchReport
    output_blocks: tuple[Any, ...]
    worker_slot: int
    submitted_at: float | None = None
    report_received_at: float | None = None


@dataclass(frozen=True, slots=True)
class DispatchTimeline:
    """Observation-only timestamps for one accepted dispatch."""
    arena: int
    stage: int
    dispatch: int
    worker_slot: int
    grains: int
    flush_reason: str
    submitted_at: float
    report_received_at: float
    committed_at: float
    worker_started_at: float | None
    worker_finished_at: float | None
    worker_rss_bytes: int | None
    status: str


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Detached source identity retained after Arena reclaim."""
    grain: GrainId
    item: ItemRef


@dataclass(frozen=True, slots=True)
class FailureSnapshot:
    """Detached Failed Grain snapshot retained after Arena reclaim."""
    grain: GrainId
    failure: GrainFailure


@dataclass(frozen=True, slots=True)
class SuppressionSnapshot:
    """Detached Suppressed Grain snapshot retained after Arena reclaim."""
    grain: GrainId
    direct_causes: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class BlockSlice:
    """One final output row held by RunResult without driver-side get."""
    block: Any
    row: int


@dataclass(frozen=True, slots=True)
class ArenaResult:
    """Detached per-Arena delivery payload safe after state reclaim."""
    outputs: tuple[Any, ...]
    failures: tuple[FailureSnapshot, ...]
    suppressions: tuple[SuppressionSnapshot, ...]
    sources: tuple[SourceSnapshot, ...]
    metrics: dict[str, float]
    timeline: tuple[DispatchTimeline, ...] = ()
