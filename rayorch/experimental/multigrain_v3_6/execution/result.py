"""Frozen, public execution results detached from mutable driver state."""

from __future__ import annotations

from dataclasses import dataclass

from .worker import WorkerSnapshot


@dataclass(frozen=True, slots=True)
class CallMetrics:
    """One Call's run-local physical metrics snapshot."""

    call_index: int
    udf_name: str
    actor_instances: int
    rpcs: int
    grains: int
    retries: int
    batch_sizes: tuple[int, ...]
    worker_snapshots: tuple[WorkerSnapshot, ...]

    @property
    def average_batch(self) -> float:
        return self.grains / self.rpcs if self.rpcs else 0.0


@dataclass(frozen=True, slots=True)
class RunResult:
    """Ordered outputs and audit snapshots with no mutable Engine references."""

    outputs: object
    elapsed_s: float
    calls: tuple[CallMetrics, ...]
    microbatches: tuple[MicrobatchMetrics, ...]
    peak_active_microbatches: int

    @property
    def rpc_count(self) -> int:
        return sum(metrics.rpcs for metrics in self.calls)

    @property
    def actor_count(self) -> int:
        return sum(metrics.actor_instances for metrics in self.calls)

    @property
    def released_values(self) -> int:
        return sum(metrics.released_values for metrics in self.microbatches)


@dataclass(frozen=True, slots=True)
class MicrobatchMetrics:
    """Read-only semantic scale snapshot for one completed source slice."""

    index: int
    entity_count: int
    item_count: int
    expansion_count: int
    grain_count: int
    released_values: int


__all__ = ["CallMetrics", "MicrobatchMetrics", "RunResult"]
