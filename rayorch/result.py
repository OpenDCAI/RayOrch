"""Frozen, public execution results detached from mutable driver state."""

from __future__ import annotations

from dataclasses import dataclass

from ._model import ItemOutcome


@dataclass(frozen=True, slots=True)
class OutputIssue:
    """A final output without a business value and its optional readable cause.

    Successful outputs remain ordinary values. This type is reserved for
    framework results and must not be used as an ordinary output value.
    """

    outcome: ItemOutcome
    cause: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.outcome, ItemOutcome)
            or self.outcome is ItemOutcome.PRESENT
        ):
            raise ValueError("OutputIssue requires a non-PRESENT ItemOutcome")
        if self.cause is not None and not isinstance(self.cause, str):
            raise TypeError("OutputIssue cause must be text or None")


@dataclass(frozen=True, slots=True)
class CallMetrics:
    """One Call's run-local physical metrics snapshot.

    ``grain_dispatches`` includes repeated dispatches of the same Grain.
    ``grain_requeues`` counts recovery enqueue events, which may not yet have
    resulted in another dispatch. Neither field is a unique Grain count.
    """

    call_index: int
    udf_name: str
    actor_instances: int
    rpcs: int
    grain_dispatches: int
    grain_requeues: int
    batch_sizes: tuple[int, ...]

    @property
    def average_batch(self) -> float:
        return self.grain_dispatches / self.rpcs if self.rpcs else 0.0


@dataclass(frozen=True, slots=True)
class RunResult:
    """Ordered outputs and execution metrics with no mutable Engine references."""

    outputs: object
    elapsed_s: float
    calls: tuple[CallMetrics, ...]
    input_batches: tuple[InputBatchMetrics, ...]
    peak_active_input_batches: int

    @property
    def rpc_count(self) -> int:
        return sum(metrics.rpcs for metrics in self.calls)

    @property
    def actor_count(self) -> int:
        return sum(metrics.actor_instances for metrics in self.calls)

    @property
    def released_values(self) -> int:
        return sum(metrics.released_values for metrics in self.input_batches)


@dataclass(frozen=True, slots=True)
class InputBatchMetrics:
    """Read-only semantic scale snapshot for one completed source slice."""

    index: int
    entity_count: int
    item_count: int
    expansion_count: int
    grain_count: int
    released_values: int


__all__ = ["CallMetrics", "InputBatchMetrics", "OutputIssue", "RunResult"]
