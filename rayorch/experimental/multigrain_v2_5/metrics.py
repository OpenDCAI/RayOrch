"""Names for the intentionally small first metrics surface."""

from __future__ import annotations

from dataclasses import dataclass


GRAINS_PER_RPC = "grains_per_rpc"
RPC_COUNT = "rpc_count"
PENDING_DISPATCHES = "pending_dispatches"
TAIL_OR_ISOLATION_RPC_FRACTION = "tail_or_isolation_rpc_fraction"
BATCH_FILL_RATIO = "batch_fill_ratio"
READY_QUEUE_HIGH_WATERMARK = "ready_queue_high_watermark"
METADATA_BYTES_PER_GRAIN = "metadata_bytes_per_grain"
METADATA_BYTES_PER_EMISSION = "metadata_bytes_per_emission"
DRIVER_RSS = "driver_rss"
WORKER_RSS = "worker_rss"


CORE_METRICS = (
    GRAINS_PER_RPC,
    RPC_COUNT,
    PENDING_DISPATCHES,
    TAIL_OR_ISOLATION_RPC_FRACTION,
    BATCH_FILL_RATIO,
    READY_QUEUE_HIGH_WATERMARK,
    METADATA_BYTES_PER_GRAIN,
    METADATA_BYTES_PER_EMISSION,
    DRIVER_RSS,
    WORKER_RSS,
)


@dataclass(frozen=True, slots=True)
class DispatchTimeline:
    arena: int
    node: int
    dispatch: int
    actor_index: int
    grains: int
    flush_reason: str
    submitted_at: float
    manifest_received_at: float
    committed_at: float
    worker_started_at: float | None
    worker_finished_at: float | None
    worker_rss_bytes: int | None
    status: str


def percentile(values: list[float], quantile: float) -> float:
    """Return a deterministic nearest-rank-style interpolated percentile."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
