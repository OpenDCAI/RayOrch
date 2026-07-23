"""Names for the intentionally small first metrics surface."""

from __future__ import annotations


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
