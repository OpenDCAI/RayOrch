"""Names for the intentionally small first metrics surface."""

from __future__ import annotations


GRAINS_PER_RPC = "grains_per_rpc"
PENDING_DISPATCHES = "pending_dispatches"
TAIL_OR_ISOLATION_RPC_FRACTION = "tail_or_isolation_rpc_fraction"
METADATA_BYTES_PER_GRAIN = "metadata_bytes_per_grain"
METADATA_BYTES_PER_EMISSION = "metadata_bytes_per_emission"
DRIVER_RSS = "driver_rss"
WORKER_RSS = "worker_rss"


CORE_METRICS = (
    GRAINS_PER_RPC,
    PENDING_DISPATCHES,
    TAIL_OR_ISOLATION_RPC_FRACTION,
    METADATA_BYTES_PER_GRAIN,
    METADATA_BYTES_PER_EMISSION,
    DRIVER_RSS,
    WORKER_RSS,
)
