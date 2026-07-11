"""Instrumentation for the multigrain executors (M2 metric #3).

Lightweight, dependency-free counters that both the local ``MultigrainExecutor``
and the Ray ``MultigrainRayExecutor`` can populate. They produce the headline
numbers M2 needs without pulling in a metrics backend:

* per-node wall time and row counts -> end-to-end makespan;
* per-shard busy time at parallel stages -> **GPU idle bubble** (load imbalance);
* lineage footprint (records + ancestor entries + approx bytes) -> **lineage
  overhead**;
* recovery counters (retried shards / recomputed rows) -> **recovery cost**.

Nothing here touches Ray or torch, so it stays importable and unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .core import PortBatch


@dataclass
class NodeMetric:
    """One executed node (or one sharded stage) worth of measurements."""

    name: str
    kind: str
    replicas: int
    rows_in: int
    rows_out: int
    wall_s: float
    # Per-shard busy seconds for a parallel stage. len == replicas actually run.
    # A single-task node records one entry equal to ``wall_s``.
    shard_busy_s: List[float] = field(default_factory=list)
    # Recovery accounting: how many shards were retried and how many rows those
    # retries recomputed (i.e. work spent on recovery, not first-pass progress).
    retries: int = 0
    recovery_rows: int = 0

    @property
    def stage_makespan_s(self) -> float:
        """Wall time of the stage = the slowest shard (shards run concurrently)."""
        return max(self.shard_busy_s) if self.shard_busy_s else self.wall_s

    @property
    def stage_ideal_s(self) -> float:
        """Perfectly balanced makespan = mean shard busy time."""
        if not self.shard_busy_s:
            return self.wall_s
        return sum(self.shard_busy_s) / len(self.shard_busy_s)

    @property
    def idle_bubble_frac(self) -> float:
        """Fraction of the stage's parallel capacity wasted to imbalance.

        ``1 - mean_busy / max_busy``. 0.0 == perfectly balanced, ->1.0 == one
        shard carries everything. This is the number LPT is meant to drive down.
        """
        mk = self.stage_makespan_s
        if mk <= 0.0 or len(self.shard_busy_s) <= 1:
            return 0.0
        return max(0.0, 1.0 - (self.stage_ideal_s / mk))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "replicas": self.replicas,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "wall_s": self.wall_s,
            "shard_busy_s": list(self.shard_busy_s),
            "stage_makespan_s": self.stage_makespan_s,
            "idle_bubble_frac": self.idle_bubble_frac,
            "retries": self.retries,
            "recovery_rows": self.recovery_rows,
        }


@dataclass
class RunMetrics:
    """Collector threaded through an executor for one ``execute`` call."""

    nodes: List[NodeMetric] = field(default_factory=list)

    def record(self, metric: NodeMetric) -> None:
        self.nodes.append(metric)

    def by_name(self, name: str) -> NodeMetric | None:
        for metric in self.nodes:
            if metric.name == name:
                return metric
        return None

    @property
    def total_wall_s(self) -> float:
        """Sum of stage makespans (serial-barrier model; matches current MVP)."""
        return sum(node.stage_makespan_s for node in self.nodes)

    @property
    def total_retries(self) -> int:
        return sum(node.retries for node in self.nodes)

    @property
    def total_recovery_rows(self) -> int:
        return sum(node.recovery_rows for node in self.nodes)

    def worst_bubble(self) -> NodeMetric | None:
        parallel = [n for n in self.nodes if len(n.shard_busy_s) > 1]
        return max(parallel, key=lambda n: n.idle_bubble_frac, default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_wall_s": self.total_wall_s,
            "total_retries": self.total_retries,
            "total_recovery_rows": self.total_recovery_rows,
            "nodes": [node.to_dict() for node in self.nodes],
        }


def _approx_str_bytes(value: str) -> int:
    # Cheap, deterministic proxy for the serialized size of a lineage token.
    return len(value.encode("utf-8", "ignore"))


def lineage_footprint(batches: Sequence[PortBatch]) -> Dict[str, int]:
    """Estimate the lineage metadata carried by a set of ports.

    Counts records, ancestor/ordinal entries, and an approximate byte size of the
    lineage-only columns (ids, display keys, ancestors, ordinals, lineage paths,
    relations). Excludes ``values`` -- we want the *overhead*, not the payload.
    """
    records = 0
    ancestor_entries = 0
    relation_entries = 0
    approx_bytes = 0
    for batch in batches:
        records += len(batch)
        for i in range(len(batch)):
            approx_bytes += _approx_str_bytes(batch.record_ids[i])
            approx_bytes += _approx_str_bytes(batch.display_keys[i])
            for key, val in batch.ancestors[i].items():
                ancestor_entries += 1
                approx_bytes += _approx_str_bytes(key) + _approx_str_bytes(val)
            for key, val in batch.ancestor_display[i].items():
                approx_bytes += _approx_str_bytes(key) + _approx_str_bytes(str(val))
            for key in batch.ordinals[i]:
                approx_bytes += _approx_str_bytes(key) + 8
            for token in batch.lineage[i]:
                approx_bytes += _approx_str_bytes(token)
            if batch.relations:
                relation_entries += len(batch.relations[i])
    return {
        "records": records,
        "ancestor_entries": ancestor_entries,
        "relation_entries": relation_entries,
        "approx_bytes": approx_bytes,
        "bytes_per_record": (approx_bytes // records) if records else 0,
    }


__all__ = ["NodeMetric", "RunMetrics", "lineage_footprint"]
