"""Low-overhead, framework-neutral MinerU profiling helpers.

The benchmark workers write one JSONL file per process/actor so recording an
event never requires a driver RPC or a shared-file lock.  The files live on the
experiment's already-mounted output filesystem and are merged only after the
terminal materialization barrier.
"""

from __future__ import annotations

import json
import os
import re
import socket
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


def _safe(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return text or "unknown"


def _ray_identity() -> tuple[str, str]:
    try:
        import ray

        context = ray.get_runtime_context()
        node_id = str(context.get_node_id())
        actor_id = str(context.get_actor_id())
    except Exception:
        node_id = socket.gethostname()
        actor_id = "driver"
    return _safe(node_id), _safe(actor_id)


class ProfileEventWriter:
    """Append schema-compatible events to one actor-local JSONL stream."""

    def __init__(
        self,
        profile_dir: str | None,
        *,
        system: str,
        stage: str,
        role: str,
    ) -> None:
        self.profile_dir = str(profile_dir or "")
        self.system = system
        self.stage = stage
        self.role = role
        self.node_id, self.actor_id = _ray_identity()
        self.actor_id = f"{self.actor_id}-{os.getpid()}"
        self.path: Path | None = None
        if self.profile_dir:
            root = Path(self.profile_dir) / "raw_events"
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / (
                f"{_safe(system)}-{_safe(stage)}-{self.node_id}-"
                f"{self.actor_id}-{uuid.uuid4().hex}.jsonl"
            )

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def emit(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        payload = {
            **event,
            "system": self.system,
            "node_id": self.node_id,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def actor_ready(
        self,
        *,
        started_epoch_s: float,
        started_monotonic_s: float,
    ) -> None:
        ready_epoch_s = time.time()
        ready_monotonic_s = time.perf_counter()
        self.emit(
            {
                "type": "actor_init",
                "stage": self.stage,
                "role": self.role,
                "actor_id": self.actor_id,
                "started_epoch_s": started_epoch_s,
                "ready_epoch_s": ready_epoch_s,
                "started_monotonic_s": started_monotonic_s,
                "ready_monotonic_s": ready_monotonic_s,
            }
        )

    def stage_batch(
        self,
        *,
        started_epoch_s: float,
        started_monotonic_s: float,
        items: int,
        batch_size: int | None = None,
        stage: str | None = None,
        role: str | None = None,
        **extra: Any,
    ) -> None:
        self.emit(
            {
                "type": "stage_batch",
                "stage": stage or self.stage,
                "role": role or self.role,
                "actor_id": self.actor_id,
                "started_epoch_s": started_epoch_s,
                "finished_epoch_s": time.time(),
                "started_monotonic_s": started_monotonic_s,
                "finished_monotonic_s": time.perf_counter(),
                "items": int(items),
                "batch_size": int(items if batch_size is None else batch_size),
                **extra,
            }
        )


def write_driver_event(
    profile_dir: str | None,
    event: dict[str, Any],
    *,
    system: str,
) -> None:
    writer = ProfileEventWriter(
        profile_dir,
        system=system,
        stage="driver",
        role="collect",
    )
    writer.emit(event)


def merge_profile_events(profile_dir: str, output: str | Path) -> dict[str, Any]:
    """Merge actor streams deterministically after all producers are done."""

    root = Path(profile_dir) / "raw_events"
    rows: list[dict[str, Any]] = []
    files = tuple(sorted(root.glob("*.jsonl"))) if root.is_dir() else ()
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number} is not a JSON object")
                rows.append(value)

    def timestamp(row: dict[str, Any]) -> float:
        for key in ("started_epoch_s", "epoch_s", "ready_epoch_s"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return float("inf")

    rows.sort(
        key=lambda row: (
            timestamp(row),
            str(row.get("type", "")),
            str(row.get("stage", row.get("name", ""))),
            str(row.get("actor_id", "")),
        )
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return {"files": len(files), "events": len(rows), "path": str(destination)}


def write_gpu_samples(
    records: Iterable[dict[str, Any]],
    output: str | Path,
) -> dict[str, Any]:
    """Persist node-local samples with paired clocks and stable GPU identity."""

    node_rows: list[dict[str, Any]] = []
    for record in records:
        node_id = str(record.get("node_id") or "unknown")
        for sample in record.get("samples") or []:
            epoch_s = sample.get("epoch_s", sample.get("time"))
            monotonic_s = sample.get("monotonic_s")
            if not isinstance(epoch_s, (int, float)) or not isinstance(
                monotonic_s, (int, float)
            ):
                continue
            utilization = list(sample.get("utilization") or [])
            memory = list(sample.get("memory_used") or [])
            power = list(sample.get("power_w") or [])
            node_rows.append(
                {
                    "epoch_s": float(epoch_s),
                    "monotonic_s": float(monotonic_s),
                    "node_id": node_id,
                    "gpu_ids": [f"{node_id}:{index}" for index in range(len(utilization))],
                    "utilization": utilization,
                    "memory_used": memory,
                    "power_w": power,
                }
            )
    rows = aggregate_gpu_rows(node_rows)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return {"samples": len(rows), "path": str(destination)}


def aggregate_gpu_rows(node_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Align independently sampled nodes into cluster-wide one-second buckets."""

    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in node_rows:
        epoch_s = row.get("epoch_s")
        monotonic_s = row.get("monotonic_s")
        if not isinstance(epoch_s, (int, float)) or not isinstance(
            monotonic_s, (int, float)
        ):
            continue
        buckets.setdefault(int(float(epoch_s)), []).append(row)
    rows = []
    for _, values in sorted(buckets.items()):
        ordered = sorted(values, key=lambda row: str(row.get("node_id") or ""))
        rows.append(
            {
                "epoch_s": statistics.fmean(float(row["epoch_s"]) for row in ordered),
                "monotonic_s": statistics.fmean(
                    float(row["monotonic_s"]) for row in ordered
                ),
                "node_ids": [str(row.get("node_id") or "unknown") for row in ordered],
                "gpu_ids": [gpu for row in ordered for gpu in row.get("gpu_ids") or []],
                "utilization": [
                    value for row in ordered for value in row.get("utilization") or []
                ],
                "memory_used": [
                    value for row in ordered for value in row.get("memory_used") or []
                ],
                "power_w": [
                    value for row in ordered for value in row.get("power_w") or []
                ],
            }
        )
    return rows


__all__ = [
    "ProfileEventWriter",
    "aggregate_gpu_rows",
    "merge_profile_events",
    "write_driver_event",
    "write_gpu_samples",
]
