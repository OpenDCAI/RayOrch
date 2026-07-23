"""Fair parent-bound versus elastic Ray benchmark using identical UDFs."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import rayorch.experimental.multigrain_v2_5 as mg


@dataclass(frozen=True, slots=True)
class ChildWork:
    parent: str
    ordinal: int
    delay_s: float
    keep: bool


@dataclass(frozen=True, slots=True)
class ParentWork:
    name: str
    children: tuple[ChildWork, ...]


@dataclass(frozen=True, slots=True)
class SyntheticWorkload:
    seed: int
    parents: tuple[ParentWork, ...]

    @property
    def total_children(self) -> int:
        return sum(len(parent.children) for parent in self.parents)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    mode: str
    seed: int
    parents: int
    total_children: int
    kept_children: int
    wall_time_s: float
    throughput_children_s: float
    rpc_count: float
    grains_per_rpc: float
    batch_fill_ratio: float
    tail_rpc_fraction: float
    parent_p50_s: float
    parent_p95_s: float
    parent_p99_s: float
    actor_bubble_ratio: float
    flush_full: float
    flush_timeout: float
    flush_port_sealed: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BenchmarkExpand:
    def run(self, parents):
        return [list(parent.children) for parent in parents]


class BenchmarkMap:
    def run(self, children):
        outputs = []
        for child in children:
            time.sleep(child.delay_s)
            outputs.append(child)
        return outputs


class BenchmarkFilter:
    def run(self, children):
        return [child.keep for child in children]


class BenchmarkReduce:
    def run(self, parents, members):
        return [
            (parent.name, tuple((child.ordinal for child in group)))
            for parent, group in zip(parents, members)
        ]


class SyntheticPipeline(mg.Pipeline):
    def __init__(
        self,
        *,
        mode: str,
        batch_size: int,
        max_batch_wait_ms: float,
        replicas: int,
    ) -> None:
        scope = "elastic" if mode == "elastic" else "parent_bound"
        self.expand = mg.Expand(BenchmarkExpand).ray_options(
            replicas=min(4, replicas),
            batch_size=max(1, min(16, batch_size)),
        )
        self.map = mg.Map(BenchmarkMap).ray_options(
            replicas=replicas,
            batch_size=batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_scope=scope,
        )
        self.filter = mg.Filter(BenchmarkFilter).ray_options(
            replicas=replicas,
            batch_size=batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_scope=scope,
        )
        self.reduce = mg.Reduce(BenchmarkReduce).ray_options(
            replicas=min(4, replicas),
            batch_size=batch_size,
        )

    def forward(self, parents):
        children = self.expand(parents)
        mapped = self.map(children)
        kept = self.filter(mapped)
        return self.reduce(anchor=parents, members=kept)


def _fanout(randomizer: random.Random, mode: str, scale: float) -> int:
    if mode == "constant":
        return max(0, int(scale))
    if mode == "uniform":
        return randomizer.randint(0, max(1, int(2 * scale)))
    if mode == "lognormal":
        return max(0, int(randomizer.lognormvariate(math.log(max(scale, 1)), 0.8)))
    if mode == "pareto":
        return max(0, int(scale * randomizer.paretovariate(2.0)) - int(scale))
    if mode == "zipf":
        rank = randomizer.randint(1, 16)
        return max(0, int(scale * 8 / rank))
    raise ValueError(f"unknown fanout mode: {mode}")


def _delay(randomizer: random.Random, mode: str, mean_s: float) -> float:
    if mode == "constant":
        return mean_s
    if mode == "uniform":
        return randomizer.uniform(0.25 * mean_s, 1.75 * mean_s)
    if mode == "lognormal":
        return randomizer.lognormvariate(
            math.log(max(mean_s, 1e-6)),
            0.7,
        )
    raise ValueError(f"unknown service mode: {mode}")


def generate_workload(
    *,
    seed: int,
    parent_count: int,
    fanout_mode: str,
    fanout_scale: float,
    service_mode: str,
    mean_service_s: float,
    drop_probability: float = 0.0,
) -> SyntheticWorkload:
    """Generate deterministic fan-out, service-time, and Filter patterns."""

    randomizer = random.Random(seed)
    parents = []
    for parent_index in range(parent_count):
        name = f"parent-{parent_index}"
        count = _fanout(randomizer, fanout_mode, fanout_scale)
        children = tuple(
            ChildWork(
                parent=name,
                ordinal=ordinal,
                delay_s=_delay(
                    randomizer,
                    service_mode,
                    mean_service_s,
                ),
                keep=randomizer.random() >= drop_probability,
            )
            for ordinal in range(count)
        )
        parents.append(ParentWork(name, children))
    return SyntheticWorkload(seed, tuple(parents))


def _bubble_ratio(result: mg.RunResult) -> float:
    intervals = [
        (event.worker_started_at, event.worker_finished_at)
        for event in result.timeline
        if event.worker_started_at is not None
        and event.worker_finished_at is not None
    ]
    if not intervals:
        return 0.0
    start = min(interval[0] for interval in intervals)
    stop = max(interval[1] for interval in intervals)
    actors = {
        (event.node, event.actor_index)
        for event in result.timeline
        if event.worker_started_at is not None
    }
    capacity = max(stop - start, 1e-9) * max(len(actors), 1)
    busy = sum(interval[1] - interval[0] for interval in intervals)
    return max(0.0, min(1.0, 1.0 - busy / capacity))


def run_benchmark(
    workload: SyntheticWorkload,
    *,
    mode: str,
    batch_size: int,
    max_batch_wait_ms: float,
    replicas: int,
) -> tuple[BenchmarkReport, tuple[object, ...]]:
    """Run one real-Ray benchmark mode and return metrics plus final outputs."""

    if mode not in {"elastic", "parent_bound"}:
        raise ValueError("mode must be 'elastic' or 'parent_bound'")
    pipeline = SyntheticPipeline(
        mode=mode,
        batch_size=batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        replicas=replicas,
    )
    started = time.monotonic()
    result = mg.Executor(pipeline).run(list(workload.parents))
    outputs = result.get()
    wall = time.monotonic() - started
    kept = sum(
        child.keep
        for parent in workload.parents
        for child in parent.children
    )
    metrics = result.metrics
    report = BenchmarkReport(
        mode=mode,
        seed=workload.seed,
        parents=len(workload.parents),
        total_children=workload.total_children,
        kept_children=kept,
        wall_time_s=wall,
        throughput_children_s=(
            workload.total_children / wall if wall else 0.0
        ),
        rpc_count=metrics["rpc_count"],
        grains_per_rpc=metrics["grains_per_rpc"],
        batch_fill_ratio=metrics["batch_fill_ratio"],
        tail_rpc_fraction=metrics["tail_or_isolation_rpc_fraction"],
        parent_p50_s=metrics["parent_completion_p50_s"],
        parent_p95_s=metrics["parent_completion_p95_s"],
        parent_p99_s=metrics["parent_completion_p99_s"],
        actor_bubble_ratio=_bubble_ratio(result),
        flush_full=metrics["flush_full"],
        flush_timeout=metrics["flush_timeout"],
        flush_port_sealed=metrics["flush_port_sealed"],
    )
    return report, outputs


def write_reports(
    reports: Iterable[BenchmarkReport],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Write deterministic JSON and CSV summaries for experiment scripts."""

    rows = [report.to_dict() for report in reports]
    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(BenchmarkReport.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(rows)
