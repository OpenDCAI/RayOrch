"""Real-Ray benchmark runner sharing the production Pipeline and transport."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import time

from ..api import Expand, Filter, Map, Pipeline, Reduce
from ..executor import Executor, RunResult
from ..graph import Primitive
from .report import BenchmarkReport
from .workload import SyntheticWorkload


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
            (parent.name, tuple(child.ordinal for child in group))
            for parent, group in zip(parents, members)
        ]


class SyntheticPipeline(Pipeline):
    def __init__(
        self,
        *,
        mode: str,
        batch_size: int,
        max_batch_wait_ms: float,
        replicas: int,
    ) -> None:
        scope = "elastic" if mode == "elastic" else "parent_bound"
        self.expand = Expand(BenchmarkExpand).ray_options(
            replicas=min(4, replicas),
            batch_size=max(1, min(16, batch_size)),
        )
        self.map = Map(BenchmarkMap).ray_options(
            replicas=replicas,
            batch_size=batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_scope=scope,
        )
        self.filter = Filter(BenchmarkFilter).ray_options(
            replicas=replicas,
            batch_size=batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_scope=scope,
        )
        self.reduce = Reduce(BenchmarkReduce).ray_options(
            replicas=min(4, replicas),
            batch_size=batch_size,
        )

    def forward(self, parents):
        children = self.expand(parents)
        mapped = self.map(children)
        kept = self.filter(mapped)
        return self.reduce(anchor=parents, members=kept)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _output_digest(outputs: tuple[object, ...]) -> str:
    payload = json.dumps(
        sorted(outputs, key=repr),
        default=repr,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _node_bubble_ratio(result: RunResult, node: int) -> float:
    events = [
        event
        for event in result.timeline
        if event.node == node
        and event.worker_started_at is not None
        and event.worker_finished_at is not None
    ]
    if not events:
        return 0.0
    start = min(event.worker_started_at for event in events)
    stop = max(event.worker_finished_at for event in events)
    actor_count = result.metrics.get(f"actor_count_node_{node}", 1.0)
    capacity = max(stop - start, 1e-9) * max(actor_count, 1.0)
    busy = sum(
        event.worker_finished_at - event.worker_started_at
        for event in events
    )
    return max(0.0, min(1.0, 1.0 - busy / capacity))


def run_benchmark(
    workload: SyntheticWorkload,
    *,
    mode: str,
    batch_size: int,
    max_batch_wait_ms: float,
    replicas: int,
    repetition: int = 0,
    order_index: int = 0,
) -> tuple[BenchmarkReport, tuple[object, ...]]:
    """Run one measured mode after the Executor actor-readiness barrier."""

    if mode not in {"elastic", "parent_bound"}:
        raise ValueError("mode must be 'elastic' or 'parent_bound'")
    pipeline = SyntheticPipeline(
        mode=mode,
        batch_size=batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        replicas=replicas,
    )
    executor = Executor(pipeline)
    node_by_kind = {
        node.kind: node.id
        for node in executor.graph.nodes
        if node.kind is not Primitive.SOURCE
    }
    result = executor.run(list(workload.parents))
    outputs = result.get()
    metrics = result.metrics
    measured_wall = metrics["measured_wall_time_s"]

    try:
        import ray

        ray_version = ray.__version__
    except Exception:
        ray_version = "unknown"

    report = BenchmarkReport(
        mode=mode,
        seed=workload.seed,
        repetition=repetition,
        order_index=order_index,
        fanout_mode=workload.fanout_mode,
        service_mode=workload.service_mode,
        parent_count=len(workload.parents),
        total_children=workload.total_children,
        kept_children=workload.kept_children,
        replicas=replicas,
        batch_size=batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        startup_time_s=metrics["startup_time_s"],
        measured_wall_time_s=measured_wall,
        end_to_end_wall_time_s=metrics["end_to_end_wall_time_s"],
        throughput_children_s=(
            workload.total_children / measured_wall
            if measured_wall
            else 0.0
        ),
        rpc_count=metrics["rpc_count"],
        grains_per_rpc=metrics["grains_per_rpc"],
        batch_fill_ratio=metrics["batch_fill_ratio"],
        tail_rpc_fraction=metrics["tail_or_isolation_rpc_fraction"],
        parent_p50_s=metrics["parent_completion_p50_s"],
        parent_p95_s=metrics["parent_completion_p95_s"],
        parent_p99_s=metrics["parent_completion_p99_s"],
        expand_bubble_ratio=_node_bubble_ratio(
            result,
            node_by_kind[Primitive.EXPAND],
        ),
        map_bubble_ratio=_node_bubble_ratio(
            result,
            node_by_kind[Primitive.MAP],
        ),
        filter_bubble_ratio=_node_bubble_ratio(
            result,
            node_by_kind[Primitive.FILTER],
        ),
        reduce_bubble_ratio=_node_bubble_ratio(
            result,
            node_by_kind[Primitive.REDUCE],
        ),
        flush_full=metrics["flush_full"],
        flush_timeout=metrics["flush_timeout"],
        flush_port_sealed=metrics["flush_port_sealed"],
        output_digest=_output_digest(outputs),
        git_commit=_git_commit(),
        python_version=platform.python_version(),
        ray_version=ray_version,
    )
    return report, outputs


def run_paired_repetitions(
    workload: SyntheticWorkload,
    *,
    batch_size: int,
    elastic_wait_ms: float,
    replicas: int,
    warmups: int,
    repetitions: int,
) -> tuple[BenchmarkReport, ...]:
    """Run paired modes in deterministic randomized order and retain raw trials."""

    if warmups < 0 or repetitions <= 0:
        raise ValueError("warmups must be >=0 and repetitions must be >0")
    for _ in range(warmups):
        run_benchmark(
            workload,
            mode="parent_bound",
            batch_size=batch_size,
            max_batch_wait_ms=0.0,
            replicas=replicas,
        )
        run_benchmark(
            workload,
            mode="elastic",
            batch_size=batch_size,
            max_batch_wait_ms=elastic_wait_ms,
            replicas=replicas,
        )

    reports = []
    for repetition in range(repetitions):
        order = ["parent_bound", "elastic"]
        random.Random(workload.seed + repetition).shuffle(order)
        digests: set[str] = set()
        for order_index, mode in enumerate(order):
            report, _ = run_benchmark(
                workload,
                mode=mode,
                batch_size=batch_size,
                max_batch_wait_ms=(
                    elastic_wait_ms if mode == "elastic" else 0.0
                ),
                replicas=replicas,
                repetition=repetition,
                order_index=order_index,
            )
            reports.append(report)
            digests.add(report.output_digest)
        if len(digests) != 1:
            raise AssertionError(
                "parent-bound and elastic modes produced different outputs"
            )
    return tuple(reports)
