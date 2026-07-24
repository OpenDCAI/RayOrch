"""End-to-end regressions for dispatch timelines and parent latency metrics."""

from __future__ import annotations

import time

import pytest

import rayorch.experimental.multigrain_v2_5 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class OneChild:
    def run(self, parents):
        return [[f"{parent}:child"] for parent in parents]


class SleepChild:
    def run(self, children):
        time.sleep(0.01)
        return [f"mapped({child})" for child in children]


class Assemble:
    def run(self, anchors, members):
        return [
            f"{anchor}=>{group[0]}" for anchor, group in zip(anchors, members)
        ]


class TimelinePipeline(mg.Pipeline):
    def __init__(self):
        self.expand = mg.Expand(OneChild).ray_options(batch_size=2)
        self.map = mg.Map(SleepChild).ray_options(batch_size=2)
        self.reduce = mg.Reduce(Assemble).ray_options(batch_size=2)

    def forward(self, parents):
        children = self.expand(parents)
        mapped = self.map(children)
        return self.reduce(anchor=parents, members=mapped)


def test_timeline_records_worker_and_driver_phases_per_coarse_rpc():
    """Every Ray RPC exposes bounded timing metadata without payload access."""

    result = mg.Executor(TimelinePipeline()).run(["a", "b"])
    assert result.get() == (
        "a=>mapped(a:child)",
        "b=>mapped(b:child)",
    )
    assert len(result.timeline) == 3
    assert {event.node for event in result.timeline} == {1, 2, 3}
    for event in result.timeline:
        assert event.grains == 2
        assert event.submitted_at <= event.manifest_received_at
        assert event.manifest_received_at <= event.committed_at
        assert event.worker_started_at is not None
        assert event.worker_finished_at is not None
        assert event.worker_started_at <= event.worker_finished_at
        assert event.status == "accepted"


def test_reduce_completion_metrics_are_detached_and_monotonic():
    """RunResult carries parent completion percentiles after arena reclamation."""

    result = mg.Executor(TimelinePipeline()).run(["a", "b", "c", "d"])
    metrics = result.metrics
    assert metrics["startup_time_s"] >= 0
    assert metrics["measured_wall_time_s"] > 0
    assert (
        metrics["end_to_end_wall_time_s"]
        >= metrics["measured_wall_time_s"]
    )
    assert metrics["parent_completion_count"] == 4.0
    assert 0 < metrics["parent_completion_p50_s"]
    assert (
        metrics["parent_completion_p50_s"]
        <= metrics["parent_completion_p95_s"]
        <= metrics["parent_completion_p99_s"]
    )
