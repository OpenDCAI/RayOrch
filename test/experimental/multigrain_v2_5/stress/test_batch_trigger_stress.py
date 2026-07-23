"""Slow load-shape regressions for batch triggering and Ray rebatching."""

from __future__ import annotations

import time

import pytest

import rayorch.experimental.multigrain_v2_5 as mg

from ..unit.test_batch_trigger import _arena, _enqueue


@pytest.mark.slow
def test_ten_thousand_ready_grains_use_bounded_full_batches():
    """A 10k-grain burst stays count-bounded and creates no tiny middle RPCs."""

    _, arena, map_spec, sources = _arena(
        batch_size=64,
        wait_ms=5,
        count=10_000,
    )
    _enqueue(arena, map_spec, sources)
    sizes = []
    while arena.ready_count(1) >= 64:
        plan = arena.reserve_dispatch(1, admission_closed=False)
        assert plan is not None
        sizes.append(len(plan.entries))
    tail = arena.reserve_dispatch(1, admission_closed=True)
    assert tail is not None
    sizes.append(len(tail.entries))

    assert sizes[:-1] == [64] * 156
    assert sizes[-1] == 16
    assert sum(sizes) == 10_000
    metrics = arena.metrics_snapshot()
    assert metrics["rpc_count"] == 157.0
    assert metrics["flush_full"] == 156.0
    assert metrics["flush_port_sealed"] == 1.0
    assert metrics["ready_queue_high_watermark"] == 10_000.0


class SkewExpand:
    def run(self, parents):
        outputs = []
        for name, count, delay in parents:
            time.sleep(delay)
            outputs.append(
                [f"{name}:child:{ordinal}" for ordinal in range(count)]
            )
        return outputs


class ObserveBatch:
    def run(self, children):
        return [(child, len(children)) for child in children]


class SkewPipeline(mg.Pipeline):
    def __init__(self, wait_ms: float):
        self.expand = mg.Expand(SkewExpand).ray_options(
            replicas=4,
            batch_size=1,
        )
        self.map = mg.Map(ObserveBatch).ray_options(
            replicas=2,
            batch_size=8,
            max_batch_wait_ms=wait_ms,
        )

    def forward(self, parents):
        return self.map(self.expand(parents))


@pytest.mark.slow
@pytest.mark.usefixtures("ray_cluster")
def test_ray_skewed_fanout_wait_policy_preserves_results_and_reduces_fragmentation():
    """Skewed delayed fan-out keeps semantics while a wait cap coalesces RPCs."""

    counts = (0, 1, 2, 8, 16)
    parents = [
        (
            f"parent-{index}",
            counts[index % len(counts)],
            (index % 4) * 0.003,
        )
        for index in range(40)
    ]
    no_wait = mg.Executor(SkewPipeline(0.0)).run(parents)
    coalesced = mg.Executor(SkewPipeline(20.0)).run(parents)

    no_wait_rows = no_wait.get()
    coalesced_rows = coalesced.get()
    assert {row[0] for row in no_wait_rows} == {
        row[0] for row in coalesced_rows
    }
    assert coalesced.metrics["rpc_count"] <= no_wait.metrics["rpc_count"]
    assert (
        coalesced.metrics["grains_per_rpc"]
        >= no_wait.metrics["grains_per_rpc"]
    )
    assert no_wait.metrics["flush_timeout"] >= 1.0
