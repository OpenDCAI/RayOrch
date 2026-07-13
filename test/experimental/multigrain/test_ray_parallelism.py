"""Ray parallelism coverage for the experimental multigrain IR.

Uses dummy CPU sleep operators to prove the passive IR + PhysicalHints can drive
real Ray parallelism via ``MultigrainRayExecutor``:

* replica parallelism  -- row-sharded Map runs N/replicas faster;
* microbatch overlap   -- bounded in-flight window pipelines microbatches.

These are wall-clock assertions, so they are marked ``slow`` and require a Ray
cluster (``--runslow`` + the shared ``ray_cluster`` fixture).
"""
from __future__ import annotations

import time
import uuid

import pytest
import ray

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.graph import PhysicalHints
from rayorch.experimental.multigrain.ray_executor import MultigrainRayExecutor

from test.experimental.multigrain.dummy_ops import (
    SLEEP,
    InitCountingEmbed,
    MergeColumns,
    SlowDrop,
    SlowEmbed,
)

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("ray_cluster")]


@ray.remote(num_cpus=0)
class _InitCounter:
    def __init__(self) -> None:
        self.value = 0

    def add(self, amount: int) -> None:
        self.value += amount

    def get(self) -> int:
        return self.value


# --------------------------------------------------------------------------
# Correctness: Ray executor matches the local executor
# --------------------------------------------------------------------------
class MapFilterPipe(mg.Pipeline):
    def __init__(self, replicas: int = 1) -> None:
        super().__init__()
        self.embed = mg.Map(SlowEmbed, physical=PhysicalHints(replicas=replicas))
        self.keep = mg.Filter(SlowDrop, physical=PhysicalHints(replicas=replicas))

    def forward(self, chunks):
        return self.keep(self.embed(chunks))


def test_ray_executor_matches_local_executor() -> None:
    ir = MapFilterPipe(replicas=4).compile()
    chunks = mg.source([f"c{i}" if i % 3 else f"x{i}" for i in range(9)], name="chunks")

    local = mg.MultigrainExecutor().execute(ir, {"chunks": chunks})
    distributed = MultigrainRayExecutor().execute(ir, {"chunks": chunks})

    assert distributed.values == local.values
    assert distributed.record_ids == local.record_ids
    assert distributed.lineage == local.lineage


# --------------------------------------------------------------------------
# Replica parallelism: row-sharded Map speeds up wall-clock time
# --------------------------------------------------------------------------
class MapPipe(mg.Pipeline):
    def __init__(self, replicas: int) -> None:
        super().__init__()
        self.embed = mg.Map(SlowEmbed, physical=PhysicalHints(replicas=replicas))

    def forward(self, chunks):
        return self.embed(chunks)


def test_replicas_speed_up_row_sharded_map() -> None:
    chunks = mg.source([f"c{i}" for i in range(8)], name="chunks")

    serial_ir = MapPipe(replicas=1).compile()
    parallel_ir = MapPipe(replicas=4).compile()
    serial_executor = MultigrainRayExecutor()
    parallel_executor = MultigrainRayExecutor()
    # Compare steady-state execution, not one-time actor/process construction.
    serial_executor.warm_pools(serial_ir)
    parallel_executor.warm_pools(parallel_ir)
    try:
        start = time.time()
        serial_out = serial_executor.execute(serial_ir, {"chunks": chunks})
        serial = time.time() - start

        start = time.time()
        parallel_out = parallel_executor.execute(parallel_ir, {"chunks": chunks})
        parallel = time.time() - start
    finally:
        serial_executor.shutdown()
        parallel_executor.shutdown()

    assert parallel_out.values == serial_out.values
    # 8 rows * 0.2s: serial ~1.6s, 4-way ~0.4s. Allow generous Ray overhead.
    assert parallel < serial * 0.6


# --------------------------------------------------------------------------
# Microbatch overlap: bounded in-flight window pipelines microbatches
# --------------------------------------------------------------------------
class ChainPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.a = mg.Map(SlowEmbed, name="a")
        self.b = mg.Map(SlowEmbed, name="b")

    def forward(self, chunks):
        return self.b(self.a(chunks))


def test_microbatch_overlap_speeds_up_throughput() -> None:
    ir = ChainPipe().compile()
    microbatches = [
        {"chunks": mg.source([f"m{m}-{i}" for i in range(2)], name="chunks")}
        for m in range(4)
    ]
    executor = MultigrainRayExecutor()

    start = time.time()
    serial = executor.execute_microbatches(ir, microbatches, max_inflight=1)
    serial_time = time.time() - start

    start = time.time()
    overlapped = executor.execute_microbatches(ir, microbatches, max_inflight=4)
    overlap_time = time.time() - start

    assert [r.values for r in serial] == [r.values for r in overlapped]
    # Node-level scheduling pays one Ray task launch per stage (rather than one
    # task per whole graph), but bounded microbatches must still overlap
    # materially. Model-holding stages use persistent actors (covered below).
    assert overlap_time < serial_time * 0.8


class CountingPipe(mg.Pipeline):
    def __init__(self, counter_name: str) -> None:
        super().__init__()
        self.embed = mg.Map(
            InitCountingEmbed,
            counter_name,
            physical=PhysicalHints(replicas=2),
        )

    def forward(self, chunks):
        return self.embed(chunks)


def test_stream_reuses_persistent_actor_pool_across_microbatches() -> None:
    counter_name = f"mg-init-{uuid.uuid4().hex}"
    counter = _InitCounter.options(name=counter_name).remote()
    executor = MultigrainRayExecutor()
    try:
        ir = CountingPipe(counter_name).compile()
        inputs = (
            {"chunks": mg.source([f"m{m}-{i}" for i in range(4)], name="chunks")}
            for m in range(5)
        )

        outputs = list(executor.execute_stream(ir, inputs, max_inflight=3))

        assert len(outputs) == 5
        assert [out.values[0] for out in outputs] == [
            f"counted:m{m}-0" for m in range(5)
        ]
        # Exactly one constructor call per persistent replica, not per microbatch.
        assert ray.get(counter.get.remote()) == 2
    finally:
        executor.shutdown()
        ray.kill(counter)


class BranchPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.left = mg.Map(SlowEmbed, name="left")
        self.right = mg.Map(SlowEmbed, name="right")
        self.merge = mg.Map(MergeColumns)

    def forward(self, chunks):
        return self.merge(self.left(chunks), self.right(chunks))


def test_stream_coordinator_handles_branch_fan_in() -> None:
    ir = BranchPipe().compile()
    inputs = [
        {"chunks": mg.source([f"m{m}-0", f"m{m}-1"], name="chunks")}
        for m in range(3)
    ]
    executor = MultigrainRayExecutor()
    try:
        outputs = list(executor.execute_stream(ir, inputs, max_inflight=2))
    finally:
        executor.shutdown()

    assert [out.values for out in outputs] == [
        [
            f"emb:m{m}-0|emb:m{m}-0",
            f"emb:m{m}-1|emb:m{m}-1",
        ]
        for m in range(3)
    ]


class SameNameMapPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.op = mg.Map(
            SlowEmbed,
            name="shared-name",
            physical=PhysicalHints(replicas=2),
        )

    def forward(self, chunks):
        return self.op(chunks)


class SameNameFilterPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.op = mg.Filter(
            SlowDrop,
            name="shared-name",
            physical=PhysicalHints(replicas=2),
        )

    def forward(self, chunks):
        return self.op(chunks)


def test_executor_rejects_cross_graph_actor_pool_name_collision() -> None:
    executor = MultigrainRayExecutor()
    rows = mg.source(["a", "b"], name="chunks")
    try:
        executor.execute(SameNameMapPipe().compile(), {"chunks": rows})
        with pytest.raises(ValueError, match="actor pool name collision"):
            executor.execute(SameNameFilterPipe().compile(), {"chunks": rows})
    finally:
        executor.shutdown()
