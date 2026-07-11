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

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.graph import PhysicalHints
from rayorch.experimental.multigrain.ray_executor import MultigrainRayExecutor

from test.experimental.multigrain.dummy_ops import SLEEP, SlowDrop, SlowEmbed

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("ray_cluster")]


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
    start = time.time()
    serial_out = MultigrainRayExecutor().execute(serial_ir, {"chunks": chunks})
    serial = time.time() - start

    parallel_ir = MapPipe(replicas=4).compile()
    start = time.time()
    parallel_out = MultigrainRayExecutor().execute(parallel_ir, {"chunks": chunks})
    parallel = time.time() - start

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
    # each microbatch ~0.8s; 4 serial ~3.2s, overlapped ~0.8s.
    assert overlap_time < serial_time * 0.6
