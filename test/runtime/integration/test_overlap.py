from __future__ import annotations

import time

import pytest

from rayorch import DagPipeline, RuntimeDagExecutor, RuntimeRayModule
from rayorch.runtime import MicroBatch

from test.runtime.helpers import cleanup_modules

pytestmark = pytest.mark.usefixtures("ray_cluster")


class SleepOp:
    def __init__(self, seconds: float = 0.08):
        self.seconds = seconds

    def run(self, x):
        time.sleep(self.seconds)
        return [value + 1 for value in x]


class RuntimeSleepPipe(DagPipeline):
    def __init__(self):
        self.a = RuntimeRayModule(
            SleepOp, replicas=1, max_inflight=2, num_outputs=1
        ).pre_init()
        self.b = RuntimeRayModule(
            SleepOp, replicas=1, max_inflight=2, num_outputs=1
        ).pre_init()
        self.c = RuntimeRayModule(
            SleepOp, replicas=1, max_inflight=2, num_outputs=1
        ).pre_init()
        super().__init__()

    def forward(self, x: list[int]) -> list[int]:
        y = self.a(x)
        y = self.b(y)
        y = self.c(y)
        return y


def test_runtime_executor_uses_ray_replicas() -> None:
    class ReplicaPipe(DagPipeline):
        def __init__(self):
            self.sleep = RuntimeRayModule(
                SleepOp, replicas=4, max_inflight=4
            ).pre_init(seconds=0.1)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            y = self.sleep(x)
            return y

    pipe = ReplicaPipe()
    executor = RuntimeDagExecutor(pipe)
    try:
        executor.run(MicroBatch.source({"x": [-1]}, dataset="replica-warmup"))
        source = MicroBatch.source({"x": list(range(8))}, dataset="replica-sleep")
        start = time.perf_counter()
        result = executor.run(source)
        elapsed = time.perf_counter() - start

        assert result.batch.columns["y"] == [value + 1 for value in range(8)]
        assert elapsed < 0.8
    finally:
        executor.close()


def test_runtime_executor_overlaps_microbatches() -> None:
    sequential_pipe = RuntimeSleepPipe()
    overlap_pipe = RuntimeSleepPipe()
    try:
        runtime_batches = [
            MicroBatch.source({"x": [i]}, dataset=f"overlap-{i}")
            for i in range(4)
        ]

        sequential_executor = RuntimeDagExecutor(
            sequential_pipe, max_batches_inflight=1
        )
        sequential_executor.run(
            MicroBatch.source({"x": [-1]}, dataset="sequential-warmup")
        )
        start = time.perf_counter()
        sequential = sequential_executor.run(runtime_batches)
        sequential_time = time.perf_counter() - start
        sequential_executor.close()

        overlap_executor = RuntimeDagExecutor(
            overlap_pipe, max_batches_inflight=4
        )
        overlap_executor.run(
            MicroBatch.source({"x": [-1]}, dataset="overlap-warmup")
        )
        start = time.perf_counter()
        overlapped = overlap_executor.run(runtime_batches)
        runtime_time = time.perf_counter() - start
        overlap_executor.close()

        assert [result.batch.columns["y"] for result in overlapped] == [
            [3],
            [4],
            [5],
            [6],
        ]
        assert [result.batch.columns["y"] for result in sequential] == [
            [3],
            [4],
            [5],
            [6],
        ]
        assert runtime_time < sequential_time * 0.8
    finally:
        cleanup_modules(
            sequential_pipe.a,
            sequential_pipe.b,
            sequential_pipe.c,
            overlap_pipe.a,
            overlap_pipe.b,
            overlap_pipe.c,
        )
