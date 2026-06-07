from __future__ import annotations

import time

import ray

from rayorch import DagExecutor, DagPipeline, PipeRef, RayModule, RuntimeDagExecutor
from rayorch import RuntimeRayModule
from rayorch.runtime import MicroBatch

from test.runtime_test_utils import cleanup_modules


class SleepOp:
    def __init__(self, seconds: float = 0.35):
        self.seconds = seconds

    def run(self, x):
        time.sleep(self.seconds)
        return [value + 1 for value in x]


class NormalSleepPipe(DagPipeline):
    def __init__(self):
        self.a = RayModule(SleepOp, replicas=1, max_inflight=2).pre_init()
        self.b = RayModule(SleepOp, replicas=1, max_inflight=2).pre_init()
        self.c = RayModule(SleepOp, replicas=1, max_inflight=2).pre_init()
        super().__init__()

    def forward(self, x: PipeRef):
        return self.c(self.b(self.a(x)))


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

    def forward(self, x: PipeRef):
        y = self.a(x)
        y = self.b(y)
        y = self.c(y)
        return y


def test_runtime_executor_uses_ray_replicas() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=8)

    class ReplicaPipe(DagPipeline):
        def __init__(self):
            self.sleep = RuntimeRayModule(
                SleepOp, replicas=4, max_inflight=4, num_outputs=1
            ).pre_init(seconds=2.0)
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.sleep(x)
            return y

    pipe = ReplicaPipe()
    try:
        source = MicroBatch.source({"x": list(range(16))}, dataset="replica-sleep")
        start = time.perf_counter()
        result = RuntimeDagExecutor().run(pipe, source)
        elapsed = time.perf_counter() - start

        assert result.batch.columns["y"] == [value + 1 for value in range(16)]
        assert elapsed < 4.5
    finally:
        cleanup_modules(pipe.sleep)
        ray.shutdown()


def test_runtime_executor_overlaps_microbatches_like_dag_executor() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=12)
    normal_pipe = NormalSleepPipe()
    runtime_pipe = RuntimeSleepPipe()
    try:
        normal_batches = [[i] for i in range(4)]
        runtime_batches = [
            MicroBatch.source({"x": [i]}, dataset=f"overlap-{i}")
            for i in range(4)
        ]

        start = time.perf_counter()
        sequential = RuntimeDagExecutor(max_batches_inflight=1).run(
            runtime_pipe, runtime_batches
        )
        sequential_time = time.perf_counter() - start

        start = time.perf_counter()
        overlapped = RuntimeDagExecutor(max_batches_inflight=4).run(
            runtime_pipe, runtime_batches
        )
        runtime_time = time.perf_counter() - start

        start = time.perf_counter()
        generic = DagExecutor(max_batches_inflight=4).run(normal_pipe, normal_batches)
        generic_time = time.perf_counter() - start

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
        assert generic == [[3], [4], [5], [6]]
        assert runtime_time < sequential_time * 0.8
        assert runtime_time < generic_time * 1.6 + 0.5
    finally:
        cleanup_modules(runtime_pipe.a, runtime_pipe.b, runtime_pipe.c)
        cleanup_modules(normal_pipe.a, normal_pipe.b, normal_pipe.c)
        ray.shutdown()
