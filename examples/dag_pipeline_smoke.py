"""Small manual smoke test for the DAG pipeline API."""
from __future__ import annotations

import ray

from rayorch import DagExecutor, DagPipeline, Dispatch, PipeRef, RayModule


def _cleanup_pipe_modules(pipe: DagPipeline) -> None:
    for value in pipe.__dict__.values():
        if isinstance(value, RayModule):
            for actor in getattr(value, "actors", []):
                try:
                    ray.kill(actor)
                except Exception:
                    pass


class SplitParityOp:
    def run(self, x: list[int]) -> tuple[list[int], list[int]]:
        return [v for v in x if v % 2 == 0], [v for v in x if v % 2 == 1]


class ScaleEvenOp:
    def run(self, x: list[int]) -> list[int]:
        return [2 * v for v in x]


class ScaleOddOp:
    def run(self, x: list[int]) -> list[int]:
        return [3 * v for v in x]


class DemoPipe(DagPipeline):
    def __init__(self, replicas: int):
        self.split = RayModule(
            SplitParityOp,
            replicas=replicas,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=2,
        ).pre_init()
        self.left = RayModule(ScaleEvenOp, replicas=1, max_inflight=2).pre_init()
        self.right = RayModule(ScaleOddOp, replicas=1, max_inflight=2).pre_init()
        super().__init__()

    def forward(self, x: PipeRef) -> tuple[PipeRef, PipeRef]:
        even_in, odd_in = self.split(x)
        return self.left(even_in), self.right(odd_in)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, num_cpus=8)
    pipe = DemoPipe(replicas=3)
    xs = [list(range(10)), list(range(7))]
    out = DagExecutor(max_batches_inflight=4).run(pipe, xs)
    print(out)
    _cleanup_pipe_modules(pipe)
    ray.shutdown()
