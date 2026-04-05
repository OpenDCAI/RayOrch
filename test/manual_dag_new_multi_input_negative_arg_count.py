from __future__ import annotations

import ray

from rayorch import RayModule
from rayorch.dag_new_pipeline import DagPipeline, PipeRef


class Sum3Op:
    def run(self, a: list[int], b: list[int], c: list[int]) -> list[int]:
        return [x + y + z for x, y, z in zip(a, b, c)]


class MultiInput3Pipe(DagPipeline):
    def __init__(self):
        self.add3 = RayModule(Sum3Op, replicas=1, max_inflight=2).pre_init()
        super().__init__()

    def forward(self, a: PipeRef, b: PipeRef, c: PipeRef) -> PipeRef:
        return self.add3(a, b, c)


def _cleanup(pipe: MultiInput3Pipe) -> None:
    for actor in getattr(pipe.add3, "actors", []):
        try:
            ray.kill(actor)
        except Exception:
            pass


def main() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=2)
    pipe = MultiInput3Pipe()
    try:
        # 负例：forward(a,b,c)，实际传 4 个位置参数
        a_batches = [[1, 2], [3, 4]]
        b_batches = [[10, 20], [30, 40]]
        c_batches = [[100, 200], [300, 400]]
        d_batches = [[1000, 2000], [3000, 4000]]
        # 预期抛 ValueError 并打印完整 traceback
        pipe(a_batches, b_batches, c_batches, d_batches)
    finally:
        _cleanup(pipe)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

