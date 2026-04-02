from __future__ import annotations

import ray

from rayorch import RayModule
from rayorch.dag_new_pipeline import DagPipeline, PipeRef


class SumOp:
    def run(self, a: list[int], b: list[int]) -> list[int]:
        return [x + y for x, y in zip(a, b)]


class MultiInputPipe(DagPipeline):
    def __init__(self):
        self.add = RayModule(SumOp, replicas=1).pre_init()
        super().__init__(max_batches_inflight=2, stage_options={"add": {"compute_inflight": 2}})

    def forward(self, a: PipeRef, b: PipeRef) -> PipeRef:
        return self.add(a, b)


def _cleanup(pipe: MultiInputPipe) -> None:
    for actor in getattr(pipe.add, "actors", []):
        try:
            ray.kill(actor)
        except Exception:
            pass


def main() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=2)
    pipe = MultiInputPipe()
    try:
        # 故意构造负例：两列 batch 数不一致（2 vs 1）
        a_batches = [[1, 2, 3], [4, 5, 6]]
        b_batches = [[10, 20, 30]]
        # 预期抛 ValueError，并显示完整 traceback
        pipe(a_batches, b_batches)
    finally:
        _cleanup(pipe)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

