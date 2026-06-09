"""
Complex DAG stress test for ``rayorch.dag_new_pipeline``.

This test validates:
- Multi-batch execution under irregular operator latency.
- Complex DAG dependencies with both 1->3 and 3->1 argument patterns.
- Deterministic output correctness checks.
- Timeline export via ``ray.timeline(...)``.
"""

from __future__ import annotations

import time
from typing import Tuple

import ray

from rayorch import RayModule
from rayorch.dag_new_pipeline import DagExecutor, DagPipeline


def _cleanup_modules(*modules: RayModule) -> None:
    for m in modules:
        for actor in getattr(m, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


class _JitterSleepMixin:
    """Per-actor deterministic jitter for fast stress tests."""

    def __init__(self, base_sleep_s: float):
        self.base_sleep_s = float(base_sleep_s)
        self.call_idx = 0

    def _sleep(self) -> None:
        jitter = (-0.01, 0.0, 0.01)[self.call_idx % 3]
        self.call_idx += 1
        delay = max(0.01, min(1.0, float(self.base_sleep_s + jitter)))
        time.sleep(delay)


class PreOp(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> list[int]:
        self._sleep()
        return [v + 10 for v in x]


class SplitAOp(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> tuple[list[int], list[int], list[int]]:
        self._sleep()
        return ([v + 1 for v in x], [v * 2 for v in x], [v - 1 for v in x])


class Branch1Op(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> list[int]:
        self._sleep()
        return [2 * v for v in x]


class Branch2Op(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> list[int]:
        self._sleep()
        return [v + 5 for v in x]


class Branch3Op(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> list[int]:
        self._sleep()
        return [v - 3 for v in x]


class JoinAOp(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, a: list[int], b: list[int], c: list[int]) -> list[int]:
        self._sleep()
        return [x + y + z for x, y, z in zip(a, b, c)]


class SplitBOp(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> tuple[list[int], list[int], list[int]]:
        self._sleep()
        return ([v for v in x], [v + 2 for v in x], [2 * v for v in x])


class Tail1Op(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> list[int]:
        self._sleep()
        return [3 * v for v in x]


class Tail2Op(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> list[int]:
        self._sleep()
        return [v - 1 for v in x]


class Tail3Op(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, x: list[int]) -> list[int]:
        self._sleep()
        return [v // 2 for v in x]


class JoinBOp(_JitterSleepMixin):
    def __init__(self, base_sleep_s: int):
        super().__init__(base_sleep_s)

    def run(self, a: list[int], b: list[int], c: list[int]) -> list[int]:
        self._sleep()
        return [x + y + z for x, y, z in zip(a, b, c)]


class ComplexDagNewPipe(DagPipeline):
    """
    DAG shape:
    input -> pre -> split_a(1->3) -> (b1,b2,b3) -> join_a(3->1)
          -> split_b(1->3) -> (t1,t2,t3) -> join_b(3->1) -> output
    """

    def __init__(self):
        self.pre = RayModule(PreOp, replicas=1, max_inflight=2).pre_init(0.03)
        self.split_a = RayModule(SplitAOp, replicas=1, max_inflight=2).pre_init(0.04)
        self.branch_1 = RayModule(Branch1Op, replicas=1, max_inflight=2).pre_init(0.07)
        self.branch_2 = RayModule(Branch2Op, replicas=1, max_inflight=2).pre_init(0.08)
        self.branch_3 = RayModule(Branch3Op, replicas=1, max_inflight=2).pre_init(0.09)
        self.join_a = RayModule(JoinAOp, replicas=1, max_inflight=2).pre_init(0.05)
        self.split_b = RayModule(SplitBOp, replicas=1, max_inflight=2).pre_init(0.10)
        self.tail_1 = RayModule(Tail1Op, replicas=1, max_inflight=2).pre_init(0.11)
        self.tail_2 = RayModule(Tail2Op, replicas=1, max_inflight=2).pre_init(0.07)
        self.tail_3 = RayModule(Tail3Op, replicas=1, max_inflight=2).pre_init(0.04)
        self.join_b = RayModule(JoinBOp, replicas=1, max_inflight=2).pre_init(0.08)
        super().__init__()

    def forward(self, x: list[int]) -> list[int]:
        y = self.pre(x)
        a1, a2, a3 = self.split_a(y)  # 1 -> 3
        j1 = self.join_a(self.branch_1(a1), self.branch_2(a2), self.branch_3(a3))  # 3 -> 1
        b1, b2, b3 = self.split_b(j1)  # 1 -> 3
        out = self.join_b(self.tail_1(b1), self.tail_2(b2), self.tail_3(b3))  # 3 -> 1
        return out


def _expected(batch: list[int]) -> list[int]:
    # Derived closed-form output for each element x:
    # a = x + 10
    # d = 5a + 3
    # out = 5d + 1 = 25x + 266
    return [25 * x + 266 for x in batch]


def test_dag_new_pipeline_complex_multi_batch_with_timeline():
    ray.init(ignore_reinit_error=True, num_cpus=12)
    pipe = None
    try:
        pipe = ComplexDagNewPipe()
        inputs = [
            list(range(0, 5)),
            list(range(7, 11)),
            [13, 17, 19],
        ]
        executor = DagExecutor(pipe, max_batches_inflight=4)
        outputs = executor.run(inputs)
        expected = [_expected(b) for b in inputs]

        assert outputs == expected

        # Export timeline for visual inspection of overlap behavior.
        ray.timeline("dag_new_pipeline_complex_timeline.json")
    finally:
        if pipe is not None:
            _cleanup_modules(
                pipe.pre,
                pipe.split_a,
                pipe.branch_1,
                pipe.branch_2,
                pipe.branch_3,
                pipe.join_a,
                pipe.split_b,
                pipe.tail_1,
                pipe.tail_2,
                pipe.tail_3,
                pipe.join_b,
            )
        if ray.is_initialized():
            ray.shutdown()

