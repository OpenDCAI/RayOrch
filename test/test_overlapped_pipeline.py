import time

import ray

from rayorch import OverlappedPipeline, RayModule
from rayorch.dispatch_mode import dispatch_one_to_all


def _collect_first_identical(rm, outs):
    """ONE_TO_ALL：各 replica 结果相同，对外与 replicas=1 一致。"""
    return outs[0]


def _sleep_module(sleep_s: float, replicas: int) -> RayModule:
    if replicas == 1:
        return RayModule(SleepStageOp, replicas=1).pre_init(sleep_s)
    return RayModule(
        SleepStageOp,
        replicas=replicas,
        dispatch_fn=dispatch_one_to_all,
        collect_fn=_collect_first_identical,
    ).pre_init(sleep_s)


def _merge_module(replicas: int) -> RayModule:
    if replicas == 1:
        return RayModule(MergeSumOp, replicas=1).pre_init()
    return RayModule(
        MergeSumOp,
        replicas=replicas,
        dispatch_fn=dispatch_one_to_all,
        collect_fn=_collect_first_identical,
    ).pre_init()


class SleepStageOp:
    def __init__(self, sleep_s: float):
        self.sleep_s = sleep_s

    def run(self, x):
        time.sleep(self.sleep_s)
        return x + 1


class MergeSumOp:
    def run(self, a, b):
        return a + b


def _cleanup_modules(*modules: RayModule) -> None:
    for m in modules:
        for actor in getattr(m, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


def run_overlapped_complex_dag(n_items: int = 8, replicas: int = 1) -> tuple[list[int], float]:
    num_cpus = 32 if replicas > 1 else 16
    ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
    pre = l1 = l2 = r1 = r2 = merge_left = merge_right = merge_final = tail = None
    try:
        pre = _sleep_module(0.03, replicas)
        l1 = _sleep_module(0.03, replicas)
        l2 = _sleep_module(0.03, replicas)
        r1 = _sleep_module(0.03, replicas)
        r2 = _sleep_module(0.03, replicas)
        merge_left = _merge_module(replicas)
        merge_right = _merge_module(replicas)
        merge_final = _merge_module(replicas)
        tail = _sleep_module(0.03, replicas)

        class OverlappedComplexPipe(OverlappedPipeline):
            def __init__(self):
                self.pre = pre
                self.l1 = l1
                self.l2 = l2
                self.r1 = r1
                self.r2 = r2
                self.merge_left = merge_left
                self.merge_right = merge_right
                self.merge_final = merge_final
                self.tail = tail
                super().__init__(max_inflight=8)

            def forward(self, x):
                y = self.pre(x)
                left = self.merge_left(self.l1(y), self.l2(y))
                right = self.merge_right(self.r1(y), self.r2(y))
                merged = self.merge_final(left, right)
                return self.tail(merged)

        pipe = OverlappedComplexPipe()
        t0 = time.perf_counter()
        out = pipe(list(range(n_items)))
        elapsed = time.perf_counter() - t0
        ray.timeline("overlapped_complex_dag_timeline.json")
        return out, elapsed
    finally:
        if all(
            x is not None
            for x in (pre, l1, l2, r1, r2, merge_left, merge_right, merge_final, tail)
        ):
            _cleanup_modules(pre, l1, l2, r1, r2, merge_left, merge_right, merge_final, tail)
        if ray.is_initialized():
            ray.shutdown()


def test_overlapped_pipeline_complex_dag_batch8():
    out, elapsed = run_overlapped_complex_dag(n_items=8)
    assert out == [4 * x + 9 for x in range(8)]
    assert elapsed < 6.0


def test_overlapped_pipeline_complex_dag_batch8_multi_replica():
    out, elapsed = run_overlapped_complex_dag(n_items=8, replicas=3)
    assert out == [4 * x + 9 for x in range(8)]
    assert elapsed < 12.0


if __name__ == "__main__":
    out, elapsed = run_overlapped_complex_dag(n_items=8)
    print("Overlapped complex DAG output:", out)
    print(f"Overlapped complex DAG elapsed: {elapsed:.3f}s")
    print("Timeline saved to: overlapped_complex_dag_timeline.json")
