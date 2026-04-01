import time

import pytest
import ray

from rayorch import OverlappedPipeline, RayModule
from rayorch.dispatch_mode import collect_concat, dispatch_broadcast, dispatch_shard_all_args_mod


def _collect_first_identical(rm, outs):
    """BROADCAST：各 replica 结果相同，对外与 replicas=1 一致。"""
    return outs[0]


def _sleep_module(sleep_s: float, replicas: int) -> RayModule:
    if replicas == 1:
        return RayModule(SleepStageOp, replicas=1).pre_init(sleep_s)
    return RayModule(
        SleepStageOp,
        replicas=replicas,
        dispatch_fn=dispatch_broadcast,
        collect_fn=_collect_first_identical,
    ).pre_init(sleep_s)


def _merge_module(replicas: int) -> RayModule:
    if replicas == 1:
        return RayModule(MergeSumOp, replicas=1).pre_init()
    return RayModule(
        MergeSumOp,
        replicas=replicas,
        dispatch_fn=dispatch_broadcast,
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


class ShardPlusOneOp:
    def run(self, xs):
        return [x + 1 for x in xs]


class SumListOp:
    def run(self, xs):
        return sum(xs)


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


def test_overlapped_pipeline_rejects_multi_ref_shard_dependency():
    """
    上游是 shard 多 replica（future 含多个 completion refs），下游在 graphless
    OverlappedPipeline 中直接消费该 future 时，必须显式报错，避免静默 refs[0] 退化。
    """
    ray.init(ignore_reinit_error=True, num_cpus=16)
    shard = sink = None
    try:
        shard = RayModule(
            ShardPlusOneOp,
            replicas=3,
            dispatch_fn=dispatch_shard_all_args_mod,
            collect_fn=collect_concat,
        ).pre_init()
        sink = RayModule(
            SumListOp,
            replicas=1,
        ).pre_init()

        class ShardThenSinkPipe(OverlappedPipeline):
            def __init__(self):
                self.shard = shard
                self.sink = sink
                super().__init__(max_inflight=2)

            def forward(self, x):
                y = self.shard(x)
                return self.sink(y)

        pipe = ShardThenSinkPipe()
        with pytest.raises(ValueError, match="multiple completion refs"):
            pipe([list(range(8))])
    finally:
        if shard is not None and sink is not None:
            _cleanup_modules(shard, sink)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    out, elapsed = run_overlapped_complex_dag(n_items=8)
    print("Overlapped complex DAG output:", out)
    print(f"Overlapped complex DAG elapsed: {elapsed:.3f}s")
    print("Timeline saved to: overlapped_complex_dag_timeline.json")
