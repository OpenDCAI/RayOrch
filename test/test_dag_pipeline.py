"""
DAG path: :class:`rayorch.DagPipeline`, :class:`rayorch.DagPipelineExecutor`, ``PipeRef``,
and linear :class:`rayorch.PipelineExecutor`.

Graphless overlap: ``test_overlapped_pipeline.py`` (:class:`OverlappedPipeline`).
"""

import time

import ray

from rayorch import DagNode, DagPipeline, DagPipelineExecutor, PipelineExecutor, RayModule
from rayorch.dispatch_mode import dispatch_broadcast


def _collect_first_identical(rm, outs):
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


def _cleanup_modules(*modules: RayModule) -> None:
    for m in modules:
        for actor in getattr(m, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


def run_linear_two_stage_demo(
    manage_ray: bool = True,
) -> tuple[list[int], list[int], float, float]:
    """Two-stage chain via :class:`PipelineExecutor` (wraps :class:`DagPipelineExecutor`)."""
    if manage_ray and not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
    stage1 = None
    stage2 = None
    try:
        stage1 = RayModule(SleepStageOp, replicas=1).pre_init(0.5)
        stage2 = RayModule(SleepStageOp, replicas=1).pre_init(0.5)
        inputs = [0, 10, 20]

        t0 = time.perf_counter()
        baseline = []
        for x in inputs:
            baseline.append(stage2(stage1(x)))
        baseline_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        executor = PipelineExecutor([stage1, stage2], max_inflight=[2, 2])
        pipelined = executor.run(inputs)
        pipeline_t = time.perf_counter() - t1

        ray.timeline("pipeline_linear_overlap_timeline.json")

        return baseline, pipelined, baseline_t, pipeline_t
    finally:
        if stage1 is not None and stage2 is not None:
            _cleanup_modules(stage1, stage2)
        if manage_ray and ray.is_initialized():
            ray.shutdown()


def test_pipeline_executor_overlaps_batches():
    baseline, pipelined, baseline_t, pipeline_t = run_linear_two_stage_demo()
    assert baseline == [2, 12, 22]
    assert pipelined == baseline
    assert pipeline_t < baseline_t - 0.5


def test_dag_pipeline_executor_explicit_tape():
    """``DagPipelineExecutor`` without subclassing: hand-built :class:`DagNode` list."""
    ray.init(ignore_reinit_error=True, num_cpus=4)
    pre = a = b = merge = None
    try:
        pre = RayModule(SleepStageOp, replicas=1).pre_init(0.05)
        a = RayModule(SleepStageOp, replicas=1).pre_init(0.05)
        b = RayModule(SleepStageOp, replicas=1).pre_init(0.05)
        merge = RayModule(MergeSumOp, replicas=1).pre_init()
        nodes = [
            DagNode("pre", pre, ("input",), {}, max_inflight=4),
            DagNode("a", a, ("pre",), {}, max_inflight=4),
            DagNode("b", b, ("pre",), {}, max_inflight=4),
            DagNode("merge", merge, ("a", "b"), {}, max_inflight=4),
        ]
        ex = DagPipelineExecutor(nodes)
        out = ex.run([0, 10, 20], outputs=("merge",))
        assert out == [4, 24, 44]
    finally:
        if all(x is not None for x in (pre, a, b, merge)):
            _cleanup_modules(pre, a, b, merge)
        if ray.is_initialized():
            ray.shutdown()


def run_fork_join_class_demo(manage_ray: bool = True) -> tuple[list[int], float]:
    """Subclass ``DagPipeline``: fork after ``pre``, merge two branches."""
    if manage_ray and not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
    preprocess = branch_a = branch_b = merge = None
    try:
        preprocess = RayModule(SleepStageOp, replicas=1).pre_init(0.2)
        branch_a = RayModule(SleepStageOp, replicas=1).pre_init(0.3)
        branch_b = RayModule(SleepStageOp, replicas=1).pre_init(0.3)
        merge = RayModule(MergeSumOp, replicas=1).pre_init()

        class ForkJoinPipe(DagPipeline):
            def __init__(self):
                self.pre = preprocess
                self.a = branch_a
                self.b = branch_b
                self.merge = merge
                super().__init__(
                    stage_options={
                        "pre": {"max_inflight": 2},
                        "a": {"max_inflight": 2},
                        "b": {"max_inflight": 2},
                        "merge": {"max_inflight": 2},
                    }
                )

            def forward(self, x):
                y = self.pre(x)
                return self.merge(self.a(y), self.b(y))

        pipe = ForkJoinPipe()
        t0 = time.perf_counter()
        outputs = pipe([0, 10, 20])
        elapsed = time.perf_counter() - t0
        ray.timeline("dag_fork_join_timeline.json")
        return outputs, elapsed
    finally:
        if all(x is not None for x in (preprocess, branch_a, branch_b, merge)):
            _cleanup_modules(preprocess, branch_a, branch_b, merge)
        if manage_ray and ray.is_initialized():
            ray.shutdown()


def test_dag_fork_join_subclass_outputs():
    outputs, _ = run_fork_join_class_demo()
    assert outputs == [4, 24, 44]


def run_dag_stress_demo(
    n_items: int = 64, manage_ray: bool = True
) -> tuple[list[int], list[int], float, float]:
    if manage_ray and not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=8)
    pre = a = b = merge = None
    try:
        pre = RayModule(SleepStageOp, replicas=1).pre_init(0.05)
        a = RayModule(SleepStageOp, replicas=1).pre_init(0.05)
        b = RayModule(SleepStageOp, replicas=1).pre_init(0.05)
        merge = RayModule(MergeSumOp, replicas=1).pre_init()

        class StressPipe(DagPipeline):
            def __init__(self):
                self.pre = pre
                self.a = a
                self.b = b
                self.merge = merge
                super().__init__(
                    stage_options={
                        "pre": {"max_inflight": 8},
                        "a": {"max_inflight": 8},
                        "b": {"max_inflight": 8},
                        "merge": {"max_inflight": 8},
                    }
                )

            def forward(self, x):
                y = self.pre(x)
                return self.merge(self.a(y), self.b(y))

        pipe = StressPipe()
        inputs = list(range(n_items))

        t0 = time.perf_counter()
        baseline = []
        for x in inputs:
            y = pre(x)
            baseline.append(merge(a(y), b(y)))
        baseline_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        pipelined = pipe(inputs)
        pipeline_t = time.perf_counter() - t1
        ray.timeline("dag_stress_timeline.json")
        return baseline, pipelined, baseline_t, pipeline_t
    finally:
        if all(x is not None for x in (pre, a, b, merge)):
            _cleanup_modules(pre, a, b, merge)
        if manage_ray and ray.is_initialized():
            ray.shutdown()


def test_dag_many_batches_overlap():
    baseline, pipelined, baseline_t, pipeline_t = run_dag_stress_demo(n_items=64)
    assert pipelined == baseline
    assert pipelined == [2 * x + 4 for x in range(64)]
    assert pipeline_t < baseline_t


def run_complex_dag_demo(
    n_items: int = 8, manage_ray: bool = True, replicas: int = 1
) -> tuple[list[int], list[int], float, float]:
    if manage_ray and not ray.is_initialized():
        num_cpus = 32 if replicas > 1 else 16
        ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
    pre = l1 = l2 = r1 = r2 = ml = mr = mf = tail = None
    try:
        pre = _sleep_module(0.03, replicas)
        l1 = _sleep_module(0.03, replicas)
        l2 = _sleep_module(0.03, replicas)
        r1 = _sleep_module(0.03, replicas)
        r2 = _sleep_module(0.03, replicas)
        ml = _merge_module(replicas)
        mr = _merge_module(replicas)
        mf = _merge_module(replicas)
        tail = _sleep_module(0.03, replicas)

        class ComplexPipe(DagPipeline):
            def __init__(self):
                self.pre = pre
                self.l1 = l1
                self.l2 = l2
                self.r1 = r1
                self.r2 = r2
                self.merge_left = ml
                self.merge_right = mr
                self.merge_final = mf
                self.tail = tail
                super().__init__(
                    stage_options={
                        "pre": {"max_inflight": 4},
                        "l1": {"max_inflight": 4},
                        "l2": {"max_inflight": 4},
                        "r1": {"max_inflight": 4},
                        "r2": {"max_inflight": 4},
                        "merge_left": {"max_inflight": 4},
                        "merge_right": {"max_inflight": 4},
                        "merge_final": {"max_inflight": 4},
                        "tail": {"max_inflight": 4},
                    }
                )

            def forward(self, x):
                y = self.pre(x)
                left = self.merge_left(self.l1(y), self.l2(y))
                right = self.merge_right(self.r1(y), self.r2(y))
                merged = self.merge_final(left, right)
                return self.tail(merged)

        pipe = ComplexPipe()
        inputs = list(range(n_items))

        t0 = time.perf_counter()
        baseline = []
        for x in inputs:
            y = pre(x)
            left = ml(l1(y), l2(y))
            right = mr(r1(y), r2(y))
            merged = mf(left, right)
            baseline.append(tail(merged))
        baseline_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        pipelined = pipe(inputs)
        pipeline_t = time.perf_counter() - t1
        ray.timeline("dag_complex_timeline.json")
        return baseline, pipelined, baseline_t, pipeline_t
    finally:
        if all(
            x is not None for x in (pre, l1, l2, r1, r2, ml, mr, mf, tail)
        ):
            _cleanup_modules(pre, l1, l2, r1, r2, ml, mr, mf, tail)
        if manage_ray and ray.is_initialized():
            ray.shutdown()


def test_dag_deep_graph_correctness():
    baseline, pipelined, baseline_t, pipeline_t = run_complex_dag_demo(n_items=8)
    assert pipelined == baseline
    assert pipelined == [4 * x + 9 for x in range(8)]
    assert pipeline_t < baseline_t


def test_dag_deep_graph_multi_replica():
    baseline, pipelined, baseline_t, pipeline_t = run_complex_dag_demo(
        n_items=8, replicas=3
    )
    assert pipelined == baseline
    assert pipelined == [4 * x + 9 for x in range(8)]
    assert pipeline_t < baseline_t


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
