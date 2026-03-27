import time

import ray

from rayorch import DummyRunPipeline, PipelineExecutor, RayModule


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


def run_linear_demo(manage_ray: bool = True) -> tuple[list[int], list[int], float, float]:
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

        ray.timeline("pipeline_overlap_timeline.json")

        return baseline, pipelined, baseline_t, pipeline_t
    finally:
        if stage1 is not None and stage2 is not None:
            _cleanup_modules(stage1, stage2)
        if manage_ray and ray.is_initialized():
            ray.shutdown()


def run_dag_demo(manage_ray: bool = True) -> tuple[list[int], float]:
    if manage_ray and not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
    preprocess = None
    branch_a = None
    branch_b = None
    merge = None
    try:
        preprocess = RayModule(SleepStageOp, replicas=1).pre_init(0.2)  # x+1
        branch_a = RayModule(SleepStageOp, replicas=1).pre_init(0.3)  # +1
        branch_b = RayModule(SleepStageOp, replicas=1).pre_init(0.3)  # +1
        merge = RayModule(MergeSumOp, replicas=1).pre_init()  # (a+b)

        class SleepDagPipe(DummyRunPipeline):
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

            def dummy_run(self, x):
                y = self.pre(x)
                a = self.a(y)
                b = self.b(y)
                return self.merge(a, b)

        dag = SleepDagPipe()
        inputs = [0, 10, 20]

        t0 = time.perf_counter()
        outputs = dag.run(inputs)
        elapsed = time.perf_counter() - t0
        ray.timeline("pipeline_overlap_timeline.json")
        return outputs, elapsed
    finally:
        if all(x is not None for x in (preprocess, branch_a, branch_b, merge)):
            _cleanup_modules(preprocess, branch_a, branch_b, merge)
        if manage_ray and ray.is_initialized():
            ray.shutdown()


def test_pipeline_executor_overlaps_batches():
    baseline, pipelined, baseline_t, pipeline_t = run_linear_demo()
    assert baseline == [2, 12, 22]
    assert pipelined == baseline
    # Expect overlap: 3 microbatches x 2 stages x 0.5s = 3.0s serial.
    # With pipelining, wall time should be clearly smaller.
    assert pipeline_t < baseline_t - 0.5


def test_dag_pipeline_outputs():
    outputs, _ = run_dag_demo()
    # x -> pre(+1), then two branches each +1 => (x+2) + (x+2) = 2x + 4
    assert outputs == [4, 24, 44]


def run_dummyrun_stress_demo(
    n_items: int = 64, manage_ray: bool = True
) -> tuple[list[int], list[int], float, float]:
    if manage_ray and not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=8)
    pre = None
    a = None
    b = None
    merge = None
    try:
        pre = RayModule(SleepStageOp, replicas=1).pre_init(0.05)  # x+1
        a = RayModule(SleepStageOp, replicas=1).pre_init(0.05)  # +1
        b = RayModule(SleepStageOp, replicas=1).pre_init(0.05)  # +1
        merge = RayModule(MergeSumOp, replicas=1).pre_init()  # (a+b)

        class StressPipe(DummyRunPipeline):
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

            def dummy_run(self, x):
                y = self.pre(x)
                return self.merge(self.a(y), self.b(y))

        pipe = StressPipe()
        inputs = list(range(n_items))

        # Serial baseline on same modules (without pipeline overlap).
        t0 = time.perf_counter()
        baseline = []
        for x in inputs:
            y = pre(x)
            baseline.append(merge(a(y), b(y)))
        baseline_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        pipelined = pipe.run(inputs)
        pipeline_t = time.perf_counter() - t1
        ray.timeline("dummyrun_stress_timeline.json")
        return baseline, pipelined, baseline_t, pipeline_t
    finally:
        if all(x is not None for x in (pre, a, b, merge)):
            _cleanup_modules(pre, a, b, merge)
        if manage_ray and ray.is_initialized():
            ray.shutdown()


def test_dummyrun_pipeline_stress():
    baseline, pipelined, baseline_t, pipeline_t = run_dummyrun_stress_demo(n_items=64)
    assert pipelined == baseline
    assert pipelined == [2 * x + 4 for x in range(64)]
    # With enough data points, overlap should reduce wall time.
    assert pipeline_t < baseline_t


def run_complex_dag_demo(
    n_items: int = 8, manage_ray: bool = True
) -> tuple[list[int], list[int], float, float]:
    if manage_ray and not ray.is_initialized():
        # Complex graph has 9 long-lived actors; reserve enough logical CPUs.
        ray.init(ignore_reinit_error=True, num_cpus=16)
    pre = None
    l1 = None
    l2 = None
    r1 = None
    r2 = None
    merge_left = None
    merge_right = None
    merge_final = None
    tail = None
    try:
        pre = RayModule(SleepStageOp, replicas=1).pre_init(0.03)
        l1 = RayModule(SleepStageOp, replicas=1).pre_init(0.03)
        l2 = RayModule(SleepStageOp, replicas=1).pre_init(0.03)
        r1 = RayModule(SleepStageOp, replicas=1).pre_init(0.03)
        r2 = RayModule(SleepStageOp, replicas=1).pre_init(0.03)
        merge_left = RayModule(MergeSumOp, replicas=1).pre_init()
        merge_right = RayModule(MergeSumOp, replicas=1).pre_init()
        merge_final = RayModule(MergeSumOp, replicas=1).pre_init()
        tail = RayModule(SleepStageOp, replicas=1).pre_init(0.03)

        class ComplexPipe(DummyRunPipeline):
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

            def dummy_run(self, x):
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
            left = merge_left(l1(y), l2(y))
            right = merge_right(r1(y), r2(y))
            merged = merge_final(left, right)
            baseline.append(tail(merged))
        baseline_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        pipelined = pipe.run(inputs)
        pipeline_t = time.perf_counter() - t1
        ray.timeline("complex_dag_timeline.json")
        return baseline, pipelined, baseline_t, pipeline_t
    finally:
        if all(
            x is not None
            for x in (pre, l1, l2, r1, r2, merge_left, merge_right, merge_final, tail)
        ):
            _cleanup_modules(pre, l1, l2, r1, r2, merge_left, merge_right, merge_final, tail)
        if manage_ray and ray.is_initialized():
            ray.shutdown()


def test_complex_dag_with_batch_8():
    baseline, pipelined, baseline_t, pipeline_t = run_complex_dag_demo(n_items=8)
    assert pipelined == baseline
    # Formula: tail( merge(merge(l1(pre(x)), l2(pre(x))), merge(r1(pre(x)), r2(pre(x)))) )
    # = 4*x + 9
    assert pipelined == [4 * x + 9 for x in range(8)]
    assert pipeline_t < baseline_t


if __name__ == "__main__":
    # Run all demos in one process; allocate enough CPUs for the largest graph.
    ray.init(ignore_reinit_error=True, num_cpus=16)
    try:
        baseline, pipelined, baseline_t, pipeline_t = run_linear_demo(manage_ray=False)
        print("Linear baseline, Done!")
        dag_out, dag_t = run_dag_demo(manage_ray=False)
        print("DAG, Done!")
        s_base, s_pipe, s_base_t, s_pipe_t = run_dummyrun_stress_demo(n_items=64, manage_ray=False)
        print("Stress, Done!")
        c_base, c_pipe, c_base_t, c_pipe_t = run_complex_dag_demo(n_items=8, manage_ray=False)
        print("Complex DAG, Done!")
        print("Baseline output:", baseline)
        print("Pipelined output:", pipelined)
        print(f"Baseline time: {baseline_t:.3f}s")
        print(f"Pipelined time: {pipeline_t:.3f}s")
        print("DAG output:", dag_out)
        print(f"DAG time: {dag_t:.3f}s")
        print("Stress baseline head:", s_base[:8], "...")
        print("Stress pipeline head:", s_pipe[:8], "...")
        print(f"Stress baseline time: {s_base_t:.3f}s")
        print(f"Stress pipeline time: {s_pipe_t:.3f}s")
        print("Stress timeline saved to: dummyrun_stress_timeline.json")
        print("Complex DAG baseline:", c_base)
        print("Complex DAG pipeline:", c_pipe)
        print(f"Complex DAG baseline time: {c_base_t:.3f}s")
        print(f"Complex DAG pipeline time: {c_pipe_t:.3f}s")
        print("Complex DAG timeline saved to: complex_dag_timeline.json")
        print("Timeline saved to: pipeline_overlap_timeline.json")
    finally:
        if ray.is_initialized():
            ray.shutdown()
