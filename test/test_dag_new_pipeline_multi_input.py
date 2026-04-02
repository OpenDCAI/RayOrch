from __future__ import annotations

import json
from pathlib import Path

import ray

from rayorch import RayModule
from rayorch.dag_new_pipeline import DagExecutor, DagPipeline, PipeRef


def _cleanup_modules(*modules: RayModule) -> None:
    for m in modules:
        for actor in getattr(m, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


class SumOp:
    def run(self, a: list[int], b: list[int]) -> list[int]:
        return [x + y for x, y in zip(a, b)]


class ScaleOp:
    def run(self, x: list[int]) -> list[int]:
        return [3 * v for v in x]


class MultiInputPipe(DagPipeline):
    def __init__(self):
        self.add = RayModule(SumOp, replicas=1, max_inflight=2).pre_init()
        self.scale = RayModule(ScaleOp, replicas=1, max_inflight=2).pre_init()
        super().__init__()

    def forward(self, a: PipeRef, b: PipeRef) -> PipeRef:
        return self.scale(self.add(a, b))


def _dump_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def test_dag_new_pipeline_multi_root_inputs_and_file_output(tmp_path: Path):
    ray.init(ignore_reinit_error=True, num_cpus=4)
    pipe = None
    try:
        pipe = MultiInputPipe()

        a_batches = [[1, 2, 3], [10, 20]]
        b_batches = [[4, 5, 6], [1, 2]]

        # serial positional multi-root call
        outputs_positional = pipe(a_batches, b_batches)
        # serial keyword multi-root call
        outputs_named = pipe(a=a_batches, b=b_batches)
        expected = [[15, 21, 27], [33, 66]]

        assert outputs_positional == expected
        assert outputs_named == expected

        # overlapped via DagExecutor
        dag = DagExecutor(max_batches_inflight=4)
        outputs_dag = dag.run(pipe, a_batches, b_batches)
        assert outputs_dag == expected

        out_file = tmp_path / "dag_new_multi_input_output.json"
        _dump_json(out_file, outputs_positional)
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded == expected
    finally:
        if pipe is not None:
            _cleanup_modules(pipe.add, pipe.scale)
        if ray.is_initialized():
            ray.shutdown()

