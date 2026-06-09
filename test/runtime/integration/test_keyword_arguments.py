"""Runtime DAG execution through positional and keyword operator calls."""
from __future__ import annotations

import pytest

from rayorch import DagPipeline, RuntimeDagExecutor, RuntimeRayModule
from rayorch.runtime import BadRecordError, MicroBatch

pytestmark = pytest.mark.usefixtures("ray_cluster")


class CombineOp:
    def run(self, first, second, third):
        return [
            f"{a}|{b}|{c}"
            for a, b, c in zip(first, second, third)
        ]


class RejectSecondOp:
    def run(self, first, second):
        for index, value in enumerate(second):
            if value == "bad":
                raise BadRecordError("bad second input", index=index)
        return [f"{a}:{b}" for a, b in zip(first, second)]


def _source() -> MicroBatch:
    return MicroBatch.source(
        {
            "a": ["a0", "a1"],
            "b": ["b0", "b1"],
            "c": ["c0", "c1"],
        },
        dataset="kwargs",
    )


def test_runtime_executes_all_keyword_args_in_parameter_order() -> None:
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.combine = RuntimeRayModule(CombineOp)
            super().__init__()

        def forward(self, a, b, c):
            output = self.combine(third=c, first=a, second=b)
            return output

    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(_source())

    assert result.batch.columns["output"] == [
        "a0|b0|c0",
        "a1|b1|c1",
    ]
    assert result.trace_row("kwargs:0") == ["combine"]


def test_runtime_executes_mixed_positional_and_keyword_args() -> None:
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.combine = RuntimeRayModule(CombineOp)
            super().__init__()

        def forward(self, a, b, c):
            output = self.combine(a, third=c, second=b)
            return output

    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(_source())

    assert result.batch.columns["output"] == [
        "a0|b0|c0",
        "a1|b1|c1",
    ]


def test_runtime_keyword_fanin_aligns_rows_and_reports_failure() -> None:
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.reject = RuntimeRayModule(RejectSecondOp)
            super().__init__()

        def forward(self, a, b):
            output = self.reject(second=b, first=a)
            return output

    source = MicroBatch.source(
        {"a": ["a0", "a1", "a2"], "b": ["ok", "bad", "ok2"]},
        dataset="kwargs-error",
    )
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.row_ids == [source.row_ids[0], source.row_ids[2]]
    assert result.batch.columns["output"] == ["a0:ok", "a2:ok2"]
    assert len(result.quarantined) == 1
    error = result.quarantined[0]
    assert error.row_id == source.row_ids[1]
    assert error.op == "reject"
    assert error.values == {"first": "a1", "second": "bad"}


def test_executor_accepts_pipeline_source_columns_as_keywords() -> None:
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.combine = RuntimeRayModule(CombineOp)
            super().__init__()

        def forward(self, a, b, c):
            output = self.combine(first=a, second=b, third=c)
            return output

    with RuntimeDagExecutor(Pipe(), batch_size=2, dataset="source-kwargs") as executor:
        results = executor.run(
            c=["c0", "c1"],
            a=["a0", "a1"],
            b=["b0", "b1"],
        )

    assert results[0].batch.columns["output"] == [
        "a0|b0|c0",
        "a1|b1|c1",
    ]
    assert results[0].batch.row_ids == ["source-kwargs:0", "source-kwargs:1"]
