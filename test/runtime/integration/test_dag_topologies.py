"""Runtime lineage and error semantics across representative DAG shapes.

All operators preserve document-level record cardinality. A branch may remove
bad rows through quarantine, and required fan-in inputs then use the healthy
row-id intersection.
"""
from __future__ import annotations

from typing import Any

from rayorch import DagPipeline, RuntimeDagExecutor, RuntimeRayModule
from rayorch.runtime import BadRecordError, MicroBatch

import pytest

pytestmark = pytest.mark.usefixtures("ray_cluster")


class AddOp:
    def __init__(self, amount: int) -> None:
        self.amount = amount

    def run(self, values: list[int]) -> list[int]:
        return [value + self.amount for value in values]


class ScaleOp:
    def __init__(self, factor: int) -> None:
        self.factor = factor

    def run(self, values: list[int]) -> list[int]:
        return [value * self.factor for value in values]


class Sum2Op:
    def run(self, left: list[int], right: list[int]) -> list[int]:
        return [a + b for a, b in zip(left, right)]


class Sum3Op:
    def run(
        self,
        first: list[int],
        second: list[int],
        third: list[int],
    ) -> list[int]:
        return [a + b + c for a, b, c in zip(first, second, third)]


class SplitPairOp:
    def run(self, values: list[int]) -> tuple[list[int], list[int]]:
        return (
            [value + 1 for value in values],
            [value * 10 for value in values],
        )


class RejectTextOp:
    def __init__(self, token: str) -> None:
        self.token = token

    def run(self, values: list[str]) -> list[int]:
        for index, value in enumerate(values):
            if self.token in value:
                raise BadRecordError(f"rejected token {self.token!r}", index=index)
        return [len(value) for value in values]


class RejectAllOp:
    def run(self, values: list[str]) -> list[int]:
        raise BadRecordError("all rows rejected")


class FailMergedRowOp:
    def __init__(self, bad_left: int) -> None:
        self.bad_left = bad_left

    def run(self, left: list[int], right: list[int]) -> list[int]:
        for index, value in enumerate(left):
            if value == self.bad_left:
                raise BadRecordError("merged row failed", index=index)
        return [a + b for a, b in zip(left, right)]


class FailValueOp:
    def __init__(self, bad_value: int) -> None:
        self.bad_value = bad_value

    def run(self, values: list[int]) -> list[int]:
        for index, value in enumerate(values):
            if value == self.bad_value:
                raise BadRecordError("downstream row failed", index=index)
        return [value + 1 for value in values]


def _run(pipeline: DagPipeline, columns: dict[str, list[Any]]):
    with RuntimeDagExecutor(pipeline) as executor:
        return executor.run(MicroBatch.source(columns, dataset="topology"))


def test_three_way_fanin_preserves_shared_ancestor_once() -> None:
    # source -> base -> (left, middle, right) -> merge
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.base = RuntimeRayModule(AddOp).pre_init(1)
            self.left = RuntimeRayModule(AddOp).pre_init(1)
            self.middle = RuntimeRayModule(ScaleOp).pre_init(2)
            self.right = RuntimeRayModule(ScaleOp).pre_init(10)
            self.merge = RuntimeRayModule(Sum3Op)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            shared = self.base(x)
            merged = self.merge(
                self.left(shared),
                self.middle(shared),
                self.right(shared),
            )
            return merged

    result = _run(Pipe(), {"x": [1, 2]})

    assert result.batch.columns["merged"] == [27, 40]
    assert result.trace_row("topology:0") == [
        "base",
        "left",
        "middle",
        "right",
        "merge",
    ]


def test_nested_diamonds_keep_complete_lineage() -> None:
    # source -> base -> diamond -> merge -> diamond -> final
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.base = RuntimeRayModule(AddOp).pre_init(1)
            self.left = RuntimeRayModule(AddOp).pre_init(1)
            self.right = RuntimeRayModule(ScaleOp).pre_init(2)
            self.merge = RuntimeRayModule(Sum2Op)
            self.tail_left = RuntimeRayModule(AddOp).pre_init(1)
            self.tail_right = RuntimeRayModule(ScaleOp).pre_init(2)
            self.final = RuntimeRayModule(Sum2Op)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            shared = self.base(x)
            merged = self.merge(self.left(shared), self.right(shared))
            output = self.final(
                self.tail_left(merged),
                self.tail_right(merged),
            )
            return output

    result = _run(Pipe(), {"x": [1, 2]})

    assert result.batch.columns["output"] == [22, 31]
    assert result.trace_row("topology:0") == [
        "base",
        "left",
        "right",
        "merge",
        "tail_left",
        "tail_right",
        "final",
    ]


def test_source_can_join_a_deeper_derived_branch() -> None:
    # source ---------------------> merge
    # source -> derived ----------> merge
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.derived = RuntimeRayModule(AddOp).pre_init(1)
            self.merge = RuntimeRayModule(Sum2Op)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            merged = self.merge(x, self.derived(x))
            return merged

    result = _run(Pipe(), {"x": [1, 2]})

    assert result.batch.columns["merged"] == [3, 5]
    assert result.trace_row("topology:0") == ["derived", "merge"]


def test_outputs_from_same_node_do_not_create_false_branch_lineage() -> None:
    # source -> split -(left, right)-> merge
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.split = RuntimeRayModule(SplitPairOp, num_outputs=2)
            self.merge = RuntimeRayModule(Sum2Op)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            left, right = self.split(x)
            merged = self.merge(left, right)
            return merged

    result = _run(Pipe(), {"x": [1, 2]})

    assert result.batch.columns["merged"] == [12, 23]
    merge_path = result.batch.path_ids[0]
    assert len(result.paths[merge_path].parents) == 1
    assert result.trace_row("topology:0") == ["split", "merge"]


def test_common_ancestor_failure_is_quarantined_once_before_fanout() -> None:
    # source -> clean -> (left, right) -> merge
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.clean = RuntimeRayModule(RejectTextOp).pre_init("bad")
            self.left = RuntimeRayModule(AddOp).pre_init(1)
            self.right = RuntimeRayModule(ScaleOp).pre_init(10)
            self.merge = RuntimeRayModule(Sum2Op)
            super().__init__()

        def forward(self, x: list[str]) -> list[int]:
            shared = self.clean(x)
            merged = self.merge(self.left(shared), self.right(shared))
            return merged

    source = MicroBatch.source(
        {"x": ["a", "bad", "ccc"]},
        dataset="ancestor-error",
    )
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.row_ids == [source.row_ids[0], source.row_ids[2]]
    assert result.batch.columns["merged"] == [12, 34]
    assert [(record.row_id, record.op) for record in result.quarantined] == [
        (source.row_ids[1], "clean"),
    ]
    assert result.trace_row(source.row_ids[0]) == [
        "clean",
        "left",
        "right",
        "merge",
    ]


def test_independent_branch_failures_use_row_intersection() -> None:
    # source -> reject "left" --\
    # source -> reject "right" --+-> merge
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.left = RuntimeRayModule(RejectTextOp).pre_init("left")
            self.right = RuntimeRayModule(RejectTextOp).pre_init("right")
            self.merge = RuntimeRayModule(Sum2Op)
            super().__init__()

        def forward(self, x: list[str]) -> list[int]:
            merged = self.merge(self.left(x), self.right(x))
            return merged

    source = MicroBatch.source(
        {"x": ["ok", "left-bad", "right-bad", "ok2"]},
        dataset="branch-filter",
    )
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.row_ids == [source.row_ids[0], source.row_ids[3]]
    assert result.batch.columns["merged"] == [4, 6]
    assert {(record.row_id, record.op) for record in result.quarantined} == {
        (source.row_ids[1], "left"),
        (source.row_ids[2], "right"),
    }


def test_same_row_can_have_independent_errors_on_two_branches() -> None:
    # Independent branch execution produces two error events for the same row.
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.left = RuntimeRayModule(RejectTextOp).pre_init("bad")
            self.right = RuntimeRayModule(RejectTextOp).pre_init("bad")
            self.merge = RuntimeRayModule(Sum2Op)
            super().__init__()

        def forward(self, x: list[str]) -> list[int]:
            merged = self.merge(self.left(x), self.right(x))
            return merged

    source = MicroBatch.source(
        {"x": ["ok", "bad"]},
        dataset="double-error",
    )
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.row_ids == [source.row_ids[0]]
    assert result.batch.columns["merged"] == [4]
    assert {(record.row_id, record.op) for record in result.quarantined} == {
        (source.row_ids[1], "left"),
        (source.row_ids[1], "right"),
    }


def test_empty_required_branch_skips_downstream_call() -> None:
    # A completely quarantined branch makes the required fan-in empty.
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.empty = RuntimeRayModule(RejectAllOp)
            self.length = RuntimeRayModule(RejectTextOp).pre_init("<never>")
            self.merge = RuntimeRayModule(Sum2Op)
            super().__init__()

        def forward(self, x: list[str]) -> list[int]:
            merged = self.merge(self.empty(x), self.length(x))
            return merged

    source = MicroBatch.source(
        {"x": ["first", "second"]},
        dataset="empty-branch",
    )
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.row_ids == []
    assert result.batch.columns["merged"] == []
    assert [record.row_id for record in result.quarantined] == source.row_ids


def test_failure_at_fanin_reports_all_successful_parent_branches() -> None:
    # The failing node is reported separately; its path points to both parents.
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.left = RuntimeRayModule(AddOp).pre_init(1)
            self.right = RuntimeRayModule(ScaleOp).pre_init(10)
            self.merge = RuntimeRayModule(FailMergedRowOp).pre_init(3)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            merged = self.merge(self.left(x), self.right(x))
            return merged

    source = MicroBatch.source({"x": [1, 2, 3]}, dataset="fanin-error")
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.columns["merged"] == [12, 34]
    assert result.batch.row_ids == [source.row_ids[0], source.row_ids[2]]
    assert len(result.quarantined) == 1
    error = result.quarantined[0]
    assert error.row_id == source.row_ids[1]
    assert error.op == "merge"
    assert result.trace(error.path_id) == ["left", "right"]


def test_failure_after_fanin_retains_the_complete_merged_history() -> None:
    # source -> (left, right) -> merge -> sink(fails)
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.left = RuntimeRayModule(AddOp).pre_init(1)
            self.right = RuntimeRayModule(ScaleOp).pre_init(10)
            self.merge = RuntimeRayModule(Sum2Op)
            self.sink = RuntimeRayModule(FailValueOp).pre_init(23)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            merged = self.merge(self.left(x), self.right(x))
            output = self.sink(merged)
            return output

    source = MicroBatch.source({"x": [1, 2, 3]}, dataset="post-fanin-error")
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.columns["output"] == [13, 35]
    assert result.batch.row_ids == [source.row_ids[0], source.row_ids[2]]
    error = result.quarantined[0]
    assert error.row_id == source.row_ids[1]
    assert error.op == "sink"
    assert result.trace(error.path_id) == ["left", "right", "merge"]


def test_multiple_graph_outputs_get_a_combined_result_lineage() -> None:
    # source -> (left, right) -> graph outputs
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.left = RuntimeRayModule(AddOp).pre_init(1)
            self.right = RuntimeRayModule(ScaleOp).pre_init(10)
            super().__init__()

        def forward(
            self,
            x: list[int],
        ) -> tuple[list[int], list[int]]:
            left = self.left(x)
            right = self.right(x)
            return left, right

    result = _run(Pipe(), {"x": [1, 2]})

    assert result.batch.columns == {
        "left": [2, 3],
        "right": [10, 20],
    }
    assert result.trace_row("topology:0") == ["left", "right"]


def test_multiple_graph_outputs_use_their_healthy_row_intersection() -> None:
    # Final RuntimeResult remains row-aligned even without an explicit join op.
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.left = RuntimeRayModule(RejectTextOp).pre_init("left")
            self.right = RuntimeRayModule(RejectTextOp).pre_init("right")
            super().__init__()

        def forward(
            self,
            x: list[str],
        ) -> tuple[list[int], list[int]]:
            left = self.left(x)
            right = self.right(x)
            return left, right

    source = MicroBatch.source(
        {"x": ["ok", "left-bad", "right-bad", "ok2"]},
        dataset="multi-output-filter",
    )
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.row_ids == [source.row_ids[0], source.row_ids[3]]
    assert result.batch.columns == {
        "left": [2, 3],
        "right": [2, 3],
    }
    assert result.trace_row(source.row_ids[0]) == ["left", "right"]
