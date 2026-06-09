from __future__ import annotations

import pytest

from rayorch import DagPipeline, RuntimeDagExecutor, RuntimeRayModule
from rayorch.runtime import BadRecordError, MicroBatch

from test.runtime.helpers import trace_path

pytestmark = pytest.mark.usefixtures("ray_cluster")


class DropBadOp:
    def run(self, x):
        for index, value in enumerate(x):
            if "bad" in value:
                raise BadRecordError("bad row", index=index)
        return [len(value) for value in x]


class AddOneOp:
    def run(self, x):
        return [value + 1 for value in x]


class MulTenOp:
    def run(self, x):
        return [value * 10 for value in x]


class MergeOp:
    def run(self, left, right):
        return [l + r for l, r in zip(left, right)]


class LengthOp:
    def run(self, x):
        return [len(value) for value in x]


def test_all_bad_rows_flow_as_empty_batch_to_downstream_node() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.drop = RuntimeRayModule(DropBadOp, replicas=2)
            self.next = RuntimeRayModule(AddOneOp, replicas=2)
            super().__init__()

        def forward(self, x: list[str]) -> list[int]:
            return self.next(self.drop(x))

    source = MicroBatch.source({"x": ["bad-0", "bad-1"]}, dataset="all-bad")
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.columns["next.out0"] == []
    assert [record.row_id for record in result.quarantined] == source.row_ids


def test_reused_module_keeps_per_node_runtime_spec() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.add = RuntimeRayModule(AddOneOp)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            y = self.add(x)
            return self.add(y)

    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(MicroBatch.source({"x": [1, 2]}))

    assert result.batch.columns["add_1.out0"] == [3, 4]
    assert trace_path(result.paths, result.batch.path_ids[0]) == ["add", "add_1"]


def test_fanout_allows_multiple_consumers_of_same_rows() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.add = RuntimeRayModule(AddOneOp)
            self.left = RuntimeRayModule(AddOneOp)
            self.right = RuntimeRayModule(MulTenOp)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            shared = self.add(x)
            left = self.left(shared)
            self.right(shared)
            return left

    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(MicroBatch.source({"x": [1, 2]}))

    assert result.batch.columns["left"] == [3, 4]
    assert trace_path(result.paths, result.batch.path_ids[0]) == ["add", "left"]


def test_fanin_preserves_diverged_parent_paths() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.left = RuntimeRayModule(AddOneOp)
            self.right = RuntimeRayModule(MulTenOp)
            self.merge = RuntimeRayModule(MergeOp)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            merged = self.merge(self.left(x), self.right(x))
            return merged

    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(MicroBatch.source({"x": [1, 2]}))

    assert result.batch.columns["merged"] == [12, 23]
    final_path = result.batch.path_ids[0]
    join_path = result.paths[final_path].parents[0]
    assert result.paths[join_path].op is None
    assert len(result.paths[join_path].parents) == 2
    assert result.trace(final_path) == ["left", "right", "merge"]
    assert result.trace_row(result.batch.row_ids[0]) == ["left", "right", "merge"]


def test_fanin_uses_healthy_row_intersection() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.clean = RuntimeRayModule(DropBadOp)
            self.length = RuntimeRayModule(LengthOp)
            self.merge = RuntimeRayModule(MergeOp)
            super().__init__()

        def forward(self, x: list[str]) -> list[int]:
            merged = self.merge(self.clean(x), self.length(x))
            return merged

    source = MicroBatch.source(
        {"x": ["a", "bad-row", "ccc"]},
        dataset="fanin-filter",
    )
    with RuntimeDagExecutor(Pipe()) as executor:
        result = executor.run(source)

    assert result.batch.row_ids == [source.row_ids[0], source.row_ids[2]]
    assert result.batch.columns["merged"] == [2, 6]
    assert [record.row_id for record in result.quarantined] == [source.row_ids[1]]
    assert result.trace(result.batch.path_ids[0]) == [
        "clean",
        "length",
        "merge",
    ]


def test_runtime_rejects_duplicate_row_ids() -> None:
    batch = MicroBatch(
        columns={"x": [1, 2]},
        row_ids=["duplicate", "duplicate"],
        path_ids=["source", "source"],
    )

    class Pipe(DagPipeline):
        def __init__(self):
            self.add = RuntimeRayModule(AddOneOp)
            super().__init__()

        def forward(self, x: list[int]) -> list[int]:
            return self.add(x)

    with RuntimeDagExecutor(Pipe()) as executor:
        with pytest.raises(ValueError, match="row_ids must be unique"):
            executor.run(batch)
