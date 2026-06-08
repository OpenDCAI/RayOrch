from __future__ import annotations

import pytest
import ray

from rayorch import DagPipeline, PipeRef, RuntimeDagExecutor, RuntimeRayModule
from rayorch.runtime import BadRecordError, MicroBatch

from test.runtime_test_utils import cleanup_pipeline, trace_path


class DropBadOp:
    """Drop rows whose value contains ``bad`` using direct BadRecordError index."""

    def run(self, x):
        for index, value in enumerate(x):
            if "bad" in value:
                raise BadRecordError("bad row", index=index)
        return [f"ok:{value}" for value in x]


class SplitRetryBadOp:
    """Force split-and-retry by omitting the bad row index."""

    def run(self, x):
        if any("bad" in value for value in x):
            raise BadRecordError("bad row")
        return [f"split-ok:{value}" for value in x]


class ExceptionBadOp:
    """Force split-and-retry via a normal exception."""

    def run(self, x):
        if any("boom" in value for value in x):
            raise RuntimeError("boom row")
        return [f"exc-ok:{value}" for value in x]


class AddOneOp:
    def run(self, x):
        return [value + 1 for value in x]


class MulTenOp:
    def run(self, x):
        return [value * 10 for value in x]


class MergeOp:
    def run(self, left, right):
        return [l + r for l, r in zip(left, right)]


def test_all_bad_microbatch_flows_as_empty_through_downstream_stage() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)

    class Pipe(DagPipeline):
        def __init__(self):
            self.drop = RuntimeRayModule(
                DropBadOp, replicas=2, max_inflight=2, num_outputs=1
            ).pre_init()
            self.next = RuntimeRayModule(
                AddOneOp, replicas=2, max_inflight=2, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.drop(x)
            z = self.next(y)
            return z

    pipe = Pipe()
    try:
        source = MicroBatch.source(
            {"x": ["bad-0", "bad-1", "bad-2"]},
            dataset="all-bad",
        )

        result = RuntimeDagExecutor(max_batches_inflight=2).run(pipe, source)

        assert len(result.batch) == 0
        assert result.batch.columns["z"] == []
        assert [record.row_id for record in result.quarantined] == source.row_ids
        assert {record.op for record in result.quarantined} == {"drop"}
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_bad_record_without_index_uses_split_and_retry() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)

    class Pipe(DagPipeline):
        def __init__(self):
            self.op = RuntimeRayModule(
                SplitRetryBadOp, replicas=2, max_inflight=2, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.op(x)
            return y

    pipe = Pipe()
    try:
        source = MicroBatch.source(
            {"x": ["a", "bad", "b", "c"]},
            dataset="split-retry",
        )

        result = RuntimeDagExecutor().run(pipe, source)

        assert result.batch.columns["y"] == ["split-ok:a", "split-ok:b", "split-ok:c"]
        assert [record.values["x"] for record in result.quarantined] == ["bad"]
        assert result.quarantined[0].op == "op"
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_normal_exception_uses_split_and_retry() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)

    class Pipe(DagPipeline):
        def __init__(self):
            self.op = RuntimeRayModule(
                ExceptionBadOp, replicas=2, max_inflight=2, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.op(x)
            return y

    pipe = Pipe()
    try:
        source = MicroBatch.source(
            {"x": ["a", "boom", "b"]},
            dataset="normal-exception",
        )

        result = RuntimeDagExecutor().run(pipe, source)

        assert result.batch.columns["y"] == ["exc-ok:a", "exc-ok:b"]
        assert [record.values["x"] for record in result.quarantined] == ["boom"]
        assert "RuntimeError: boom row" in result.quarantined[0].error
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_multiple_bad_rows_in_one_stage_are_all_quarantined() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)

    class Pipe(DagPipeline):
        def __init__(self):
            self.op = RuntimeRayModule(
                SplitRetryBadOp, replicas=2, max_inflight=2, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.op(x)
            return y

    pipe = Pipe()
    try:
        source = MicroBatch.source(
            {"x": ["bad-0", "a", "bad-1", "b", "bad-2"]},
            dataset="multi-bad",
        )

        result = RuntimeDagExecutor().run(pipe, source)

        assert result.batch.columns["y"] == ["split-ok:a", "split-ok:b"]
        assert [record.values["x"] for record in result.quarantined] == [
            "bad-0",
            "bad-1",
            "bad-2",
        ]
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_reusing_same_runtime_module_uses_per_node_runtime_spec() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)

    class Pipe(DagPipeline):
        def __init__(self):
            self.add = RuntimeRayModule(
                AddOneOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.add(x)
            z = self.add(y)
            return z

    pipe = Pipe()
    try:
        source = MicroBatch.source({"x": [1, 2, 3]}, dataset="reuse-module")

        result = RuntimeDagExecutor().run(pipe, source)

        assert result.batch.columns["z"] == [3, 4, 5]
        assert trace_path(result.paths, result.batch.path_ids[0]) == ["add", "add_1"]
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_fanout_allows_multiple_downstream_consumers() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)

    class Pipe(DagPipeline):
        def __init__(self):
            self.add = RuntimeRayModule(
                AddOneOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            self.left = RuntimeRayModule(
                AddOneOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            self.right = RuntimeRayModule(
                MulTenOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            y = self.add(x)
            left = self.left(y)
            right = self.right(y)
            return left

    pipe = Pipe()
    try:
        source = MicroBatch.source({"x": [1, 2]}, dataset="fanout")

        result = RuntimeDagExecutor().run(pipe, source)

        assert result.batch.columns["left"] == [3, 4]
        assert trace_path(result.paths, result.batch.path_ids[0]) == ["add", "left"]
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()


def test_fanin_rejects_diverged_paths_in_mvp() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=4)

    class Pipe(DagPipeline):
        def __init__(self):
            self.left = RuntimeRayModule(
                AddOneOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            self.right = RuntimeRayModule(
                MulTenOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            self.merge = RuntimeRayModule(
                MergeOp, replicas=1, max_inflight=2, num_outputs=1
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef):
            left = self.left(x)
            right = self.right(x)
            merged = self.merge(left, right)
            return merged

    pipe = Pipe()
    try:
        source = MicroBatch.source({"x": [1, 2]}, dataset="fanin")

        with pytest.raises(ValueError, match="aligned path_ids"):
            RuntimeDagExecutor().run(pipe, source)
    finally:
        cleanup_pipeline(pipe)
        ray.shutdown()
