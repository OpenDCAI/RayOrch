"""Recovery presets and grain-level containment through the V3 public API."""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class Poison:
    def run(self, rows):
        for index, row in enumerate(rows):
            if row == "bad":
                raise mg.BadRecordError("bad row", index=index)
        return [f"ok:{row}" for row in rows]


class ExactBadPipeline(mg.Pipeline):
    def __init__(self):
        self.map = mg.Map(Poison).ray_options(batch_size=4)

    def forward(self, rows):
        return self.map(rows)


def test_bad_record_fails_one_grain_and_reexecutes_healthy_siblings():
    """Explicit row attribution contains failure without failing the Arena."""

    result = mg.Executor(ExactBadPipeline()).run(["a", "bad", "b"])
    assert result.get() == ("ok:a", "ok:b")
    assert len(result.failures) == 1
    assert result.failures[0].failure.kind == "bad_record"


class GenericPoison:
    def run(self, rows):
        if "bad" in rows:
            raise RuntimeError("opaque failure")
        return [f"ok:{row}" for row in rows]


class IsolatePipeline(mg.Pipeline):
    def __init__(self):
        self.map = mg.Map(GenericPoison).ray_options(
            batch_size=4,
            recovery="isolate_tail",
            max_recovery_attempts=32,
            max_extra_rpcs=64,
        )

    def forward(self, rows):
        return self.map(rows)


def test_isolate_tail_retries_then_bisects_to_singleton():
    """Opaque errors defer, retry, bisect, and fail only the poison Grain."""

    result = mg.Executor(IsolatePipeline()).run(["a", "bad", "b", "c"])
    assert result.get() == ("ok:a", "ok:b", "ok:c")
    assert len(result.failures) == 1
    assert result.failures[0].failure.kind == "udf_error"


class FailBatchPipeline(mg.Pipeline):
    def __init__(self):
        self.map = mg.Map(GenericPoison).ray_options(
            batch_size=4,
            recovery="fail_batch",
        )

    def forward(self, rows):
        return self.map(rows)


def test_fail_batch_marks_every_grain_failed_without_retry():
    """The explicit fail_batch preset trades attribution for progress."""

    result = mg.Executor(FailBatchPipeline()).run(["a", "bad", "b"])
    assert result.get() == ()
    assert len(result.failures) == 3


class RetryOnce:
    def __init__(self):
        self.failed = False

    def run(self, rows):
        if not self.failed:
            self.failed = True
            raise RuntimeError("transient")
        return [f"ok:{row}" for row in rows]


class RetryBatchPipeline(mg.Pipeline):
    def __init__(self):
        self.map = mg.Map(RetryOnce).ray_options(
            batch_size=3,
            recovery="retry_batch",
        )

    def forward(self, rows):
        return self.map(rows)


def test_retry_batch_reexecutes_same_logical_grains():
    """A transient UDF error retries the whole batch with stable GrainIds."""

    result = mg.Executor(RetryBatchPipeline()).run(["a", "b", "c"])
    assert result.get() == ("ok:a", "ok:b", "ok:c")
    assert result.failures == ()


class RetryTailPipeline(mg.Pipeline):
    def __init__(self):
        self.map = mg.Map(RetryOnce).ray_options(
            batch_size=3,
            recovery="retry_tail",
        )

    def forward(self, rows):
        return self.map(rows)


def test_retry_tail_defers_then_reexecutes_failed_batch():
    """A deferred whole-batch retry cannot let the Arena finish early."""

    result = mg.Executor(RetryTailPipeline()).run(["a", "b", "c"])
    assert result.get() == ("ok:a", "ok:b", "ok:c")


class RaisePipeline(mg.Pipeline):
    def __init__(self):
        self.map = mg.Map(GenericPoison).ray_options(batch_size=2)

    def forward(self, rows):
        return self.map(rows)


def test_raise_is_the_default_udf_error_policy():
    """Generic UDF errors abort unless a recovery preset opts in."""

    with pytest.raises(mg.ExecutionError, match="UDF error"):
        mg.Executor(RaisePipeline()).run(["bad"])
