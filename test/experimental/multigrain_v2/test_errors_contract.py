from __future__ import annotations

from dataclasses import dataclass

import pytest

from rayorch.experimental.multigrain_v2 import (
    BadRecordError,
    CompileError,
    ExecutionError,
    MultigrainError,
)


def test_public_exception_shapes() -> None:
    compile_error = CompileError("bad graph", code="BAD_GRAPH", path="layers[0]")
    assert isinstance(compile_error, MultigrainError)
    assert compile_error.args == ("bad graph",)
    assert (compile_error.code, compile_error.path) == ("BAD_GRAPH", "layers[0]")

    execution_error = ExecutionError(
        "aborted",
        code="ABORTED",
        batch_id="b",
        node_id="n",
        causes=("one", "two"),
    )
    assert (
        execution_error.code,
        execution_error.batch_id,
        execution_error.node_id,
        execution_error.causes,
    ) == ("ABORTED", "b", "n", ("one", "two"))

    bad_record = BadRecordError("bad row", index=3)
    assert bad_record.index == 3
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            BadRecordError("bad row", index=invalid)  # type: ignore[arg-type]


def test_execution_error_from_summary_is_detached() -> None:
    @dataclass
    class Summary:
        code: str
        message: str
        causes: tuple[str, ...]

    error = ExecutionError.from_summary(
        Summary("NODE_FAILED", "node failed", ("actor died",))
    )
    assert error.args == ("node failed",)
    assert error.code == "NODE_FAILED"
    assert error.causes == ("actor died",)
    assert error.batch_id is None
    assert error.node_id is None

    unknown = ExecutionError.from_summary(None)
    assert unknown.code == "UNKNOWN_EXECUTION_ERROR"
