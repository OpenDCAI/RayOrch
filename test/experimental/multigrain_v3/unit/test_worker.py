"""Worker ABI tests without Ray."""

from __future__ import annotations

import rayorch.experimental.multigrain_v3 as mg
from rayorch.experimental.multigrain_v3.protocol import (
    BatchCall,
    Invocation,
    MissingTake,
    RowTake,
    ValueTake,
)
from rayorch.experimental.multigrain_v3.dag import Primitive
from rayorch.experimental.multigrain_v3.model import AttemptToken, GrainId
from rayorch.experimental.multigrain_v3.worker import execute_call


class TupleFilter:
    def run(self, left, right):
        return [a + b > 0 for a, b in zip(left, right)]


class OptionalMap:
    def run(self, rows, optional_rows):
        return [
            row if value is mg.MISSING else (row, value)
            for row, value in zip(rows, optional_rows)
        ]


def _token(index: int) -> AttemptToken:
    return AttemptToken(0, 0, GrainId(bytes([index]) * 16), 1)


def test_filter_returns_only_mask_report_and_no_business_blocks():
    """Tuple-preserving Filter never copies its input payloads."""

    stage = mg.Pipeline  # keep public import exercised
    del stage
    from rayorch.experimental.multigrain_v3.dag import (
        ExecutionSpec,
        InputSpec,
        StageSpec,
        UdfSpec,
    )

    spec = StageSpec(
        0,
        Primitive.FILTER,
        (
            InputSpec("left", mg.PortId(1, 0)),
            InputSpec("right", mg.PortId(2, 0)),
        ),
        2,
        0,
        UdfSpec(TupleFilter),
        ExecutionSpec(),
    )
    call = BatchCall(
        0,
        0,
        (
            Invocation(
                _token(1),
                (
                    ValueTake((RowTake(0, 0),)),
                    ValueTake((RowTake(1, 0),)),
                ),
            ),
            Invocation(
                _token(2),
                (
                    ValueTake((RowTake(0, 1),)),
                    ValueTake((RowTake(1, 1),)),
                ),
            ),
        ),
    )
    report, blocks = execute_call(spec, TupleFilter(), call, ((1, -2), (1, 1)))
    assert blocks == ()
    assert [ack.keep for ack in report.acks] == [True, False]


def test_missing_take_materializes_framework_sentinel():
    """OPTIONAL_ONE absence reaches the UDF as MISSING, not None."""

    from rayorch.experimental.multigrain_v3.dag import (
        ExecutionSpec,
        InputMode,
        InputSpec,
        StageSpec,
        UdfSpec,
    )

    spec = StageSpec(
        0,
        Primitive.MAP,
        (
            InputSpec("rows", mg.PortId(1, 0)),
            InputSpec("optional_rows", mg.PortId(2, 0), InputMode.OPTIONAL_ONE),
        ),
        1,
        0,
        UdfSpec(OptionalMap),
        ExecutionSpec(),
    )
    call = BatchCall(
        0,
        0,
        (
            Invocation(
                _token(1),
                (ValueTake((RowTake(0, 0),)), MissingTake()),
            ),
        ),
    )
    report, blocks = execute_call(spec, OptionalMap(), call, (("a",),))
    assert report.column_lengths == (1,)
    assert blocks == (("a",),)
