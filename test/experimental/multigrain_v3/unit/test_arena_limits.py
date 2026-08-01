"""Boundary-driven tests for bounded V3 Arena metadata."""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3 as mg
from rayorch.experimental.multigrain_v3.arena import (
    ArenaAbort,
    ArenaEngine,
    ArenaLimits,
)
from rayorch.experimental.multigrain_v3.protocol import (
    BatchReport,
    DispatchCompletion,
    ValueAck,
)


class Fanout:
    def run(self, rows):
        return [list(range(row)) for row in rows]


class Gather:
    def run(self, groups):
        return [tuple(group) for group in groups]


class Pipeline(mg.Pipeline):
    def __init__(self):
        self.expand = mg.Expand(Fanout).ray_options(batch_size=1)
        self.reduce = mg.Reduce(Gather).ray_options(batch_size=1)

    def forward(self, rows):
        children = self.expand(rows)
        return self.reduce(anchor=rows, members=children)


def test_reduce_slot_limit_aborts_before_partial_accumulator_publication():
    """A slot overflow is an Arena abort, not a failed logical Grain."""

    compiled = Pipeline().compile()
    arena = ArenaEngine(
        0,
        compiled.dag,
        bytes(range(16)),
        limits=ArenaLimits(max_reduce_slots=3),
    )
    arena.admit_sources(((3,),), position_starts=(0,))
    arena.advance()
    intent = arena.reserve_dispatch(1)
    assert intent is not None
    ack = ValueAck(intent.call.invocations[0].token, (3,))
    report = BatchReport(intent.call.dispatch, (ack,), (3,))
    with pytest.raises(ArenaAbort, match="max_reduce_slots"):
        arena.commit(
            DispatchCompletion(
                arena.id,
                intent.call,
                report,
                ((0, 1, 2),),
                worker_slot=0,
            )
        )
