"""Generation fencing and stale-completion confluence for ArenaEngine."""

from __future__ import annotations

import rayorch.experimental.multigrain_v3 as mg
from rayorch.experimental.multigrain_v3.arena import ArenaEngine
from rayorch.experimental.multigrain_v3.protocol import (
    BatchReport,
    DispatchCompletion,
    DispatchFailure,
    FailureKind,
    InvocationAck,
)


class Identity:
    def run(self, rows):
        return list(rows)


class Pipeline(mg.Pipeline):
    def __init__(self):
        self.map = mg.Map(Identity).ray_options(
            batch_size=1,
            max_infra_retries=2,
        )

    def forward(self, rows):
        return self.map(rows)


def test_late_completion_from_retried_generation_is_ignored():
    """An old physical RPC cannot overwrite the accepted retry generation."""

    compiled = Pipeline().compile()
    arena = ArenaEngine(0, compiled.dag, bytes(range(16)))
    arena.admit_sources((("value",),), position_starts=(0,))
    arena.advance()

    old = arena.reserve_dispatch(1)
    assert old is not None
    arena.handle_failure(
        DispatchFailure(
            old.call.dispatch,
            FailureKind.INFRA_FAILURE,
            "worker died",
            worker_slot=0,
        )
    )
    retry = arena.reserve_dispatch(1)
    assert retry is not None

    retry_report = BatchReport(
        retry.call.dispatch,
        (InvocationAck(retry.call.invocations[0].token, (1,)),),
        (1,),
    )
    arena.commit(
        DispatchCompletion(
            arena.id,
            retry.call,
            retry_report,
            (("new",),),
            worker_slot=1,
        )
    )

    old_report = BatchReport(
        old.call.dispatch,
        (InvocationAck(old.call.invocations[0].token, (1,)),),
        (1,),
    )
    arena.commit(
        DispatchCompletion(
            arena.id,
            old.call,
            old_report,
            (("old",),),
            worker_slot=0,
        )
    )
    arena.advance()
    result = arena.finish()
    assert result.outputs[0].block == ("new",)
