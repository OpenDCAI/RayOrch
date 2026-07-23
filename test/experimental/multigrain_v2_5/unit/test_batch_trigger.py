from __future__ import annotations

from dataclasses import replace

from rayorch.experimental.multigrain_v2_5.executor import Arena, ArenaLimits
from rayorch.experimental.multigrain_v2_5.graph import (
    BindingReceipt,
    admit_source,
    compile_graph,
    plan_map,
)
from rayorch.experimental.multigrain_v2_5.worker import DispatchErrorReport

from ..reference_semantics.semantic_cases import RUN_SALT
from ..support import map_node, source_node


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, milliseconds: float) -> None:
        self.now += milliseconds / 1000.0


def _arena(*, batch_size: int, wait_ms: float, count: int, isolate=False):
    clock = FakeClock()
    source_spec = source_node(0)
    map_spec = map_node(1, source_spec.output_ports[0])
    assert map_spec.execution is not None
    map_spec = replace(
        map_spec,
        execution=replace(
            map_spec.execution,
            batch_size=batch_size,
            max_batch_wait_ms=wait_ms,
            error_policy=("isolate" if isolate else "raise"),
        ),
    )
    graph = compile_graph((source_spec, map_spec))
    arena = Arena(
        1,
        graph,
        RUN_SALT,
        limits=ArenaLimits(
            max_grains=max(100, count * 3),
            max_pending_dispatches=max(64, count),
        ),
        clock=clock,
    )
    sources = []
    for position in range(count):
        source = admit_source(source_spec, RUN_SALT, position)
        arena.admit_source(source, value=position)
        sources.append(source)
    return clock, arena, map_spec, sources


def _enqueue(arena, map_spec, sources):
    decisions = tuple(
        plan_map(
            map_spec,
            RUN_SALT,
            (
                BindingReceipt.present(
                    "primary",
                    source.output_slots[0],
                ),
            ),
        )
        for source in sources
    )
    return arena.ensure_plans(decisions)


def test_underfilled_open_queue_waits_then_flushes_on_timeout():
    """An open underfilled queue must wait exactly until its tail deadline."""
    clock, arena, map_spec, sources = _arena(
        batch_size=8,
        wait_ms=2,
        count=3,
    )
    _enqueue(arena, map_spec, sources)

    assert arena.reserve_dispatch(1, admission_closed=False) is None
    clock.advance_ms(1.9)
    assert arena.reserve_dispatch(1, admission_closed=False) is None
    clock.advance_ms(0.1)
    plan = arena.reserve_dispatch(1, admission_closed=False)
    assert plan is not None and len(plan.entries) == 3
    assert arena.dispatch_flush_reason(plan) == "timeout"


def test_trickle_reaching_batch_size_flushes_full_without_timeout():
    """Trickled grains that fill the batch before deadline use a full RPC."""
    clock, arena, map_spec, sources = _arena(
        batch_size=8,
        wait_ms=5,
        count=8,
    )
    _enqueue(arena, map_spec, sources[:4])
    clock.advance_ms(1)
    assert arena.reserve_dispatch(1, admission_closed=False) is None
    _enqueue(arena, map_spec, sources[4:])

    plan = arena.reserve_dispatch(1, admission_closed=False)
    assert plan is not None and len(plan.entries) == 8
    assert arena.dispatch_flush_reason(plan) == "full"


def test_admission_close_flushes_tail_without_waiting():
    """A node with no future candidates flushes its tail immediately."""
    _, arena, map_spec, sources = _arena(
        batch_size=16,
        wait_ms=100,
        count=3,
    )
    _enqueue(arena, map_spec, sources)
    plan = arena.reserve_dispatch(1, admission_closed=True)
    assert plan is not None and len(plan.entries) == 3
    assert arena.dispatch_flush_reason(plan) == "port_sealed"


def test_tail_timer_restarts_after_full_batch_is_removed():
    """Removing a full batch starts a fresh deadline for the remaining tail."""
    clock, arena, map_spec, sources = _arena(
        batch_size=4,
        wait_ms=2,
        count=6,
    )
    _enqueue(arena, map_spec, sources)
    clock.advance_ms(50)
    full = arena.reserve_dispatch(1, admission_closed=False)
    assert full is not None and len(full.entries) == 4
    assert arena.dispatch_flush_reason(full) == "full"

    clock.advance_ms(1.9)
    assert arena.reserve_dispatch(1, admission_closed=False) is None
    clock.advance_ms(0.1)
    tail = arena.reserve_dispatch(1, admission_closed=False)
    assert tail is not None and len(tail.entries) == 2
    assert arena.dispatch_flush_reason(tail) == "timeout"


def test_isolation_group_flushes_immediately_without_mixing_normal_queue():
    """Isolation groups bypass waiting and never absorb normal ready grains."""
    _, arena, map_spec, sources = _arena(
        batch_size=2,
        wait_ms=100,
        count=4,
        isolate=True,
    )
    records = _enqueue(arena, map_spec, sources)
    root = arena.reserve_dispatch(1, admission_closed=False)
    assert root is not None and len(root.entries) == 2
    arena.handle_error(
        root,
        DispatchErrorReport(root.id, "generic_udf", None, "opaque"),
    )

    isolated = arena.reserve_dispatch(1, admission_closed=False)
    assert isolated is not None and len(isolated.entries) == 1
    assert arena.dispatch_flush_reason(isolated) == "isolation"
    assert arena.ready_count(1) == 2
    assert all(record.outcome is None for record in records)


def test_large_burst_produces_full_batches_and_one_sealed_tail():
    """A 1000-grain burst produces only full batches plus one final tail."""
    _, arena, map_spec, sources = _arena(
        batch_size=16,
        wait_ms=10,
        count=1000,
    )
    _enqueue(arena, map_spec, sources)
    plans = []
    while arena.ready_count(1) >= 16:
        plan = arena.reserve_dispatch(1, admission_closed=False)
        assert plan is not None
        plans.append(plan)
    tail = arena.reserve_dispatch(1, admission_closed=True)
    assert tail is not None
    plans.append(tail)

    assert len(plans) == 63
    assert [len(plan.entries) for plan in plans[:-1]] == [16] * 62
    assert len(plans[-1].entries) == 8
    assert all(
        arena.dispatch_flush_reason(plan) == "full"
        for plan in plans[:-1]
    )
    assert arena.dispatch_flush_reason(plans[-1]) == "port_sealed"


def test_new_arrivals_do_not_extend_an_existing_tail_deadline():
    """Continuous trickle arrivals cannot postpone an older tail indefinitely."""

    clock, arena, map_spec, sources = _arena(
        batch_size=8,
        wait_ms=2,
        count=4,
    )
    _enqueue(arena, map_spec, sources[:2])
    clock.advance_ms(1.0)
    _enqueue(arena, map_spec, sources[2:3])
    clock.advance_ms(0.9)
    _enqueue(arena, map_spec, sources[3:])
    assert arena.reserve_dispatch(1, admission_closed=False) is None
    clock.advance_ms(0.1)

    plan = arena.reserve_dispatch(1, admission_closed=False)
    assert plan is not None and len(plan.entries) == 4
    assert arena.dispatch_flush_reason(plan) == "timeout"


def test_zero_wait_flushes_underfilled_open_queue_immediately():
    """max_batch_wait_ms=0 preserves the explicit low-latency behavior."""

    _, arena, map_spec, sources = _arena(
        batch_size=8,
        wait_ms=0,
        count=2,
    )
    _enqueue(arena, map_spec, sources)
    plan = arena.reserve_dispatch(1, admission_closed=False)
    assert plan is not None and len(plan.entries) == 2
    assert arena.dispatch_flush_reason(plan) == "timeout"


def test_arena_drain_flushes_tail_before_timeout():
    """Arena drain is an immediate tail flush distinct from port sealing."""

    _, arena, map_spec, sources = _arena(
        batch_size=8,
        wait_ms=100,
        count=3,
    )
    _enqueue(arena, map_spec, sources)
    plan = arena.reserve_dispatch(
        1,
        admission_closed=False,
        draining=True,
    )
    assert plan is not None and len(plan.entries) == 3
    assert arena.dispatch_flush_reason(plan) == "arena_drain"


def test_same_turn_arrival_wins_full_trigger_over_expired_timeout():
    """Commit/enqueue-before-trigger ordering prefers full over timeout."""

    clock, arena, map_spec, sources = _arena(
        batch_size=4,
        wait_ms=2,
        count=4,
    )
    _enqueue(arena, map_spec, sources[:3])
    clock.advance_ms(2)
    _enqueue(arena, map_spec, sources[3:])

    plan = arena.reserve_dispatch(1, admission_closed=False)
    assert plan is not None and len(plan.entries) == 4
    assert arena.dispatch_flush_reason(plan) == "full"
