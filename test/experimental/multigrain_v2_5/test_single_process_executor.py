from __future__ import annotations

from dataclasses import replace

import pytest

from rayorch.experimental.multigrain_v2_5.executor import (
    Arena,
    ArenaAbort,
    ArenaLimits,
    ArenaState,
    CommitStatus,
)
from rayorch.experimental.multigrain_v2_5.graph import (
    BindingReceipt,
    ExecutionOptions,
    admit_source,
    compile_graph,
    plan_expand,
    plan_filter,
    plan_map,
    plan_reduce,
)
from rayorch.experimental.multigrain_v2_5.grain import (
    Failed,
    FiberBarrier,
    FiberId,
    GrainPhase,
    ItemRef,
)
from rayorch.experimental.multigrain_v2_5.worker import DispatchErrorReport

from .reference_semantics.semantic_cases import RUN_SALT
from .support import (
    expand_node,
    filter_node,
    map_node,
    reduce_node,
    source_node,
)


def _with_execution(node, **changes):
    assert node.execution is not None
    return replace(
        node,
        execution=replace(node.execution, **changes),
    )


def _map_arena(
    count: int,
    *,
    batch_size: int,
    error_policy: str = "raise",
    limits: ArenaLimits = ArenaLimits(),
):
    source_spec = source_node(0)
    map_spec = _with_execution(
        map_node(1, source_spec.output_ports[0]),
        batch_size=batch_size,
        error_policy=error_policy,
    )
    graph = compile_graph((source_spec, map_spec))
    arena = Arena(1, graph, RUN_SALT, limits=limits)
    records = []
    for position in range(count):
        source = admit_source(source_spec, RUN_SALT, position)
        arena.admit_source(source, value=position)
        decision = plan_map(
            map_spec,
            RUN_SALT,
            (BindingReceipt.present("primary", source.output_slots[0]),),
        )
        records.extend(arena.ensure_plans((decision,)))
    return arena, map_spec, tuple(records)


def _map_output(plan, prefix="v"):
    return (
        tuple(
            (f"{prefix}{index}",)
            for index, _entry in enumerate(plan.entries)
        ),
    )


def test_dispatch_packs_multiple_logical_grains_into_one_rpc_unit():
    arena, _, records = _map_arena(5, batch_size=4)
    first = arena.reserve_dispatch(1)
    assert first is not None
    assert len(first.entries) == 4
    assert len({entry.token.grain for entry in first.entries}) == 4

    assert arena.commit_normalized(first, _map_output(first)) is CommitStatus.ACCEPTED
    assert all(record.phase is GrainPhase.SEALED for record in records[:4])

    tail = arena.reserve_dispatch(1)
    assert tail is not None
    assert len(tail.entries) == 1


def test_infrastructure_retry_reuses_identity_and_stale_result_is_noop():
    arena, _, records = _map_arena(2, batch_size=2)
    first = arena.reserve_dispatch(1)
    assert first is not None
    ids = tuple(entry.token.grain for entry in first.entries)

    assert arena.handle_infrastructure_failure(first) is CommitStatus.ACCEPTED
    retry = arena.reserve_dispatch(1)
    assert retry is not None
    assert tuple(entry.token.grain for entry in retry.entries) == ids
    assert all(
        new.token.generation == old.token.generation + 1
        for old, new in zip(first.entries, retry.entries)
    )

    assert arena.commit_normalized(first, _map_output(first, "stale")) is (
        CommitStatus.STALE
    )
    assert all(record.outcome is None for record in records)
    assert arena.commit_normalized(retry, _map_output(retry, "fresh")) is (
        CommitStatus.ACCEPTED
    )


def test_manifest_contract_violation_aborts_without_partial_visibility():
    arena, _, records = _map_arena(2, batch_size=2)
    plan = arena.reserve_dispatch(1)
    assert plan is not None
    malformed = (
        (
            ("ok",),
            (),
        ),
    )

    with pytest.raises(ArenaAbort, match="must emit one row"):
        arena.commit_normalized(plan, malformed)

    assert arena.state is ArenaState.RECLAIMED
    assert len(arena.grains) == 0
    assert all(record.outcome is None for record in records)


def test_pending_dispatch_limit_is_backpressure_not_semantic_failure():
    limits = ArenaLimits(max_pending_dispatches=1)
    arena, _, records = _map_arena(
        4,
        batch_size=2,
        limits=limits,
    )
    first = arena.reserve_dispatch(1)
    assert first is not None
    assert arena.reserve_dispatch(1) is None
    assert all(record.outcome is None for record in records)

    arena.commit_normalized(first, _map_output(first))
    second = arena.reserve_dispatch(1)
    assert second is not None
    assert len(second.entries) == 2


def test_explicit_bad_record_fails_one_grain_and_requeues_healthy_peers():
    arena, _, records = _map_arena(3, batch_size=3)
    plan = arena.reserve_dispatch(1)
    assert plan is not None
    bad = plan.entries[1].token
    report = DispatchErrorReport(
        dispatch=plan.id,
        kind="bad_record",
        bad_token=bad,
        message="poison",
    )

    arena.handle_error(plan, report)
    assert isinstance(records[1].outcome, Failed)
    assert records[0].phase is GrainPhase.READY
    assert records[2].phase is GrainPhase.READY

    healthy = arena.reserve_dispatch(1)
    assert healthy is not None
    assert len(healthy.entries) == 2
    arena.commit_normalized(healthy, _map_output(healthy, "healthy"))


def test_generic_isolation_splits_to_singletons_and_contains_failure():
    arena, _, records = _map_arena(
        2,
        batch_size=2,
        error_policy="isolate",
    )
    root = arena.reserve_dispatch(1)
    assert root is not None
    arena.handle_error(
        root,
        DispatchErrorReport(root.id, "generic_udf", None, "opaque"),
    )

    first = arena.reserve_dispatch(1)
    second = arena.reserve_dispatch(1)
    assert first is not None and second is not None
    assert len(first.entries) == len(second.entries) == 1

    arena.handle_error(
        first,
        DispatchErrorReport(first.id, "generic_udf", None, "poison"),
    )
    arena.commit_normalized(second, ((("healthy",),),))

    outcomes = [record.outcome for record in records]
    assert sum(isinstance(outcome, Failed) for outcome in outcomes) == 1
    assert sum(outcome is not None for outcome in outcomes) == 2


def test_retry_budget_exhaustion_is_arena_abort():
    arena, _, _ = _map_arena(
        1,
        batch_size=1,
        limits=ArenaLimits(max_infra_retries=0),
    )
    plan = arena.reserve_dispatch(1)
    assert plan is not None
    with pytest.raises(ArenaAbort, match="retry budget"):
        arena.handle_infrastructure_failure(plan)
    assert arena.state is ArenaState.RECLAIMED


def test_mixed_current_and_stale_tokens_abort_arena():
    arena, _, records = _map_arena(2, batch_size=2)
    plan = arena.reserve_dispatch(1)
    assert plan is not None
    records[0].release_for_reexecution(plan.entries[0].token)

    with pytest.raises(ArenaAbort, match="mixed current/stale"):
        arena.commit_normalized(plan, _map_output(plan))
    assert arena.state is ArenaState.RECLAIMED


def test_max_grains_preflight_is_atomic_arena_abort():
    source_spec = source_node(0)
    map_spec = map_node(1, source_spec.output_ports[0])
    graph = compile_graph((source_spec, map_spec))
    arena = Arena(
        1,
        graph,
        RUN_SALT,
        limits=ArenaLimits(max_grains=2),
    )
    sources = []
    for position in range(2):
        source = admit_source(source_spec, RUN_SALT, position)
        arena.admit_source(source, value=position)
        sources.append(source)
    decision = plan_map(
        map_spec,
        RUN_SALT,
        (BindingReceipt.present("primary", sources[0].output_slots[0]),),
    )

    with pytest.raises(ArenaAbort, match="max_grains"):
        arena.ensure_plans((decision,))
    assert arena.state is ArenaState.RECLAIMED
    assert len(arena.grains) == 0


def test_expand_cross_port_mismatch_and_fanout_limit_abort():
    source_spec = source_node(0)
    expand_spec = expand_node(1, source_spec.output_ports[0], outputs=2)
    graph = compile_graph((source_spec, expand_spec))

    mismatch = Arena(1, graph, RUN_SALT)
    source = admit_source(source_spec, RUN_SALT, 0)
    mismatch.admit_source(source, value="parent")
    mismatch.ensure_plans(
        (
            plan_expand(
                expand_spec,
                RUN_SALT,
                (
                    BindingReceipt.present(
                        "parent",
                        source.output_slots[0],
                    ),
                ),
            ),
        )
    )
    plan = mismatch.reserve_dispatch(1)
    assert plan is not None
    with pytest.raises(ArenaAbort, match="share one cardinality"):
        mismatch.commit_normalized(
            plan,
            (
                (("a", "b"),),
                (("x",),),
            ),
        )

    limited = Arena(
        2,
        graph,
        RUN_SALT,
        limits=ArenaLimits(max_fanout_per_grain=1),
    )
    source = admit_source(source_spec, RUN_SALT, 1)
    limited.admit_source(source, value="parent")
    limited.ensure_plans(
        (
            plan_expand(
                expand_spec,
                RUN_SALT,
                (
                    BindingReceipt.present(
                        "parent",
                        source.output_slots[0],
                    ),
                ),
            ),
        )
    )
    plan = limited.reserve_dispatch(1)
    assert plan is not None
    with pytest.raises(ArenaAbort, match="max_fanout"):
        limited.commit_normalized(
            plan,
            (
                (("a", "b"),),
                (("x", "y"),),
            ),
        )


def test_run_result_detaches_failures_and_sources_before_reclaim():
    arena, _, _ = _map_arena(1, batch_size=1)
    plan = arena.reserve_dispatch(1)
    assert plan is not None
    arena.handle_error(
        plan,
        DispatchErrorReport(
            plan.id,
            "bad_record",
            plan.entries[0].token,
            "poison",
        ),
    )
    result = arena.deliver(())

    assert arena.state is ArenaState.RECLAIMED
    assert len(arena.grains) == 0
    assert len(result.failures) == 1
    assert result.failures[0].failure.message == "poison"
    assert len(result.sources) == 1


def test_delivery_rejects_nonterminal_logical_grains():
    arena, _, _ = _map_arena(1, batch_size=1)
    with pytest.raises(ArenaAbort, match="non-terminal"):
        arena.deliver(())
    assert arena.state is ArenaState.RECLAIMED


def test_phase2_core_gate_cross_parent_rebatch_filter_and_reduce():
    source_spec = source_node(0)
    expand_spec = _with_execution(
        expand_node(1, source_spec.output_ports[0]),
        batch_size=2,
    )
    map_spec = _with_execution(
        map_node(2, expand_spec.output_ports[0]),
        batch_size=8,
    )
    filter_spec = _with_execution(
        filter_node(3, map_spec.output_ports[0]),
        batch_size=8,
    )
    reduce_spec = _with_execution(
        reduce_node(
            4,
            source_spec.output_ports[0],
            filter_spec.output_ports[0],
        ),
        batch_size=2,
    )
    graph = compile_graph(
        (source_spec, expand_spec, map_spec, filter_spec, reduce_spec)
    )
    arena = Arena(1, graph, RUN_SALT)

    sources = []
    for position, value in enumerate(("parent-a", "parent-b")):
        source = admit_source(source_spec, RUN_SALT, position)
        arena.admit_source(source, value=value)
        sources.append(source)
        arena.ensure_plans(
            (
                plan_expand(
                    expand_spec,
                    RUN_SALT,
                    (
                        BindingReceipt.present(
                            "parent",
                            source.output_slots[0],
                        ),
                    ),
                ),
            )
        )

    expand_dispatch = arena.reserve_dispatch(1)
    assert expand_dispatch is not None
    arena.commit_normalized(
        expand_dispatch,
        (
            (
                ("a0", "a1"),
                ("b0", "b1", "b2"),
            ),
        ),
    )

    expand_records = [
        arena.grains.get(entry.token.grain)
        for entry in expand_dispatch.entries
    ]
    child_items = [
        emission.item
        for record in expand_records
        if record is not None
        for emission in record.outcome.emissions_by_port[0]  # type: ignore[union-attr]
    ]
    for item in child_items:
        arena.ensure_plans(
            (
                plan_map(
                    map_spec,
                    RUN_SALT,
                    (BindingReceipt.present("primary", item),),
                ),
            )
        )

    map_dispatch = arena.reserve_dispatch(2)
    assert map_dispatch is not None
    assert len(map_dispatch.entries) == 5
    parent_anchors = {
        arena.expand_origins.get(
            arena.grains.get(entry.token.grain).inputs[0].items[0].entity  # type: ignore[union-attr]
        ).anchor  # type: ignore[union-attr]
        for entry in map_dispatch.entries
    }
    assert len(parent_anchors) == 2
    arena.commit_normalized(
        map_dispatch,
        (
            tuple(
                (f"mapped-{index}",)
                for index in range(len(map_dispatch.entries))
            ),
        ),
    )

    map_records = [
        arena.grains.get(entry.token.grain) for entry in map_dispatch.entries
    ]
    for record in map_records:
        assert record is not None
        arena.ensure_plans(
            (
                plan_filter(
                    filter_spec,
                    RUN_SALT,
                    (
                        BindingReceipt.present(
                            "target",
                            record.output_slots[0],
                        ),
                    ),
                ),
            )
        )

    filter_dispatch = arena.reserve_dispatch(3)
    assert filter_dispatch is not None
    keeps = (True, False, True, True, True)
    arena.commit_normalized(
        filter_dispatch,
        (
            tuple(
                ((f"kept-{index}",) if keep else ())
                for index, keep in enumerate(keeps)
            ),
        ),
    )

    barriers = {
        source.output_slots[0]: FiberBarrier(
            FiberId(4, source.output_slots[0]),
            expand_record.id,
        )
        for source, expand_record in zip(sources, expand_records)
        if expand_record is not None
    }
    for source, expand_record in zip(sources, expand_records):
        assert expand_record is not None
        barrier = barriers[source.output_slots[0]]
        barrier.set_expected(
            len(expand_record.outcome.emissions_by_port[0])  # type: ignore[union-attr]
        )

    filter_records = [
        arena.grains.get(entry.token.grain)
        for entry in filter_dispatch.entries
    ]
    for keep, record in zip(keeps, filter_records):
        assert record is not None
        origin = arena.expand_origins.get(record.output_slots[0].entity)
        assert origin is not None
        barrier = barriers[origin.anchor]
        if keep:
            barrier.settle_present(origin.ordinal, record.output_slots[0])
        else:
            barrier.settle_dropped(origin.ordinal)

    reduce_records = []
    for source in sources:
        decision = plan_reduce(
            reduce_spec,
            RUN_SALT,
            BindingReceipt.present("anchor", source.output_slots[0]),
            barriers[source.output_slots[0]],
        )
        reduce_records.extend(arena.ensure_plans((decision,)))

    reduce_dispatch = arena.reserve_dispatch(4)
    assert reduce_dispatch is not None
    assert len(reduce_dispatch.entries) == 2
    assert [len(entry.role_takes[1]) for entry in reduce_dispatch.entries] == [
        1,
        3,
    ]
    arena.commit_normalized(
        reduce_dispatch,
        (
            (
                ("assembled-a",),
                ("assembled-b",),
            ),
        ),
    )

    result = arena.deliver(
        tuple(record.output_slots[0] for record in reduce_records)
    )
    assert result.outputs == ("assembled-a", "assembled-b")
    assert result.failures == ()
    assert len(result.sources) == 2
    assert arena.state is ArenaState.RECLAIMED
    assert len(arena.grains) == 0


def _run_map_schedule(batch_size: int, *, retry_first: bool):
    arena, _, records = _map_arena(6, batch_size=batch_size)
    plans = []
    while True:
        plan = arena.reserve_dispatch(1)
        if plan is None:
            break
        plans.append(plan)
    if retry_first:
        first = plans.pop(0)
        arena.handle_infrastructure_failure(first)
        retry = arena.reserve_dispatch(1)
        assert retry is not None
        assert arena.commit_normalized(first, _map_output(first, "stale")) is (
            CommitStatus.STALE
        )
        plans.append(retry)
    for plan in reversed(plans):
        outputs = (
            tuple(
                (f"value-{entry.token.grain.hex()}",)
                for entry in plan.entries
            ),
        )
        arena.commit_normalized(plan, outputs)
    items = tuple(record.output_slots[0] for record in records)
    structural = tuple((record.id, record.output_slots[0]) for record in records)
    result = arena.deliver(items)
    return structural, result.outputs


def test_packing_retry_and_completion_order_converge():
    singleton = _run_map_schedule(1, retry_first=False)
    packed_reordered = _run_map_schedule(3, retry_first=False)
    packed_retried = _run_map_schedule(3, retry_first=True)
    assert singleton == packed_reordered == packed_retried
