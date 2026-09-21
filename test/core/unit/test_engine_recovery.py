"""Driver recovery preserves exact RPC batches while owning one partition."""

from types import SimpleNamespace

import pytest

import rayorch as ro
from rayorch._execution.executor import (
    Executor,
    _ActorSlot,
    _CallCounters,
    _InputBatchSlot,
)
from rayorch._model import GrainPhase, ItemOutcome, ItemRef
from rayorch._protocol import (
    BlockRef,
    DispatchFailure,
    DispatchFailureKind,
    GrainReport,
    PortOutputReport,
    RowBinding,
)
from rayorch._runtime.engine import InputBatchEngine


class Identity:
    def run(self, values):
        return values


class Pipeline(ro.Pipeline):
    def __init__(self, policy):
        self.identity = ro.RayModule(Identity).ray_options(
            batch_size=4, recovery=policy
        )

    def forward(self, values):
        return self.identity(values)


def driver(policy, count=5):
    plan = Pipeline(policy).compile().plan
    engine = InputBatchEngine(plan)
    engine.admit_sources(
        {
            plan.source_ports[0]: tuple(
                RowBinding(BlockRef("input"), i) for i in range(count)
            )
        }
    )
    engine.close_admission()
    (call,) = plan.calls
    pool = plan.dispatch(call).pool
    rpc_batches = []

    def remote(invocations, layouts, input_layout):
        rpc_batches.append(
            tuple((inv.grain.entity.value, inv.generation) for inv in invocations)
        )
        return len(rpc_batches)

    executor = object.__new__(Executor)
    executor.plan = plan
    executor._actors = {
        pool: [
            _ActorSlot(pool, SimpleNamespace(execute=SimpleNamespace(remote=remote)))
        ]
    }
    executor._calls_by_pool = {pool: (call,)}
    executor._pool_cursor = {pool: 0}
    executor._counters = {call: _CallCounters()}
    return executor, engine, {0: _InputBatchSlot(0, engine)}, rpc_batches, call


FAILURE = DispatchFailure(
    DispatchFailureKind.UDF_ERROR, "ValueError", "poison", "trace"
)


@pytest.mark.parametrize(
    ("policy", "poison", "expected", "requeues"),
    [
        (ro.RecoveryPolicy.retry_batch(), False, [(0, 1, 2, 3), (0, 1, 2, 3), (4,)], 4),
        (ro.RecoveryPolicy.retry_tail(), False, [(0, 1, 2, 3), (4,), (0, 1, 2, 3)], 4),
        (
            ro.RecoveryPolicy.isolate_tail(),
            True,
            [(0, 1, 2, 3), (4,), (0, 1, 2, 3), (0, 1), (2, 3), (2,), (3,)],
            10,
        ),
    ],
)
def test_recovery_preserves_rpc_membership_order_generations_and_batch_sizes(
    policy, poison, expected, requeues
):
    executor, engine, active, rpc_batches, call = driver(policy)
    pending = {}
    failures = 0
    partitions = 0
    partition = engine._partition_in_flight

    def counted(batch):
        nonlocal partitions
        partitions += 1
        return partition(batch)

    engine._partition_in_flight = counted
    while not engine.is_complete():
        assert executor._dispatch_ready(active, pending)
        _, rpc = pending.popitem()
        batch = rpc.execution_microbatch
        fail = (
            any(g.entity.value == 2 for g in batch.grains) if poison else failures == 0
        )
        if fail:
            executor._handle_dispatch_failure(engine, rpc, FAILURE)
            failures += 1
        else:
            output = engine.plan.outputs_by_call[call][0]
            engine.commit_reports(
                batch,
                tuple(
                    GrainReport(
                        grain,
                        engine.grain_snapshot(grain).generation,
                        (
                            PortOutputReport(
                                output,
                                scalar=RowBinding(
                                    BlockRef("output"), grain.entity.value
                                ),
                            ),
                        ),
                    )
                    for grain in batch.grains
                ),
            )
        rpc.actor.busy = False

    assert [tuple(entity for entity, _ in batch) for batch in rpc_batches] == expected
    attempts = {}
    for batch in rpc_batches:
        for entity, generation in batch:
            assert generation == attempts.get(entity, 0)
            attempts[entity] = generation + 1
    metrics = executor._counters[call]
    assert metrics.rpcs == len(expected)
    assert metrics.batch_sizes == [len(batch) for batch in expected]
    assert metrics.grain_requeues == requeues
    output = engine.plan.outputs_by_call[call][0]
    assert [engine.item_outcome(item) for item in engine.ordered_items(output)] == [
        ItemOutcome.FAILED if poison and i == 2 else ItemOutcome.PRESENT
        for i in range(5)
    ]
    assert partitions == failures


@pytest.mark.parametrize("all_barriered", [False, True])
def test_abort_preserves_state_but_all_barriered_work_finishes(all_barriered):
    executor, engine, active, rpc_batches, call = driver(
        ro.RecoveryPolicy.abort(), count=4
    )
    pending = {}
    executor._dispatch_ready(active, pending)
    _, rpc = pending.popitem()
    grains = rpc.execution_microbatch.grains
    for grain in grains if all_barriered else grains[:1]:
        engine._suppression_barriers.establish(call, grain.entity, "parent failed")
    before = (
        dict(engine._state.items),
        dict(engine._state.values),
        engine.grain_snapshots(),
        tuple(engine._fact_queue),
    )
    if all_barriered:
        executor._handle_dispatch_failure(engine, rpc, FAILURE)
        assert engine.is_complete()
        assert all(
            engine.grain_snapshot(grain).phase is GrainPhase.SEALED for grain in grains
        )
        output = engine.plan.outputs_by_call[call][0]
        for grain in grains:
            item = ItemRef(output, grain.entity)
            assert engine.item_outcome(item) is ItemOutcome.SUPPRESSED
            assert engine.item_cause(item) == "parent failed"
    else:
        with pytest.raises(ro.ExecutionError, match="poison"):
            executor._handle_dispatch_failure(engine, rpc, FAILURE)
        assert before == (
            engine._state.items,
            engine._state.values,
            engine.grain_snapshots(),
            tuple(engine._fact_queue),
        )
        assert engine.dispatch_priority(call) is None
    assert executor._counters[call].grain_requeues == 0
    assert len(rpc_batches) == 1


@pytest.mark.parametrize(
    "policy", [ro.RecoveryPolicy.retry_batch(), ro.RecoveryPolicy.retry_tail()]
)
@pytest.mark.parametrize("all_barriered", [False, True])
def test_exhausted_budget_never_resets_after_partition(policy, all_barriered):
    executor, engine, active, rpc_batches, call = driver(policy, count=4)
    pending = {}
    executor._dispatch_ready(active, pending)
    _, first = pending.popitem()
    executor._handle_dispatch_failure(engine, first, FAILURE)
    first.actor.busy = False
    executor._dispatch_ready(active, pending)
    _, retry = pending.popitem()
    assert retry.execution_microbatch.udf_retries == 1
    grains = retry.execution_microbatch.grains
    for grain in grains if all_barriered else grains[:1]:
        engine._suppression_barriers.establish(call, grain.entity, "parent failed")
    before = (
        dict(engine._state.items),
        engine.grain_snapshots(),
        tuple(engine._fact_queue),
    )
    if all_barriered:
        executor._handle_dispatch_failure(engine, retry, FAILURE)
        assert engine.is_complete()
    else:
        with pytest.raises(ro.ExecutionError, match="generation=1"):
            executor._handle_dispatch_failure(engine, retry, FAILURE)
        assert before == (
            engine._state.items,
            engine.grain_snapshots(),
            tuple(engine._fact_queue),
        )
    assert engine.dispatch_priority(call) is None
    assert len(rpc_batches) == 2
    assert executor._counters[call].grain_requeues == 4


@pytest.mark.parametrize("live_count", [1, 2])
def test_isolation_decision_uses_live_subset_after_barriers(live_count):
    executor, engine, active, rpc_batches, call = driver(
        ro.RecoveryPolicy.isolate_tail(), count=4
    )
    pending = {}
    executor._dispatch_ready(active, pending)
    _, first = pending.popitem()
    executor._handle_dispatch_failure(engine, first, FAILURE)
    first.actor.busy = False
    executor._dispatch_ready(active, pending)
    _, retry = pending.popitem()
    grains = retry.execution_microbatch.grains
    for grain in grains[:-live_count]:
        engine._suppression_barriers.establish(call, grain.entity, "parent failed")
    executor._handle_dispatch_failure(engine, retry, FAILURE)
    output = engine.plan.outputs_by_call[call][0]
    for grain in grains[:-live_count]:
        assert (
            engine.item_outcome(ItemRef(output, grain.entity)) is ItemOutcome.SUPPRESSED
        )
    if live_count == 1:
        assert (
            engine.item_outcome(ItemRef(output, grains[-1].entity))
            is ItemOutcome.FAILED
        )
        assert engine.is_complete()
        assert executor._counters[call].grain_requeues == 4
    else:
        # The two survivors split; they are neither merged with suppressed peers
        # nor granted a fresh whole-batch retry budget.
        for grain in grains[-live_count:]:
            batch = engine.reserve_dispatch(call, max_size=4)
            assert batch.grains == (grain,)
            assert batch.udf_retries == 1
            assert engine.grain_snapshot(grain).generation == 2
        assert executor._counters[call].grain_requeues == 6
    assert len(rpc_batches) == 2  # Recovery only enqueues; it never sends an RPC.
