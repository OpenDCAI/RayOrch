"""Failure-policy reducer and microbatch-owned recovery queue regression."""

from __future__ import annotations

import operator
from dataclasses import FrozenInstanceError

import pytest

from rayorch.experimental.multigrain_v3_6.model import (
    CallRef,
    DomainRef,
    EntityRef,
    GrainPhase,
    GrainRef,
)
from rayorch.experimental.multigrain_v3_6.recovery import (
    RecoveryAction,
    RecoveryPolicy,
    UdfRecoveryMode,
)
from rayorch.experimental.multigrain_v3_6.runtime.dispatch import (
    DispatchBatch,
    DispatchState,
)


def _grains(count: int) -> tuple[GrainRef, ...]:
    call = CallRef(0)
    domain = DomainRef(0)
    return tuple(
        GrainRef(call, EntityRef(domain, ordinal)) for ordinal in range(count)
    )


def _reserve_without_barriers(
    dispatch: DispatchState,
    call: CallRef,
    *,
    max_size: int,
    pack_by_parent: bool,
) -> DispatchBatch | None:
    batch, suppressed = dispatch.reserve_with_barriers(
        call,
        max_size=max_size,
        pack_by_parent=pack_by_parent,
    )
    assert not suppressed
    return batch


def test_normal_queues_preserve_call_fifo_and_do_not_scan_other_calls():
    dispatch = DispatchState()
    domain = DomainRef(0)
    first_call = CallRef(0)
    second_call = CallRef(1)
    first = tuple(
        GrainRef(first_call, EntityRef(domain, ordinal))
        for ordinal in range(2_000)
    )
    second = tuple(
        GrainRef(second_call, EntityRef(domain, 2_000 + ordinal))
        for ordinal in range(8)
    )
    for grain in (*first, *second):
        dispatch.inputs_ready(grain, grain.entity)

    original_is_ready = dispatch._is_ready
    visits = 0

    def counted_is_ready(grain):
        nonlocal visits
        visits += 1
        return original_is_ready(grain)

    dispatch._is_ready = counted_is_ready
    selected = _reserve_without_barriers(
        dispatch,
        second_call,
        max_size=4,
        pack_by_parent=False,
    )

    assert selected is not None
    assert selected.grains == second[:4]
    assert visits == 4
    assert dispatch.ready_count == 2_004
    assert dispatch.priority(first_call) == 1
    assert dispatch.priority(second_call) == 1


def test_unordered_ready_index_is_not_fifo_but_preserves_lineage_membership():
    dispatch = DispatchState(ready_fifo=False)
    call = CallRef(0)
    child_domain = DomainRef(1)
    parent_domain = DomainRef(0)
    grains = tuple(
        GrainRef(call, EntityRef(child_domain, ordinal))
        for ordinal in range(6)
    )
    parents = (
        EntityRef(parent_domain, 0),
        EntityRef(parent_domain, 1),
        EntityRef(parent_domain, 0),
        EntityRef(parent_domain, 1),
        EntityRef(parent_domain, 0),
        EntityRef(parent_domain, 1),
    )
    for grain, parent in zip(grains, parents):
        dispatch.inputs_ready(grain, parent)

    first = _reserve_without_barriers(
        dispatch, call, max_size=3, pack_by_parent=True
    )
    second = _reserve_without_barriers(
        dispatch, call, max_size=3, pack_by_parent=True
    )

    assert first is not None and second is not None
    assert len({dispatch.parent_anchor(grain) for grain in first.grains}) == 1
    assert len({dispatch.parent_anchor(grain) for grain in second.grains}) == 1
    assert set(first.grains) | set(second.grains) == set(grains)
    assert dispatch.ready_count == 0


def test_unordered_ready_index_applies_suppression_barrier_without_cross_call_queue():
    dispatch = DispatchState(ready_fifo=False)
    call = CallRef(0)
    child_domain = DomainRef(1)
    parent_domain = DomainRef(0)
    barriered_parent = EntityRef(parent_domain, 0)
    live_parent = EntityRef(parent_domain, 1)
    grains = tuple(
        GrainRef(call, EntityRef(child_domain, ordinal))
        for ordinal in range(4)
    )
    for grain, parent in zip(
        grains,
        (barriered_parent, live_parent, barriered_parent, live_parent),
    ):
        dispatch.inputs_ready(grain, parent)

    batch, suppressed = dispatch.reserve_with_barriers(
        call,
        max_size=4,
        pack_by_parent=False,
        barriered_anchors={barriered_parent},
    )

    assert batch is not None
    assert set(batch.grains) == {grains[1], grains[3]}
    assert set(suppressed) == {grains[0], grains[2]}
    assert dispatch.ready_count == 0


def test_single_parent_packing_preserves_parent_and_relative_fifo_order():
    dispatch = DispatchState()
    call = CallRef(0)
    child_domain = DomainRef(1)
    parent_domain = DomainRef(0)
    grains = tuple(
        GrainRef(call, EntityRef(child_domain, ordinal))
        for ordinal in range(4)
    )
    parents = (
        EntityRef(parent_domain, 0),
        EntityRef(parent_domain, 1),
        EntityRef(parent_domain, 0),
        EntityRef(parent_domain, 1),
    )
    for grain, parent in zip(grains, parents):
        dispatch.inputs_ready(grain, parent)

    first = _reserve_without_barriers(
        dispatch, call, max_size=3, pack_by_parent=True
    )
    second = _reserve_without_barriers(
        dispatch, call, max_size=3, pack_by_parent=True
    )

    assert first is not None
    assert second is not None
    assert first.grains == (grains[0], grains[2])
    assert second.grains == (grains[1], grains[3])
    assert dispatch.ready_count == 0


def test_barriered_reserve_seals_only_barriered_parents_and_preserves_fifo():
    dispatch = DispatchState()
    call = CallRef(0)
    child_domain = DomainRef(1)
    parent_domain = DomainRef(0)
    grains = tuple(
        GrainRef(call, EntityRef(child_domain, ordinal))
        for ordinal in range(4)
    )
    barriered_parent = EntityRef(parent_domain, 0)
    live_parent = EntityRef(parent_domain, 1)
    parents = (
        barriered_parent,
        barriered_parent,
        live_parent,
        barriered_parent,
    )
    for grain, parent in zip(grains, parents):
        dispatch.inputs_ready(grain, parent)

    original_is_ready = dispatch._is_ready
    visits = 0

    def counted_is_ready(grain):
        nonlocal visits
        visits += 1
        return original_is_ready(grain)

    dispatch._is_ready = counted_is_ready

    batch, suppressed = dispatch.reserve_with_barriers(
        call,
        max_size=4,
        pack_by_parent=False,
        barriered_anchors={barriered_parent},
    )

    assert batch is not None
    assert batch.grains == (grains[2],)
    assert suppressed == (grains[0], grains[1], grains[3])
    snapshots = dispatch.snapshots()
    assert snapshots[grains[2]].phase is GrainPhase.IN_FLIGHT
    assert {
        snapshots[grain].phase for grain in suppressed
    } == {GrainPhase.SEALED}
    assert dispatch.ready_count == 0
    assert visits == len(grains)


@pytest.mark.parametrize(
    "action",
    [RecoveryAction.RETRY_IMMEDIATE, RecoveryAction.RETRY_TAIL],
)
def test_barriered_recovery_partitions_exact_batch_without_resetting_budget(action):
    dispatch = DispatchState()
    call = CallRef(0)
    child_domain = DomainRef(1)
    parent_domain = DomainRef(0)
    grains = tuple(
        GrainRef(call, EntityRef(child_domain, ordinal))
        for ordinal in range(4)
    )
    barriered_parent = EntityRef(parent_domain, 0)
    live_parent = EntityRef(parent_domain, 1)
    for grain, parent in zip(
        grains,
        (barriered_parent, live_parent, barriered_parent, live_parent),
    ):
        dispatch.inputs_ready(grain, parent)

    initial = _reserve_without_barriers(
        dispatch, call, max_size=4, pack_by_parent=False
    )
    assert initial is not None
    dispatch.recover_udf(initial, action)
    live, suppressed = dispatch.reserve_with_barriers(
        call,
        max_size=4,
        pack_by_parent=False,
        barriered_anchors={barriered_parent},
    )

    assert live is not None
    assert live.grains == (grains[1], grains[3])
    assert live.udf_retries == 1
    assert suppressed == (grains[0], grains[2])
    assert {
        dispatch.snapshot(grain).generation for grain in grains
    } == {1}


def test_dispatch_snapshots_do_not_leak_mutable_grain_authority():
    grain = _grains(1)[0]
    state = DispatchState()
    state.inputs_ready(grain, grain.entity)

    snapshots = state.snapshots()
    original = snapshots[grain]
    with pytest.raises(TypeError):
        operator.setitem(snapshots, grain, original)
    with pytest.raises(FrozenInstanceError):
        setattr(original, "generation", 1)

    assert _reserve_without_barriers(
        state, grain.call, max_size=1, pack_by_parent=False
    ) is not None
    assert original.phase is GrainPhase.READY
    assert state.snapshot(grain).phase is GrainPhase.IN_FLIGHT


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_recovery_policy_rejects_ambiguous_infrastructure_budgets(value):
    error = ValueError if value == -1 else TypeError
    with pytest.raises(error):
        RecoveryPolicy.abort(infra_retries=value)


def test_recovery_records_reject_states_outside_the_closed_algebra():
    with pytest.raises(TypeError, match="UdfRecoveryMode"):
        RecoveryPolicy("retry", 1, 1)
    with pytest.raises(ValueError, match="non-empty"):
        RecoveryPolicy.retry_batch().decide_udf(
            completed_retries=0,
            grain_count=0,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        DispatchBatch(_grains(1), -1)


def test_retry_policies_have_finite_exact_attempt_budgets():
    grains = _grains(3)
    for policy, action in (
        (RecoveryPolicy.retry_batch(attempts=2), RecoveryAction.RETRY_IMMEDIATE),
        (RecoveryPolicy.retry_tail(attempts=2), RecoveryAction.RETRY_TAIL),
    ):
        assert policy.decide_udf(
            completed_retries=0,
            grain_count=len(grains),
        ) is action
        assert policy.decide_udf(
            completed_retries=1,
            grain_count=len(grains),
        ) is action
        assert policy.decide_udf(
            completed_retries=2,
            grain_count=len(grains),
        ) is RecoveryAction.ABORT


def test_isolate_tail_is_one_retry_then_finite_binary_isolation():
    grains = _grains(4)
    policy = RecoveryPolicy.isolate_tail()

    assert policy.decide_udf(
        completed_retries=0,
        grain_count=len(grains),
    ) is RecoveryAction.RETRY_TAIL
    assert policy.decide_udf(
        completed_retries=1,
        grain_count=2,
    ) is RecoveryAction.SPLIT_TAIL
    assert policy.decide_udf(
        completed_retries=1,
        grain_count=1,
    ) is RecoveryAction.FAIL_SINGLETON


def test_udf_reducer_is_total_for_every_mode_and_context_state():
    grains = _grains(2)
    policies = {
        UdfRecoveryMode.ABORT: RecoveryPolicy.abort(),
        UdfRecoveryMode.RETRY_BATCH: RecoveryPolicy.retry_batch(),
        UdfRecoveryMode.RETRY_TAIL: RecoveryPolicy.retry_tail(),
        UdfRecoveryMode.ISOLATE_TAIL: RecoveryPolicy.isolate_tail(),
    }
    completed_retries = [0, 1]
    for mode in UdfRecoveryMode:
        for completed in completed_retries:
            action = policies[mode].decide_udf(
                completed_retries=completed,
                grain_count=len(grains),
            )
            assert isinstance(action, RecoveryAction)


def _dispatch_with_four_ready() -> tuple[
    DispatchState,
    tuple[GrainRef, ...],
]:
    dispatch = DispatchState()
    grains = _grains(4)
    for grain in grains:
        dispatch.inputs_ready(grain, grain.entity)
    return dispatch, grains


def _reserve(
    dispatch: DispatchState,
    *,
    max_size: int,
):
    selection = _reserve_without_barriers(
        dispatch,
        CallRef(0),
        max_size=max_size,
        pack_by_parent=False,
    )
    assert selection is not None
    return selection


def test_immediate_recovery_precedes_ready_work_and_preserves_exact_batch():
    dispatch, _ = _dispatch_with_four_ready()
    failed = _reserve(dispatch, max_size=2)

    dispatch.recover_udf(
        failed,
        RecoveryAction.RETRY_IMMEDIATE,
    )

    assert dispatch.priority(CallRef(0)) == 0
    retried = _reserve(dispatch, max_size=4)
    assert retried.grains == failed.grains
    assert retried.udf_retries == 1
    snapshots = dispatch.snapshots()
    assert {snapshots[grain].generation for grain in failed.grains} == {1}
    assert {snapshots[grain].infra_failures for grain in failed.grains} == {0}


def test_retry_tail_yields_to_ready_work_then_preserves_exact_batch():
    dispatch, _ = _dispatch_with_four_ready()
    failed = _reserve(dispatch, max_size=2)

    dispatch.recover_udf(
        failed,
        RecoveryAction.RETRY_TAIL,
    )

    assert dispatch.priority(CallRef(0)) == 1
    ready_batch = _reserve(dispatch, max_size=4)
    assert ready_batch.udf_retries == 0
    assert ready_batch.grains != failed.grains
    assert len(ready_batch.grains) == 2
    assert dispatch.priority(CallRef(0)) == 2
    retried = _reserve(dispatch, max_size=4)
    assert retried.grains == failed.grains
    assert retried.udf_retries == 1


def test_isolation_split_releases_once_and_queues_ordered_halves():
    dispatch, _ = _dispatch_with_four_ready()
    initial = _reserve(dispatch, max_size=4)
    dispatch.recover_udf(initial, RecoveryAction.RETRY_TAIL)
    failed = _reserve(dispatch, max_size=4)

    dispatch.recover_udf(
        failed,
        RecoveryAction.SPLIT_TAIL,
    )

    left = _reserve(dispatch, max_size=4)
    right = _reserve(dispatch, max_size=4)
    assert left.grains == failed.grains[:2]
    assert right.grains == failed.grains[2:]
    assert left.udf_retries == 1
    assert right.udf_retries == 1
    snapshots = dispatch.snapshots()
    assert {snapshots[grain].generation for grain in failed.grains} == {2}


def test_infrastructure_policy_preflight_does_not_mutate_dispatch_group():
    dispatch, _ = _dispatch_with_four_ready()
    failed = _reserve(dispatch, max_size=2)
    counts = dispatch.infrastructure_failures(failed)

    assert not RecoveryPolicy.abort(infra_retries=0).allows_infrastructure_retry(
        counts
    )
    snapshots = dispatch.snapshots()
    assert {
        snapshots[grain].phase for grain in failed.grains
    } == {GrainPhase.IN_FLIGHT}
    assert {snapshots[grain].generation for grain in failed.grains} == {0}
    assert {snapshots[grain].infra_failures for grain in failed.grains} == {0}
