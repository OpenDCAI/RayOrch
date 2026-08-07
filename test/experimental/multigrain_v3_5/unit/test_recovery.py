"""Failure-policy reducer and Arena-owned recovery queue regression."""

from __future__ import annotations

import operator
from dataclasses import FrozenInstanceError

import pytest

from rayorch.experimental.multigrain_v3_5.model import (
    CallRef,
    DomainRef,
    EntityRef,
    GrainPhase,
    GrainRef,
)
from rayorch.experimental.multigrain_v3_5.recovery import (
    RecoveryAction,
    RecoveryPolicy,
    UdfRecoveryMode,
)
from rayorch.experimental.multigrain_v3_5.runtime.dispatch import (
    DispatchSelection,
    DispatchState,
)


def _grains(count: int) -> tuple[GrainRef, ...]:
    call = CallRef(0)
    domain = DomainRef(0)
    return tuple(
        GrainRef(call, EntityRef(domain, ordinal)) for ordinal in range(count)
    )


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

    assert state.reserve(grain.call, max_size=1, parent_bound=False) is not None
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
        DispatchSelection(_grains(1), -1)


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
    selection = dispatch.reserve(
        CallRef(0),
        max_size=max_size,
        parent_bound=False,
    )
    assert selection is not None
    return selection


def test_immediate_recovery_precedes_normal_work_and_preserves_exact_group():
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


def test_tail_recovery_yields_to_normal_work_then_preserves_exact_group():
    dispatch, _ = _dispatch_with_four_ready()
    failed = _reserve(dispatch, max_size=2)

    dispatch.recover_udf(
        failed,
        RecoveryAction.RETRY_TAIL,
    )

    assert dispatch.priority(CallRef(0)) == 1
    normal = _reserve(dispatch, max_size=4)
    assert normal.udf_retries == 0
    assert normal.grains != failed.grains
    assert len(normal.grains) == 2
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
