"""v3.5.1 动态状态代数的笛卡尔积完备性回归。"""

from __future__ import annotations

import itertools
from dataclasses import fields

import pytest

from rayorch.experimental.multigrain_v3_5.model import (
    GrainPhase,
    InputMode,
    ItemOutcome,
    ShapeState,
)
from rayorch.experimental.multigrain_v3_5.transitions import (
    CallAction,
    FilterCause,
    GrainEvent,
    GroupCause,
    InvalidTransition,
    broadcast_transition,
    call_transition,
    expansion_shape_transition,
    filter_transition,
    grain_transition,
    group_transition,
    item_transition,
    shape_transition,
)
from rayorch.experimental.multigrain_v3_5.runtime.dispatch import GrainRecord
from rayorch.experimental.multigrain_v3_5.runtime.state import EntityOrigin, ShapeRecord


TERMINAL_OR_PENDING = (None, *ItemOutcome)


def test_runtime_records_contain_only_authoritative_state():
    assert tuple(field.name for field in fields(EntityOrigin)) == (
        "parent_entity",
        "ordinal",
    )
    assert tuple(field.name for field in fields(GrainRecord)) == (
        "phase",
        "generation",
        "infra_failures",
    )
    assert tuple(field.name for field in fields(ShapeRecord)) == (
        "state",
        "children",
        "cause",
    )


def test_grain_phase_event_cartesian_product_is_closed():
    expected = {
        (None, GrainEvent.INPUTS_READY): GrainPhase.READY,
        (None, GrainEvent.INPUTS_TERMINAL): GrainPhase.SEALED,
        (GrainPhase.READY, GrainEvent.RESERVE): GrainPhase.IN_FLIGHT,
        (GrainPhase.IN_FLIGHT, GrainEvent.RETRY): GrainPhase.READY,
        (GrainPhase.IN_FLIGHT, GrainEvent.REPORT): GrainPhase.SEALED,
    }
    for phase, event in itertools.product((None, *GrainPhase), GrainEvent):
        if (phase, event) in expected:
            assert grain_transition(phase, event) is expected[phase, event]
        else:
            with pytest.raises(InvalidTransition):
                grain_transition(phase, event)


def test_item_and_shape_terminal_publication_cartesian_products_are_monotonic():
    for current, publication in itertools.product(
        (None, *ItemOutcome),
        ItemOutcome,
    ):
        if current is None or current is publication:
            assert item_transition(current, publication) is publication
        else:
            with pytest.raises(InvalidTransition):
                item_transition(current, publication)

    for current, publication in itertools.product(
        (None, *ShapeState),
        ShapeState,
    ):
        if current is None or current is publication:
            assert shape_transition(current, publication) is publication
        else:
            with pytest.raises(InvalidTransition):
                shape_transition(current, publication)


def test_call_input_cartesian_product_is_order_independent():
    for modes in itertools.product(InputMode, repeat=2):
        for outcomes in itertools.product(TERMINAL_OR_PENDING, repeat=2):
            if any(
                outcome in {ItemOutcome.FAILED, ItemOutcome.SUPPRESSED}
                for outcome in outcomes
            ):
                expected = CallAction.SUPPRESS_OUTPUTS
            elif None in outcomes:
                expected = CallAction.WAIT
            elif any(
                mode is InputMode.REQUIRED and outcome is ItemOutcome.DROPPED
                for mode, outcome in zip(modes, outcomes)
            ):
                expected = CallAction.DROP_OUTPUTS
            else:
                expected = CallAction.READY

            assert call_transition(modes, outcomes).action is expected
            assert call_transition(modes[::-1], outcomes[::-1]).action is expected


def test_all_optional_dropped_inputs_are_ready_without_identity_driver():
    decision = call_transition(
        (InputMode.OPTIONAL, InputMode.OPTIONAL),
        (ItemOutcome.DROPPED, ItemOutcome.DROPPED),
    )
    assert decision.action is CallAction.READY


def test_filter_source_mask_control_cartesian_product_is_closed():
    for source, mask, control in itertools.product(
        TERMINAL_OR_PENDING,
        TERMINAL_OR_PENDING,
        (None, False, True),
    ):
        if source is None:
            expected = (None, None)
        elif source is not ItemOutcome.PRESENT:
            expected = (source, FilterCause.SOURCE)
        elif mask is None:
            expected = (None, None)
        elif mask is not ItemOutcome.PRESENT:
            expected = (mask, FilterCause.MASK)
        elif control is None:
            with pytest.raises(InvalidTransition):
                filter_transition(source, mask, control)
            continue
        elif control:
            expected = (ItemOutcome.PRESENT, None)
        else:
            expected = (ItemOutcome.DROPPED, FilterCause.MASK)

        decision = filter_transition(source, mask, control)
        assert (decision.outcome, decision.cause) == expected


def test_broadcast_and_expand_cover_every_item_terminal():
    shape_by_output = {
        ItemOutcome.PRESENT: ShapeState.SUCCEEDED,
        ItemOutcome.DROPPED: ShapeState.DROPPED,
        ItemOutcome.FAILED: ShapeState.FAILED,
        ItemOutcome.SUPPRESSED: ShapeState.FAILED,
    }
    for outcome in ItemOutcome:
        assert broadcast_transition(outcome) is outcome
        assert expansion_shape_transition(outcome) is shape_by_output[outcome]


def test_group_shape_member_value_cartesian_product_is_closed():
    for shape, member, value in itertools.product(
        (None, *ShapeState),
        TERMINAL_OR_PENDING,
        TERMINAL_OR_PENDING,
    ):
        decision = group_transition(shape, (member,), (value,))
        if shape is None:
            assert decision.outcome is None
        elif shape is ShapeState.DROPPED:
            assert (decision.outcome, decision.cause) == (
                ItemOutcome.DROPPED,
                GroupCause.SHAPE,
            )
        elif shape is ShapeState.FAILED:
            assert (decision.outcome, decision.cause) == (
                ItemOutcome.SUPPRESSED,
                GroupCause.SHAPE,
            )
        elif member in {ItemOutcome.FAILED, ItemOutcome.SUPPRESSED}:
            assert (decision.outcome, decision.cause) == (
                ItemOutcome.SUPPRESSED,
                GroupCause.MEMBER,
            )
        elif member is None:
            assert decision.outcome is None
        elif member is ItemOutcome.DROPPED:
            assert decision.outcome is ItemOutcome.PRESENT
            assert decision.survivors == ()
        elif value is None:
            assert decision.outcome is None
        elif value is ItemOutcome.PRESENT:
            assert decision.outcome is ItemOutcome.PRESENT
            assert decision.survivors == (0,)
        else:
            assert (decision.outcome, decision.cause) == (
                ItemOutcome.SUPPRESSED,
                GroupCause.VALUE,
            )


def test_group_ignores_excluded_values_but_failure_absorbs_pending_members():
    excluded = group_transition(
        ShapeState.SUCCEEDED,
        (ItemOutcome.DROPPED, ItemOutcome.PRESENT),
        (ItemOutcome.FAILED, ItemOutcome.PRESENT),
    )
    assert excluded.outcome is ItemOutcome.PRESENT
    assert excluded.survivors == (1,)

    failed = group_transition(
        ShapeState.SUCCEEDED,
        (None, ItemOutcome.FAILED),
        (None, None),
    )
    assert failed.outcome is ItemOutcome.SUPPRESSED
    assert failed.cause is GroupCause.MEMBER
    assert failed.cause_index == 1
