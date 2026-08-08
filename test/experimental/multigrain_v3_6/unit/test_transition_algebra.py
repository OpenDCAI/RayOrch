"""v3.6 动态状态代数的笛卡尔积完备性回归。"""

from __future__ import annotations

import itertools
from dataclasses import fields

import pytest

from rayorch.experimental.multigrain_v3_6.model import (
    GrainPhase,
    InputMode,
    ItemOutcome,
    ExpansionOutcome,
)
from rayorch.experimental.multigrain_v3_6.runtime.transitions import (
    CallAction,
    FilterCause,
    GrainEvent,
    ReduceCause,
    InvalidTransition,
    broadcast_transition,
    call_transition,
    expansion_outcome_from_item,
    filter_transition,
    grain_transition,
    reduce_transition,
    item_transition,
    expansion_transition,
)
from rayorch.experimental.multigrain_v3_6.runtime.dispatch import GrainRecord
from rayorch.experimental.multigrain_v3_6.runtime.state import EntityParent, ExpansionRecord


TERMINAL_OR_PENDING = (None, *ItemOutcome)


def test_runtime_records_contain_only_authoritative_state():
    assert tuple(field.name for field in fields(EntityParent)) == (
        "parent_entity",
        "ordinal",
    )
    assert tuple(field.name for field in fields(GrainRecord)) == (
        "phase",
        "generation",
        "infra_failures",
    )
    assert tuple(field.name for field in fields(ExpansionRecord)) == (
        "outcome",
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


def test_item_and_expansion_terminal_publications_are_monotonic():
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
        (None, *ExpansionOutcome),
        ExpansionOutcome,
    ):
        if current is None or current is publication:
            assert expansion_transition(current, publication) is publication
        else:
            with pytest.raises(InvalidTransition):
                expansion_transition(current, publication)


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
    expansion_by_output = {
        ItemOutcome.PRESENT: ExpansionOutcome.SUCCEEDED,
        ItemOutcome.DROPPED: ExpansionOutcome.DROPPED,
        ItemOutcome.FAILED: ExpansionOutcome.FAILED,
        ItemOutcome.SUPPRESSED: ExpansionOutcome.FAILED,
    }
    for outcome in ItemOutcome:
        assert broadcast_transition(outcome) is outcome
        assert expansion_outcome_from_item(outcome) is expansion_by_output[outcome]


def test_reduce_expansion_member_value_cartesian_product_is_closed():
    for expansion, member, value in itertools.product(
        (None, *ExpansionOutcome),
        TERMINAL_OR_PENDING,
        TERMINAL_OR_PENDING,
    ):
        decision = reduce_transition(expansion, (member,), (value,))
        if expansion is None:
            assert decision.outcome is None
        elif expansion is ExpansionOutcome.DROPPED:
            assert (decision.outcome, decision.cause) == (
                ItemOutcome.DROPPED,
                ReduceCause.SHAPE,
            )
        elif expansion is ExpansionOutcome.FAILED:
            assert (decision.outcome, decision.cause) == (
                ItemOutcome.SUPPRESSED,
                ReduceCause.SHAPE,
            )
        elif member in {ItemOutcome.FAILED, ItemOutcome.SUPPRESSED}:
            assert (decision.outcome, decision.cause) == (
                ItemOutcome.SUPPRESSED,
                ReduceCause.MEMBER,
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
                ReduceCause.VALUE,
            )


def test_group_ignores_excluded_values_but_failure_absorbs_pending_members():
    excluded = reduce_transition(
        ExpansionOutcome.SUCCEEDED,
        (ItemOutcome.DROPPED, ItemOutcome.PRESENT),
        (ItemOutcome.FAILED, ItemOutcome.PRESENT),
    )
    assert excluded.outcome is ItemOutcome.PRESENT
    assert excluded.survivors == (1,)

    failed = reduce_transition(
        ExpansionOutcome.SUCCEEDED,
        (None, ItemOutcome.FAILED),
        (None, None),
    )
    assert failed.outcome is ItemOutcome.SUPPRESSED
    assert failed.cause is ReduceCause.MEMBER
    assert failed.cause_index == 1
