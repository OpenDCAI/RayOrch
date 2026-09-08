"""不依赖 runtime/Ray 的封闭动态状态转移代数。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ..model import GrainPhase, InputMode, ItemOutcome, ExpansionOutcome


class InvalidTransition(ValueError):
    """输入事实不满足某个封闭状态机的合同。"""


class GrainEvent(Enum):
    """能够改变一个 Grain phase 的完整事件集合。"""

    INPUTS_READY = auto()
    INPUTS_TERMINAL = auto()
    RESERVE = auto()
    SUPPRESS = auto()
    RETRY = auto()
    REPORT = auto()


_GRAIN_TRANSITIONS = {
    (None, GrainEvent.INPUTS_READY): GrainPhase.READY,
    (None, GrainEvent.INPUTS_TERMINAL): GrainPhase.SEALED,
    (GrainPhase.READY, GrainEvent.RESERVE): GrainPhase.IN_FLIGHT,
    (GrainPhase.READY, GrainEvent.SUPPRESS): GrainPhase.SEALED,
    (GrainPhase.IN_FLIGHT, GrainEvent.RETRY): GrainPhase.READY,
    (GrainPhase.IN_FLIGHT, GrainEvent.REPORT): GrainPhase.SEALED,
}


def grain_transition(
    phase: GrainPhase | None,
    event: GrainEvent,
) -> GrainPhase:
    """执行唯一合法的 Grain phase 转移；非法边直接失败。"""

    try:
        return _GRAIN_TRANSITIONS[(phase, event)]
    except KeyError as error:
        before = "WAITING" if phase is None else phase.name
        raise InvalidTransition(f"illegal Grain transition: {before} + {event.name}") from error


def item_transition(
    current: ItemOutcome | None,
    publication: ItemOutcome,
) -> ItemOutcome:
    """Item 只允许从 UNRESOLVED 进入终态，或幂等重放同一终态。"""

    if current is None or current is publication:
        return publication
    raise InvalidTransition(
        f"conflicting Item transition: {current.name} -> {publication.name}"
    )


def expansion_transition(
    current: ExpansionOutcome | None,
    publication: ExpansionOutcome,
) -> ExpansionOutcome:
    """Expansion 与 Item 一样，只接受一次终态 publication。"""

    if current is None or current is publication:
        return publication
    raise InvalidTransition(
        f"conflicting Expansion transition: {current.name} -> {publication.name}"
    )


class CallAction(Enum):
    """输入归约对 Grain/outputs 的唯一动作。"""

    WAIT = auto()
    READY = auto()
    DROP_OUTPUTS = auto()
    SUPPRESS_OUTPUTS = auto()


@dataclass(frozen=True, slots=True)
class CallTransition:
    action: CallAction
    decisive_input: int | None = None


def call_transition(
    modes: tuple[InputMode, ...],
    outcomes: tuple[ItemOutcome | None, ...],
) -> CallTransition:
    """以 failure > unresolved > required-drop 的交换归约分类 Call。"""

    if not modes or len(modes) != len(outcomes):
        raise InvalidTransition("Call modes/outcomes must be non-empty and aligned")

    for index, outcome in enumerate(outcomes):
        if outcome in {ItemOutcome.FAILED, ItemOutcome.SUPPRESSED}:
            return CallTransition(CallAction.SUPPRESS_OUTPUTS, index)
    if any(outcome is None for outcome in outcomes):
        return CallTransition(CallAction.WAIT)
    for index, (mode, outcome) in enumerate(zip(modes, outcomes)):
        if mode is InputMode.REQUIRED and outcome is ItemOutcome.DROPPED:
            return CallTransition(CallAction.DROP_OUTPUTS, index)
    return CallTransition(CallAction.READY)


class FilterCause(Enum):
    SOURCE = auto()
    MASK = auto()


@dataclass(frozen=True, slots=True)
class FilterTransition:
    outcome: ItemOutcome | None
    cause: FilterCause | None = None


def filter_transition(
    source: ItemOutcome | None,
    mask: ItemOutcome | None,
    mask_control: bool | None,
) -> FilterTransition:
    """按 source gate、再 mask gate 决定 Filter target。"""

    if source is None:
        return FilterTransition(None)
    if source is not ItemOutcome.PRESENT:
        return FilterTransition(source, FilterCause.SOURCE)
    if mask is None:
        return FilterTransition(None)
    if mask is not ItemOutcome.PRESENT:
        return FilterTransition(mask, FilterCause.MASK)
    if type(mask_control) is not bool:
        raise InvalidTransition("PRESENT Filter mask requires bool control")
    if mask_control:
        return FilterTransition(ItemOutcome.PRESENT)
    return FilterTransition(ItemOutcome.DROPPED, FilterCause.MASK)


def broadcast_transition(source: ItemOutcome) -> ItemOutcome:
    """Broadcast 是透明 outcome view。"""

    return source


def expansion_outcome_from_item(output: ItemOutcome) -> ExpansionOutcome:
    """把 Call output outcome 映射为 Expansion 终态。"""

    if output is ItemOutcome.PRESENT:
        return ExpansionOutcome.SUCCEEDED
    if output is ItemOutcome.DROPPED:
        return ExpansionOutcome.DROPPED
    return ExpansionOutcome.FAILED


class ReduceCause(Enum):
    SHAPE = auto()
    MEMBER = auto()
    VALUE = auto()


@dataclass(frozen=True, slots=True)
class ReduceTransition:
    outcome: ItemOutcome | None
    survivors: tuple[int, ...] = ()
    cause: ReduceCause | None = None
    cause_index: int | None = None


def reduce_transition(
    expansion: ExpansionOutcome | None,
    members: tuple[ItemOutcome | None, ...],
    values: tuple[ItemOutcome | None, ...],
) -> ReduceTransition:
    """按 Expansion→members→survivor values 的显式 gate 归约 Reduce。"""

    if len(members) != len(values):
        raise InvalidTransition("Nested-group members/values must be aligned")
    if expansion is None:
        return ReduceTransition(None)
    if expansion is ExpansionOutcome.DROPPED:
        return ReduceTransition(ItemOutcome.DROPPED, cause=ReduceCause.SHAPE)
    if expansion is ExpansionOutcome.FAILED:
        return ReduceTransition(ItemOutcome.SUPPRESSED, cause=ReduceCause.SHAPE)

    for index, outcome in enumerate(members):
        if outcome in {ItemOutcome.FAILED, ItemOutcome.SUPPRESSED}:
            return ReduceTransition(
                ItemOutcome.SUPPRESSED,
                cause=ReduceCause.MEMBER,
                cause_index=index,
            )
    if any(outcome is None for outcome in members):
        return ReduceTransition(None)

    survivors = tuple(
        index
        for index, outcome in enumerate(members)
        if outcome is ItemOutcome.PRESENT
    )
    for index in survivors:
        outcome = values[index]
        if outcome is None:
            return ReduceTransition(None)
        if outcome is not ItemOutcome.PRESENT:
            return ReduceTransition(
                ItemOutcome.SUPPRESSED,
                cause=ReduceCause.VALUE,
                cause_index=index,
            )
    return ReduceTransition(ItemOutcome.PRESENT, survivors)


__all__ = [
    "CallAction",
    "CallTransition",
    "FilterCause",
    "FilterTransition",
    "GrainEvent",
    "ReduceCause",
    "ReduceTransition",
    "InvalidTransition",
    "broadcast_transition",
    "call_transition",
    "expansion_outcome_from_item",
    "filter_transition",
    "grain_transition",
    "reduce_transition",
    "item_transition",
    "expansion_transition",
]
