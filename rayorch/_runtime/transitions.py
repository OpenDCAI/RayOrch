"""Closed dynamic transition algebra independent of runtime state and Ray."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .._model import GrainPhase, InputMode, ItemOutcome, ExpansionOutcome


class InvalidTransition(ValueError):
    """Input facts violate a closed state-machine contract."""


class GrainEvent(Enum):
    """Exhaustive event set capable of changing a Grain phase."""

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
    """Apply the sole legal Grain phase transition and reject invalid edges."""

    try:
        return _GRAIN_TRANSITIONS[(phase, event)]
    except KeyError as error:
        before = "WAITING" if phase is None else phase.name
        raise InvalidTransition(f"illegal Grain transition: {before} + {event.name}") from error


def item_transition(
    current: ItemOutcome | None,
    publication: ItemOutcome,
) -> ItemOutcome:
    """Allow only initial Item publication or idempotent replay."""

    if current is None or current is publication:
        return publication
    raise InvalidTransition(
        f"conflicting Item transition: {current.name} -> {publication.name}"
    )


def expansion_transition(
    current: ExpansionOutcome | None,
    publication: ExpansionOutcome,
) -> ExpansionOutcome:
    """Allow only initial Expansion publication or idempotent replay."""

    if current is None or current is publication:
        return publication
    raise InvalidTransition(
        f"conflicting Expansion transition: {current.name} -> {publication.name}"
    )


class CallAction(Enum):
    """Exhaustive actions produced by reducing Call input states."""

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
    """Classify a Call with failure > unresolved > required-drop precedence."""

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
    """Resolve a Filter target through the source gate and then mask gate."""

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
    """Treat Broadcast as a transparent outcome view."""

    return source


def expansion_outcome_from_item(output: ItemOutcome) -> ExpansionOutcome:
    """Map a Call output outcome to an Expansion outcome."""

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
    """Reduce through explicit Expansion, member, and survivor-value gates."""

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
