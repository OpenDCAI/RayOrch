"""Ray-free logical identities and terminal states.

This module deliberately depends on neither Ray nor compiler/runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

@dataclass(frozen=True, slots=True, order=True)
class CallRef:
    """Compact identity of one compute call site in a static Program."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("CallRef must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class PortRef:
    """Compact identity of one logical data Port in a static Program."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("PortRef must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class DomainRef:
    """Identity of an entity-granularity level; values are local to a Domain."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("DomainRef must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class EntityRef:
    """One logical occurrence inside a Domain.

    ``value`` is local to one MicrobatchEngine. The Domain is part of identity,
    preventing unrelated coordinates with equal integers from aligning.
    """

    domain: DomainRef
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("EntityRef.value must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class ItemRef:
    """The intersection of a Port and Entity: one logical data occurrence."""

    port: PortRef
    entity: EntityRef


@dataclass(frozen=True, slots=True, order=True)
class GrainRef:
    """The intersection of a Call and Entity: one logical invocation."""

    call: CallRef
    entity: EntityRef


class InputMode(Enum):
    """Propagation policy when a Call input is not PRESENT."""

    REQUIRED = auto()
    OPTIONAL = auto()


class ItemOutcome(Enum):
    """Mutually exclusive terminal Item states; only PRESENT carries a value."""

    PRESENT = auto()
    DROPPED = auto()
    FAILED = auto()
    SUPPRESSED = auto()


class GrainPhase(Enum):
    """Lifecycle phases from runnable through in-flight to sealed."""

    READY = auto()
    IN_FLIGHT = auto()
    SEALED = auto()


class ExpansionOutcome(Enum):
    """Terminal state and cardinality availability of one fan-out Expansion."""

    SUCCEEDED = auto()
    DROPPED = auto()
    FAILED = auto()


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


__all__ = [
    "CallRef",
    "DomainRef",
    "EntityRef",
    "GrainPhase",
    "GrainRef",
    "InputMode",
    "ItemOutcome",
    "ItemRef",
    "MISSING",
    "PortRef",
    "ExpansionOutcome",
]
