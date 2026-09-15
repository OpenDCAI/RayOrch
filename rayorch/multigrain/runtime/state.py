"""Passive runtime records and tables with no scheduling policy or Ray dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from ..model import (
    DomainRef,
    EntityRef,
    GrainRef,
    ItemOutcome,
    ItemRef,
    ExpansionOutcome,
)
from ..protocol import RowBinding


class CommitError(RuntimeError):
    """A commit conflicts with an already-published immutable runtime fact."""


@dataclass(frozen=True, slots=True)
class ExpansionRef:
    """Identity of one parent Entity's fan-out into a child Domain."""

    child_domain: DomainRef
    parent_entity: EntityRef


@dataclass(frozen=True, slots=True)
class EntityParent:
    """A child Entity's explicit parent reference and stable ordinal."""

    parent_entity: EntityRef
    ordinal: int


@dataclass(frozen=True, slots=True)
class ItemRecord:
    """Immutable semantic facts for one Item.

    ``outcome`` records the terminal membership/computation state, ``cause``
    records abnormal provenance, and ``control`` stores the small control-plane
    value required by scheduling (currently a Filter boolean). Business payloads
    remain only in ``RuntimeState.values``, so the engine can recover and
    continue structural propagation without reading them.
    """

    outcome: ItemOutcome
    cause: object | None = None
    control: bool | None = None


@dataclass(frozen=True, slots=True)
class ExpansionRecord:
    """Terminal state and unique ordered children of one Expansion."""

    outcome: ExpansionOutcome
    children: tuple[EntityRef, ...] | None
    cause: object | None = None

    def __post_init__(self) -> None:
        if (self.outcome is ExpansionOutcome.SUCCEEDED) != (self.children is not None):
            raise ValueError("only SUCCEEDED Expansion may contain children")

    @property
    def cardinality(self) -> int | None:
        return None if self.children is None else len(self.children)


@dataclass(frozen=True, slots=True)
class NestedGroupLayout:
    """CSR-style level offsets from one implicit root to ordered leaves."""

    offsets_by_level: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.offsets_by_level:
            raise ValueError("NestedGroupLayout requires at least one level")
        for offsets in self.offsets_by_level:
            if not offsets or offsets[0] != 0:
                raise ValueError("every NestedGroupLayout level must start at zero")
            if any(left > right for left, right in zip(offsets, offsets[1:])):
                raise ValueError("NestedGroupLayout offsets must be monotonic")

    @property
    def depth(self) -> int:
        """Return the structural depth from the implicit root to leaves."""

        return len(self.offsets_by_level)

    @classmethod
    def one_level(cls, count: int) -> NestedGroupLayout:
        """Construct a one-level group containing ``count`` ordered leaves."""

        if count < 0:
            raise ValueError("group count must be non-negative")
        return cls(((0, count),))

    @classmethod
    def nest(
        cls,
        children: tuple[NestedGroupLayout, ...],
        *,
        child_depth: int,
    ) -> NestedGroupLayout:
        """Nest equal-depth child layouts while preserving empty-group depth."""

        if child_depth <= 0:
            raise ValueError("child_depth must be positive")
        if any(child.depth != child_depth for child in children):
            raise ValueError("nested children must have one canonical depth")

        top = (0, len(children))
        if not children:
            return cls((top,) + tuple((0,) for _ in range(child_depth)))

        merged: list[tuple[int, ...]] = [top]
        for level in range(child_depth):
            offsets = [0]
            total = 0
            for child in children:
                current = child.offsets_by_level[level]
                for left, right in zip(current, current[1:]):
                    total += right - left
                    offsets.append(total)
            merged.append(tuple(offsets))
        return cls(tuple(merged))


@dataclass(frozen=True, slots=True)
class NestedGroupBinding:
    """Reference-only binding of a canonical layout to flat leaf Items."""

    layout: NestedGroupLayout
    flat_items: tuple[ItemRef, ...]

    def __post_init__(self) -> None:
        if self.layout.offsets_by_level[-1][-1] != len(self.flat_items):
            raise ValueError("NestedGroupLayout leaf count does not match flat_items")


ValueBinding: TypeAlias = RowBinding | NestedGroupBinding


@dataclass(slots=True)
class PendingGrain:
    """Unresolved Call input slots in CallSpec order."""

    slots: list[ItemRef | None]


@dataclass(slots=True)
class RuntimeState:
    """All passive semantic tables exclusively written by MicrobatchEngine."""

    items: dict[ItemRef, ItemRecord] = field(default_factory=dict)
    expansions: dict[ExpansionRef, ExpansionRecord] = field(default_factory=dict)
    entity_lineage: dict[EntityRef, EntityParent] = field(default_factory=dict)
    values: dict[ItemRef, ValueBinding] = field(default_factory=dict)
    pending_grains: dict[GrainRef, PendingGrain] = field(default_factory=dict)


__all__ = [
    "CommitError",
    "EntityParent",
    "NestedGroupBinding",
    "NestedGroupLayout",
    "ItemRecord",
    "PendingGrain",
    "RuntimeState",
    "ExpansionRef",
    "ExpansionRecord",
    "ValueBinding",
]
