"""Oracle-only Logical-Grain records.

The reference package deliberately duplicates these small records instead of
importing production Multigrain code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


DIGEST_BYTES = 16


class Primitive(Enum):
    SOURCE = "source"
    MAP = "map"
    FILTER = "filter"
    EXPAND = "expand"
    REDUCE = "reduce"
    RELATE = "relate"


@dataclass(frozen=True, slots=True)
class PortId:
    node: int
    slot: int

    def __post_init__(self) -> None:
        if self.node < 0 or self.slot < 0:
            raise ValueError("PortId fields must be non-negative")


@dataclass(frozen=True, slots=True)
class EntityId:
    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != DIGEST_BYTES:
            raise ValueError("EntityId must contain exactly 16 bytes")

    def hex(self) -> str:
        return self.raw.hex()


@dataclass(frozen=True, slots=True)
class ItemRef:
    port: PortId
    entity: EntityId


@dataclass(frozen=True, slots=True)
class RoleItems:
    role: str
    items: tuple[ItemRef, ...]

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("role must be non-empty")
        if not isinstance(self.items, tuple):
            raise TypeError("RoleItems.items must be a tuple")


@dataclass(frozen=True, slots=True)
class GrainId:
    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != DIGEST_BYTES:
            raise ValueError("GrainId must contain exactly 16 bytes")

    def hex(self) -> str:
        return self.raw.hex()


LineageCause: TypeAlias = ItemRef | GrainId


@dataclass(frozen=True, slots=True)
class Emission:
    item: ItemRef
    ordinal: int

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class Success:
    emissions_by_port: tuple[tuple[Emission, ...], ...]


@dataclass(frozen=True, slots=True)
class GrainFailure:
    kind: str
    message: str
    direct_causes: tuple[LineageCause, ...]


@dataclass(frozen=True, slots=True)
class Failed:
    failure: GrainFailure


@dataclass(frozen=True, slots=True)
class Suppressed:
    direct_causes: tuple[LineageCause, ...]


GrainOutcome: TypeAlias = Success | Failed | Suppressed


@dataclass(frozen=True, slots=True)
class GrainRecord:
    id: GrainId
    node: int
    inputs: tuple[RoleItems, ...]
    output_slots: tuple[ItemRef, ...]
    outcome: GrainOutcome


class SettlementKind(Enum):
    PRESENT = "present"
    DROPPED = "dropped"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True, slots=True)
class MemberSettlement:
    ordinal: int
    kind: SettlementKind
    item: ItemRef | None = None
    receipt: GrainId | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("settlement ordinal must be non-negative")

    @classmethod
    def present(cls, ordinal: int, item: ItemRef) -> "MemberSettlement":
        return cls(ordinal, SettlementKind.PRESENT, item=item)

    @classmethod
    def dropped(cls, ordinal: int) -> "MemberSettlement":
        return cls(ordinal, SettlementKind.DROPPED)

    @classmethod
    def failed(
        cls,
        ordinal: int,
        item: ItemRef,
        receipt: GrainId,
    ) -> "MemberSettlement":
        return cls(ordinal, SettlementKind.FAILED, item=item, receipt=receipt)

    @classmethod
    def suppressed(
        cls,
        ordinal: int,
        item: ItemRef,
        receipt: GrainId,
    ) -> "MemberSettlement":
        return cls(
            ordinal,
            SettlementKind.SUPPRESSED,
            item=item,
            receipt=receipt,
        )


@dataclass(frozen=True, slots=True)
class RelateResult:
    records: tuple[GrainRecord, ...]
    unmatched: tuple[ItemRef, ...]
    complete: bool
