"""Logical-Grain semantic records, identities, and derived indexes."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, TypeAlias


DIGEST_BYTES = 16
RUN_SALT_BYTES = 16
PERSONALIZATION = b"RayOrchMGV2.5"


class GrainInvariantError(RuntimeError):
    """A semantic table or lifecycle invariant was violated."""


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
class AttemptToken:
    arena: int
    dispatch: int
    grain: GrainId
    generation: int


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


class GrainPhase(Enum):
    READY = "ready"
    IN_FLIGHT = "in_flight"
    SEALED = "sealed"


@dataclass(slots=True)
class GrainRecord:
    id: GrainId
    node: int
    inputs: tuple[RoleItems, ...]
    output_slots: tuple[ItemRef, ...]
    outcome: GrainOutcome | None = None
    phase: GrainPhase = GrainPhase.READY
    generation: int = 0
    active: AttemptToken | None = None
    infra_failures: int = 0

    def __post_init__(self) -> None:
        self.validate_state()

    def validate_state(self) -> None:
        legal = (
            (
                self.phase is GrainPhase.READY
                and self.active is None
                and self.outcome is None
            )
            or (
                self.phase is GrainPhase.IN_FLIGHT
                and self.active is not None
                and self.outcome is None
            )
            or (
                self.phase is GrainPhase.SEALED
                and self.active is None
                and self.outcome is not None
            )
        )
        if not legal:
            raise GrainInvariantError("illegal GrainRecord phase/active/outcome state")
        if self.generation < 0 or self.infra_failures < 0:
            raise GrainInvariantError("grain counters must be non-negative")
        if self.active is not None:
            if self.active.grain != self.id:
                raise GrainInvariantError("active token targets a different grain")
            if self.active.generation != self.generation:
                raise GrainInvariantError("active token generation mismatch")

    @classmethod
    def sealed(
        cls,
        *,
        id: GrainId,
        node: int,
        inputs: tuple[RoleItems, ...],
        output_slots: tuple[ItemRef, ...],
        outcome: GrainOutcome,
    ) -> "GrainRecord":
        return cls(
            id=id,
            node=node,
            inputs=inputs,
            output_slots=output_slots,
            outcome=outcome,
            phase=GrainPhase.SEALED,
        )

    def reserve(self, arena: int, dispatch: int) -> AttemptToken:
        if self.phase is not GrainPhase.READY:
            raise GrainInvariantError("only READY grains may be reserved")
        self.generation += 1
        token = AttemptToken(arena, dispatch, self.id, self.generation)
        self.phase = GrainPhase.IN_FLIGHT
        self.active = token
        self.validate_state()
        return token

    def retry_infrastructure_failure(self, token: AttemptToken) -> bool:
        if self.active != token:
            return False
        self.active = None
        self.phase = GrainPhase.READY
        self.infra_failures += 1
        self.validate_state()
        return True

    def release_for_reexecution(self, token: AttemptToken) -> bool:
        if self.active != token:
            return False
        self.active = None
        self.phase = GrainPhase.READY
        self.validate_state()
        return True

    def seal(
        self,
        outcome: GrainOutcome,
        *,
        token: AttemptToken | None = None,
    ) -> bool:
        if self.phase is GrainPhase.SEALED:
            if self.outcome != outcome:
                raise GrainInvariantError("grain already sealed with another outcome")
            return False
        if self.phase is GrainPhase.IN_FLIGHT and self.active != token:
            return False
        if self.phase is GrainPhase.READY and token is not None:
            raise GrainInvariantError("READY grain cannot accept an attempt token")
        self.outcome = outcome
        self.active = None
        self.phase = GrainPhase.SEALED
        self.validate_state()
        return True


def _u64(value: int) -> bytes:
    if value < 0:
        raise ValueError("length/count cannot be negative")
    return struct.pack(">Q", value)


def canonical_encode(value: Any) -> bytes:
    """Encode the closed identity type set."""

    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        magnitude = abs(value)
        raw = (
            b""
            if magnitude == 0
            else magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "big")
        )
        sign = b"\x01" if value < 0 else b"\x00"
        return b"i" + sign + _u64(len(raw)) + raw
    if isinstance(value, bytes):
        return b"y" + _u64(len(value)) + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"s" + _u64(len(raw)) + raw
    if isinstance(value, tuple):
        return b"t" + _u64(len(value)) + b"".join(
            canonical_encode(part) for part in value
        )
    if isinstance(value, PortId):
        return b"p" + canonical_encode(value.node) + canonical_encode(value.slot)
    if isinstance(value, EntityId):
        return b"e" + value.raw
    if isinstance(value, ItemRef):
        return b"r" + canonical_encode(value.port) + canonical_encode(value.entity)
    if isinstance(value, RoleItems):
        return b"o" + canonical_encode(value.role) + canonical_encode(value.items)
    if isinstance(value, GrainId):
        return b"g" + value.raw
    raise TypeError(f"unsupported canonical identity type: {type(value)!r}")


def _check_run_salt(run_salt: bytes) -> None:
    if type(run_salt) is not bytes or len(run_salt) != RUN_SALT_BYTES:
        raise ValueError("run_salt must be exactly 16 bytes")


def semantic_hash(domain: str, run_salt: bytes, *parts: Any) -> bytes:
    _check_run_salt(run_salt)
    return hashlib.blake2b(
        canonical_encode((domain, run_salt, *parts)),
        digest_size=DIGEST_BYTES,
        person=PERSONALIZATION,
    ).digest()


def source_entity(
    run_salt: bytes,
    source_port: PortId,
    source_position: int,
) -> EntityId:
    return EntityId(
        semantic_hash(
            "source-entity",
            run_salt,
            source_port,
            source_position,
        )
    )


def source_grain_id(
    run_salt: bytes,
    source_port: PortId,
    source_position: int,
) -> GrainId:
    return GrainId(
        semantic_hash(
            "source-grain",
            run_salt,
            source_port,
            source_position,
        )
    )


def _bound_grain_id(
    domain: str,
    run_salt: bytes,
    node: int,
    inputs: tuple[RoleItems, ...],
) -> GrainId:
    if not isinstance(inputs, tuple):
        raise TypeError("identity role bindings must be a tuple")
    return GrainId(semantic_hash(domain, run_salt, node, inputs))


def map_grain_id(
    run_salt: bytes,
    node: int,
    inputs: tuple[RoleItems, ...],
) -> GrainId:
    return _bound_grain_id("map-grain", run_salt, node, inputs)


def filter_grain_id(
    run_salt: bytes,
    node: int,
    inputs: tuple[RoleItems, ...],
) -> GrainId:
    return _bound_grain_id("filter-grain", run_salt, node, inputs)


def expand_grain_id(
    run_salt: bytes,
    node: int,
    inputs: tuple[RoleItems, ...],
) -> GrainId:
    return _bound_grain_id("expand-grain", run_salt, node, inputs)


def expand_entity(
    run_salt: bytes,
    node: int,
    parent_entity: EntityId,
    ordinal: int,
) -> EntityId:
    return EntityId(
        semantic_hash(
            "expand-entity",
            run_salt,
            node,
            parent_entity,
            ordinal,
        )
    )


def reduce_grain_id(
    run_salt: bytes,
    node: int,
    anchor: ItemRef,
) -> GrainId:
    return GrainId(semantic_hash("reduce-grain", run_salt, node, anchor))


def relate_entity(
    run_salt: bytes,
    node: int,
    inputs: tuple[RoleItems, ...],
) -> EntityId:
    if not isinstance(inputs, tuple):
        raise TypeError("identity role bindings must be a tuple")
    return EntityId(semantic_hash("relate-entity", run_salt, node, inputs))


def relate_grain_id(
    run_salt: bytes,
    node: int,
    inputs: tuple[RoleItems, ...],
) -> GrainId:
    return _bound_grain_id("relate-grain", run_salt, node, inputs)


class ProducerIndex:
    def __init__(self) -> None:
        self._producers: dict[ItemRef, GrainId] = {}

    def add(self, item: ItemRef, producer: GrainId) -> None:
        existing = self._producers.get(item)
        if existing is not None and existing != producer:
            raise GrainInvariantError("ItemRef has more than one producer")
        self._producers[item] = producer

    def register(self, record: GrainRecord) -> None:
        for item in record.output_slots:
            self.add(item, record.id)
        if isinstance(record.outcome, Success):
            for emissions in record.outcome.emissions_by_port:
                for emission in emissions:
                    self.add(emission.item, record.id)

    def get(self, item: ItemRef) -> GrainId | None:
        return self._producers.get(item)


class ConsumerIndex:
    def __init__(self) -> None:
        self._consumers: dict[ItemRef, list[GrainId]] = {}

    def register(self, record: GrainRecord) -> None:
        for role in record.inputs:
            for item in role.items:
                consumers = self._consumers.setdefault(item, [])
                if record.id not in consumers:
                    consumers.append(record.id)

    def get(self, item: ItemRef) -> tuple[GrainId, ...]:
        return tuple(self._consumers.get(item, ()))


class PortIndex:
    def __init__(self) -> None:
        self._items: dict[PortId, dict[EntityId, ItemRef]] = {}

    def add(self, item: ItemRef) -> None:
        by_entity = self._items.setdefault(item.port, {})
        existing = by_entity.get(item.entity)
        if existing is not None and existing != item:
            raise GrainInvariantError("duplicate entity on one port")
        by_entity[item.entity] = item

    def register(self, outcome: GrainOutcome) -> None:
        if not isinstance(outcome, Success):
            return
        for emissions in outcome.emissions_by_port:
            for emission in emissions:
                self.add(emission.item)

    def get(self, port: PortId, entity: EntityId) -> ItemRef | None:
        return self._items.get(port, {}).get(entity)

    def items(self, port: PortId) -> tuple[ItemRef, ...]:
        return tuple(self._items.get(port, {}).values())


class ValueIndex:
    def __init__(self) -> None:
        self._values: dict[ItemRef, Any] = {}

    def put(self, item: ItemRef, value_location: Any) -> None:
        if item in self._values:
            raise GrainInvariantError("value location already published")
        self._values[item] = value_location

    def get(self, item: ItemRef) -> Any:
        return self._values[item]

    def contains(self, item: ItemRef) -> bool:
        return item in self._values


@dataclass(frozen=True, slots=True)
class ExpandOrigin:
    anchor: ItemRef
    ordinal: int
    origin: GrainId


class ExpandOriginIndex:
    def __init__(self) -> None:
        self._origins: dict[EntityId, ExpandOrigin] = {}

    def add(self, entity: EntityId, origin: ExpandOrigin) -> None:
        existing = self._origins.get(entity)
        if existing is not None and existing != origin:
            raise GrainInvariantError("child entity has conflicting Expand origins")
        self._origins[entity] = origin

    def get(self, entity: EntityId) -> ExpandOrigin | None:
        return self._origins.get(entity)


@dataclass(frozen=True, slots=True)
class FiberId:
    reduce_node: int
    anchor: ItemRef


class FiberState(Enum):
    OPEN = "open"
    READY = "ready"
    SUPPRESSED = "suppressed"


@dataclass(slots=True)
class FiberBarrier:
    id: FiberId
    origin: GrainId
    expected: int | None = None
    present: dict[int, ItemRef] = field(default_factory=dict)
    dropped: set[int] = field(default_factory=set)
    failed: dict[int, tuple[ItemRef, GrainId]] = field(default_factory=dict)
    blocked_by: set[GrainId] = field(default_factory=set)

    def set_expected(self, expected: int) -> None:
        if expected < 0:
            raise GrainInvariantError("fiber expected count cannot be negative")
        if self.expected is not None and self.expected != expected:
            raise GrainInvariantError("fiber expected count changed")
        if self.blocked_by:
            raise GrainInvariantError("failed origin cannot later publish cardinality")
        self.expected = expected
        self._validate()

    def block_origin(self, cause: GrainId) -> None:
        if self.expected is not None:
            raise GrainInvariantError("known-cardinality fiber cannot lose its origin")
        if self.blocked_by and cause not in self.blocked_by:
            raise GrainInvariantError("fiber has more than one origin failure")
        self.blocked_by.add(cause)

    def settle_present(self, ordinal: int, item: ItemRef) -> None:
        self._settle(ordinal, "present", item)

    def settle_dropped(self, ordinal: int) -> None:
        self._settle(ordinal, "dropped", None)

    def settle_failed(
        self,
        ordinal: int,
        item: ItemRef,
        receipt: GrainId,
    ) -> None:
        self._settle(ordinal, "failed", (item, receipt))

    def _settle(self, ordinal: int, kind: str, value: Any) -> None:
        if ordinal < 0:
            raise GrainInvariantError("fiber ordinal cannot be negative")
        if self.expected is not None and ordinal >= self.expected:
            raise GrainInvariantError("fiber ordinal exceeds expected cardinality")
        existing: tuple[str, Any] | None = None
        if ordinal in self.present:
            existing = ("present", self.present[ordinal])
        elif ordinal in self.dropped:
            existing = ("dropped", None)
        elif ordinal in self.failed:
            existing = ("failed", self.failed[ordinal])
        if existing is not None:
            if existing != (kind, value):
                raise GrainInvariantError("fiber ordinal settled inconsistently")
            return
        if kind == "present":
            self.present[ordinal] = value
        elif kind == "dropped":
            self.dropped.add(ordinal)
        else:
            self.failed[ordinal] = value
        self._validate()

    def _validate(self) -> None:
        overlap = (
            set(self.present) & self.dropped
            or set(self.present) & set(self.failed)
            or self.dropped & set(self.failed)
        )
        if overlap:
            raise GrainInvariantError("fiber settlement classes overlap")
        if self.expected is not None and self.settled_count > self.expected:
            raise GrainInvariantError("fiber settled beyond expected count")

    @property
    def settled_count(self) -> int:
        return len(self.present) + len(self.dropped) + len(self.failed)

    @property
    def state(self) -> FiberState:
        self._validate()
        if self.blocked_by:
            return FiberState.SUPPRESSED
        if self.expected is None or self.settled_count < self.expected:
            return FiberState.OPEN
        if self.failed:
            return FiberState.SUPPRESSED
        return FiberState.READY

    def present_members(self) -> tuple[ItemRef, ...]:
        return tuple(self.present[ordinal] for ordinal in sorted(self.present))

    def known_members(self) -> tuple[ItemRef, ...]:
        members: list[ItemRef] = []
        for ordinal in sorted(set(self.present) | set(self.failed)):
            if ordinal in self.present:
                members.append(self.present[ordinal])
            else:
                members.append(self.failed[ordinal][0])
        return tuple(members)

    def suppression_causes(self) -> tuple[GrainId, ...]:
        if self.blocked_by:
            return tuple(sorted(self.blocked_by, key=lambda grain: grain.raw))
        return tuple(
            self.failed[ordinal][1] for ordinal in sorted(self.failed)
        )


class JoinIndex:
    """Small derived cache placeholder for the Phase 5 sealed-port join."""

    def __init__(self, roles: tuple[str, ...]) -> None:
        self.roles = roles
        self._rows: dict[str, dict[Any, list[ItemRef]]] = {
            role: {} for role in roles
        }
        self._sealed: set[str] = set()

    def add(self, role: str, key: Any, item: ItemRef) -> None:
        if role not in self._rows:
            raise GrainInvariantError(f"unknown join role: {role}")
        self._rows[role].setdefault(key, []).append(item)

    def seal(self, role: str) -> None:
        if role not in self._rows:
            raise GrainInvariantError(f"unknown join role: {role}")
        self._sealed.add(role)

    @property
    def complete(self) -> bool:
        return self._sealed == set(self.roles)

    def rows(self, role: str, key: Any) -> tuple[ItemRef, ...]:
        return tuple(self._rows[role].get(key, ()))


def _semantic_fields(record: GrainRecord) -> tuple[Any, ...]:
    return (
        record.id,
        record.node,
        record.inputs,
        record.output_slots,
    )


class GrainTable:
    def __init__(self, records: Iterable[GrainRecord] = ()) -> None:
        self._records: dict[GrainId, GrainRecord] = {}
        for record in records:
            self._records[record.id] = record

    def get(self, grain_id: GrainId) -> GrainRecord | None:
        return self._records.get(grain_id)

    def add_terminal(self, record: GrainRecord) -> GrainRecord:
        if record.phase is not GrainPhase.SEALED or record.outcome is None:
            raise GrainInvariantError("terminal insertion needs a SEALED grain")
        existing = self._records.get(record.id)
        if existing is None:
            self._records[record.id] = record
            return record
        if _semantic_fields(existing) != _semantic_fields(record):
            raise GrainInvariantError("same GrainId has conflicting semantic fields")
        if existing.outcome != record.outcome:
            raise GrainInvariantError("same GrainId has conflicting terminal outcome")
        return existing

    def ensure_executable(self, record: GrainRecord) -> GrainRecord:
        if record.id in self._records:
            return self._ensure(record, suppressed=False)
        if record.phase is not GrainPhase.READY or record.outcome is not None:
            raise GrainInvariantError("executable template must be READY")
        return self._ensure(record, suppressed=False)

    def ensure_suppressed(self, record: GrainRecord) -> GrainRecord:
        if record.id in self._records:
            return self._ensure(record, suppressed=True)
        if not isinstance(record.outcome, Suppressed):
            raise GrainInvariantError("suppressed template needs Suppressed outcome")
        return self._ensure(record, suppressed=True)

    def _ensure(self, record: GrainRecord, *, suppressed: bool) -> GrainRecord:
        existing = self._records.get(record.id)
        if existing is None:
            self._records[record.id] = record
            return record
        if _semantic_fields(existing) != _semantic_fields(record):
            raise GrainInvariantError("same GrainId has conflicting semantic fields")
        existing_suppressed = isinstance(existing.outcome, Suppressed)
        if existing_suppressed != suppressed:
            raise GrainInvariantError("grain classification changed across planner passes")
        return existing

    def __len__(self) -> int:
        return len(self._records)

    def values(self) -> tuple[GrainRecord, ...]:
        return tuple(self._records.values())
