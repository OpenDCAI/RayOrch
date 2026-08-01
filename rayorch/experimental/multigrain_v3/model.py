"""Core immutable identities and stable runtime records for Multigrain V3."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias


DIGEST_BYTES = 16
RUN_SALT_BYTES = 16
PERSONALIZATION = b"RayOrchMGV3"


class InvariantError(RuntimeError):
    """An internal semantic or lifecycle invariant was violated."""


@dataclass(frozen=True, slots=True)
class PortId:
    """A static output coordinate in the compiled DAG."""

    stage: int
    output: int

    def __post_init__(self) -> None:
        if self.stage < 0 or self.output < 0:
            raise ValueError("PortId fields must be non-negative")


@dataclass(frozen=True, slots=True)
class EntityId:
    """A fixed-width logical occurrence identity."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != DIGEST_BYTES:
            raise ValueError("EntityId must contain exactly 16 bytes")

    def hex(self) -> str:
        return self.raw.hex()


@dataclass(frozen=True, slots=True)
class ItemRef:
    """A logical value coordinate: one entity on one output port."""

    port: PortId
    entity: EntityId


@dataclass(frozen=True, slots=True)
class GrainId:
    """A fixed-width logical operation identity."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != DIGEST_BYTES:
            raise ValueError("GrainId must contain exactly 16 bytes")

    def hex(self) -> str:
        return self.raw.hex()


@dataclass(frozen=True, slots=True)
class GroupShape:
    """Canonical nested-list shape encoded as per-level CSR offsets.

    The first offset array maps the single Reduce anchor to level-1 nodes.
    Each following array maps nodes at one Expand depth to the next depth.
    The final offset value equals the number of flat leaf ``items``.
    """

    offsets_by_level: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.offsets_by_level:
            raise ValueError("GroupShape requires at least one Expand level")
        expected_parents = 1
        for offsets in self.offsets_by_level:
            if len(offsets) != expected_parents + 1:
                raise ValueError("GroupShape offset arity is inconsistent")
            if not offsets or offsets[0] != 0:
                raise ValueError("GroupShape offsets must start at zero")
            if any(left > right for left, right in zip(offsets, offsets[1:])):
                raise ValueError("GroupShape offsets must be non-decreasing")
            expected_parents = offsets[-1]

    @property
    def depth(self) -> int:
        return len(self.offsets_by_level)

    @property
    def leaf_count(self) -> int:
        return self.offsets_by_level[-1][-1]


@dataclass(frozen=True, slots=True)
class InputBinding:
    """One compiled input name bound to ordered logical items."""

    name: str
    items: tuple[ItemRef, ...]
    group_shape: GroupShape | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("input binding name must be non-empty")
        if not isinstance(self.items, tuple):
            raise TypeError("InputBinding.items must be a tuple")
        if self.group_shape is not None and (
            self.group_shape.leaf_count != len(self.items)
        ):
            raise ValueError("GroupShape leaf count does not match items")


@dataclass(frozen=True, slots=True)
class AttemptToken:
    """Generation fence for one physical attempt of a Logical Grain."""
    arena: int
    dispatch: int
    grain: GrainId
    generation: int


@dataclass(frozen=True, slots=True)
class Emission:
    """One ordered logical output emitted on a Stage Port."""
    item: ItemRef
    ordinal: int

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("emission ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class Success:
    """Terminal success with one emission sequence per output Port."""
    emissions_by_port: tuple[tuple[Emission, ...], ...]


LineageCause: TypeAlias = ItemRef | GrainId


@dataclass(frozen=True, slots=True)
class GrainFailure:
    """User-visible failure attribution for one Logical Grain."""
    kind: str
    message: str
    direct_causes: tuple[LineageCause, ...] = ()


@dataclass(frozen=True, slots=True)
class Failed:
    """Terminal outcome for a Grain that executed and failed."""
    failure: GrainFailure


@dataclass(frozen=True, slots=True)
class Suppressed:
    """Terminal outcome for a Grain blocked by required dependencies."""
    direct_causes: tuple[LineageCause, ...]


GrainOutcome: TypeAlias = Success | Failed | Suppressed


class GrainPhase(Enum):
    """The only legal execution lifecycle states for a Grain."""
    READY = "ready"
    IN_FLIGHT = "in_flight"
    SEALED = "sealed"


@dataclass(slots=True)
class GrainRecord:
    """Logical operation state; physical execution details stay elsewhere."""

    id: GrainId
    stage: int
    inputs: tuple[InputBinding, ...]
    output_ports: tuple[PortId, ...]
    outcome: GrainOutcome | None = None
    phase: GrainPhase = GrainPhase.READY
    generation: int = 0
    active_attempt: AttemptToken | None = None
    infra_failures: int = 0

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def sealed(
        cls,
        *,
        id: GrainId,
        stage: int,
        inputs: tuple[InputBinding, ...],
        output_ports: tuple[PortId, ...],
        outcome: GrainOutcome,
    ) -> "GrainRecord":
        return cls(
            id=id,
            stage=stage,
            inputs=inputs,
            output_ports=output_ports,
            outcome=outcome,
            phase=GrainPhase.SEALED,
        )

    def validate(self) -> None:
        legal = (
            self.phase is GrainPhase.READY
            and self.outcome is None
            and self.active_attempt is None
        ) or (
            self.phase is GrainPhase.IN_FLIGHT
            and self.outcome is None
            and self.active_attempt is not None
        ) or (
            self.phase is GrainPhase.SEALED
            and self.outcome is not None
            and self.active_attempt is None
        )
        if not legal:
            raise InvariantError("illegal GrainRecord lifecycle state")
        if self.generation < 0 or self.infra_failures < 0:
            raise InvariantError("grain counters must be non-negative")
        if self.active_attempt is not None:
            if self.active_attempt.grain != self.id:
                raise InvariantError("active attempt targets another grain")
            if self.active_attempt.generation != self.generation:
                raise InvariantError("active attempt generation mismatch")

    def reserve(self, arena: int, dispatch: int) -> AttemptToken:
        if self.phase is not GrainPhase.READY:
            raise InvariantError("only READY grains can be reserved")
        self.generation += 1
        token = AttemptToken(arena, dispatch, self.id, self.generation)
        self.active_attempt = token
        self.phase = GrainPhase.IN_FLIGHT
        self.validate()
        return token

    def release(self, token: AttemptToken, *, infrastructure: bool = False) -> bool:
        if self.active_attempt != token:
            return False
        self.active_attempt = None
        self.phase = GrainPhase.READY
        if infrastructure:
            self.infra_failures += 1
        self.validate()
        return True

    def seal(self, outcome: GrainOutcome, token: AttemptToken | None = None) -> bool:
        if self.phase is GrainPhase.SEALED:
            if self.outcome != outcome:
                raise InvariantError("grain already sealed with another outcome")
            return False
        if self.phase is GrainPhase.IN_FLIGHT and self.active_attempt != token:
            return False
        if self.phase is GrainPhase.READY and token is not None:
            raise InvariantError("READY grain cannot accept an attempt token")
        self.outcome = outcome
        self.phase = GrainPhase.SEALED
        self.active_attempt = None
        self.validate()
        return True


class ItemTerminal(Enum):
    """Per-Port terminal receipt classification."""
    PRESENT = "present"
    DROPPED = "dropped"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True, slots=True)
class ItemRecord:
    """Terminal per-port lineage; physical value location is separate."""

    ref: ItemRef
    producer: GrainId | None
    terminal: ItemTerminal
    cause: GrainId | None = None

    def __post_init__(self) -> None:
        if self.terminal is ItemTerminal.PRESENT and self.producer is None:
            raise ValueError("PRESENT item must have a producer")
        if self.terminal in {ItemTerminal.FAILED, ItemTerminal.SUPPRESSED}:
            if self.producer is None:
                raise ValueError("failed/suppressed item must have a producer")


@dataclass(frozen=True, slots=True)
class EntityOrigin:
    """Parent/ordinal ancestry introduced by one Expand Stage."""
    parent_entity: EntityId
    expand_stage: int
    expand_grain: GrainId
    ordinal: int

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("entity ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class BlockRow:
    """Arena-local physical coordinate inside one coarse block."""
    block: int
    row: int

    def __post_init__(self) -> None:
        if self.block < 0 or self.row < 0:
            raise ValueError("BlockRow fields must be non-negative")


def _u64(value: int) -> bytes:
    if value < 0:
        raise ValueError("length/count cannot be negative")
    return struct.pack(">Q", value)


def canonical_encode(value: Any) -> bytes:
    """Encode the closed identity type set deterministically."""

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
        return b"i" + (b"\x01" if value < 0 else b"\x00") + _u64(len(raw)) + raw
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
        return b"p" + canonical_encode(value.stage) + canonical_encode(value.output)
    if isinstance(value, EntityId):
        return b"e" + value.raw
    if isinstance(value, ItemRef):
        return b"r" + canonical_encode(value.port) + canonical_encode(value.entity)
    if isinstance(value, GrainId):
        return b"g" + value.raw
    if isinstance(value, InputBinding):
        return (
            b"o"
            + canonical_encode(value.name)
            + canonical_encode(value.items)
            + canonical_encode(value.group_shape)
        )
    if isinstance(value, GroupShape):
        return b"h" + canonical_encode(value.offsets_by_level)
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


def source_entity(run_salt: bytes, source_position: int) -> EntityId:
    return EntityId(semantic_hash("source-entity", run_salt, source_position))


def source_grain_id(
    run_salt: bytes,
    source_port: PortId,
    source_position: int,
) -> GrainId:
    return GrainId(
        semantic_hash("source-grain", run_salt, source_port, source_position)
    )


def stage_grain_id(
    run_salt: bytes,
    stage: int,
    inputs: tuple[InputBinding, ...],
) -> GrainId:
    return GrainId(semantic_hash("stage-grain", run_salt, stage, inputs))


def expand_entity(
    run_salt: bytes,
    expand_stage: int,
    parent_entity: EntityId,
    ordinal: int,
) -> EntityId:
    return EntityId(
        semantic_hash(
            "expand-entity",
            run_salt,
            expand_stage,
            parent_entity,
            ordinal,
        )
    )
