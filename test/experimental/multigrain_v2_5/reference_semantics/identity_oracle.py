"""Independent canonical encoding and identity derivation oracle."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

from .models import EntityId, GrainId, ItemRef, PortId, RoleItems


DIGEST_BYTES = 16
RUN_SALT_BYTES = 16
PERSONALIZATION = b"RayOrchMGV2.5"


def _u64(value: int) -> bytes:
    if value < 0:
        raise ValueError("length/count cannot be negative")
    return struct.pack(">Q", value)


def canonical_encode(value: Any) -> bytes:
    """Encode the closed identity type set specified by the architecture."""

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
    payload = canonical_encode((domain, run_salt, *parts))
    return hashlib.blake2b(
        payload,
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
