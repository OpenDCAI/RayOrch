"""Physical-invariant identity and MGCV1 canonical encoding."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import TypeAlias
import unicodedata


CanonicalValue: TypeAlias = (
    None | bool | int | float | str | bytes | tuple["CanonicalValue", ...]
)


def _require_digest(value: bytes, name: str) -> None:
    if not isinstance(value, bytes) or len(value) != hashlib.sha256().digest_size:
        raise ValueError(f"{name} must contain a full SHA-256 digest")


@dataclass(frozen=True, slots=True)
class BatchId:
    value: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.value, bytes) or not self.value:
            raise ValueError("BatchId value must be non-empty bytes")


@dataclass(frozen=True, slots=True)
class NodeId:
    value: bytes

    def __post_init__(self) -> None:
        _require_digest(self.value, "NodeId")


@dataclass(frozen=True, slots=True)
class PortId:
    value: bytes

    def __post_init__(self) -> None:
        _require_digest(self.value, "PortId")


@dataclass(frozen=True, slots=True)
class DomainId:
    value: bytes

    def __post_init__(self) -> None:
        _require_digest(self.value, "DomainId")


@dataclass(frozen=True, slots=True)
class EntityId:
    value: bytes

    def __post_init__(self) -> None:
        _require_digest(self.value, "EntityId")


@dataclass(frozen=True, slots=True)
class RowId:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("RowId value must be an int")
        if self.value < 0:
            raise ValueError("RowId value must be non-negative")


@dataclass(frozen=True, slots=True)
class MicrobatchAttempt:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("MicrobatchAttempt value must be an int")
        if self.value < 0:
            raise ValueError("MicrobatchAttempt value must be non-negative")


@dataclass(frozen=True, slots=True)
class RowRef:
    batch: BatchId
    port: PortId
    row: RowId


@dataclass(frozen=True, slots=True)
class IdentityKey:
    batch: BatchId
    domain: DomainId
    entity: EntityId


def _uleb128(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ULEB128 value must be an int")
    if value < 0:
        raise ValueError("ULEB128 value must be non-negative")
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            encoded.append(byte | 0x80)
        else:
            encoded.append(byte)
            return bytes(encoded)


def _frame(payload: bytes) -> bytes:
    return _uleb128(len(payload)) + payload


def canonical_encode(value: CanonicalValue) -> bytes:
    """Encode one value using the type-sensitive MGCV1 byte grammar."""
    value_type = type(value)
    if value is None:
        tag, payload = b"N", b""
    elif value_type is bool:
        tag, payload = b"B", b"\x01" if value else b"\x00"
    elif value_type is int:
        tag, payload = b"I", str(value).encode("ascii")
    elif value_type is float:
        if not math.isfinite(value):
            raise ValueError("MGCV1 does not support non-finite floats")
        normalized = 0.0 if value == 0.0 else value
        tag, payload = b"F", struct.pack(">d", normalized)
    elif value_type is str:
        tag = b"S"
        payload = unicodedata.normalize("NFC", value).encode("utf-8")
    elif value_type is bytes:
        tag, payload = b"Y", value
    elif value_type is tuple:
        tag = b"T"
        children = []
        for child in value:
            encoded = canonical_encode(child)
            children.append(_frame(encoded))
        payload = b"".join(children)
    else:
        raise TypeError(
            "MGCV1 supports only None, bool, int, finite float, str, bytes, "
            "and recursive tuples"
        )
    return tag + _frame(payload)


def stable_hash(
    domain_tag: CanonicalValue,
    *fields: CanonicalValue,
) -> bytes:
    parts = [b"MGID/v1", _frame(canonical_encode(domain_tag))]
    parts.extend(_frame(canonical_encode(field)) for field in fields)
    return hashlib.sha256(b"".join(parts)).digest()


def derive_node_id(
    pipeline_module: str,
    pipeline_qualname: str,
    attribute_path: str,
    primitive_kind: str,
) -> NodeId:
    return NodeId(
        stable_hash(
            "node",
            pipeline_module,
            pipeline_qualname,
            attribute_path,
            primitive_kind,
        )
    )


def derive_port_id(node: NodeId, slot: int) -> PortId:
    if type(slot) is not int or slot < 0:
        raise ValueError("port slot must be non-negative")
    return PortId(stable_hash("port", node.value, slot))


def derive_input_salt(
    pipeline_module: str,
    pipeline_qualname: str,
    parameter_name: str,
    parameter_position: int,
) -> bytes:
    if type(parameter_position) is not int or parameter_position < 0:
        raise ValueError("parameter position must be non-negative")
    return stable_hash(
        "input",
        pipeline_module,
        pipeline_qualname,
        parameter_name,
        parameter_position,
    )


def derive_input_port_id(input_salt: bytes) -> PortId:
    return PortId(stable_hash("input-port", input_salt))


def derive_source_domain(input_salt: bytes) -> DomainId:
    return DomainId(stable_hash("source-domain", input_salt))


def derive_expand_domain(node: NodeId) -> DomainId:
    return DomainId(stable_hash("expand-domain", node.value))


def derive_relate_domain(node: NodeId) -> DomainId:
    return DomainId(stable_hash("relate-domain", node.value))


def derive_source_position_entity(domain: DomainId, ordinal: int) -> EntityId:
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("source ordinal must be non-negative")
    return EntityId(stable_hash("source-position", domain.value, ordinal))


def derive_source_key_entity(
    domain: DomainId,
    key: CanonicalValue,
) -> EntityId:
    return EntityId(
        stable_hash("source-key", domain.value, canonical_encode(key))
    )


def derive_expand_entity(
    node: NodeId,
    parent_domain: DomainId,
    parent_entity: EntityId,
    sibling_ordinal: int,
) -> EntityId:
    if type(sibling_ordinal) is not int or sibling_ordinal < 0:
        raise ValueError("sibling ordinal must be non-negative")
    return EntityId(
        stable_hash(
            node.value,
            parent_domain.value,
            parent_entity.value,
            sibling_ordinal,
        )
    )


def derive_relate_entity(
    node: NodeId,
    ordered_parents: tuple[tuple[DomainId, EntityId], ...],
) -> EntityId:
    parents: CanonicalValue = tuple(
        (domain.value, entity.value) for domain, entity in ordered_parents
    )
    return EntityId(stable_hash(node.value, parents))


__all__ = [
    "BatchId",
    "CanonicalValue",
    "DomainId",
    "EntityId",
    "IdentityKey",
    "MicrobatchAttempt",
    "NodeId",
    "PortId",
    "RowId",
    "RowRef",
    "canonical_encode",
    "derive_expand_domain",
    "derive_expand_entity",
    "derive_input_port_id",
    "derive_input_salt",
    "derive_node_id",
    "derive_port_id",
    "derive_relate_domain",
    "derive_relate_entity",
    "derive_source_domain",
    "derive_source_key_entity",
    "derive_source_position_entity",
    "stable_hash",
]
