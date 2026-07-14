"""Typed operation calls stored in the passive execution graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias


@dataclass(frozen=True)
class OperatorFactorySpec:
    """Importable class plus constructor arguments, instantiated on a worker."""

    import_path: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KeyJoinSpec:
    """Declarative per-role field names for an equi-join."""

    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RelationAdapterSpec:
    """Importable adapter producing invocation-local relation evidence."""

    import_path: str


RelationMatcherSpec: TypeAlias = (
    KeyJoinSpec | RelationAdapterSpec
)


@dataclass(frozen=True)
class MapOp:
    factory: OperatorFactorySpec


@dataclass(frozen=True)
class FilterOp:
    factory: OperatorFactorySpec


@dataclass(frozen=True)
class ExpandOp:
    factory: OperatorFactorySpec


@dataclass(frozen=True)
class ReduceOp:
    factory: OperatorFactorySpec


@dataclass(frozen=True)
class RelateOp:
    factory: OperatorFactorySpec
    matcher: RelationMatcherSpec


@dataclass(frozen=True)
class FilterByMaskOp:
    """Internal Select lowering: filter all inputs by one mask input."""

    mask_input: int


OperationSpec: TypeAlias = (
    MapOp
    | FilterOp
    | ExpandOp
    | ReduceOp
    | RelateOp
    | FilterByMaskOp
)


def operation_name(operation: OperationSpec) -> str:
    return type(operation).__name__.removesuffix("Op").upper()


def operator_factory(operation: OperationSpec) -> OperatorFactorySpec | None:
    return getattr(operation, "factory", None)


__all__ = [
    "ExpandOp",
    "FilterByMaskOp",
    "FilterOp",
    "KeyJoinSpec",
    "MapOp",
    "OperationSpec",
    "OperatorFactorySpec",
    "ReduceOp",
    "RelateOp",
    "RelationAdapterSpec",
    "RelationMatcherSpec",
    "operation_name",
    "operator_factory",
]
