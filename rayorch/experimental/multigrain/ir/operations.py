"""Typed operation calls stored in the passive execution graph."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Callable, Mapping, TypeAlias


@dataclass(frozen=True)
class OperatorFactorySpec:
    """Importable class plus constructor arguments, instantiated on a worker."""

    import_path: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KeyJoinSpec:
    """Equi-join field names aligned with the relation's ordered roles."""

    fields: tuple[str, ...]


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
class ByAncestor:
    """Group a descendant by its unique anchor-domain ancestry."""


@dataclass(frozen=True)
class ByRole:
    """Group a relation-aware descendant by one direct parent role."""

    role: str


GroupSelector: TypeAlias = ByAncestor | ByRole


@dataclass(frozen=True)
class ReduceOp:
    factory: OperatorFactorySpec
    selectors: tuple[GroupSelector, ...] = ()


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


def resolve_operator_factory(spec: OperatorFactorySpec) -> type[Any]:
    """Resolve and validate the importable operator class in a passive spec."""
    parts = spec.import_path.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError("operator factory path must be 'package.module.Object'")
    target: Any | None = None
    attributes: list[str] = []
    for boundary in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:boundary])
        try:
            target = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise
            continue
        attributes = parts[boundary:]
        break
    if target is None:
        raise ImportError(f"cannot import operator factory {spec.import_path!r}")
    for part in attributes:
        target = getattr(target, part)
    if not isinstance(target, type):
        raise TypeError(f"operator factory target {spec.import_path!r} is not a class")
    return target


def resolve_relation_adapter(spec: RelationAdapterSpec) -> Callable[..., Any]:
    """Resolve and validate a ``package.module:callable`` adapter reference."""
    module_name, separator, attribute = spec.import_path.partition(":")
    if (
        separator != ":"
        or not module_name
        or not attribute
        or ":" in attribute
    ):
        raise ValueError("relation adapter path must be 'package.module:callable'")
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        if not part:
            raise ValueError(
                "relation adapter path must be 'package.module:callable'"
            )
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(
            f"relation adapter target {spec.import_path!r} is not callable"
        )
    return target


__all__ = [
    "ByAncestor",
    "ByRole",
    "ExpandOp",
    "GroupSelector",
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
    "resolve_operator_factory",
    "resolve_relation_adapter",
]
