"""Shared helpers for experimental multigrain operator wrappers."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Sequence

from ..data.batch import PortBatch, _normalize_output_lists
from ..ir.model import SymbolicPort


def op_name(op_cls: Any, name: str | None) -> str:
    if name is not None:
        return name
    return getattr(op_cls, "__name__", type(op_cls).__name__)


def op_ref(op_cls: Any) -> str:
    cls = op_cls if isinstance(op_cls, type) else type(op_cls)
    return f"{cls.__module__}.{cls.__qualname__}"


def require_compilable_factory(op_cls: Any, primitive: str) -> None:
    """Reject live instances when a wrapper enters symbolic compilation.

    Eager wrappers may intentionally use an already-created object.  Passive IR
    cannot: it records a class recipe, not arbitrary live instance state.
    """
    if not isinstance(op_cls, type):
        raise TypeError(
            f"{primitive} compiled graphs require an importable operator class; "
            "live operator instances are eager-only"
        )
    if op_cls.__module__ == "__main__" or "<locals>" in op_cls.__qualname__:
        raise TypeError(
            f"{primitive} compiled graphs require an importable operator class; "
            "classes defined in __main__ or a local scope are not importable"
        )
    target: Any = import_module(op_cls.__module__)
    for part in op_cls.__qualname__.split("."):
        target = getattr(target, part, None)
    if target is not op_cls:
        raise TypeError(
            f"{primitive} operator class "
            f"{op_cls.__module__}.{op_cls.__qualname__} is not importable"
        )


def validate_output_count(value: Any, *, field: str = "num_outputs") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def checked_output_lists(
    raw: Any,
    *,
    expected: int,
    primitive: str,
    name: str,
) -> tuple[list[Any], ...]:
    outputs = _normalize_output_lists(raw)
    if len(outputs) != expected:
        raise ValueError(
            f"{primitive} '{name}' expected {expected} outputs, got {len(outputs)}"
        )
    return outputs


class LazyOp:
    """Deferred operator factory (RayModule-style).

    The wrapper holds the operator *class + init args* and only instantiates the
    operator on first use, not when the graph is built. This mirrors
    ``RayModule``: ``op.__init__`` (which may load a heavy model) runs at execution
    time -- once per replica/executor -- so compiling / tracing / serializing the
    graph never acquires a GPU or loads a model. An already-constructed instance
    is used as-is.
    """

    __slots__ = ("_op_cls", "_args", "_kwargs", "_cached")

    def __init__(self, op_cls: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._op_cls = op_cls
        self._args = args
        self._kwargs = kwargs
        self._cached = None if isinstance(op_cls, type) else op_cls

    def get(self) -> Any:
        if self._cached is None:
            self._cached = self._op_cls(*self._args, **self._kwargs)
        return self._cached


def output_name(base: str, index: int, count: int) -> str:
    return base if count == 1 or index == 0 else f"{base}_{index}"


def as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def check_symbolic_same_grain(name: str, ports: Sequence[SymbolicPort]) -> str:
    grain = ports[0].grain
    if any(port.grain != grain for port in ports[1:]):
        raise ValueError(
            f"{name} requires same-grain inputs, got {[port.grain for port in ports]}"
        )
    return grain


def normalize_mask(mask: Any, count: int) -> list[bool]:
    if not isinstance(mask, list):
        raise TypeError("Filter mask must be a list of bools")
    if len(mask) != count:
        raise ValueError(f"Filter mask length {len(mask)} does not match input {count}")
    if any(not isinstance(value, bool) for value in mask):
        raise TypeError("Filter mask must contain bool values")
    return mask


def lineage_union(
    paths: Sequence[tuple[str, ...]],
    op_name: str | None = None,
) -> tuple[str, ...]:
    lineage: list[str] = []
    for path in paths:
        for step in path:
            if step not in lineage:
                lineage.append(step)
    if op_name is not None:
        lineage.append(op_name)
    return tuple(lineage)


def take_with_lineage(
    batch: PortBatch,
    indices: Sequence[int],
    *,
    name: str,
    op_name: str,
) -> PortBatch:
    kept = batch.take(indices, name=name)
    kept.lineage = [tuple((*path, op_name)) for path in kept.lineage]
    return kept
