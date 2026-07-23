"""Small public API surface for the V2.5 prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .grain import PortId


class CompileError(ValueError):
    """The user graph cannot be represented by the current V2.5 scope."""


class ExecutionError(RuntimeError):
    """A microbatch arena failed at run-control level."""


class BadRecordError(Exception):
    """Explicitly attribute a UDF error to one dispatch grain."""

    def __init__(self, message: str, *, index: int) -> None:
        super().__init__(message)
        if index < 0:
            raise ValueError("bad record index must be non-negative")
        self.index = index


@dataclass(frozen=True, slots=True)
class Port:
    id: PortId


@dataclass(frozen=True, slots=True)
class KeyedPort:
    port: Port
    by: Any


def keyed(port: Port | PortId, *, by: Any) -> KeyedPort:
    public_port = port if isinstance(port, Port) else Port(port)
    return KeyedPort(public_port, by)


class Pipeline:
    """User authoring base class; graph tracing is introduced after Phase 1."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


PrimitiveT = TypeVar("PrimitiveT", bound="_ConfiguredPrimitive")


class _ConfiguredPrimitive(Generic[PrimitiveT]):
    """RayModule-style constructor and execution option capture."""

    def __init__(self, udf: Any) -> None:
        self.udf = udf
        self.init_args: tuple[Any, ...] = ()
        self.init_kwargs: dict[str, Any] = {}
        self.options: dict[str, Any] = {}

    def pre_init(
        self: PrimitiveT,
        *args: Any,
        **kwargs: Any,
    ) -> PrimitiveT:
        self.init_args = tuple(args)
        self.init_kwargs = dict(kwargs)
        return self

    def ray_options(
        self: PrimitiveT,
        **options: Any,
    ) -> PrimitiveT:
        self.options.update(options)
        return self


class Map(_ConfiguredPrimitive["Map"]):
    pass


class Filter(_ConfiguredPrimitive["Filter"]):
    pass


class Expand(_ConfiguredPrimitive["Expand"]):
    pass


class Reduce(_ConfiguredPrimitive["Reduce"]):
    pass


class Relate(_ConfiguredPrimitive["Relate"]):
    pass
