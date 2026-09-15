"""Stable public authoring API for RayOrch dataflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Self, overload

from ._model import PortRef

if TYPE_CHECKING:
    from ._program.plan import CompiledProgram


@dataclass(frozen=True, slots=True)
class Port:
    """Public symbolic handle for one logical Port."""

    ref: PortRef
    _owner: int


@dataclass(frozen=True, slots=True)
class OptionalInput:
    """Port wrapper that changes input policy without creating another Port."""

    port: Port


class RayModule:
    """Declarative UDF recipe; actor handles exist only in the execution layer."""

    def __init__(self, udf: Any, *, num_outputs: int = 1) -> None:
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive")
        self.udf = udf
        self.num_outputs = int(num_outputs)
        self.init_args: tuple[Any, ...] = ()
        self.init_kwargs: dict[str, Any] = {}
        self.options: dict[str, Any] = {}

    def returns(self, count: int) -> Self:
        """Declare the number of logical output Ports produced by one Call."""

        if count <= 0:
            raise ValueError("output count must be positive")
        self.num_outputs = int(count)
        return self

    def pre_init(self, *args: Any, **kwargs: Any) -> Self:
        """Record UDF constructor arguments without instantiating it at compile time."""

        self.init_args = tuple(args)
        self.init_kwargs = dict(kwargs)
        return self

    def ray_options(self, **options: Any) -> Self:
        """Record physical execution options and return this recipe."""

        if "num_outputs" in options:
            raise ValueError("use RayModule.returns(...) for logical outputs")
        self.options.update(options)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Port | tuple[Port, ...]:
        from ._builder import active_builder

        return active_builder("RayModule calls").call(self, args, kwargs)


@overload
def function(fn: Callable[..., Any], *, num_outputs: int = 1) -> RayModule: ...


@overload
def function(
    fn: None = None,
    *,
    num_outputs: int = 1,
) -> Callable[[Callable[..., Any]], RayModule]: ...


def function(
    fn: Callable[..., Any] | None = None,
    *,
    num_outputs: int = 1,
) -> RayModule | Callable[[Callable[..., Any]], RayModule]:
    """Adapt a callable into a stateless RayModule recipe."""

    def wrap(target: Callable[..., Any]) -> RayModule:
        return RayModule(target, num_outputs=num_outputs)

    return wrap if fn is None else wrap(fn)


class Pipeline:
    """Base class for declarative dataflows compiled by one symbolic trace."""

    def forward(self, *args: Any) -> Any:
        """Declare a graph by composing RayModules and structural Port operations."""

        raise NotImplementedError

    def compile(self, *, optimize: bool = True) -> CompiledProgram:
        """Trace ``forward`` and compile it into an immutable RuntimePlan."""

        from ._builder import compile_pipeline

        return compile_pipeline(self, optimize=optimize)


__all__ = ["OptionalInput", "Pipeline", "Port", "RayModule", "function"]
