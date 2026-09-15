"""Pure symbolic relationships between RayOrch pipeline ports."""

from __future__ import annotations

from ._builder import active_builder
from .api import OptionalInput, Port
from .errors import CompileError


def expand(port: Port) -> Port:
    """Expand one grouped Port into a newly created child Domain."""

    return active_builder().expand((port,))[0]


def expand_aligned(*ports: Port) -> tuple[Port, ...]:
    """Expand aligned grouped Ports into one child Domain."""

    return active_builder().expand(tuple(ports))


def reduce(port: Port, *, members: Port | None = None) -> Port:
    """Reduce one Domain level while preserving ordered membership."""

    return active_builder().reduce((port,), members)[0]


def reduce_aligned(
    *ports: Port,
    members: Port | None = None,
) -> tuple[Port, ...]:
    """Reduce multiple value Ports over the same member set."""

    return active_builder().reduce(ports, members)


def broadcast(port: Port, *, like: Port) -> Port:
    """Project an ancestor Port into the descendant Domain containing ``like``."""

    return active_builder().broadcast(port, like)


def filter(port: Port, mask: Port) -> Port:
    """Filter membership with a boolean mask without changing the source Domain."""

    return active_builder().filter(port, mask)


def optional(port: Port) -> OptionalInput:
    """Mark a Port as an optional Call input without creating a graph node."""

    if not isinstance(port, Port):
        raise CompileError("optional requires a Port")
    return OptionalInput(port)


__all__ = [
    "broadcast",
    "expand",
    "expand_aligned",
    "filter",
    "optional",
    "reduce",
    "reduce_aligned",
]
