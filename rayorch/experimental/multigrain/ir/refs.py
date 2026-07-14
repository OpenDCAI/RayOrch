"""Stable references used by the passive execution graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class GraphInputRef:
    """Reference to one named graph input."""

    name: str


@dataclass(frozen=True)
class NodeOutputRef:
    """Reference to one named output of a graph node."""

    node: str
    output: str = "out"


PortRef: TypeAlias = GraphInputRef | NodeOutputRef


def ref_label(ref: PortRef) -> str:
    if isinstance(ref, GraphInputRef):
        return f"input:{ref.name}"
    return f"{ref.node}.{ref.output}"


__all__ = ["GraphInputRef", "NodeOutputRef", "PortRef", "ref_label"]
