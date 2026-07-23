"""Per-output identity relations for the multigrain execution graph."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .refs import PortRef


class IncompleteGroupPolicy(str, Enum):
    """How Reduce treats an anchor with permanently missing descendants."""

    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class SameAs:
    """The output reuses one source's identity and grain."""

    source: PortRef


@dataclass(frozen=True)
class SubsetOf:
    """The output is a zero-or-one subset of one identity source."""

    source: PortRef


@dataclass(frozen=True)
class ChildrenOf:
    """The output creates children of one direct identity parent."""

    parent: PortRef


@dataclass(frozen=True)
class AggregateOf:
    """The output returns to an anchor after consuming descendant groups."""

    anchor: PortRef
    incomplete: IncompleteGroupPolicy = IncompleteGroupPolicy.FAIL_OPEN


@dataclass(frozen=True)
class RelatedFrom:
    """The output identity derives from inputs under ordered role names."""

    roles: tuple[str, ...]


OutputRelation: TypeAlias = (
    SameAs | SubsetOf | ChildrenOf | AggregateOf | RelatedFrom
)


__all__ = [
    "AggregateOf",
    "ChildrenOf",
    "IncompleteGroupPolicy",
    "OutputRelation",
    "RelatedFrom",
    "SameAs",
    "SubsetOf",
]
