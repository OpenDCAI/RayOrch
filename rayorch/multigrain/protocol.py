"""Stable Ray-free DTOs shared by the runtime, executor, and Worker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .model import GrainRef, PortRef


@dataclass(frozen=True, slots=True)
class BlockRef:
    """Opaque handle to an immutable coarse block; RowBinding selects a row."""
    handle: Any


@dataclass(frozen=True, slots=True)
class RowBinding:
    """Physical location of one row without carrying its business payload."""

    block: BlockRef
    row: int

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("row must be non-negative")


@dataclass(frozen=True, slots=True)
class ExpandedRows:
    """Ordered rows and optional per-row control values for an expanded Port."""

    port: PortRef
    rows: tuple[RowBinding, ...]
    controls: tuple[bool, ...] | None = None


@dataclass(frozen=True, slots=True)
class PortOutputReport:
    """One output Port's physical binding report inside a Grain report."""

    port: PortRef
    scalar: RowBinding | None = None
    expansions: tuple[ExpandedRows, ...] = ()
    control: bool | None = None


@dataclass(frozen=True, slots=True)
class GrainReport:
    """Atomic Worker completion report for one Grain."""

    grain: GrainRef
    generation: int
    outputs: tuple[PortOutputReport, ...]


@dataclass(frozen=True, slots=True)
class NestedGroupInput:
    """Read rows and reconstruct one ordered nested value group."""

    bindings: tuple[RowBinding, ...]
    offsets_by_level: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MissingInput:
    """Optional+DROPPED input reconstructed as the singleton MISSING value."""


GrainInput = RowBinding | NestedGroupInput | MissingInput


@dataclass(frozen=True, slots=True)
class GrainInvocation:
    """One Grain generation and its resolved physical UDF inputs."""

    grain: GrainRef
    generation: int
    inputs: tuple[GrainInput, ...]


@dataclass(frozen=True, slots=True)
class RecordFailure:
    """Value sentinel marking one Grain as a business failure."""

    cause: Any


@dataclass(frozen=True, slots=True)
class GroupFailure:
    """Fail one Grain and suppress siblings with the same Call and direct parent."""

    cause: Any


@dataclass(frozen=True, slots=True)
class GrainFailureReport:
    """Generation-fenced Worker failure report for one Grain."""

    grain: GrainRef
    generation: int
    cause: Any
    suppress_siblings: bool = False


WorkerReport = GrainReport | GrainFailureReport


class DispatchFailureKind(Enum):
    """A whole Worker dispatch failed before per-Grain reports existed."""

    UDF_ERROR = auto()
    CONTRACT_ERROR = auto()


@dataclass(frozen=True, slots=True)
class DispatchFailure:
    """Serializable exception snapshot for one whole-dispatch failure."""

    kind: DispatchFailureKind
    error_type: str
    message: str
    traceback: str


WorkerDispatchResult = tuple[WorkerReport, ...] | DispatchFailure


@dataclass(frozen=True, slots=True)
class CallInputLayout:
    """Stable positional and keyword Worker invocation layout for one Call."""

    positional_count: int
    keyword_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.positional_count < 0:
            raise ValueError("positional_count must be non-negative")
        if any(not name for name in self.keyword_names):
            raise ValueError("keyword input names must be non-empty")
        if len(set(self.keyword_names)) != len(self.keyword_names):
            raise ValueError("keyword input names must be unique")

    @property
    def input_count(self) -> int:
        """Return the number of logical input slots covered by this layout."""

        return self.positional_count + len(self.keyword_names)


@dataclass(frozen=True, slots=True)
class CallOutputLayout:
    """Compiler-generated output layout and control-manifest requirements."""

    port: PortRef
    expanded_ports: tuple[PortRef, ...] = ()
    control_ports: frozenset[PortRef] = frozenset()


def restore_nested_group(
    leaves: list[Any],
    offsets_by_level: tuple[tuple[int, ...], ...],
) -> list[Any]:
    """Apply CSR offsets from leaves upward to restore one rooted nested group."""

    nodes: list[Any] = leaves
    for offsets in reversed(offsets_by_level):
        nodes = [
            nodes[offsets[index] : offsets[index + 1]]
            for index in range(len(offsets) - 1)
        ]
    if len(nodes) != 1:
        raise ValueError("NestedGroupLayout does not have one root")
    return nodes[0]


__all__ = [
    "BlockRef",
    "GrainFailureReport",
    "GrainReport",
    "DispatchFailure",
    "DispatchFailureKind",
    "ExpandedRows",
    "GroupFailure",
    "NestedGroupInput",
    "CallInputLayout",
    "GrainInput",
    "GrainInvocation",
    "MissingInput",
    "CallOutputLayout",
    "PortOutputReport",
    "RecordFailure",
    "RowBinding",
    "WorkerReport",
    "WorkerDispatchResult",
    "restore_nested_group",
]
