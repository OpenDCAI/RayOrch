"""Phase boundaries for the future single-process and Ray executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .api import ExecutionError
from .grain import GrainFailure, PortId
from .graph import CompiledGraph


class SourcePositionAllocator:
    """Run-scoped per-source-port logical ordinal allocator."""

    def __init__(self) -> None:
        self._next: dict[PortId, int] = {}

    def allocate(self, source_port: PortId) -> int:
        position = self._next.get(source_port, 0)
        self._next[source_port] = position + 1
        return position

    def peek(self, source_port: PortId) -> int:
        return self._next.get(source_port, 0)


@dataclass(frozen=True, slots=True)
class RunResult:
    outputs: tuple[Any, ...] = ()
    failures: tuple[GrainFailure, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)


class Executor:
    """Public placeholder; execution semantics start in Phase 2."""

    def __init__(self, graph: CompiledGraph) -> None:
        self.graph = graph

    def run(self, *sources: Any) -> RunResult:
        del sources
        raise ExecutionError(
            "Multigrain V2.5 execution is not implemented before Phase 2"
        )
