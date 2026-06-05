"""DAG graph data structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..ray_module import RayModule


@dataclass(frozen=True)
class PipeRef:
    """Symbolic reference to one output slot of a DAG node."""

    node: str
    index: int = 0


@dataclass(frozen=True)
class NodeSpec:
    """Immutable definition of a single compute node in the DAG."""

    name: str
    module: RayModule
    args: Tuple[PipeRef, ...]
    kw_args: Dict[str, PipeRef]
    max_inflight: int = 1
    num_outputs: int = 1
    input_names: Tuple[str, ...] = ()
    output_names: Tuple[str, ...] = ()


@dataclass
class CompiledGraph:
    """Immutable DAG topology produced by ``Pipeline.compile()``."""

    nodes: Dict[str, NodeSpec]
    topo_order: Tuple[str, ...]
    deps: Dict[str, Tuple[str, ...]]
    consumers: Dict[str, Tuple[str, ...]]
    graph_outputs: Tuple[PipeRef, ...]
    input_keys: Tuple[str, ...]
