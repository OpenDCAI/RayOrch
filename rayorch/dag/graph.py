"""DAG graph data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from ..ray_module import RayModule


@dataclass(frozen=True)
class PipeRef:
    """Internal symbolic reference to one output slot of a DAG node."""

    node: str
    index: int = 0
    value_type: Any = field(default=Any, compare=False, repr=False)

    @staticmethod
    def _symbolic_error(operation: str) -> TypeError:
        return TypeError(
            f"cannot use a symbolic pipeline value with {operation}. "
            "DagPipeline.forward() is traced during compilation, so "
            "data-dependent Python control flow is not supported."
        )

    def __bool__(self) -> bool:
        raise self._symbolic_error("bool()/if")

    def __len__(self) -> int:
        raise self._symbolic_error("len()")

    def __iter__(self):
        raise self._symbolic_error("iteration")

    def __getitem__(self, key: Any) -> Any:
        raise self._symbolic_error("indexing")


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
    input_types: Tuple[Any, ...] = ()
    output_types: Tuple[Any, ...] = ()


@dataclass
class CompiledGraph:
    """Immutable DAG topology produced by ``Pipeline.compile()``."""

    nodes: Dict[str, NodeSpec]
    topo_order: Tuple[str, ...]
    deps: Dict[str, Tuple[str, ...]]
    consumers: Dict[str, Tuple[str, ...]]
    graph_outputs: Tuple[PipeRef, ...]
    input_keys: Tuple[str, ...]
    input_types: Dict[str, Any] = field(default_factory=dict)
    output_types: Tuple[Any, ...] = ()
