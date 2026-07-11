"""Symbolic DAG tracing and IR for the experimental multi-grain API.

The classes here intentionally model a richer IR than the current local runtime
needs. The goal is to keep the prototype honest: graph validation, lineage,
rebatching, materialization, and future Ray lowering should all consume the same
passive data structure instead of growing separate ad hoc metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import inspect
from typing import Any, Dict, Iterable, List, Mapping, Sequence


class NodeKind(str, Enum):
    MAP = "MAP"
    EXPAND = "EXPAND"
    FILTER = "FILTER"
    REDUCE = "REDUCE"
    RELATE = "RELATE"
    PROJECT = "PROJECT"
    REBATCH = "REBATCH"
    MATERIALIZE = "MATERIALIZE"


class RelationKind(str, Enum):
    PRESERVE = "PRESERVE"
    EXPAND = "EXPAND"
    FILTER = "FILTER"
    REDUCE = "REDUCE"
    RELATE = "RELATE"
    GROUPED = "GROUPED"


class PayloadKind(str, Enum):
    OBJECT = "object"
    TABLE = "table"
    TENSOR = "tensor"


class OrdinalPolicy(str, Enum):
    PRESERVE = "preserve"
    CHILD_INDEX = "child_index"
    NONE = "none"


class MissingChildPolicy(str, Enum):
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"
    RETRY_FIRST = "retry_first"
    PARTIAL = "partial"


class MaterializePolicy(str, Enum):
    NEVER = "never"
    DEBUG_ONLY = "debug_only"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    CHECKPOINT = "checkpoint"


class MaterializeReason(str, Enum):
    TRACE = "trace"
    REPLAY_BOUNDARY = "replay_boundary"
    EXPENSIVE_OP = "expensive_op"
    NONDETERMINISTIC = "nondeterministic"
    SIDE_EFFECT = "side_effect"
    USER_REQUEST = "user_request"


@dataclass(frozen=True)
class IRPortRef:
    """Reference to a node output port."""

    node: str
    port: str = "out"
    index: int = 0

    @property
    def is_input(self) -> bool:
        return self.node.startswith("__input__")


@dataclass(frozen=True)
class DisplayKeySpec:
    """How to produce user-facing labels for trace views."""

    source: str | None = None
    template: str | None = None


@dataclass(frozen=True)
class OrdinalKeySpec:
    """How child ordering should be restored after physical reordering."""

    parent_grain: str | None = None
    key: str = "ordinal"


@dataclass(frozen=True)
class IRPortSpec:
    """Typed logical port spec.

    This is a fully passive port description: it carries the reference plus the
    logical metadata (grain/name/display) but never a live tracer/builder. It is
    the object that actually lives inside the IR, mirroring FX ``Node`` /
    ``jaxpr`` vars rather than the trace-time ``SymbolicPort`` token.
    """

    ref: IRPortRef
    grain: str
    name: str | None = None
    payload: PayloadKind = PayloadKind.OBJECT
    display_key: DisplayKeySpec | None = None
    ordinal_key: OrdinalKeySpec | None = None
    materialize: MaterializePolicy = MaterializePolicy.NEVER

    @property
    def node(self) -> str:
        return self.ref.node

    @property
    def index(self) -> int:
        return self.ref.index

    @property
    def port(self) -> str:
        return self.ref.port


@dataclass(frozen=True)
class RelationSpec:
    """How one output port relates to input ports."""

    output: IRPortRef
    relation: RelationKind
    parents: tuple[IRPortRef, ...]
    roles: tuple[str, ...] = ()
    parent_input: int | None = None
    anchor: IRPortRef | None = None
    ordinal: OrdinalPolicy = OrdinalPolicy.PRESERVE
    missing: MissingChildPolicy = MissingChildPolicy.FAIL_OPEN


@dataclass(frozen=True)
class CardinalityContract:
    """Per-node cardinality contract."""

    kind: NodeKind
    input_grains: tuple[str, ...]
    output_grains: tuple[str, ...]
    relations: tuple[RelationSpec, ...]


@dataclass(frozen=True)
class OperatorRecipe:
    """Passive recipe for reconstructing an operator."""

    cls_ref: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorProperties:
    deterministic: bool = True
    side_effect: bool = False
    idempotent: bool = True
    retryable: bool = True
    expensive: bool = False
    gpu_heavy: bool = False
    stateful: bool = False


@dataclass(frozen=True)
class PhysicalHints:
    replicas: int = 1
    num_gpus_per_replica: float = 0.0
    max_inflight: int = 1
    batch_size: int | None = None
    prefer_rebatch: bool = False
    engine: str | None = None


@dataclass(frozen=True)
class MaterializationSpec:
    port: IRPortRef
    policy: MaterializePolicy
    reason: MaterializeReason
    storage: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SymbolicPort:
    """A symbolic port produced while tracing ``Pipeline.forward``.

    ``SymbolicPort`` is a *trace-time only* token. It carries a live ``tracer``
    so that torch-like wrapper calls (``self.op(x)``) can register nodes as the
    ``forward`` runs. It is deliberately **not** stored inside the IR: the
    tracer builds passive ``IRPortSpec``/``IRPortRef`` objects instead, so the
    compiled ``MultigrainIR`` stays picklable and engine-agnostic. This mirrors
    PyTorch FX (``Proxy`` traces, ``Node`` is stored) and JAX (``Tracer``
    traces, ``Var`` is stored).
    """

    node: str
    index: int
    name: str
    grain: str
    tracer: "GraphTracer"
    port: str = "out"

    @property
    def ref(self) -> IRPortRef:
        return IRPortRef(self.node, self.port, self.index)

    @property
    def spec(self) -> IRPortSpec:
        return _port_spec_from_symbolic(self)


@dataclass(frozen=True)
class IRNode:
    """One logical multi-grain node in a compiled graph.

    All fields are passive: inputs/outputs are described by ``IRPortSpec`` (ref
    + grain + metadata), never by ``SymbolicPort``. ``input_refs``/``inputs``/
    ``outputs`` are convenience views for passes and display.
    """

    name: str
    kind: NodeKind
    input_specs: tuple[IRPortSpec, ...]
    output_specs: tuple[IRPortSpec, ...]
    contract: CardinalityContract
    op: OperatorRecipe
    properties: OperatorProperties = field(default_factory=OperatorProperties)
    physical: PhysicalHints = field(default_factory=PhysicalHints)
    parent_input: int | None = None
    grouped: bool = False

    @property
    def input_refs(self) -> tuple[IRPortRef, ...]:
        return tuple(spec.ref for spec in self.input_specs)

    @property
    def output_refs(self) -> tuple[IRPortRef, ...]:
        return tuple(spec.ref for spec in self.output_specs)

    @property
    def inputs(self) -> tuple[IRPortSpec, ...]:
        """Passive view of input ports (compat alias for ``input_specs``)."""
        return self.input_specs

    @property
    def outputs(self) -> tuple[IRPortSpec, ...]:
        """Passive view of output ports (compat alias for ``output_specs``)."""
        return self.output_specs


@dataclass(frozen=True)
class MultigrainIR:
    """Immutable experimental multi-grain DAG."""

    name: str
    inputs: tuple[IRPortSpec, ...]
    nodes: tuple[IRNode, ...]
    outputs: tuple[IRPortRef, ...]
    materialization: tuple[MaterializationSpec, ...] = ()

    @property
    def topo_order(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    @property
    def graph_outputs(self) -> tuple[IRPortRef, ...]:
        return self.outputs

    @property
    def input_ports(self) -> tuple[str, ...]:
        return tuple(port.name for port in self.inputs)

    @property
    def deps(self) -> dict[str, tuple[str, ...]]:
        deps: dict[str, tuple[str, ...]] = {}
        for node in self.nodes:
            seen: list[str] = []
            for ref in node.input_refs:
                if not ref.is_input and ref.node not in seen:
                    seen.append(ref.node)
            deps[node.name] = tuple(seen)
        return deps

    @property
    def consumers(self) -> dict[str, tuple[str, ...]]:
        consumers: dict[str, list[str]] = {node.name: [] for node in self.nodes}
        for node in self.nodes:
            for dep in self.deps[node.name]:
                consumers.setdefault(dep, []).append(node.name)
        return {name: tuple(values) for name, values in consumers.items()}

    def node(self, name: str) -> IRNode:
        for spec in self.nodes:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def with_materialization(
        self,
        port: SymbolicPort | IRPortSpec | IRPortRef,
        *,
        policy: MaterializePolicy,
        reason: MaterializeReason,
        storage: Mapping[str, Any] | None = None,
    ) -> "MultigrainIR":
        ref = port.ref if isinstance(port, (SymbolicPort, IRPortSpec)) else port
        spec = MaterializationSpec(
            port=ref,
            policy=policy,
            reason=reason,
            storage=storage,
        )
        return replace(self, materialization=(*self.materialization, spec))

    def describe(self) -> str:
        lines = [
            f"MultigrainIR({self.name})",
            f"  inputs: {list(self.input_ports)}",
        ]
        for node in self.nodes:
            inputs = ", ".join(
                f"{port.name}:{port.grain}" for port in node.inputs
            )
            outputs = ", ".join(
                f"{spec.ref.port}:{spec.grain}/{spec.ref.node}"
                for spec in node.output_specs
            )
            lines.append(
                f"  [{node.kind.value}] {node.name}\n"
                f"        in:  {inputs or '-'}\n"
                f"        out: {outputs or '-'}"
            )
        lines.append(
            "  outputs: "
            + ", ".join(f"{ref.node}.{ref.port}" for ref in self.graph_outputs)
        )
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for port in self.inputs:
            lines.append(f"    {port.node}([in: {port.name}:{port.grain}])")
        for node in self.nodes:
            lines.append(f'    {node.name}["{node.name}<br/>{node.kind.value}"]')
            for ref in node.input_refs:
                lines.append(f"    {ref.node} -->|{ref.port}| {node.name}")
        lines.append("    __sink__([out])")
        for ref in self.graph_outputs:
            lines.append(f"    {ref.node} --> __sink__")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(
            {
                "name": self.name,
                "inputs": [asdict(port) for port in self.inputs],
                "nodes": [_node_to_dict(node) for node in self.nodes],
                "topo_order": self.topo_order,
                "deps": self.deps,
                "consumers": self.consumers,
                "graph_outputs": [asdict(ref) for ref in self.graph_outputs],
                "materialization": [asdict(item) for item in self.materialization],
            }
        )


class GraphTracer:
    """Collect logical operator calls made with symbolic ports."""

    def __init__(self, *, name: str = "pipeline") -> None:
        self.name = name
        self.nodes: List[IRNode] = []
        self._name_counts: Dict[str, int] = {}

    def source(self, name: str) -> SymbolicPort:
        return SymbolicPort(
            node=f"__input__{name}",
            index=0,
            name=name,
            grain=name,
            tracer=self,
        )

    def unique_name(self, base: str) -> str:
        count = self._name_counts.get(base, 0)
        self._name_counts[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    def add_node(
        self,
        *,
        name: str,
        kind: str | NodeKind,
        inputs: Sequence[SymbolicPort],
        num_outputs: int,
        output_grain: str,
        parent_input: int | None = None,
        grouped: bool = False,
        op: OperatorRecipe | None = None,
        properties: OperatorProperties | None = None,
        physical: PhysicalHints | None = None,
        relation_roles: Sequence[str] = (),
    ) -> SymbolicPort | tuple[SymbolicPort, ...]:
        if not inputs:
            raise ValueError("a traced node requires at least one input")
        for port in inputs:
            if port.tracer is not self:
                raise ValueError("cannot mix symbolic ports from different traces")
        node_name = self.unique_name(name)
        node_kind = kind if isinstance(kind, NodeKind) else NodeKind(kind)
        outputs = tuple(
            SymbolicPort(
                node=node_name,
                index=index,
                name=node_name if num_outputs == 1 or index == 0 else f"{node_name}_{index}",
                grain=output_grain,
                tracer=self,
                port="out" if index == 0 else f"out_{index}",
            )
            for index in range(num_outputs)
        )
        input_specs = tuple(_port_spec_from_symbolic(port) for port in inputs)
        input_refs = tuple(port.ref for port in inputs)
        output_specs = tuple(_port_spec_from_symbolic(port) for port in outputs)
        relations = tuple(
            RelationSpec(
                output=port.ref,
                relation=_relation_kind_for(node_kind),
                parents=input_refs,
                roles=tuple(relation_roles),
                parent_input=parent_input,
                anchor=input_refs[0] if node_kind is NodeKind.REDUCE else None,
                ordinal=(
                    OrdinalPolicy.CHILD_INDEX
                    if node_kind is NodeKind.EXPAND
                    else OrdinalPolicy.PRESERVE
                ),
            )
            for port in outputs
        )
        contract = CardinalityContract(
            kind=node_kind,
            input_grains=tuple(port.grain for port in inputs),
            output_grains=tuple(port.grain for port in outputs),
            relations=relations,
        )
        self.nodes.append(
            IRNode(
                name=node_name,
                kind=node_kind,
                input_specs=input_specs,
                output_specs=output_specs,
                contract=contract,
                op=op or OperatorRecipe(cls_ref=name),
                properties=properties or OperatorProperties(),
                physical=physical or PhysicalHints(),
                parent_input=parent_input,
                grouped=grouped,
            )
        )
        return outputs[0] if num_outputs == 1 else outputs

    def build(
        self,
        inputs: Sequence[SymbolicPort],
        outputs: Any,
    ) -> MultigrainIR:
        return MultigrainIR(
            name=self.name,
            inputs=tuple(_port_spec_from_symbolic(port) for port in inputs),
            nodes=tuple(self.nodes),
            outputs=tuple(
                port.ref for port in _flatten_symbolic_outputs(outputs)
            ),
        )


class Pipeline:
    """Torch-like experimental pipeline with symbolic compile support."""

    def __init__(self) -> None:
        self._compiled: CompiledGraph | None = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def compile(self) -> CompiledGraph:
        tracer = GraphTracer(name=type(self).__name__)
        sig = inspect.signature(self.forward)
        inputs: List[SymbolicPort] = []
        args: List[SymbolicPort] = []
        kwargs: Dict[str, SymbolicPort] = {}
        for param in sig.parameters.values():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    "forward() cannot use *args/**kwargs in the multigrain MVP"
                )
            port = tracer.source(param.name)
            inputs.append(port)
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                args.append(port)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[param.name] = port

        outputs = self.forward(*args, **kwargs)
        self._compiled = tracer.build(inputs, outputs)
        return self._compiled


def _flatten_symbolic_outputs(value: Any) -> List[SymbolicPort]:
    if isinstance(value, SymbolicPort):
        return [value]
    if isinstance(value, tuple):
        ports: List[SymbolicPort] = []
        for item in value:
            ports.extend(_flatten_symbolic_outputs(item))
        return ports
    raise TypeError(
        "Pipeline.forward() must return SymbolicPort or tuple[SymbolicPort, ...]"
    )


def ensure_symbolic_ports(values: Sequence[Any]) -> tuple[SymbolicPort, ...] | None:
    """Return symbolic ports if this call is in trace mode."""
    if not values:
        return None
    if isinstance(values[0], SymbolicPort):
        if not all(isinstance(value, SymbolicPort) for value in values):
            raise TypeError("cannot mix symbolic and eager ports")
        return tuple(values)
    return None


def _port_spec_from_symbolic(port: SymbolicPort) -> IRPortSpec:
    return IRPortSpec(
        ref=port.ref,
        grain=port.grain,
        name=port.name,
        display_key=DisplayKeySpec(source=port.name),
        ordinal_key=OrdinalKeySpec(parent_grain=port.grain),
    )


def _relation_kind_for(kind: NodeKind) -> RelationKind:
    table = {
        NodeKind.MAP: RelationKind.PRESERVE,
        NodeKind.EXPAND: RelationKind.EXPAND,
        NodeKind.FILTER: RelationKind.FILTER,
        NodeKind.REDUCE: RelationKind.REDUCE,
        NodeKind.RELATE: RelationKind.RELATE,
        NodeKind.PROJECT: RelationKind.PRESERVE,
        NodeKind.REBATCH: RelationKind.PRESERVE,
        NodeKind.MATERIALIZE: RelationKind.PRESERVE,
    }
    return table[kind]


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def _node_to_dict(node: IRNode) -> dict[str, Any]:
    return {
        "name": node.name,
        "kind": node.kind,
        "input_specs": [asdict(spec) for spec in node.input_specs],
        "output_specs": [asdict(spec) for spec in node.output_specs],
        "input_refs": [asdict(ref) for ref in node.input_refs],
        "contract": asdict(node.contract),
        "op": asdict(node.op),
        "properties": asdict(node.properties),
        "physical": asdict(node.physical),
        "parent_input": node.parent_input,
        "grouped": node.grouped,
    }


__all__ = [
    "CardinalityContract",
    "CompiledGraph",
    "DisplayKeySpec",
    "GraphTracer",
    "IRNode",
    "IRPortRef",
    "IRPortSpec",
    "MaterializationSpec",
    "MaterializePolicy",
    "MaterializeReason",
    "MissingChildPolicy",
    "MultigrainIR",
    "NodeSpec",
    "NodeKind",
    "OperatorProperties",
    "OperatorRecipe",
    "OrdinalKeySpec",
    "OrdinalPolicy",
    "PayloadKind",
    "PhysicalHints",
    "Pipeline",
    "RelationKind",
    "RelationSpec",
    "SymbolicPort",
    "ensure_symbolic_ports",
]


CompiledGraph = MultigrainIR
NodeSpec = IRNode

