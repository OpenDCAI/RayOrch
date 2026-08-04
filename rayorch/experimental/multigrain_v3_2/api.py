"""Multigrain V3.2 的 RayModule + Port-functional authoring API。"""

from __future__ import annotations

import contextvars
import inspect
from dataclasses import dataclass
from typing import Any, Self

from rayorch.experimental.multigrain_v3.api import CompiledPipeline

from .ir import (
    CallInput,
    CallNode,
    ConsumerEdge,
    DirectValue,
    DomainId,
    DomainSpec,
    ExpandNode,
    ExpansionId,
    ExpansionSpec,
    GroupValue,
    LogicalCompileError,
    LogicalDAG,
    LogicalPortId,
    ModuleSpec,
    PortSpec,
    ReduceNode,
    SourceNode,
    freeze_mapping,
)


@dataclass(frozen=True, slots=True)
class Port:
    """一个逻辑 Port 的公开符号句柄。"""

    id: LogicalPortId


@dataclass(frozen=True, slots=True)
class OptionalPort:
    port: Port


def optional(port: Port) -> OptionalPort:
    if not isinstance(port, Port):
        raise LogicalCompileError("optional(...) requires a Port")
    return OptionalPort(port)


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    """新的逻辑 DAG，以及经过验证、可执行的 V3 物理计划。"""

    logical: LogicalDAG
    physical: CompiledPipeline


class Pipeline:
    """通过 forward 方法 trace Port-first 逻辑 DAG 的基类。"""

    def forward(self, *args: Any) -> Any:
        raise NotImplementedError

    def compile(self) -> CompiledProgram:
        parameters = tuple(inspect.signature(self.forward).parameters.values())
        if any(
            parameter.kind
            not in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            for parameter in parameters
        ):
            raise LogicalCompileError(
                "Pipeline.forward supports positional sources only"
            )
        builder = _GraphBuilder(tuple(parameter.name for parameter in parameters))
        token = _ACTIVE_TRACE.set(builder)
        try:
            result = self.forward(*(Port(port) for port in builder.source_ports))
        finally:
            _ACTIVE_TRACE.reset(token)
        outputs = builder.normalize_outputs(result)
        logical = builder.build(tuple(port.id for port in outputs))
        from .lowering import lower_to_v3

        return CompiledProgram(logical, lower_to_v3(logical))


class RayModule:
    """声明式 UDF/actor 配方；actor handle 归 Executor 所有。"""

    def __init__(self, udf: Any) -> None:
        self.udf = udf
        self.init_args: tuple[Any, ...] = ()
        self.init_kwargs: dict[str, Any] = {}
        self.options: dict[str, Any] = {}

    def pre_init(self, *args: Any, **kwargs: Any) -> Self:
        self.init_args = tuple(args)
        self.init_kwargs = dict(kwargs)
        return self

    def ray_options(self, **options: Any) -> Self:
        self.options.update(options)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        builder = _ACTIVE_TRACE.get()
        if builder is None:
            raise LogicalCompileError(
                "RayModule calls are only valid in Pipeline.forward"
            )
        return builder.call(self, tuple(args), dict(kwargs))


class _GraphBuilder:
    """生成不可变 LogicalDAG 的可变 trace builder。"""

    def __init__(self, parameters: tuple[str, ...]) -> None:
        self.nodes: list[Any] = []
        self.ports: dict[LogicalPortId, PortSpec] = {}
        self.domains: dict[DomainId, DomainSpec] = {
            DomainId(0): DomainSpec(DomainId(0))
        }
        self.expansions: dict[ExpansionId, ExpansionSpec] = {}
        self.consumers: dict[LogicalPortId, list[ConsumerEdge]] = {}
        self.next_domain = 1
        self.next_expansion = 0
        sources = []
        for parameter in parameters:
            node_id = len(self.nodes)
            port = LogicalPortId(node_id, 0)
            self.nodes.append(SourceNode(node_id, port, parameter))
            self.ports[port] = PortSpec(port, DomainId(0), DirectValue())
            sources.append(port)
        self.source_ports = tuple(sources)

    def call(
        self,
        module: RayModule,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Port | tuple[Port, ...]:
        if not args:
            raise LogicalCompileError("RayModule requires at least one input")
        inputs: list[CallInput] = []
        for index, value in enumerate(args):
            port, is_optional = self._port_value(value)
            inputs.append(CallInput(f"arg_{index}", port.id, is_optional))
        for name, value in kwargs.items():
            port, is_optional = self._port_value(value)
            inputs.append(CallInput(name, port.id, is_optional))

        domains = {self.ports[item.port].domain for item in inputs}
        if len(domains) != 1:
            raise LogicalCompileError(
                "RayModule inputs belong to different entity domains; use an "
                "explicit expand/reduce/broadcast relation"
            )
        domain = next(iter(domains))
        options = dict(module.options)
        output_count = int(options.get("num_outputs", 1))
        if output_count <= 0:
            raise LogicalCompileError("num_outputs must be positive")
        node_id = len(self.nodes)
        outputs = tuple(
            LogicalPortId(node_id, output) for output in range(output_count)
        )
        spec = ModuleSpec(
            module.udf,
            module.init_args,
            tuple(module.init_kwargs.items()),
            tuple(module.options.items()),
        )
        self.nodes.append(CallNode(node_id, spec, tuple(inputs), outputs))
        for output in outputs:
            self.ports[output] = PortSpec(output, domain, DirectValue())
        for index, item in enumerate(inputs):
            self.consumers.setdefault(item.port, []).append(
                ConsumerEdge(node_id, index)
            )
        public = tuple(Port(port) for port in outputs)
        return public[0] if len(public) == 1 else public

    def expand(self, values: tuple[Port, ...]) -> tuple[Port, ...]:
        ports = self._ports(values, "expand")
        domains = {self.ports[port.id].domain for port in ports}
        if len(domains) != 1:
            raise LogicalCompileError("aligned expand inputs need one parent domain")
        parent_domain = next(iter(domains))
        expansion = ExpansionId(self.next_expansion)
        self.next_expansion += 1
        child_domain = DomainId(self.next_domain)
        self.next_domain += 1
        self.domains[child_domain] = DomainSpec(
            child_domain,
            parent_domain,
            expansion,
        )
        node_id = len(self.nodes)
        outputs = tuple(
            LogicalPortId(node_id, output) for output in range(len(ports))
        )
        input_ids = tuple(port.id for port in ports)
        self.expansions[expansion] = ExpansionSpec(
            expansion,
            parent_domain,
            child_domain,
            input_ids,
        )
        self.nodes.append(ExpandNode(node_id, expansion, input_ids, outputs))
        for output in outputs:
            self.ports[output] = PortSpec(output, child_domain, DirectValue())
        for index, port in enumerate(ports):
            self.consumers.setdefault(port.id, []).append(
                ConsumerEdge(node_id, index)
            )
        return tuple(Port(port) for port in outputs)

    def reduce(self, values: tuple[Port, ...]) -> tuple[Port, ...]:
        ports = self._ports(values, "reduce")
        domains = {self.ports[port.id].domain for port in ports}
        if len(domains) != 1:
            raise LogicalCompileError("aligned reduce inputs need one child domain")
        child_domain = next(iter(domains))
        domain = self.domains[child_domain]
        if domain.parent is None or domain.via_expansion is None:
            raise LogicalCompileError("reduce input has no parent expansion")
        node_id = len(self.nodes)
        outputs = tuple(
            LogicalPortId(node_id, output) for output in range(len(ports))
        )
        input_ids = tuple(port.id for port in ports)
        self.nodes.append(
            ReduceNode(node_id, domain.via_expansion, input_ids, outputs)
        )
        for output, leaf in zip(outputs, input_ids):
            self.ports[output] = PortSpec(
                output,
                domain.parent,
                GroupValue(leaf, (domain.via_expansion,)),
            )
        for index, port in enumerate(ports):
            self.consumers.setdefault(port.id, []).append(
                ConsumerEdge(node_id, index)
            )
        return tuple(Port(port) for port in outputs)

    def normalize_outputs(self, value: Any) -> tuple[Port, ...]:
        values = (value,) if isinstance(value, Port) else value
        if not (
            isinstance(values, tuple)
            and values
            and all(isinstance(port, Port) for port in values)
        ):
            raise LogicalCompileError(
                "Pipeline.forward must return Port or non-empty tuple[Port]"
            )
        return values

    def build(self, outputs: tuple[LogicalPortId, ...]) -> LogicalDAG:
        for output in outputs:
            if output not in self.ports:
                raise LogicalCompileError(f"unknown graph output: {output}")
        return LogicalDAG(
            nodes=tuple(self.nodes),
            ports=freeze_mapping(self.ports),
            domains=freeze_mapping(self.domains),
            expansions=freeze_mapping(self.expansions),
            consumers_by_port=freeze_mapping(
                {port: tuple(edges) for port, edges in self.consumers.items()}
            ),
            source_ports=self.source_ports,
            output_ports=outputs,
        )

    def _ports(self, values: tuple[Port, ...], label: str) -> tuple[Port, ...]:
        if not values or any(not isinstance(port, Port) for port in values):
            raise LogicalCompileError(f"{label} requires one or more Ports")
        return values

    @staticmethod
    def _port_value(value: Any) -> tuple[Port, bool]:
        if isinstance(value, OptionalPort):
            return value.port, True
        if not isinstance(value, Port):
            raise LogicalCompileError(f"expected Port, got {type(value)!r}")
        return value, False


_ACTIVE_TRACE: contextvars.ContextVar[_GraphBuilder | None] = contextvars.ContextVar(
    "multigrain_v3_2_trace", default=None
)


def _expand_ports(values: tuple[Port, ...]) -> tuple[Port, ...]:
    builder = _ACTIVE_TRACE.get()
    if builder is None:
        raise LogicalCompileError("expand is only valid in Pipeline.forward")
    return builder.expand(values)


def _reduce_ports(values: tuple[Port, ...]) -> tuple[Port, ...]:
    builder = _ACTIVE_TRACE.get()
    if builder is None:
        raise LogicalCompileError("reduce is only valid in Pipeline.forward")
    return builder.reduce(values)
