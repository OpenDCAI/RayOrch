"""Small public API surface for the V2.5 prototype."""

from __future__ import annotations

import contextvars
import inspect
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .grain import PortId


class CompileError(ValueError):
    """The user graph cannot be represented by the current V2.5 scope."""


class ExecutionError(RuntimeError):
    """A microbatch arena failed at run-control level."""


class BadRecordError(Exception):
    """Explicitly attribute a UDF error to one dispatch grain."""

    def __init__(self, message: str, *, index: int) -> None:
        super().__init__(message)
        if index < 0:
            raise ValueError("bad record index must be non-negative")
        self.index = index


@dataclass(frozen=True, slots=True)
class Port:
    id: PortId


@dataclass(frozen=True, slots=True)
class KeyedPort:
    port: Port
    by: Any


def keyed(port: Port | PortId, *, by: Any) -> KeyedPort:
    public_port = port if isinstance(port, Port) else Port(port)
    return KeyedPort(public_port, by)


class Pipeline:
    """User authoring base class traced into the compact compiled graph."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def compile(self) -> "CompiledPipeline":
        from .graph import NodeSpec, Primitive, compile_graph

        signature = inspect.signature(self.forward)
        parameters = tuple(signature.parameters.values())
        if any(
            parameter.kind
            not in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            for parameter in parameters
        ):
            raise CompileError(
                "Phase 3 Pipeline.forward supports positional source parameters"
            )
        source_nodes = tuple(
            NodeSpec(
                id=index,
                kind=Primitive.SOURCE,
                inputs=(),
                output_ports=(PortId(index, 0),),
                udf_recipe=None,
                execution=None,
            )
            for index in range(len(parameters))
        )
        context = _TraceContext(list(source_nodes), len(source_nodes))
        token = _ACTIVE_TRACE.set(context)
        try:
            result = self.forward(
                *(Port(node.output_ports[0]) for node in source_nodes)
            )
        finally:
            _ACTIVE_TRACE.reset(token)
        outputs = _normalize_pipeline_outputs(result)
        return CompiledPipeline(
            graph=compile_graph(tuple(context.nodes)),
            source_ports=tuple(
                Port(node.output_ports[0]) for node in source_nodes
            ),
            outputs=outputs,
        )


@dataclass(frozen=True, slots=True)
class CompiledPipeline:
    graph: Any
    source_ports: tuple[Port, ...]
    outputs: tuple[Port, ...]


@dataclass(slots=True)
class _TraceContext:
    nodes: list[Any]
    next_node: int

    def call(
        self,
        primitive: "_ConfiguredPrimitive[Any]",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Port | tuple[Port, ...]:
        from .graph import (
            ExecutionOptions,
            InputBinding,
            KeyProjection,
            NodeSpec,
            Primitive,
            UdfRecipe,
        )

        kind_by_type = {
            Map: Primitive.MAP,
            Filter: Primitive.FILTER,
            Expand: Primitive.EXPAND,
            Reduce: Primitive.REDUCE,
            Relate: Primitive.RELATE,
        }
        kind = kind_by_type[type(primitive)]
        node_id = self.next_node
        self.next_node += 1
        options = dict(primitive.options)
        output_arity = int(options.pop("num_outputs", 1))
        if output_arity <= 0:
            raise CompileError("num_outputs must be positive")
        output_ports = tuple(
            PortId(node_id, slot) for slot in range(output_arity)
        )

        reduce_anchor = None
        reduce_members = None
        relate_keys = ()
        if kind is Primitive.REDUCE:
            if args or set(kwargs) != {"anchor", "members"}:
                raise CompileError(
                    "Reduce must be called as reduce(anchor=..., members=...)"
                )
            anchor = _require_port(kwargs["anchor"])
            members = _require_port(kwargs["members"])
            bindings = (
                InputBinding("anchor", anchor.id),
                InputBinding("members", members.id),
            )
            reduce_anchor = anchor.id
            reduce_members = members.id
        elif kind is Primitive.RELATE:
            if args or len(kwargs) < 2:
                raise CompileError("Relate requires at least two keyed roles")
            keyed_roles = tuple(
                (role, _require_keyed(value))
                for role, value in kwargs.items()
            )
            bindings = tuple(
                InputBinding(role, value.port.id)
                for role, value in keyed_roles
            )
            relate_keys = tuple(
                KeyProjection(role, value.by)
                for role, value in keyed_roles
            )
        else:
            if not args:
                raise CompileError(f"{kind.value} requires a driving Port")
            driving = _require_port(args[0])
            driving_role = {
                Primitive.MAP: "primary",
                Primitive.FILTER: "target",
                Primitive.EXPAND: "parent",
            }[kind]
            bindings_list = [InputBinding(driving_role, driving.id)]
            for index, value in enumerate(args[1:], start=1):
                bindings_list.append(
                    InputBinding(f"input_{index}", _require_port(value).id)
                )
            bindings_list.extend(
                InputBinding(role, _require_port(value).id)
                for role, value in kwargs.items()
            )
            bindings = tuple(bindings_list)

        execution = ExecutionOptions(
            replicas=int(options.pop("replicas", 1)),
            batch_size=int(options.pop("batch_size", 1)),
            max_batch_wait_ms=float(
                options.pop("max_batch_wait_ms", 2.0)
            ),
            batch_scope=str(options.pop("batch_scope", "elastic")),
            error_policy=str(options.pop("error_policy", "raise")),
            max_retries=int(options.pop("max_retries", 0)),
            options=tuple(options.items()),
        )
        node = NodeSpec(
            id=node_id,
            kind=kind,
            inputs=bindings,
            output_ports=output_ports,
            udf_recipe=UdfRecipe(
                primitive.udf,
                init_args=primitive.init_args,
                init_kwargs=tuple(primitive.init_kwargs.items()),
            ),
            execution=execution,
            reduce_anchor=reduce_anchor,
            reduce_members=reduce_members,
            relate_keys=relate_keys,
        )
        self.nodes.append(node)
        ports = tuple(Port(port) for port in output_ports)
        return ports[0] if len(ports) == 1 else ports


_ACTIVE_TRACE: contextvars.ContextVar[_TraceContext | None] = (
    contextvars.ContextVar("multigrain_v2_5_trace", default=None)
)


def _require_port(value: Any) -> Port:
    if not isinstance(value, Port):
        raise CompileError(f"expected symbolic Port, got {type(value)!r}")
    return value


def _require_keyed(value: Any) -> KeyedPort:
    if not isinstance(value, KeyedPort):
        raise CompileError("Relate inputs must be wrapped with keyed(...)")
    return value


def _normalize_pipeline_outputs(value: Any) -> tuple[Port, ...]:
    if isinstance(value, Port):
        return (value,)
    if isinstance(value, tuple) and all(
        isinstance(port, Port) for port in value
    ):
        return value
    raise CompileError("Pipeline.forward must return Port or tuple[Port, ...]")


PrimitiveT = TypeVar("PrimitiveT", bound="_ConfiguredPrimitive")


class _ConfiguredPrimitive(Generic[PrimitiveT]):
    """RayModule-style constructor and execution option capture."""

    def __init__(self, udf: Any) -> None:
        self.udf = udf
        self.init_args: tuple[Any, ...] = ()
        self.init_kwargs: dict[str, Any] = {}
        self.options: dict[str, Any] = {}

    def pre_init(
        self: PrimitiveT,
        *args: Any,
        **kwargs: Any,
    ) -> PrimitiveT:
        self.init_args = tuple(args)
        self.init_kwargs = dict(kwargs)
        return self

    def ray_options(
        self: PrimitiveT,
        **options: Any,
    ) -> PrimitiveT:
        self.options.update(options)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        context = _ACTIVE_TRACE.get()
        if context is None:
            raise CompileError(
                "primitive calls are only valid while tracing Pipeline.forward"
            )
        return context.call(self, tuple(args), dict(kwargs))


class Map(_ConfiguredPrimitive["Map"]):
    pass


class Filter(_ConfiguredPrimitive["Filter"]):
    pass


class Expand(_ConfiguredPrimitive["Expand"]):
    pass


class Reduce(_ConfiguredPrimitive["Reduce"]):
    pass


class Relate(_ConfiguredPrimitive["Relate"]):
    pass
