"""RayModule-like authoring API for the Multigrain V3 prototype."""

from __future__ import annotations

import contextvars
import inspect
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .dag import (
    CompiledDAG,
    CompileError,
    ExecutionSpec,
    InputMode,
    InputSpec,
    Primitive,
    RecoveryLimits,
    RecoveryPreset,
    RecoverySpec,
    ReduceSpec,
    StageSpec,
    UdfSpec,
    compile_dag,
)
from .model import PortId


class ExecutionError(RuntimeError):
    """A run or bounded Arena failed."""


class BadRecordError(Exception):
    """Explicitly attribute a UDF error to one dispatch row."""

    def __init__(self, message: str, *, index: int) -> None:
        super().__init__(message)
        if index < 0:
            raise ValueError("bad record index must be non-negative")
        self.index = index


class _Missing:
    """Pickle-stable sentinel for explicit optional input absence."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __reduce__(self):
        return (_missing_singleton, ())


def _missing_singleton() -> "_Missing":
    return MISSING


MISSING = _Missing()


@dataclass(frozen=True, slots=True)
class Port:
    """Symbolic output handle used only while tracing Pipeline.forward."""
    id: PortId


@dataclass(frozen=True, slots=True)
class OptionalPort:
    """Wrapper marking one aligned input as OPTIONAL_ONE."""
    port: Port


def optional(port: Port) -> OptionalPort:
    if not isinstance(port, Port):
        raise CompileError("optional(...) requires a symbolic Port")
    return OptionalPort(port)


@dataclass(frozen=True, slots=True)
class CompiledPipeline:
    """Immutable DAG plus ordered public source and output Ports."""
    dag: CompiledDAG
    source_ports: tuple[Port, ...]
    outputs: tuple[Port, ...]


class Pipeline:
    """User authoring base class traced once into an immutable CompiledDAG."""

    def forward(self, *args: Any) -> Any:
        raise NotImplementedError

    def compile(self) -> CompiledPipeline:
        parameters = tuple(inspect.signature(self.forward).parameters.values())
        if any(
            parameter.kind
            not in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            for parameter in parameters
        ):
            raise CompileError("Pipeline.forward supports positional sources only")

        source_stages = tuple(
            StageSpec(
                id=index,
                kind=Primitive.SOURCE,
                inputs=(),
                output_count=1,
                driving_input=None,
                udf=None,
                execution=None,
            )
            for index in range(len(parameters))
        )
        context = _TraceContext(list(source_stages), len(source_stages))
        token = _ACTIVE_TRACE.set(context)
        try:
            result = self.forward(
                *(
                    Port(PortId(index, 0))
                    for index in range(len(source_stages))
                )
            )
        finally:
            _ACTIVE_TRACE.reset(token)
        outputs = _normalize_outputs(result)
        source_ports = tuple(
            Port(PortId(index, 0)) for index in range(len(source_stages))
        )
        dag = compile_dag(
            tuple(context.stages),
            source_ports=tuple(port.id for port in source_ports),
            output_ports=tuple(port.id for port in outputs),
        )
        return CompiledPipeline(dag, source_ports, outputs)


@dataclass(slots=True)
class _TraceContext:
    """Mutable compiler state scoped to one symbolic forward trace."""
    stages: list[StageSpec]
    next_stage: int

    def call(
        self,
        primitive: "_ConfiguredPrimitive[Any]",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Port | tuple[Port, ...]:
        kind = {
            Map: Primitive.MAP,
            Filter: Primitive.FILTER,
            Expand: Primitive.EXPAND,
            Reduce: Primitive.REDUCE,
        }[type(primitive)]
        stage_id = self.next_stage
        self.next_stage += 1
        options = dict(primitive.options)
        output_count_option = options.pop("num_outputs", None)

        if kind is Primitive.REDUCE:
            if args or not {"anchor", "members"}.issubset(kwargs):
                raise CompileError("Reduce requires anchor=... and members=...")
            anchor = _port(kwargs.pop("anchor"))
            members = _port(kwargs.pop("members"))
            bindings: list[InputSpec] = [
                InputSpec("anchor", anchor.id, InputMode.ANCHOR),
                InputSpec("members", members.id, InputMode.GROUP),
            ]
            origin_expand = self._origin_expand(members.id)
            for name, value in kwargs.items():
                public_port, optional_value = _port_value(value)
                mode = self._reduce_mode(
                    anchor,
                    public_port,
                    origin_expand=origin_expand,
                    optional_value=optional_value,
                )
                bindings.append(InputSpec(name, public_port.id, mode))
            reduce_spec = ReduceSpec(
                members_input=1,
                origin_expand=origin_expand,
            )
            driving_input = None
            output_count = int(output_count_option or 1)
        else:
            if not args:
                raise CompileError(f"{kind.value} requires a driving Port")
            inputs: list[InputSpec] = []
            for index, value in enumerate(args):
                public_port, optional_value = _port_value(value)
                name = {
                    Primitive.MAP: "primary",
                    Primitive.FILTER: "target",
                    Primitive.EXPAND: "parent",
                }[kind] if index == 0 else f"input_{index}"
                inputs.append(
                    InputSpec(
                        name,
                        public_port.id,
                        (
                            InputMode.OPTIONAL_ONE
                            if optional_value
                            else InputMode.ONE
                        ),
                    )
                )
            for name, value in kwargs.items():
                public_port, optional_value = _port_value(value)
                inputs.append(
                    InputSpec(
                        name,
                        public_port.id,
                        (
                            InputMode.OPTIONAL_ONE
                            if optional_value
                            else InputMode.ONE
                        ),
                    )
                )
            bindings = inputs
            driving_input = 0
            reduce_spec = None
            output_count = (
                len(bindings)
                if kind is Primitive.FILTER
                else int(output_count_option or 1)
            )
            if kind is Primitive.FILTER and output_count_option not in {
                None,
                len(bindings),
            }:
                raise CompileError("Filter output_count always equals input count")

        if output_count <= 0:
            raise CompileError("num_outputs must be positive")
        execution = _execution_spec(options)
        stage = StageSpec(
            id=stage_id,
            kind=kind,
            inputs=tuple(bindings),
            output_count=output_count,
            driving_input=driving_input,
            udf=UdfSpec(
                primitive.udf,
                primitive.init_args,
                tuple(primitive.init_kwargs.items()),
            ),
            execution=execution,
            reduce=reduce_spec,
        )
        self.stages.append(stage)
        ports = tuple(Port(port) for port in stage.output_ports())
        return ports[0] if len(ports) == 1 else ports

    def _origin_expand(self, port: PortId) -> int:
        scope = self._scope(port)
        if scope:
            return scope[-1]
        raise CompileError("Reduce members do not have a unique origin Expand")

    def _reduce_mode(
        self,
        anchor: Port,
        value: Port,
        *,
        origin_expand: int,
        optional_value: bool,
    ) -> InputMode:
        anchor_scope = self._scope(anchor.id)
        value_scope = self._scope(value.id)
        if value_scope == anchor_scope:
            return (
                InputMode.OPTIONAL_ONE
                if optional_value
                else InputMode.ONE
            )
        if value_scope == (*anchor_scope, origin_expand):
            if optional_value:
                raise CompileError("optional(...) cannot wrap a Reduce GROUP")
            return InputMode.GROUP
        raise CompileError("Reduce input is not anchor- or group-aligned")

    def _scope(self, port: PortId) -> tuple[int, ...]:
        producer = self.stages[port.stage]
        if producer.kind is Primitive.SOURCE:
            return ()
        if producer.kind in {Primitive.MAP, Primitive.FILTER}:
            assert producer.driving_input is not None
            return self._scope(producer.inputs[producer.driving_input].port)
        if producer.kind is Primitive.EXPAND:
            assert producer.driving_input is not None
            parent_scope = self._scope(
                producer.inputs[producer.driving_input].port
            )
            return (*parent_scope, producer.id)
        assert producer.kind is Primitive.REDUCE
        anchor = next(
            spec
            for spec in producer.inputs
            if spec.mode is InputMode.ANCHOR
        )
        return self._scope(anchor.port)


_ACTIVE_TRACE: contextvars.ContextVar[_TraceContext | None] = (
    contextvars.ContextVar("multigrain_v3_trace", default=None)
)


def _port(value: Any) -> Port:
    if not isinstance(value, Port):
        raise CompileError(f"expected symbolic Port, got {type(value)!r}")
    return value


def _port_value(value: Any) -> tuple[Port, bool]:
    if isinstance(value, OptionalPort):
        return value.port, True
    return _port(value), False


def _normalize_outputs(value: Any) -> tuple[Port, ...]:
    if isinstance(value, Port):
        return (value,)
    if isinstance(value, tuple) and value and all(
        isinstance(port, Port) for port in value
    ):
        return value
    raise CompileError("Pipeline.forward must return Port or non-empty tuple[Port]")


def _execution_spec(options: dict[str, Any]) -> ExecutionSpec:
    preset_value = str(options.pop("recovery", options.pop("error_policy", "raise")))
    try:
        preset = RecoveryPreset(preset_value)
    except ValueError as error:
        raise CompileError(f"unknown recovery preset: {preset_value}") from error
    limits = RecoveryLimits(
        max_infra_retries=int(options.pop("max_infra_retries", 1)),
        max_recovery_attempts=int(options.pop("max_recovery_attempts", 2)),
        max_split_depth=int(options.pop("max_split_depth", 16)),
        max_extra_rpcs=int(options.pop("max_extra_rpcs", 1024)),
        max_reexecuted_grains=int(
            options.pop("max_reexecuted_grains", 100_000)
        ),
    )
    return ExecutionSpec(
        replicas=int(options.pop("replicas", 1)),
        batch_size=int(options.pop("batch_size", 1)),
        max_batch_wait_ms=float(options.pop("max_batch_wait_ms", 2.0)),
        batch_scope=str(options.pop("batch_scope", "elastic")),
        recovery=RecoverySpec(preset, limits),
        ray_options=tuple(options.items()),
    )


PrimitiveT = TypeVar("PrimitiveT", bound="_ConfiguredPrimitive")


class _ConfiguredPrimitive(Generic[PrimitiveT]):
    """RayModule-like UDF construction and execution-option builder."""
    def __init__(self, udf: Any) -> None:
        self.udf = udf
        self.init_args: tuple[Any, ...] = ()
        self.init_kwargs: dict[str, Any] = {}
        self.options: dict[str, Any] = {}

    def pre_init(self: PrimitiveT, *args: Any, **kwargs: Any) -> PrimitiveT:
        self.init_args = tuple(args)
        self.init_kwargs = dict(kwargs)
        return self

    def ray_options(self: PrimitiveT, **options: Any) -> PrimitiveT:
        self.options.update(options)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        context = _ACTIVE_TRACE.get()
        if context is None:
            raise CompileError("primitive calls are only valid in Pipeline.forward")
        return context.call(self, tuple(args), dict(kwargs))


class Map(_ConfiguredPrimitive["Map"]):
    """Entity-preserving value transformation."""
    pass


class Filter(_ConfiguredPrimitive["Filter"]):
    """Tuple-preserving bool mask over required aligned inputs."""
    pass


class Expand(_ConfiguredPrimitive["Expand"]):
    """Dynamic one-to-many Stage with stable ordinal child identities."""
    pass


class Reduce(_ConfiguredPrimitive["Reduce"]):
    """Ordered many-to-one Stage with a semantic-only anchor."""
    pass
