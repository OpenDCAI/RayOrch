"""Multigrain V3 的 RayModule-like 用户编排 API。

用户通过 Pipeline.forward 连接 symbolic Port；`pre_init` 描述 persistent UDF 构造参数，
`ray_options` 描述副本、合批、恢复和 Ray 资源。该模块只负责 authoring/trace，不执行
Arena 调度或 Ray RPC。
"""

from __future__ import annotations

import contextvars
import inspect
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .contracts import BadRecordError, ExecutionError, MISSING
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


@dataclass(frozen=True, slots=True)
class Port:
    """仅在 trace Pipeline.forward 时使用的 symbolic output handle。"""
    id: PortId


@dataclass(frozen=True, slots=True)
class OptionalPort:
    """把一个 aligned input 显式标记为 OPTIONAL_ONE 的 symbolic wrapper。"""
    port: Port


def optional(port: Port) -> OptionalPort:
    """声明某个 aligned Port 正常缺失时向 UDF 传 MISSING。"""

    if not isinstance(port, Port):
        raise CompileError("optional(...) requires a symbolic Port")
    return OptionalPort(port)


@dataclass(frozen=True, slots=True)
class CompiledPipeline:
    """不可变 CompiledDAG 与有序 public source/output Ports 的组合。"""
    dag: CompiledDAG
    source_ports: tuple[Port, ...]
    outputs: tuple[Port, ...]


class Pipeline:
    """用户 Pipeline 基类；forward 会被 symbolic trace 为 immutable CompiledDAG。"""

    def forward(self, *args: Any) -> Any:
        """声明 symbolic DAG；子类必须实现且不能直接执行业务数据。"""

        raise NotImplementedError

    def compile(self) -> CompiledPipeline:
        """执行一次 symbolic forward trace，并完成静态 DAG 校验。"""

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
    """一次 symbolic forward trace 内部使用的可变 compiler state。"""
    stages: list[StageSpec]
    next_stage: int

    def call(
        self,
        primitive: "_ConfiguredPrimitive[Any]",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Port | tuple[Port, ...]:
        """把一次 primitive 调用编译成 StageSpec，并返回 symbolic output Ports。"""

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
            scope_path = self._reduce_scope_path(anchor.id, members.id)
            for name, value in kwargs.items():
                public_port, optional_value = _port_value(value)
                mode = self._reduce_mode(
                    anchor,
                    public_port,
                    scope_path=scope_path,
                    optional_value=optional_value,
                )
                bindings.append(InputSpec(name, public_port.id, mode))
            reduce_spec = ReduceSpec(
                members_input=1,
                scope_path=scope_path,
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

    def _reduce_scope_path(
        self,
        anchor_port: PortId,
        members_port: PortId,
    ) -> tuple[int, ...]:
        """根据 anchor/member scopes 推导唯一的 descendant Expand path。"""

        anchor_scope = self._scope(anchor_port)
        member_scope = self._scope(members_port)
        if (
            len(member_scope) <= len(anchor_scope)
            or member_scope[: len(anchor_scope)] != anchor_scope
        ):
            raise CompileError(
                "Reduce members scope is not a descendant of anchor scope"
            )
        return member_scope[len(anchor_scope) :]

    def _reduce_mode(
        self,
        anchor: Port,
        value: Port,
        *,
        scope_path: tuple[int, ...],
        optional_value: bool,
    ) -> InputMode:
        """把额外 Reduce input 分类为 anchor-aligned scalar 或同路径 GROUP。"""

        anchor_scope = self._scope(anchor.id)
        value_scope = self._scope(value.id)
        if value_scope == anchor_scope:
            return (
                InputMode.OPTIONAL_ONE
                if optional_value
                else InputMode.ONE
            )
        if value_scope == (*anchor_scope, *scope_path):
            if optional_value:
                raise CompileError("optional(...) cannot wrap a Reduce GROUP")
            return InputMode.GROUP
        raise CompileError("Reduce input is not anchor- or group-aligned")

    def _scope(self, port: PortId) -> tuple[int, ...]:
        """递归推导一个 Port 所处的 Expand scope signature。"""

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
    """校验 trace 参数是普通 symbolic Port。"""

    if not isinstance(value, Port):
        raise CompileError(f"expected symbolic Port, got {type(value)!r}")
    return value


def _port_value(value: Any) -> tuple[Port, bool]:
    """解包普通/OptionalPort，并返回是否 OPTIONAL_ONE。"""

    if isinstance(value, OptionalPort):
        return value.port, True
    return _port(value), False


def _normalize_outputs(value: Any) -> tuple[Port, ...]:
    """把 forward 返回值规范化为非空、有序 Port tuple。"""

    if isinstance(value, Port):
        return (value,)
    if isinstance(value, tuple) and value and all(
        isinstance(port, Port) for port in value
    ):
        return value
    raise CompileError("Pipeline.forward must return Port or non-empty tuple[Port]")


def _execution_spec(options: dict[str, Any]) -> ExecutionSpec:
    """把 `.ray_options()` 捕获值编译为 immutable ExecutionSpec。"""

    outstanding = options.pop("max_outstanding_per_actor", None)
    legacy_outstanding = options.pop("max_pending_per_actor", None)
    if outstanding is not None and legacy_outstanding is not None:
        raise CompileError(
            "use only max_outstanding_per_actor; "
            "max_pending_per_actor is a compatibility alias"
        )
    if outstanding is None:
        outstanding = legacy_outstanding
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
        max_outstanding_per_actor=(
            None if outstanding is None else int(outstanding)
        ),
        recovery=RecoverySpec(preset, limits),
        ray_options=tuple(options.items()),
    )


PrimitiveT = TypeVar("PrimitiveT", bound="_ConfiguredPrimitive")


class _ConfiguredPrimitive(Generic[PrimitiveT]):
    """RayModule-like UDF 构造和执行配置 builder。"""
    def __init__(self, udf: Any) -> None:
        """保存 UDF target；初始化参数与执行参数后续链式设置。"""

        self.udf = udf
        self.init_args: tuple[Any, ...] = ()
        self.init_kwargs: dict[str, Any] = {}
        self.options: dict[str, Any] = {}

    def pre_init(self: PrimitiveT, *args: Any, **kwargs: Any) -> PrimitiveT:
        """记录每个 persistent actor 构造 UDF instance 的参数。"""

        self.init_args = tuple(args)
        self.init_kwargs = dict(kwargs)
        return self

    def ray_options(self: PrimitiveT, **options: Any) -> PrimitiveT:
        """记录 replicas、batch、recovery 和 Ray resource 配置。"""

        self.options.update(options)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        context = _ACTIVE_TRACE.get()
        if context is None:
            raise CompileError("primitive calls are only valid in Pipeline.forward")
        return context.call(self, tuple(args), dict(kwargs))


class Map(_ConfiguredPrimitive["Map"]):
    """保持 EntityId 的普通 value transformation 原语。"""
    pass


class Filter(_ConfiguredPrimitive["Filter"]):
    """对 required aligned tuple 计算 bool mask，并同步转发或 drop。"""
    pass


class Expand(_ConfiguredPrimitive["Expand"]):
    """动态 one-to-many 原语，为每个 child 派生稳定 ordinal identity。"""
    pass


class Reduce(_ConfiguredPrimitive["Reduce"]):
    """有序 many-to-one 原语；anchor 只参与语义，不进入 Actor payload。"""
    pass
