"""Multigrain V3.1 声明式 API。

V3.1 保持 V3 编译与运行时模型不变，只调整 authoring：用户 UDF
由 `RayModule` 描述，grain/scope 变化由 functional view 表达。
trace 最终把这些 view 降低回 V3 `MAP/EXPAND/REDUCE` StageSpec。
"""

from __future__ import annotations

import contextvars
import inspect
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Self

from rayorch.experimental.multigrain_v3.api import (
    CompiledPipeline,
    Pipeline as _V3Pipeline,
    Port as _V3Port,
)
from rayorch.experimental.multigrain_v3.dag import (
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
from rayorch.experimental.multigrain_v3.model import PortId


class _View(Enum):
    """一个物理输出 Port 的纯 trace 解释。"""

    AUTO = "auto"
    SCALAR = "scalar"
    GROUP = "group"


@dataclass(frozen=True, slots=True)
class Port:
    """符号 V3 Port，以及只在 trace 阶段存在的 scalar/group view。

    `view` 不进入 V3 `PortId` 或 compiled DAG；它只在 lowering 时消费，
    永远不会到达 Arena、RPC 或 Worker 状态。
    """

    id: PortId
    _view: _View = _View.AUTO


@dataclass(frozen=True, slots=True)
class OptionalPort:
    """只在 trace 阶段存在的 OPTIONAL_ONE 包装。"""

    port: Port


def optional(port: Port) -> OptionalPort:
    """把标量输入标记为允许正常缺失。"""

    if not isinstance(port, Port):
        raise CompileError("optional(...) requires a symbolic Port")
    if port._view is _View.GROUP:
        raise CompileError("optional(...) cannot wrap a grouped view")
    return OptionalPort(port)


def expand(group: Port) -> Port:
    """打开一个动态 group，但不新增 Stage。

    producer RayModule 从默认 MAP 降低为 EXPAND；原 `group` 仍是 grouped view，
    返回的 Port 是同一物理输出 Port 的 scalar child view。
    """

    context = _ACTIVE_TRACE.get()
    if context is None:
        raise CompileError("expand(...) is only valid in Pipeline.forward")
    return context.expand(group)


def reduce(port: Port) -> Port:
    """把最近的 Expand scope 关闭成惰性 grouped view。

    这里不创建 Stage 或 actor；下游 RayModule 消费 grouped view 时，
    才降低为物理 REDUCE Stage。
    """

    if _ACTIVE_TRACE.get() is None:
        raise CompileError("reduce(...) is only valid in Pipeline.forward")
    if not isinstance(port, Port):
        raise CompileError("reduce(...) requires a symbolic Port")
    if port._view is _View.GROUP:
        raise CompileError("reduce(...) received an already grouped view")
    return Port(port.id, _View.GROUP)


class Pipeline(_V3Pipeline):
    """把 RayModule + functional authoring 降低到未改动的 V3 DAG。"""

    def compile(self) -> CompiledPipeline:
        """trace `forward`，并把 functional view 降低为 V3 StageSpec。"""

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
                *(Port(PortId(index, 0)) for index in range(len(source_stages)))
            )
        finally:
            _ACTIVE_TRACE.reset(token)

        outputs = context.normalize_outputs(result)
        source_ports = tuple(
            Port(PortId(index, 0)) for index in range(len(source_stages))
        )
        dag = compile_dag(
            tuple(context.stages),
            source_ports=tuple(port.id for port in source_ports),
            output_ports=tuple(port.id for port in outputs),
        )
        # trace-only view metadata 不能进入编译与运行时合同；
        # CompiledPipeline 因而只保存原生 V3 Port。
        return CompiledPipeline(
            dag,
            tuple(_V3Port(port.id) for port in source_ports),
            tuple(_V3Port(port.id) for port in outputs),
        )


@dataclass(frozen=True, slots=True)
class _GroupContract:
    """一个 grouped leaf Port 的纯 trace lowering 事实。"""

    leaf: PortId
    anchor: PortId
    scope_path: tuple[int, ...]


@dataclass(slots=True)
class _TraceContext:
    """可变 authoring IR；最终输出只包含 V3 StageSpec。"""

    stages: list[StageSpec]
    next_stage: int

    def call(
        self,
        module: "RayModule",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Port | tuple[Port, ...]:
        """把 RayModule 调用降低为 MAP；存在 group 时降低为 REDUCE。"""

        if not args:
            raise CompileError("RayModule requires at least one input Port")
        values: list[tuple[str, Port, bool]] = []
        for index, value in enumerate(args):
            port, optional_value = self._port_value(value)
            values.append((f"arg_{index}", port, optional_value))
        for name, value in kwargs.items():
            if name == "__anchor__":
                raise CompileError("__anchor__ is a reserved input name")
            port, optional_value = self._port_value(value)
            values.append((name, port, optional_value))

        groups = [
            (index, self._group_contract(port))
            for index, (_, port, optional_value) in enumerate(values)
            if self._is_group(port, optional_value=optional_value)
        ]
        stage_id = self.next_stage
        self.next_stage += 1
        options = dict(module.options)
        output_count = int(options.pop("num_outputs", 1))
        if output_count <= 0:
            raise CompileError("num_outputs must be positive")

        if groups:
            contracts = tuple(contract for _, contract in groups)
            first = contracts[0]
            if any(
                contract.anchor != first.anchor
                or contract.scope_path != first.scope_path
                for contract in contracts[1:]
            ):
                raise CompileError(
                    "grouped RayModule inputs have different anchors or scopes"
                )
            bindings = [
                InputSpec("__anchor__", first.anchor, InputMode.ANCHOR)
            ]
            members_input: int | None = None
            for name, port, optional_value in values:
                if self._is_group(port, optional_value=optional_value):
                    mode = InputMode.GROUP
                    if members_input is None:
                        members_input = len(bindings)
                else:
                    mode = (
                        InputMode.OPTIONAL_ONE
                        if optional_value
                        else InputMode.ONE
                    )
                bindings.append(InputSpec(name, port.id, mode))
            assert members_input is not None
            kind = Primitive.REDUCE
            driving_input = None
            reduce_spec = ReduceSpec(members_input, first.scope_path)
        else:
            bindings = [
                InputSpec(
                    name,
                    port.id,
                    InputMode.OPTIONAL_ONE if optional_value else InputMode.ONE,
                )
                for name, port, optional_value in values
            ]
            kind = Primitive.MAP
            driving_input = 0
            reduce_spec = None

        stage = StageSpec(
            id=stage_id,
            kind=kind,
            inputs=tuple(bindings),
            output_count=output_count,
            driving_input=driving_input,
            udf=UdfSpec(
                module.udf,
                module.init_args,
                tuple(module.init_kwargs.items()),
            ),
            execution=_execution_spec(options),
            reduce=reduce_spec,
        )
        self.stages.append(stage)
        ports = tuple(Port(port) for port in stage.output_ports())
        return ports[0] if len(ports) == 1 else ports

    def expand(self, value: Port) -> Port:
        """把尚未消费的默认 MAP producer 回写为物理 EXPAND。"""

        if not isinstance(value, Port) or value._view is not _View.AUTO:
            raise CompileError("expand(...) requires an unexpanded RayModule output")
        stage = self.stages[value.id.stage]
        if stage.kind is not Primitive.MAP:
            raise CompileError("expand(...) producer must be a default RayModule call")
        if stage.output_count != 1 or value.id.output != 0:
            raise CompileError("V3.1 expand(...) currently requires one output Port")
        if any(
            spec.port == value.id
            for later in self.stages[stage.id + 1 :]
            for spec in later.inputs
        ):
            raise CompileError("expand(...) must precede all consumers of its group")
        self.stages[stage.id] = replace(stage, kind=Primitive.EXPAND)
        return Port(value.id, _View.SCALAR)

    def normalize_outputs(self, value: Any) -> tuple[Port, ...]:
        """要求物理标量输出；惰性 group 必须由下游 module 消费。"""

        ports = (value,) if isinstance(value, Port) else value
        if not (
            isinstance(ports, tuple)
            and ports
            and all(isinstance(port, Port) for port in ports)
        ):
            raise CompileError("Pipeline.forward must return Port or non-empty tuple[Port]")
        for port in ports:
            if self._is_group(port, optional_value=False):
                raise CompileError(
                    "Pipeline output is a lazy group; consume it with RayModule "
                    "or materialize it explicitly"
                )
        return ports

    def _is_group(self, port: Port, *, optional_value: bool) -> bool:
        if optional_value and port._view is _View.GROUP:
            raise CompileError("optional(...) cannot wrap a grouped view")
        if port._view is _View.GROUP:
            return True
        return (
            port._view is _View.AUTO
            and self.stages[port.id.stage].kind is Primitive.EXPAND
        )

    def _group_contract(self, port: Port) -> _GroupContract:
        member_scope = self._scope(port.id)
        if not member_scope:
            raise CompileError("reduce/group view has no Expand scope to close")
        expand_stage_id = member_scope[-1]
        expand_stage = self.stages[expand_stage_id]
        if expand_stage.kind is not Primitive.EXPAND:
            raise CompileError("group scope does not terminate at Expand")
        assert expand_stage.driving_input is not None
        anchor = expand_stage.inputs[expand_stage.driving_input].port
        return _GroupContract(port.id, anchor, (expand_stage_id,))

    def _scope(self, port: PortId) -> tuple[int, ...]:
        stage = self.stages[port.stage]
        if stage.kind is Primitive.SOURCE:
            return ()
        if stage.kind in {Primitive.MAP, Primitive.EXPAND}:
            assert stage.driving_input is not None
            parent_scope = self._scope(stage.inputs[stage.driving_input].port)
            return (
                (*parent_scope, stage.id)
                if stage.kind is Primitive.EXPAND
                else parent_scope
            )
        if stage.kind is Primitive.REDUCE:
            anchor = next(
                spec for spec in stage.inputs if spec.mode is InputMode.ANCHOR
            )
            return self._scope(anchor.port)
        raise CompileError(f"unsupported V3.1 producer kind: {stage.kind.value}")

    @staticmethod
    def _port_value(value: Any) -> tuple[Port, bool]:
        if isinstance(value, OptionalPort):
            return value.port, True
        if not isinstance(value, Port):
            raise CompileError(f"expected symbolic Port, got {type(value)!r}")
        return value, False


_ACTIVE_TRACE: contextvars.ContextVar[_TraceContext | None] = contextvars.ContextVar(
    "multigrain_v3_1_trace", default=None
)


class RayModule:
    """声明式持久 UDF 描述符；永远不持有 actor handle。"""

    def __init__(self, udf: Any) -> None:
        self.udf = udf
        self.init_args: tuple[Any, ...] = ()
        self.init_kwargs: dict[str, Any] = {}
        self.options: dict[str, Any] = {}

    def pre_init(self, *args: Any, **kwargs: Any) -> Self:
        """记录每个持久 Stage actor 的构造参数。"""

        self.init_args = tuple(args)
        self.init_kwargs = dict(kwargs)
        return self

    def ray_options(self, **options: Any) -> Self:
        """记录 replicas、batching、recovery 与 Ray 资源配置。"""

        self.options.update(options)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        context = _ACTIVE_TRACE.get()
        if context is None:
            raise CompileError("RayModule calls are only valid in Pipeline.forward")
        return context.call(self, tuple(args), dict(kwargs))


def _execution_spec(options: dict[str, Any]) -> ExecutionSpec:
    """把 RayModule options 编译为未改动的 V3 ExecutionSpec。"""

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
        max_reexecuted_grains=int(options.pop("max_reexecuted_grains", 100_000)),
    )
    return ExecutionSpec(
        replicas=int(options.pop("replicas", 1)),
        batch_size=int(options.pop("batch_size", 1)),
        max_batch_wait_ms=float(options.pop("max_batch_wait_ms", 2.0)),
        batch_scope=str(options.pop("batch_scope", "elastic")),
        recovery=RecoverySpec(preset, limits),
        ray_options=tuple(options.items()),
    )
