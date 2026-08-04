"""把 V3.2 Port-first 逻辑 IR 降低到经过验证的 V3 物理 runtime。"""

from __future__ import annotations

from typing import Any

from rayorch.experimental.multigrain_v3.api import (
    CompiledPipeline,
    Port as V3Port,
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

from .ir import CallNode, ExpandNode, LogicalDAG, ReduceNode, SourceNode


class _ExpandAlignedValues:
    """解包一个或多个 aligned group column 的结构 actor UDF。"""

    def __init__(self, width: int) -> None:
        self.width = width

    def run(self, *columns: list[Any]) -> Any:
        if len(columns) != self.width:
            raise ValueError("aligned expand width changed")
        outputs = tuple(
            [list(group) for group in column]
            for column in columns
        )
        return outputs[0] if self.width == 1 else outputs


class _PackAlignedGroups:
    """物化 canonical grouped input 的结构 actor UDF。"""

    def __init__(self, width: int) -> None:
        self.width = width

    def run(self, *columns: list[list[Any]]) -> Any:
        if len(columns) != self.width:
            raise ValueError("aligned reduce width changed")
        outputs = tuple(
            [list(group) for group in column]
            for column in columns
        )
        return outputs[0] if self.width == 1 else outputs


def lower_to_v3(logical: LogicalDAG) -> CompiledPipeline:
    """创建 correctness-first 物理计划，不使用 authoring backpatch。

    每个逻辑 Port transform 都成为显式的现有 V3 Stage；后续 optimizer 只有在
    证明等价后才能融合 stage，correctness 永远不能依赖 fusion。
    """

    stages: list[StageSpec] = []
    ports: dict[Any, PortId] = {}
    expansions: dict[Any, tuple[int, PortId]] = {}

    for node in logical.nodes:
        stage_id = len(stages)
        if isinstance(node, SourceNode):
            stage = StageSpec(
                id=stage_id,
                kind=Primitive.SOURCE,
                inputs=(),
                output_count=1,
                driving_input=None,
                udf=None,
                execution=None,
            )
            stages.append(stage)
            ports[node.output] = stage.output_ports()[0]
            continue

        if isinstance(node, CallNode):
            options = dict(node.module.options)
            declared_outputs = int(options.pop("num_outputs", len(node.outputs)))
            if declared_outputs != len(node.outputs):
                raise CompileError("logical/physical RayModule output count changed")
            inputs = tuple(
                InputSpec(
                    item.name,
                    ports[item.port],
                    InputMode.OPTIONAL_ONE if item.optional else InputMode.ONE,
                )
                for item in node.inputs
            )
            stage = StageSpec(
                id=stage_id,
                kind=Primitive.MAP,
                inputs=inputs,
                output_count=len(node.outputs),
                driving_input=0,
                udf=UdfSpec(
                    node.module.target,
                    node.module.init_args,
                    node.module.init_kwargs,
                ),
                execution=_execution_spec(options),
            )
        elif isinstance(node, ExpandNode):
            input_ports = tuple(ports[port] for port in node.inputs)
            stage = StageSpec(
                id=stage_id,
                kind=Primitive.EXPAND,
                inputs=tuple(
                    InputSpec(f"group_{index}", port, InputMode.ONE)
                    for index, port in enumerate(input_ports)
                ),
                output_count=len(node.outputs),
                driving_input=0,
                udf=UdfSpec(
                    _ExpandAlignedValues,
                    (len(node.inputs),),
                    (),
                ),
                execution=_structural_execution(),
            )
            expansions[node.expansion] = (stage_id, input_ports[0])
        elif isinstance(node, ReduceNode):
            try:
                expand_stage, anchor = expansions[node.expansion]
            except KeyError as error:
                raise CompileError("Reduce references an unlowered expansion") from error
            group_ports = tuple(ports[port] for port in node.inputs)
            inputs = (
                InputSpec("__anchor__", anchor, InputMode.ANCHOR),
                *(
                    InputSpec(f"group_{index}", port, InputMode.GROUP)
                    for index, port in enumerate(group_ports)
                ),
            )
            stage = StageSpec(
                id=stage_id,
                kind=Primitive.REDUCE,
                inputs=tuple(inputs),
                output_count=len(node.outputs),
                driving_input=None,
                udf=UdfSpec(
                    _PackAlignedGroups,
                    (len(node.inputs),),
                    (),
                ),
                execution=_structural_execution(),
                reduce=ReduceSpec(1, (expand_stage,)),
            )
        else:
            raise TypeError(f"unknown logical node: {type(node)!r}")

        stages.append(stage)
        for logical_port, physical_port in zip(node.outputs, stage.output_ports()):
            ports[logical_port] = physical_port

    source_ports = tuple(ports[port] for port in logical.source_ports)
    output_ports = tuple(ports[port] for port in logical.output_ports)
    dag = compile_dag(
        tuple(stages),
        source_ports=source_ports,
        output_ports=output_ports,
    )
    return CompiledPipeline(
        dag,
        tuple(V3Port(port) for port in source_ports),
        tuple(V3Port(port) for port in output_ports),
    )


def _structural_execution() -> ExecutionSpec:
    return ExecutionSpec(
        replicas=1,
        batch_size=64,
        max_batch_wait_ms=2.0,
        batch_scope="elastic",
        recovery=RecoverySpec(),
    )


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
