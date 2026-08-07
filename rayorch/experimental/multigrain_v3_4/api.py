"""Multigrain v3.4 的 RayModule 与符号化 Pipeline 编写 API。"""

from __future__ import annotations

import contextvars
import inspect
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Self, TypeVar, assert_never, overload
from .protocol import OutputLayout

from .model import CallRef, CompileError, DomainRef, InputMode, PoolRef, PortRef
from .program import (
    BroadcastOrigin,
    CallConsumer,
    CallOutputOrigin,
    CallSpec,
    CompiledProgram,
    DomainSpec,
    ExecutionPlan,
    ExpandOrigin,
    FilterOrigin,
    GroupOrigin,
    InputSpec,
    KernelSpec,
    PoolSpec,
    PortSpec,
    PortOrigin,
    Program,
    SourceOrigin,
    ViewConsumer,
    freeze_mapping,
)


@dataclass(frozen=True, slots=True)
class Port:
    """一个逻辑 Port 的公开符号句柄。"""

    ref: PortRef
    _owner: int


@dataclass(frozen=True, slots=True)
class OptionalPort:
    """仅改变 Call 输入策略的 Port 包装；不创建新 Port。"""

    port: Port


_TRACE_IDS = itertools.count()
_ACTIVE_TRACE: contextvars.ContextVar[_ProgramBuilder | None] = (
    contextvars.ContextVar("multigrain_v3_4_trace", default=None)
)


class RayModule:
    """声明式 UDF 配方；actor handle 只能存在于执行层。"""

    def __init__(self, udf: Any, *, num_outputs: int = 1) -> None:
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive")
        self.udf = udf
        self.num_outputs = int(num_outputs)
        self.init_args: tuple[Any, ...] = ()
        self.init_kwargs: dict[str, Any] = {}
        self.options: dict[str, Any] = {}
        self.driving_input: int | str = 0

    def returns(self, count: int) -> Self:
        """声明一次 Call 产生的逻辑输出 Port 数。"""

        if count <= 0:
            raise ValueError("output count must be positive")
        self.num_outputs = int(count)
        return self

    def pre_init(self, *args: Any, **kwargs: Any) -> Self:
        """记录 actor/UDF 实例化参数，不在编译期执行构造。"""

        self.init_args = tuple(args)
        self.init_kwargs = dict(kwargs)
        return self

    def ray_options(self, **options: Any) -> Self:
        """记录该 Call 的物理执行选项并返回自身以便链式配置。"""

        if "num_outputs" in options:
            raise ValueError("use RayModule.returns(...) for logical outputs")
        self.options.update(options)
        return self

    def driven_by(self, input_: int | str) -> Self:
        """显式选择决定执行 Domain 的必需输入槽。"""

        self.driving_input = input_
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Port | tuple[Port, ...]:
        builder = _ACTIVE_TRACE.get()
        if builder is None:
            raise CompileError("RayModule calls are only valid in Pipeline.forward")
        return builder.call(self, args, kwargs)


F = TypeVar("F", bound=Callable[..., Any])


@overload
def function(fn: F, *, num_outputs: int = 1) -> RayModule: ...


@overload
def function(fn: None = None, *, num_outputs: int = 1) -> Callable[[F], RayModule]: ...


def function(
    fn: F | None = None,
    *,
    num_outputs: int = 1,
) -> RayModule | Callable[[F], RayModule]:
    """把普通 callable 适配成无状态的 RayModule 配方。"""

    def wrap(target: F) -> RayModule:
        return RayModule(target, num_outputs=num_outputs)

    return wrap if fn is None else wrap(fn)


class Pipeline:
    """用户声明数据流的入口；compile 仅执行一次符号追踪。"""

    def forward(self, *args: Any) -> Any:
        """声明 Pipeline；子类应只组合 RayModule 与 Port 结构操作。"""

        raise NotImplementedError

    def compile(self) -> CompiledProgram:
        """追踪 forward 并冻结逻辑 Program 与物理 ExecutionPlan。"""

        parameters = tuple(inspect.signature(self.forward).parameters.values())
        allowed = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        if not parameters:
            raise CompileError("Pipeline.forward requires at least one source")
        if any(parameter.kind not in allowed for parameter in parameters):
            raise CompileError("Pipeline.forward supports positional sources only")

        builder = _ProgramBuilder(tuple(parameter.name for parameter in parameters))
        token = _ACTIVE_TRACE.set(builder)
        try:
            sources = tuple(builder.public(port) for port in builder.source_ports)
            result = self.forward(*sources)
        finally:
            _ACTIVE_TRACE.reset(token)
        output_tree = builder.normalize_outputs(result)
        return builder.build(output_tree)


class _ProgramBuilder:
    """编译期可变 builder；所有导出的 Program 对象均不可变。"""

    def __init__(self, source_names: tuple[str, ...]) -> None:
        self.owner = next(_TRACE_IDS)
        self.next_port = 0
        self.next_call = 0
        self.next_domain = 1
        self.calls: dict[CallRef, CallSpec] = {}
        self.ports: dict[PortRef, PortSpec] = {}
        self.domains: dict[DomainRef, DomainSpec] = {
            DomainRef(0): DomainSpec(DomainRef(0), debug_name="root")
        }
        self.call_options: dict[CallRef, tuple[tuple[str, Any], ...]] = {}
        self._view_intern: dict[tuple[Any, ...], tuple[PortRef, ...]] = {}

        sources = []
        for index, name in enumerate(source_names):
            ref = self._new_port(
                DomainRef(0),
                SourceOrigin(index, name),
            )
            sources.append(ref)
        self.source_ports = tuple(sources)

    def public(self, ref: PortRef) -> Port:
        """把内部 PortRef 包装为绑定当前 trace 的公开句柄。"""

        return Port(ref, self.owner)

    def spec(self, port: Port, label: str = "operation") -> PortSpec:
        """校验 Port 属于当前 trace，并返回对应静态定义。"""

        if not isinstance(port, Port) or port._owner != self.owner:
            raise CompileError(f"{label} requires a Port from the active Pipeline")
        try:
            return self.ports[port.ref]
        except KeyError as error:
            raise CompileError(f"unknown Port: {port.ref}") from error

    def call(
        self,
        module: RayModule,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Port | tuple[Port, ...]:
        """登记唯一计算边界，并为每个逻辑输出创建独立 Port。"""

        if not args and not kwargs:
            raise CompileError("RayModule requires at least one Port input")

        inputs: list[InputSpec] = []
        for index, value in enumerate(args):
            port, mode = self._input(value, f"arg_{index}")
            inputs.append(InputSpec(f"arg_{index}", port.ref, mode))
        for name, value in kwargs.items():
            port, mode = self._input(value, name)
            inputs.append(InputSpec(name, port.ref, mode))

        domains = {self.ports[item.port].domain for item in inputs}
        if len(domains) != 1:
            raise CompileError(
                "RayModule inputs belong to different Domains; use explicit "
                "broadcast/reduce or an aligned relation"
            )
        execution_domain = next(iter(domains))
        driving = self._driving_index(module.driving_input, inputs)
        if inputs[driving].mode is InputMode.OPTIONAL:
            raise CompileError("driving input cannot be optional")

        call = CallRef(self.next_call)
        self.next_call += 1
        kernel = KernelSpec(
            module.udf,
            module.init_args,
            tuple(module.init_kwargs.items()),
        )
        self.calls[call] = CallSpec(
            call,
            kernel,
            execution_domain,
            tuple(inputs),
            driving,
        )
        self.call_options[call] = tuple(module.options.items())

        outputs = tuple(
            self._new_port(
                execution_domain,
                CallOutputOrigin(call, output_index),
            )
            for output_index in range(module.num_outputs)
        )
        public = tuple(self.public(ref) for ref in outputs)
        return public[0] if len(public) == 1 else public

    def expand(self, ports: tuple[Port, ...]) -> tuple[Port, ...]:
        """创建 child Domain；aligned 输入共享同一 Shape。"""

        specs = self._ports(ports, "expand")
        if len({spec.domain for spec in specs}) != 1:
            raise CompileError("aligned expand inputs must share one parent Domain")
        if len({port.ref for port in ports}) != len(ports):
            raise CompileError("aligned expand inputs must be distinct Ports")

        signature = ("expand", tuple(port.ref for port in ports))
        existing = self._view_intern.get(signature)
        if existing is not None:
            return tuple(self.public(ref) for ref in existing)

        producers = tuple(spec.origin for spec in specs)
        if not all(isinstance(origin, CallOutputOrigin) for origin in producers):
            raise CompileError(
                "expand inputs must be outputs of one producer Call"
            )
        if len(ports) > 1:
            calls = {origin.call for origin in producers}
            if len(calls) != 1:
                raise CompileError(
                    "aligned expand inputs must be outputs of one producer Call"
                )

        already_expanded = {
            spec.origin.group_port
            for spec in self.ports.values()
            if isinstance(spec.origin, ExpandOrigin)
        }
        duplicate_sources = already_expanded.intersection(port.ref for port in ports)
        if duplicate_sources:
            raise CompileError(
                "a group Port already belongs to another expand relation; "
                "reuse the existing expanded Port or declare alignment once"
            )

        parent = specs[0].domain
        child = DomainRef(self.next_domain)
        self.next_domain += 1
        self.domains[child] = DomainSpec(
            child,
            parent,
            debug_name=f"expand_{child.value}",
        )
        outputs = tuple(
            self._new_port(child, ExpandOrigin(port.ref)) for port in ports
        )
        self._view_intern[signature] = outputs
        return tuple(self.public(ref) for ref in outputs)

    def reduce(
        self,
        ports: tuple[Port, ...],
        members: Port | None,
    ) -> tuple[Port, ...]:
        """沿 Domain parent 回收一级，并显式记录成员 Port。"""

        specs = self._ports(ports, "reduce")
        if len({spec.domain for spec in specs}) != 1:
            raise CompileError("aligned reduce inputs must share one child Domain")
        child = specs[0].domain
        domain = self.domains[child]
        if domain.parent is None:
            raise CompileError("reduce input is already in a root Domain")

        member_port = ports[0] if members is None else members
        member_spec = self.spec(member_port, "reduce members")
        if member_spec.domain != child:
            raise CompileError("reduce members must share the values' child Domain")

        signature = (
            "reduce",
            tuple(port.ref for port in ports),
            member_port.ref,
        )
        existing = self._view_intern.get(signature)
        if existing is not None:
            return tuple(self.public(ref) for ref in existing)

        outputs = tuple(
            self._new_port(
                domain.parent,
                GroupOrigin(port.ref, member_port.ref),
            )
            for port in ports
        )
        self._view_intern[signature] = outputs
        return tuple(self.public(ref) for ref in outputs)

    def broadcast(self, source: Port, like: Port) -> Port:
        """把祖先值投影到后代 Domain，不复制 payload。"""

        source_spec = self.spec(source, "broadcast")
        target_spec = self.spec(like, "broadcast like")
        if source_spec.domain == target_spec.domain:
            return source
        if not self._is_ancestor(source_spec.domain, target_spec.domain):
            raise CompileError("broadcast source Domain must be a target ancestor")
        signature = ("broadcast", source.ref, target_spec.domain)
        existing = self._view_intern.get(signature)
        if existing is not None:
            return self.public(existing[0])
        output = self._new_port(
            target_spec.domain,
            BroadcastOrigin(source.ref),
        )
        self._view_intern[signature] = (output,)
        return self.public(output)

    def filter(self, source: Port, mask: Port) -> Port:
        """保持 Domain 不变，仅由布尔 mask 改变成员状态。"""

        source_spec = self.spec(source, "filter")
        mask_spec = self.spec(mask, "filter mask")
        if source_spec.domain != mask_spec.domain:
            raise CompileError("filter source and mask must share one Domain")
        signature = ("filter", source.ref, mask.ref)
        existing = self._view_intern.get(signature)
        if existing is not None:
            return self.public(existing[0])
        output = self._new_port(
            source_spec.domain,
            FilterOrigin(source.ref, mask.ref),
        )
        self._view_intern[signature] = (output,)
        return self.public(output)

    def normalize_outputs(self, value: Any) -> object:
        """把公开 Port 输出树转换为只含 PortRef 的不可变 tuple 树。"""

        if isinstance(value, Port):
            self.spec(value, "Pipeline output")
            return value.ref
        if isinstance(value, tuple) and value:
            return tuple(self.normalize_outputs(item) for item in value)
        raise CompileError(
            "Pipeline.forward must return a Port or a non-empty nested tuple of Ports"
        )

    def build(self, output_tree: object) -> CompiledProgram:
        """冻结索引，并把结构推导的 Worker 布局写入执行计划。"""

        consumers: dict[PortRef, list[Any]] = {}
        outputs_by_call: dict[CallRef, list[PortRef]] = {}
        shape_reporters: dict[DomainRef, list[PortRef]] = {}

        for call, spec in self.calls.items():
            for index, input_ in enumerate(spec.inputs):
                consumers.setdefault(input_.port, []).append(
                    CallConsumer(call, index)
                )

        for port, spec in self.ports.items():
            origin = spec.origin
            if isinstance(origin, CallOutputOrigin):
                outputs_by_call.setdefault(origin.call, []).append(port)
            elif isinstance(origin, ExpandOrigin):
                consumers.setdefault(origin.group_port, []).append(
                    ViewConsumer(port, "group")
                )
                shape_reporters.setdefault(spec.domain, []).append(
                    origin.group_port
                )
            elif isinstance(origin, GroupOrigin):
                consumers.setdefault(origin.value_port, []).append(
                    ViewConsumer(port, "value")
                )
                consumers.setdefault(origin.members_port, []).append(
                    ViewConsumer(port, "members")
                )
            elif isinstance(origin, BroadcastOrigin):
                consumers.setdefault(origin.source_port, []).append(
                    ViewConsumer(port, "source")
                )
            elif isinstance(origin, FilterOrigin):
                consumers.setdefault(origin.source_port, []).append(
                    ViewConsumer(port, "source")
                )
                consumers.setdefault(origin.mask_port, []).append(
                    ViewConsumer(port, "mask")
                )

        # Filter 只消费 control manifest，不允许 Arena 为调度读取业务 payload。
        # demand 从 mask Port 反向穿过透明结构视图：broadcast 复制一个标量
        # control；expand 则把 group 中的每个 bool 映射到对应 child Item。
        control_ports = {
            spec.origin.mask_port
            for spec in self.ports.values()
            if isinstance(spec.origin, FilterOrigin)
        }
        pending = list(control_ports)
        while pending:
            port = pending.pop()
            for upstream in self._control_predecessors(self.ports[port].origin):
                if upstream not in control_ports:
                    control_ports.add(upstream)
                    pending.append(upstream)

        pools = {}
        call_to_pool = {}
        for call in self.calls:
            pool = PoolRef(call.value)
            pools[pool] = PoolSpec(pool, call, self.call_options[call])
            call_to_pool[call] = pool

        program = Program(
            calls=freeze_mapping(self.calls),
            ports=freeze_mapping(self.ports),
            domains=freeze_mapping(self.domains),
            source_ports=self.source_ports,
            output_tree=output_tree,
            consumers_by_port=freeze_mapping(
                {key: tuple(value) for key, value in consumers.items()}
            ),
            outputs_by_call=freeze_mapping(
                {
                    key: tuple(sorted(value, key=lambda item: item.value))
                    for key, value in outputs_by_call.items()
                }
            ),
            shape_reporters_by_domain=freeze_mapping(
                {
                    key: tuple(dict.fromkeys(value))
                    for key, value in shape_reporters.items()
                }
            ),
            control_ports=frozenset(control_ports),
        )
        # Expand/Filter 的结构语义只在编译期解释一次；执行器与 Worker
        # 只消费 OutputLayout，避免跨层读取 PortOrigin。
        layouts_by_call = {}
        for call, outputs in program.outputs_by_call.items():
            layouts = []
            for output in outputs:
                expanded = tuple(
                    port
                    for port, spec in program.ports.items()
                    if isinstance(spec.origin, ExpandOrigin)
                    and spec.origin.group_port == output
                )
                demanded = frozenset(
                    port
                    for port in (output, *expanded)
                    if port in program.control_ports
                )
                layouts.append(OutputLayout(output, expanded, demanded))
            layouts_by_call[call] = tuple(layouts)
        execution = ExecutionPlan(
            freeze_mapping(pools),
            freeze_mapping(call_to_pool),
            freeze_mapping(layouts_by_call),
        )
        return CompiledProgram(program, execution)

    @staticmethod
    def _control_predecessors(origin: PortOrigin) -> tuple[PortRef, ...]:
        """穷尽解释一个 Port 的 control demand 如何反向传播。"""

        match origin:
            case SourceOrigin() | CallOutputOrigin():
                return ()
            case BroadcastOrigin(source_port=source):
                return (source,)
            case ExpandOrigin(group_port=group):
                return (group,)
            case FilterOrigin(source_port=source):
                return (source,)
            case GroupOrigin():
                raise CompileError(
                    "group-valued Port cannot be used as a scalar filter mask"
                )
            case _:
                assert_never(origin)

    def _new_port(self, domain: DomainRef, origin: Any) -> PortRef:
        ref = PortRef(self.next_port)
        self.next_port += 1
        self.ports[ref] = PortSpec(ref, domain, origin)
        return ref

    def _ports(self, ports: tuple[Port, ...], label: str) -> tuple[PortSpec, ...]:
        if not ports:
            raise CompileError(f"{label} requires at least one Port")
        return tuple(self.spec(port, label) for port in ports)

    def _input(self, value: Any, name: str) -> tuple[Port, InputMode]:
        if isinstance(value, OptionalPort):
            self.spec(value.port, f"input {name}")
            return value.port, InputMode.OPTIONAL
        if isinstance(value, Port):
            self.spec(value, f"input {name}")
            return value, InputMode.REQUIRED
        raise CompileError(f"RayModule input {name!r} must be a Port")

    @staticmethod
    def _driving_index(
        configured: int | str,
        inputs: list[InputSpec],
    ) -> int:
        if isinstance(configured, int):
            if not 0 <= configured < len(inputs):
                raise CompileError("driving input index is out of range")
            return configured
        matches = [index for index, item in enumerate(inputs) if item.name == configured]
        if len(matches) != 1:
            raise CompileError(f"unknown or ambiguous driving input: {configured!r}")
        return matches[0]

    def _is_ancestor(self, ancestor: DomainRef, child: DomainRef) -> bool:
        cursor: DomainRef | None = child
        while cursor is not None:
            if cursor == ancestor:
                return True
            cursor = self.domains[cursor].parent
        return False


def active_builder() -> _ProgramBuilder:
    """返回当前符号追踪 builder；离开 ``forward`` 时拒绝 Port 操作。"""

    builder = _ACTIVE_TRACE.get()
    if builder is None:
        raise CompileError("Port operations are only valid in Pipeline.forward")
    return builder


__all__ = [
    "OptionalPort",
    "Pipeline",
    "Port",
    "RayModule",
    "active_builder",
    "function",
]
