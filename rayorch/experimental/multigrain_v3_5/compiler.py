"""v3.5 的固定、纯函数式编译流水线。

流水线刻意不是可插拔 PassManager：verify → analyze → optional canonicalize
→ lower → verify。关闭 canonicalization 时得到 correctness baseline。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, assert_never

from .analysis import CallUse, DerivedFacts, PrimitiveUse, analyze
from .logical import CallOutputOrigin, LogicalProgram, SourceOrigin
from .model import CallRef, CompileError, DomainRef, PoolRef, PortRef
from .plan import (
    BroadcastRoute,
    BroadcastRule,
    CallInputRoute,
    CanonicalRewrite,
    CompiledProgram,
    ExpansionRule,
    ExplainPlan,
    FilterRoute,
    FilterRule,
    GroupRoute,
    GroupRule,
    PoolSpec,
    PortExplanation,
    RuntimePlan,
    RuntimeRoute,
    freeze_mapping,
)
from .protocol import InputLayout, OutputLayout
from .semantics import InputRole, PrimitiveKind, PrimitiveSemantics


@dataclass(frozen=True, slots=True)
class _CanonicalForm:
    broadcast_sources: Mapping[PortRef, PortRef]
    rewrites: tuple[CanonicalRewrite, ...]
    optimized: bool


def compile_logical(
    logical: LogicalProgram,
    call_options: Mapping[CallRef, tuple[tuple[str, object], ...]],
    *,
    optimize: bool = True,
) -> CompiledProgram:
    """编译一个已经冻结的逻辑图。"""

    verify_logical(logical)
    facts = analyze(logical)
    canonical = _canonicalize(logical, facts, enabled=optimize)
    runtime, explain = _lower(logical, facts, canonical, call_options)
    verify_runtime_plan(logical, facts, runtime)
    return CompiledProgram(logical, facts, runtime, explain)


def verify_logical(logical: LogicalProgram) -> None:
    """验证引用闭包、Domain 关系和每种 primitive 的局部合同。"""

    if not logical.source_ports:
        raise CompileError("LogicalProgram requires at least one source Port")
    if any(key != value.ref for key, value in logical.calls.items()):
        raise CompileError("Call mapping key does not match CallSpec.ref")
    if any(key != value.ref for key, value in logical.ports.items()):
        raise CompileError("Port mapping key does not match PortSpec.ref")
    if any(key != value.ref for key, value in logical.domains.items()):
        raise CompileError("Domain mapping key does not match DomainSpec.ref")

    roots = [domain for domain, spec in logical.domains.items() if spec.parent is None]
    if len(roots) != 1:
        raise CompileError("LogicalProgram requires exactly one root Domain")
    for domain, spec in logical.domains.items():
        if spec.parent is not None and spec.parent not in logical.domains:
            raise CompileError(f"Domain {domain} has an unknown parent")
        seen: set[DomainRef] = set()
        cursor: DomainRef | None = domain
        while cursor is not None:
            if cursor in seen:
                raise CompileError("Domain parent relation contains a cycle")
            seen.add(cursor)
            cursor = logical.domain(cursor).parent

    semantics = analyze_origins(logical)
    _verify_acyclic(logical, semantics)
    for call, spec in logical.calls.items():
        if spec.execution_domain not in logical.domains:
            raise CompileError(f"Call {call} has an unknown execution Domain")
        keyword_started = False
        keyword_names: set[str] = set()
        for input_ in spec.inputs:
            if input_.port not in logical.ports:
                raise CompileError(f"Call {call} references an unknown input Port")
            if logical.port(input_.port).domain != spec.execution_domain:
                raise CompileError(f"Call {call} input Domain mismatch")
            if input_.keyword:
                keyword_started = True
                if not input_.name:
                    raise CompileError("keyword input name must be non-empty")
                if input_.name in keyword_names:
                    raise CompileError("keyword input names must be unique")
                keyword_names.add(input_.name)
            elif keyword_started:
                raise CompileError("positional input cannot follow keyword inputs")

    source_indices: list[int] = []
    for port in logical.source_ports:
        if port not in logical.ports:
            raise CompileError("source_ports contains an unknown Port")
        origin = logical.port(port).origin
        if not isinstance(origin, SourceOrigin):
            raise CompileError("source_ports must contain only Source origins")
        source_indices.append(origin.source_index)
    if source_indices != list(range(len(source_indices))):
        raise CompileError("source indices must be contiguous and ordered")
    root = roots[0]

    expanded_sources: set[PortRef] = set()
    for port, spec in logical.ports.items():
        semantic = semantics[port]
        for input_ in semantic.inputs:
            if input_.port not in logical.ports:
                raise CompileError(f"Port {port} references an unknown Port")

        match semantic.kind:
            case PrimitiveKind.SOURCE:
                if spec.domain != root:
                    raise CompileError("source Port must belong to root Domain")
            case PrimitiveKind.CALL_OUTPUT:
                call = semantic.producing_call
                if call not in logical.calls:
                    raise CompileError("CallOutput references an unknown Call")
                if spec.domain != logical.call(call).execution_domain:
                    raise CompileError("CallOutput Domain mismatch")
            case PrimitiveKind.EXPAND:
                group = semantic.inputs[0].port
                if group in expanded_sources:
                    raise CompileError("a group Port cannot drive two Expand relations")
                expanded_sources.add(group)
                if not isinstance(logical.port(group).origin, CallOutputOrigin):
                    raise CompileError("Expand source must be a Call output")
                if logical.domain(spec.domain).parent != logical.port(group).domain:
                    raise CompileError("Expand target Domain must be one level below source")
            case PrimitiveKind.GROUP:
                value, members = (item.port for item in semantic.inputs)
                child = logical.port(value).domain
                if logical.port(members).domain != child:
                    raise CompileError("Group value and members Domains must match")
                if logical.domain(child).parent != spec.domain:
                    raise CompileError("Group target must be the child Domain parent")
            case PrimitiveKind.BROADCAST:
                source = semantic.inputs[0].port
                if not _is_ancestor(logical, logical.port(source).domain, spec.domain):
                    raise CompileError("Broadcast source Domain must be a target ancestor")
                if logical.port(source).domain == spec.domain:
                    raise CompileError("same-Domain Broadcast must be canonicalized by builder")
            case PrimitiveKind.FILTER:
                source, mask = (item.port for item in semantic.inputs)
                if logical.port(source).domain != spec.domain:
                    raise CompileError("Filter source Domain mismatch")
                if logical.port(mask).domain != spec.domain:
                    raise CompileError("Filter mask Domain mismatch")
            case _:
                assert_never(semantic.kind)

    outputs_by_call: dict[CallRef, list[int]] = {}
    for semantic in semantics.values():
        if semantic.kind is PrimitiveKind.CALL_OUTPUT:
            assert semantic.producing_call is not None
            assert semantic.output_index is not None
            outputs_by_call.setdefault(semantic.producing_call, []).append(
                semantic.output_index
            )
    for call in logical.calls:
        indices = sorted(outputs_by_call.get(call, ()))
        if indices != list(range(len(indices))) or not indices:
            raise CompileError(f"Call {call} outputs must be non-empty and contiguous")

    for output in _tree_ports(logical.output_tree):
        if output not in logical.ports:
            raise CompileError("output_tree contains an unknown Port")


def analyze_origins(logical: LogicalProgram):
    """只构造穷尽 primitive 描述，供 verifier 在 full analysis 前使用。"""

    from .semantics import describe_origin

    return {
        port: describe_origin(spec.origin) for port, spec in logical.ports.items()
    }


def _verify_acyclic(
    logical: LogicalProgram,
    semantics: Mapping[PortRef, PrimitiveSemantics],
) -> None:
    """把 Call 输入和 structural inputs 统一视为 Port dependency DAG。"""

    complete: set[PortRef] = set()
    active: set[PortRef] = set()

    def visit(port: PortRef) -> None:
        if port in complete:
            return
        if port in active:
            raise CompileError("logical Port dependency cycle")
        active.add(port)
        semantic = semantics[port]
        dependencies = [input_.port for input_ in semantic.inputs]
        if semantic.kind is PrimitiveKind.CALL_OUTPUT:
            assert semantic.producing_call is not None
            if semantic.producing_call not in logical.calls:
                raise CompileError("CallOutput references an unknown Call")
            dependencies.extend(
                input_.port for input_ in logical.call(semantic.producing_call).inputs
            )
        for dependency in dependencies:
            if dependency not in semantics:
                raise CompileError(f"Port {port} references an unknown Port")
            visit(dependency)
        active.remove(port)
        complete.add(port)

    for port in logical.ports:
        visit(port)


def _canonicalize(
    logical: LogicalProgram,
    facts: DerivedFacts,
    *,
    enabled: bool,
) -> _CanonicalForm:
    """只折叠透明 Broadcast 链；不删除 Call，也不融合 Filter。"""

    direct: dict[PortRef, PortRef] = {}
    rewrites: list[CanonicalRewrite] = []
    visiting: set[PortRef] = set()

    def resolve(port: PortRef) -> PortRef:
        known = direct.get(port)
        if known is not None:
            return known
        semantic = facts.semantics_by_port[port]
        if semantic.kind is not PrimitiveKind.BROADCAST:
            direct[port] = port
            return port
        source = semantic.inputs[0].port
        if not enabled:
            direct[port] = source
            return source
        if port in visiting:
            raise CompileError("Broadcast relation contains a cycle")
        visiting.add(port)
        effective = resolve(source)
        visiting.remove(port)
        direct[port] = effective
        if effective != source:
            rewrites.append(
                CanonicalRewrite(
                    "collapse-broadcast-chain",
                    port,
                    (source,),
                    (effective,),
                )
            )
        return effective

    for port, semantic in facts.semantics_by_port.items():
        if semantic.kind is PrimitiveKind.BROADCAST:
            resolve(port)

    return _CanonicalForm(freeze_mapping(direct), tuple(rewrites), enabled)


def _lower(
    logical: LogicalProgram,
    facts: DerivedFacts,
    canonical: _CanonicalForm,
    call_options: Mapping[CallRef, tuple[tuple[str, object], ...]],
) -> tuple[RuntimePlan, ExplainPlan]:
    routes: dict[PortRef, list[RuntimeRoute]] = {}
    expansions: dict[PortRef, list[ExpansionRule]] = {}
    filters: dict[PortRef, FilterRule] = {}
    groups: dict[PortRef, GroupRule] = {}
    broadcasts: dict[PortRef, BroadcastRule] = {}
    groups_by_domain: dict[DomainRef, list[PortRef]] = {}
    broadcasts_by_domain: dict[DomainRef, list[PortRef]] = {}

    def add_route(source: PortRef, route: RuntimeRoute) -> None:
        bucket = routes.setdefault(source, [])
        if route not in bucket:
            bucket.append(route)

    # LogicalUse 是 analysis 的统一反向索引。lowering 必须显式消费每个
    # dependency role；即使 Expand/Broadcast 在这里是 pass，也不能靠遗漏实现。
    for source, uses in facts.consumers_by_port.items():
        for use in uses:
            if isinstance(use, CallUse):
                add_route(source, CallInputRoute(use.call, use.input_index))
                continue
            if not isinstance(use, PrimitiveUse):  # pragma: no cover
                raise CompileError(f"unsupported LogicalUse: {use!r}")
            match use.role:
                case InputRole.EXPAND_GROUP:
                    # Expand 由 Worker expanded layout 与 commit_success 直达，
                    # 不创建 publication receipt route。
                    pass
                case InputRole.GROUP_VALUE | InputRole.GROUP_MEMBERS:
                    add_route(source, GroupRoute(use.port))
                case InputRole.BROADCAST_SOURCE:
                    # canonicalization 可能改写 source，下面按最终 rule 加 route。
                    pass
                case InputRole.FILTER_SOURCE | InputRole.FILTER_MASK:
                    add_route(source, FilterRoute(use.port))
                case _:
                    assert_never(use.role)

    for port, semantic in facts.semantics_by_port.items():
        match semantic.kind:
            case PrimitiveKind.SOURCE | PrimitiveKind.CALL_OUTPUT:
                pass
            case PrimitiveKind.EXPAND:
                source = semantic.inputs[0].port
                expansions.setdefault(source, []).append(
                    ExpansionRule(
                        port,
                        logical.port(port).domain,
                        port in facts.control_ports,
                    )
                )
            case PrimitiveKind.FILTER:
                source, mask = (item.port for item in semantic.inputs)
                filters[port] = FilterRule(
                    source,
                    mask,
                    port in facts.control_ports,
                )
            case PrimitiveKind.GROUP:
                value, members = (item.port for item in semantic.inputs)
                child_domain = logical.port(value).domain
                groups[port] = GroupRule(
                    value,
                    members,
                    child_domain,
                    facts.group_depth_by_port[value],
                )
                groups_by_domain.setdefault(child_domain, []).append(port)
            case PrimitiveKind.BROADCAST:
                immediate = semantic.inputs[0].port
                source = canonical.broadcast_sources.get(port, immediate)
                rule = BroadcastRule(
                    source,
                    logical.port(source).domain,
                    logical.port(port).domain,
                    port in facts.control_ports,
                )
                broadcasts[port] = rule
                broadcasts_by_domain.setdefault(rule.target_domain, []).append(port)
                add_route(source, BroadcastRoute(port))
            case _:
                assert_never(semantic.kind)

    pools = {}
    call_to_pool = {}
    for call in logical.calls:
        pool = PoolRef(call.value)
        pools[pool] = PoolSpec(pool, call, tuple(call_options.get(call, ())))
        call_to_pool[call] = pool

    layouts = {}
    input_layouts = {}
    for call, outputs in facts.outputs_by_call.items():
        spec = logical.call(call)
        positional_count = sum(not input_.keyword for input_ in spec.inputs)
        input_layouts[call] = InputLayout(
            positional_count,
            tuple(input_.name for input_ in spec.inputs if input_.keyword),
        )
        call_layouts = []
        for output in outputs:
            expanded = tuple(rule.port for rule in expansions.get(output, ()))
            demanded = frozenset(
                port
                for port in (output, *expanded)
                if port in facts.control_ports
            )
            call_layouts.append(OutputLayout(output, expanded, demanded))
        layouts[call] = tuple(call_layouts)

    runtime = RuntimePlan(
        calls=logical.calls,
        domains=logical.domains,
        port_domains=freeze_mapping(
            {port: spec.domain for port, spec in logical.ports.items()}
        ),
        source_ports=logical.source_ports,
        output_tree=logical.output_tree,
        routes_by_port=freeze_mapping(
            {port: tuple(items) for port, items in routes.items()}
        ),
        outputs_by_call=facts.outputs_by_call,
        expansions_by_source=freeze_mapping(
            {port: tuple(items) for port, items in expansions.items()}
        ),
        shape_reporters_by_domain=facts.shape_reporters_by_domain,
        control_ports=facts.control_ports,
        filter_rules=freeze_mapping(filters),
        group_rules=freeze_mapping(groups),
        broadcast_rules=freeze_mapping(broadcasts),
        groups_by_child_domain=freeze_mapping(
            {domain: tuple(ports) for domain, ports in groups_by_domain.items()}
        ),
        broadcasts_by_target_domain=freeze_mapping(
            {domain: tuple(ports) for domain, ports in broadcasts_by_domain.items()}
        ),
        pools=freeze_mapping(pools),
        call_to_pool=freeze_mapping(call_to_pool),
        output_layouts_by_call=freeze_mapping(layouts),
        input_layouts_by_call=freeze_mapping(input_layouts),
    )

    rewrites_by_port: dict[PortRef, list[CanonicalRewrite]] = {}
    for rewrite in canonical.rewrites:
        rewrites_by_port.setdefault(rewrite.port, []).append(rewrite)
    explanations = {}
    for port, semantic in facts.semantics_by_port.items():
        rule = _physical_rule(port, semantic.kind, runtime)
        explanations[port] = PortExplanation(
            port,
            semantic.kind.name.lower(),
            logical.port(port).domain,
            tuple(input_.port for input_ in semantic.inputs),
            rule,
            port in facts.control_ports,
            tuple(rewrites_by_port.get(port, ())),
        )
    explain = ExplainPlan(
        canonical.optimized,
        freeze_mapping(explanations),
        canonical.rewrites,
    )
    return runtime, explain


def verify_runtime_plan(
    logical: LogicalProgram,
    facts: DerivedFacts,
    runtime: RuntimePlan,
) -> None:
    """保证 lowering 没有漏掉任何 logical primitive 或 Call ABI。"""

    if set(runtime.calls) != set(logical.calls):
        raise CompileError("RuntimePlan Call set is incomplete")
    if set(runtime.port_domains) != set(logical.ports):
        raise CompileError("RuntimePlan Port domain table is incomplete")
    if set(runtime.outputs_by_call) != set(logical.calls):
        raise CompileError("RuntimePlan Call output table is incomplete")
    if len(runtime.pools) != len(logical.calls):
        raise CompileError("RuntimePlan requires exactly one pool per Call")
    if set(runtime.call_to_pool) != set(logical.calls):
        raise CompileError("RuntimePlan pool mapping is incomplete")
    if set(runtime.output_layouts_by_call) != set(logical.calls):
        raise CompileError("RuntimePlan Worker layouts are incomplete")
    if set(runtime.input_layouts_by_call) != set(logical.calls):
        raise CompileError("RuntimePlan Worker input layouts are incomplete")

    for port, semantic in facts.semantics_by_port.items():
        match semantic.kind:
            case PrimitiveKind.SOURCE | PrimitiveKind.CALL_OUTPUT:
                pass
            case PrimitiveKind.EXPAND:
                source = semantic.inputs[0].port
                if not any(
                    rule.port == port
                    for rule in runtime.expansions_by_source.get(source, ())
                ):
                    raise CompileError("RuntimePlan omitted an Expand rule")
            case PrimitiveKind.FILTER:
                if port not in runtime.filter_rules:
                    raise CompileError("RuntimePlan omitted a Filter rule")
                rule = runtime.filter_rules[port]
                route = FilterRoute(port)
                if route not in runtime.routes_by_port.get(rule.source_port, ()):
                    raise CompileError("RuntimePlan omitted a Filter source route")
                if route not in runtime.routes_by_port.get(rule.mask_port, ()):
                    raise CompileError("RuntimePlan omitted a Filter mask route")
            case PrimitiveKind.GROUP:
                if port not in runtime.group_rules:
                    raise CompileError("RuntimePlan omitted a Group rule")
                rule = runtime.group_rules[port]
                route = GroupRoute(port)
                if route not in runtime.routes_by_port.get(rule.value_port, ()):
                    raise CompileError("RuntimePlan omitted a Group value route")
                if route not in runtime.routes_by_port.get(rule.members_port, ()):
                    raise CompileError("RuntimePlan omitted a Group members route")
            case PrimitiveKind.BROADCAST:
                if port not in runtime.broadcast_rules:
                    raise CompileError("RuntimePlan omitted a Broadcast rule")
                rule = runtime.broadcast_rules[port]
                if BroadcastRoute(port) not in runtime.routes_by_port.get(
                    rule.source_port, ()
                ):
                    raise CompileError("RuntimePlan omitted a Broadcast source route")
            case _:
                assert_never(semantic.kind)

    for call, spec in logical.calls.items():
        if runtime.input_layouts_by_call[call].input_count != len(spec.inputs):
            raise CompileError("RuntimePlan Worker input layout arity mismatch")
        for index, input_ in enumerate(spec.inputs):
            if CallInputRoute(call, index) not in runtime.routes_by_port.get(
                input_.port, ()
            ):
                raise CompileError("RuntimePlan omitted a Call input route")


def _physical_rule(port: PortRef, kind: PrimitiveKind, runtime: RuntimePlan) -> str:
    match kind:
        case PrimitiveKind.SOURCE:
            return "source-admission"
        case PrimitiveKind.CALL_OUTPUT:
            return "worker-output"
        case PrimitiveKind.EXPAND:
            return "expanded-output-layout"
        case PrimitiveKind.FILTER:
            return "arena-filter"
        case PrimitiveKind.GROUP:
            return "arena-group"
        case PrimitiveKind.BROADCAST:
            source = runtime.broadcast_rules[port].source_port
            return f"arena-broadcast(p{source.value})"
        case _:
            assert_never(kind)


def _is_ancestor(
    logical: LogicalProgram,
    ancestor: DomainRef,
    descendant: DomainRef,
) -> bool:
    cursor: DomainRef | None = descendant
    while cursor is not None:
        if cursor == ancestor:
            return True
        cursor = logical.domain(cursor).parent
    return False


def _tree_ports(tree: object) -> tuple[PortRef, ...]:
    if isinstance(tree, PortRef):
        return (tree,)
    if isinstance(tree, tuple) and tree:
        return tuple(port for child in tree for port in _tree_ports(child))
    raise CompileError("output_tree must be a PortRef or non-empty tuple tree")


__all__ = [
    "compile_logical",
    "verify_logical",
    "verify_runtime_plan",
]
