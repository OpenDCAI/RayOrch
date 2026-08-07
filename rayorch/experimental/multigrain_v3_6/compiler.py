"""v3.6 的固定、纯函数式编译流水线。

流水线刻意不是可插拔 PassManager：verify → analyze → optional canonicalize
→ lower → verify。关闭 canonicalization 时得到 correctness baseline。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, assert_never, cast

from .analysis import CallUse, ProgramAnalysis, PrimitiveUse, analyze
from .logical import CallSpec, LogicalProgram
from .model import CallRef, CompileError, DomainRef, PortRef
from .plan import (
    BroadcastEffect,
    CallInputEffect,
    CanonicalRewrite,
    CompiledProgram,
    ExpandEffect,
    ProgramExplanation,
    FilterEffect,
    ReduceEffect,
    ItemEffect,
    ActorPoolSpec,
    PortExplanation,
    RuntimePlan,
    StructuralEffect,
    freeze_mapping,
)
from .protocol import CallInputLayout, CallOutputLayout
from .recovery import DEFAULT_RECOVERY_POLICY
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
    analysis = analyze(logical)
    canonical = _canonicalize(logical, analysis, enabled=optimize)
    plan, explanation = _lower(logical, analysis, canonical, call_options)
    verify_runtime_plan(logical, analysis, plan)
    return CompiledProgram(logical, analysis, plan, explanation)


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
        for input_ in spec.ordered_inputs:
            if input_.port not in logical.ports:
                raise CompileError(f"Call {call} references an unknown input Port")
            if logical.port(input_.port).domain != spec.execution_domain:
                raise CompileError(f"Call {call} input Domain mismatch")

    source_indices: list[int] = []
    for port in logical.source_ports:
        if port not in logical.ports:
            raise CompileError("source_ports contains an unknown Port")
        semantic = semantics[port]
        if semantic.kind is not PrimitiveKind.SOURCE:
            raise CompileError("source_ports must contain only Source origins")
        assert semantic.source_index is not None
        source_indices.append(semantic.source_index)
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
                assert call is not None
                if call not in logical.calls:
                    raise CompileError("CallOutput references an unknown Call")
                if spec.domain != logical.call(call).execution_domain:
                    raise CompileError("CallOutput Domain mismatch")
            case PrimitiveKind.EXPAND:
                group = semantic.inputs[0].port
                if group in expanded_sources:
                    raise CompileError("a group Port cannot drive two Expand relations")
                expanded_sources.add(group)
                if semantics[group].kind is not PrimitiveKind.CALL_OUTPUT:
                    raise CompileError("Expand source must be a Call output")
                if logical.domain(spec.domain).parent != logical.port(group).domain:
                    raise CompileError("Expand target Domain must be one level below source")
            case PrimitiveKind.REDUCE:
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
                input_.port
                for input_ in logical.call(semantic.producing_call).ordered_inputs
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
    analysis: ProgramAnalysis,
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
        semantic = analysis.semantics_by_port[port]
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

    for port, semantic in analysis.semantics_by_port.items():
        if semantic.kind is PrimitiveKind.BROADCAST:
            resolve(port)

    return _CanonicalForm(freeze_mapping(direct), tuple(rewrites), enabled)


def _lower(
    logical: LogicalProgram,
    analysis: ProgramAnalysis,
    canonical: _CanonicalForm,
    call_options: Mapping[CallRef, tuple[tuple[str, object], ...]],
) -> tuple[RuntimePlan, ProgramExplanation]:
    item_effects_by_source: dict[PortRef, list[ItemEffect]] = {}
    expansions: dict[PortRef, list[ExpandEffect]] = {}
    structural_effects_by_target: dict[PortRef, StructuralEffect] = {}
    reductions_by_domain: dict[DomainRef, list[ReduceEffect]] = {}
    broadcasts_by_domain: dict[DomainRef, list[BroadcastEffect]] = {}

    def add_item_effect(source: PortRef, effect: ItemEffect) -> None:
        bucket = item_effects_by_source.setdefault(source, [])
        if effect not in bucket:
            bucket.append(effect)

    # 先构造每个 structural Port 唯一的不可变 Effect。后续各触发索引只持有
    # 这些对象本身，不再用 target-only Route 回查第二份 Rule。
    for port, semantic in analysis.semantics_by_port.items():
        match semantic.kind:
            case PrimitiveKind.SOURCE | PrimitiveKind.CALL_OUTPUT:
                pass
            case PrimitiveKind.EXPAND:
                source = semantic.inputs[0].port
                expansions.setdefault(source, []).append(
                    ExpandEffect(
                        port,
                        logical.port(port).domain,
                        port in analysis.control_ports,
                    )
                )
            case PrimitiveKind.FILTER:
                source, mask = (item.port for item in semantic.inputs)
                structural_effects_by_target[port] = FilterEffect(
                    port,
                    source,
                    mask,
                    port in analysis.control_ports,
                )
            case PrimitiveKind.REDUCE:
                value, members = (item.port for item in semantic.inputs)
                child_domain = logical.port(value).domain
                effect = ReduceEffect(
                    port,
                    value,
                    members,
                    child_domain,
                    analysis.group_depth_by_port[value],
                )
                structural_effects_by_target[port] = effect
                reductions_by_domain.setdefault(child_domain, []).append(effect)
            case PrimitiveKind.BROADCAST:
                immediate = semantic.inputs[0].port
                source = canonical.broadcast_sources.get(port, immediate)
                effect = BroadcastEffect(
                    port,
                    source,
                    logical.port(source).domain,
                    logical.port(port).domain,
                    port in analysis.control_ports,
                )
                structural_effects_by_target[port] = effect
                broadcasts_by_domain.setdefault(effect.target_domain, []).append(
                    effect
                )
            case _:
                assert_never(semantic.kind)

    # LogicalUse 是 analysis 的统一反向索引。lowering 必须显式消费每个
    # dependency role；即使 Expand/Broadcast 在这里是 pass，也不能靠遗漏实现。
    for source, uses in analysis.consumers_by_port.items():
        for use in uses:
            if isinstance(use, CallUse):
                add_item_effect(source, CallInputEffect(use.call, use.input_index))
                continue
            if not isinstance(use, PrimitiveUse):  # pragma: no cover
                raise CompileError(f"unsupported LogicalUse: {use!r}")
            match use.role:
                case InputRole.EXPAND_GROUP:
                    # Expand 由 Worker expanded layout 与 commit_success 直达，
                    # 不创建 Item publication effect。
                    pass
                case InputRole.REDUCE_VALUE | InputRole.REDUCE_MEMBERS:
                    effect = structural_effects_by_target.get(use.port)
                    if not isinstance(effect, ReduceEffect):  # pragma: no cover
                        raise CompileError("Reduce use has no matching ReduceEffect")
                    add_item_effect(source, effect)
                case InputRole.BROADCAST_SOURCE:
                    # canonicalization 可能改写 source；在下面按最终 source 建索引。
                    pass
                case InputRole.FILTER_SOURCE | InputRole.FILTER_MASK:
                    effect = structural_effects_by_target.get(use.port)
                    if not isinstance(effect, FilterEffect):  # pragma: no cover
                        raise CompileError("Filter use has no matching FilterEffect")
                    add_item_effect(source, effect)
                case _:
                    assert_never(use.role)

    for effect in structural_effects_by_target.values():
        if isinstance(effect, BroadcastEffect):
            add_item_effect(effect.source_port, effect)

    actor_pools_by_call = {}
    for call in logical.calls:
        actor_pools_by_call[call] = _compile_pool_spec(
            call_options.get(call, ()),
        )

    layouts = {}
    input_layouts = {}
    for call, outputs in analysis.outputs_by_call.items():
        spec = logical.call(call)
        input_layouts[call] = _compile_input_layout(spec)
        call_layouts = []
        for output in outputs:
            expanded = tuple(rule.port for rule in expansions.get(output, ()))
            demanded = frozenset(
                port
                for port in (output, *expanded)
                if port in analysis.control_ports
            )
            call_layouts.append(CallOutputLayout(output, expanded, demanded))
        layouts[call] = tuple(call_layouts)

    plan = RuntimePlan(
        calls=logical.calls,
        domains=logical.domains,
        port_domains=freeze_mapping(
            {port: spec.domain for port, spec in logical.ports.items()}
        ),
        source_ports=logical.source_ports,
        output_tree=logical.output_tree,
        item_effects_by_source=freeze_mapping(
            {port: tuple(items) for port, items in item_effects_by_source.items()}
        ),
        outputs_by_call=analysis.outputs_by_call,
        expand_effects_by_source=freeze_mapping(
            {port: tuple(items) for port, items in expansions.items()}
        ),
        expansion_sources_by_domain=analysis.expansion_sources_by_domain,
        control_ports=analysis.control_ports,
        structural_effects_by_target=freeze_mapping(structural_effects_by_target),
        reduce_effects_by_child_domain=freeze_mapping(
            {
                domain: tuple(effects)
                for domain, effects in reductions_by_domain.items()
            }
        ),
        broadcast_effects_by_target_domain=freeze_mapping(
            {
                domain: tuple(effects)
                for domain, effects in broadcasts_by_domain.items()
            }
        ),
        actor_pools_by_call=freeze_mapping(actor_pools_by_call),
        output_layouts_by_call=freeze_mapping(layouts),
        input_layouts_by_call=freeze_mapping(input_layouts),
    )

    rewrites_by_port: dict[PortRef, list[CanonicalRewrite]] = {}
    for rewrite in canonical.rewrites:
        rewrites_by_port.setdefault(rewrite.port, []).append(rewrite)
    explanations = {}
    for port, semantic in analysis.semantics_by_port.items():
        rule = _physical_rule(port, semantic.kind, plan)
        explanations[port] = PortExplanation(
            port,
            semantic.kind.name.lower(),
            logical.port(port).domain,
            tuple(input_.port for input_ in semantic.inputs),
            rule,
            port in analysis.control_ports,
            tuple(rewrites_by_port.get(port, ())),
        )
    explanation = ProgramExplanation(
        canonical.optimized,
        freeze_mapping(explanations),
        canonical.rewrites,
    )
    return plan, explanation


def _compile_pool_spec(
    options: tuple[tuple[str, object], ...],
) -> ActorPoolSpec:
    """Normalize authoring options into one typed physical Call contract."""

    raw = dict(options)
    if len(raw) != len(options):
        raise CompileError("RayModule options must have unique names")
    if "max_retries" in raw:
        raise CompileError(
            "max_retries is ambiguous; use recovery=RecoveryPolicy(...)"
        )

    try:
        return ActorPoolSpec(
            replicas=cast(Any, raw.pop("replicas", 1)),
            batch_size=cast(Any, raw.pop("batch_size", 1)),
            batch_scope=cast(Any, raw.pop("batch_scope", "elastic")),
            recovery=cast(Any, raw.pop("recovery", DEFAULT_RECOVERY_POLICY)),
            ray_options=tuple(raw.items()),
        )
    except (TypeError, ValueError) as error:
        raise CompileError(str(error)) from error


def _compile_input_layout(spec: CallSpec) -> CallInputLayout:
    """Lower one logical args/kwargs arrangement into the sole Worker ABI layout."""

    return CallInputLayout(
        len(spec.args),
        tuple(name for name, _ in spec.kwargs),
    )


def verify_runtime_plan(
    logical: LogicalProgram,
    analysis: ProgramAnalysis,
    plan: RuntimePlan,
) -> None:
    """保证 lowering 没有漏掉任何 logical primitive 或 Call ABI。"""

    if set(plan.calls) != set(logical.calls):
        raise CompileError("RuntimePlan Call set is incomplete")
    if set(plan.port_domains) != set(logical.ports):
        raise CompileError("RuntimePlan Port domain table is incomplete")
    if set(plan.outputs_by_call) != set(logical.calls):
        raise CompileError("RuntimePlan Call output table is incomplete")
    if set(plan.actor_pools_by_call) != set(logical.calls):
        raise CompileError("RuntimePlan requires exactly one pool per Call")
    if set(plan.output_layouts_by_call) != set(logical.calls):
        raise CompileError("RuntimePlan Worker layouts are incomplete")
    if set(plan.input_layouts_by_call) != set(logical.calls):
        raise CompileError("RuntimePlan Worker input layouts are incomplete")

    expected_structural = {
        port
        for port, semantic in analysis.semantics_by_port.items()
        if semantic.kind
        in {PrimitiveKind.FILTER, PrimitiveKind.REDUCE, PrimitiveKind.BROADCAST}
    }
    if set(plan.structural_effects_by_target) != expected_structural:
        raise CompileError("RuntimePlan structural Effect catalog is incomplete")

    def indexed_by_item(port: PortRef, effect: ItemEffect) -> bool:
        # identity check is intentional: indexes must reference the one catalog
        # Effect, not an equal clone that could evolve into a second truth.
        return any(
            indexed is effect
            for indexed in plan.item_effects_by_source.get(port, ())
        )

    for port, semantic in analysis.semantics_by_port.items():
        match semantic.kind:
            case PrimitiveKind.SOURCE | PrimitiveKind.CALL_OUTPUT:
                pass
            case PrimitiveKind.EXPAND:
                source = semantic.inputs[0].port
                if not any(
                    rule.port == port
                    for rule in plan.expand_effects_by_source.get(source, ())
                ):
                    raise CompileError("RuntimePlan omitted an Expand rule")
            case PrimitiveKind.FILTER:
                effect = plan.structural_effects_by_target[port]
                if not isinstance(effect, FilterEffect):
                    raise CompileError("RuntimePlan has the wrong Filter Effect")
                if not indexed_by_item(effect.source_port, effect):
                    raise CompileError("RuntimePlan omitted a Filter source index")
                if not indexed_by_item(effect.mask_port, effect):
                    raise CompileError("RuntimePlan omitted a Filter mask index")
            case PrimitiveKind.REDUCE:
                effect = plan.structural_effects_by_target[port]
                if not isinstance(effect, ReduceEffect):
                    raise CompileError("RuntimePlan has the wrong Reduce Effect")
                if not indexed_by_item(effect.value_port, effect):
                    raise CompileError("RuntimePlan omitted a Reduce value index")
                if not indexed_by_item(effect.members_port, effect):
                    raise CompileError("RuntimePlan omitted a Reduce members index")
                if not any(
                    indexed is effect
                    for indexed in plan.reduce_effects_by_child_domain.get(
                        effect.child_domain, ()
                    )
                ):
                    raise CompileError("RuntimePlan omitted a Reduce domain index")
            case PrimitiveKind.BROADCAST:
                effect = plan.structural_effects_by_target[port]
                if not isinstance(effect, BroadcastEffect):
                    raise CompileError("RuntimePlan has the wrong Broadcast Effect")
                if not indexed_by_item(effect.source_port, effect):
                    raise CompileError("RuntimePlan omitted a Broadcast source index")
                if not any(
                    indexed is effect
                    for indexed in plan.broadcast_effects_by_target_domain.get(
                        effect.target_domain, ()
                    )
                ):
                    raise CompileError("RuntimePlan omitted a Broadcast domain index")
            case _:
                assert_never(semantic.kind)

    for call, spec in logical.calls.items():
        if plan.input_layouts_by_call[call] != _compile_input_layout(spec):
            raise CompileError("RuntimePlan Worker input layout does not match CallSpec")
        for index, input_ in enumerate(spec.ordered_inputs):
            if CallInputEffect(call, index) not in plan.item_effects_by_source.get(
                input_.port, ()
            ):
                raise CompileError("RuntimePlan omitted a Call input Effect")


def _physical_rule(port: PortRef, kind: PrimitiveKind, plan: RuntimePlan) -> str:
    match kind:
        case PrimitiveKind.SOURCE:
            return "source-admission"
        case PrimitiveKind.CALL_OUTPUT:
            return "worker-output"
        case PrimitiveKind.EXPAND:
            return "expanded-output-layout"
        case PrimitiveKind.FILTER:
            return "filter-effect"
        case PrimitiveKind.REDUCE:
            return "reduce-effect"
        case PrimitiveKind.BROADCAST:
            effect = plan.structural_effects_by_target[port]
            if not isinstance(effect, BroadcastEffect):  # pragma: no cover
                raise CompileError("Broadcast Port has no BroadcastEffect")
            source = effect.source_port
            return f"broadcast-effect(p{source.value})"
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
