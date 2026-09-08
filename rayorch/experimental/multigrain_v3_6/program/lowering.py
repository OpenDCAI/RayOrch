"""ProgramAnalysis -> RuntimePlan 的纯 lowering 与可选 canonicalization。

Lowering 先创建每个 structural Port 唯一的 Effect，再让所有触发索引引用同一对象；
随后编译 Call pool、Worker ABI 和 explanation，最后一次性冻结 RuntimePlan。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, assert_never, cast

from ..model import CallRef, CompileError, DomainRef, PortRef
from ..protocol import CallInputLayout, CallOutputLayout
from ..recovery import DEFAULT_RECOVERY_POLICY
from .analysis import CallUse, PrimitiveUse, ProgramAnalysis
from .logical import CallSpec, LogicalProgram, freeze_mapping
from .plan import (
    ActorPoolSpec,
    BroadcastEffect,
    CallInputEffect,
    CanonicalRewrite,
    ExpandEffect,
    FilterEffect,
    ItemEffect,
    PortExplanation,
    ProgramExplanation,
    ReduceEffect,
    RuntimePlan,
    StructuralEffect,
)
from .semantics import InputRole, PrimitiveKind


@dataclass(frozen=True, slots=True)
class _CanonicalForm:
    broadcast_sources: Mapping[PortRef, PortRef]
    rewrites: tuple[CanonicalRewrite, ...]
    optimized: bool


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
    """Lower one verified logical graph into immutable runtime wiring."""

    item_effects_by_source: dict[PortRef, list[ItemEffect]] = {}
    expansions: dict[PortRef, list[ExpandEffect]] = {}
    structural_effects_by_target: dict[PortRef, StructuralEffect] = {}
    reductions_by_domain: dict[DomainRef, list[ReduceEffect]] = {}
    broadcasts_by_domain: dict[DomainRef, list[BroadcastEffect]] = {}

    def add_item_effect(source: PortRef, effect: ItemEffect) -> None:
        bucket = item_effects_by_source.setdefault(source, [])
        if effect not in bucket:
            bucket.append(effect)

    # ── Phase 1: canonical Effect catalog ────────────────────────────────
    # Each structural target owns exactly one immutable Effect. Trigger
    # indexes below retain this object itself, never a target-only route or
    # equal clone that would require a second lookup.
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

    # ── Phase 2: exhaustive trigger indexes ──────────────────────────────
    # LogicalUse is analysis's sole reverse index. Every dependency role is
    # handled explicitly; a pass means the primitive is committed by another
    # named route, never that lowering forgot it.
    for source, uses in analysis.consumers_by_port.items():
        for use in uses:
            if isinstance(use, CallUse):
                add_item_effect(source, CallInputEffect(use.call, use.input_index))
                continue
            if not isinstance(use, PrimitiveUse):  # pragma: no cover
                raise CompileError(f"unsupported LogicalUse: {use!r}")
            match use.role:
                case InputRole.EXPAND_GROUP:
                    # Expanded Worker layouts are committed atomically by
                    # MicrobatchEngine.commit_reports, not an Item event.
                    pass
                case InputRole.REDUCE_VALUE | InputRole.REDUCE_MEMBERS:
                    effect = structural_effects_by_target.get(use.port)
                    if not isinstance(effect, ReduceEffect):  # pragma: no cover
                        raise CompileError("Reduce use has no matching ReduceEffect")
                    add_item_effect(source, effect)
                case InputRole.BROADCAST_SOURCE:
                    # Canonicalization may replace the immediate source; the
                    # effective source is indexed after this exhaustive loop.
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

    # ── Phase 3: physical Call pools and Worker ABI ──────────────────────
    actor_pools_by_call = {
        call: _compile_pool_spec(call_options.get(call, ()))
        for call in logical.calls
    }

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

    # ── Phase 4: freeze the complete static execution contract ───────────
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
    if "batch_scope" in raw:
        raise CompileError(
            "batch_scope was replaced by batching_policy; use "
            "'any_parent' or 'single_parent'"
        )

    try:
        return ActorPoolSpec(
            replicas=cast(Any, raw.pop("replicas", 1)),
            batch_size=cast(Any, raw.pop("batch_size", 1)),
            batching_policy=cast(Any, raw.pop("batching_policy", "any_parent")),
            recovery=cast(Any, raw.pop("recovery", DEFAULT_RECOVERY_POLICY)),
            ray_options=tuple(raw.items()),
        )
    except (TypeError, ValueError) as error:
        raise CompileError(str(error)) from error


def _compile_input_layout(spec: CallSpec) -> CallInputLayout:
    """Lower logical args/kwargs into the sole dense Worker ABI layout."""

    return CallInputLayout(
        len(spec.args),
        tuple(name for name, _ in spec.kwargs),
    )


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


__all__: list[str] = []
