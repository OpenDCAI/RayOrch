"""LogicalProgram 与 RuntimePlan 的静态边界验证。

Verifier 只读取冻结对象，不构造 derived state。前半段证明用户声明的图合法，后半段
证明 lowering 没有漏掉任何 primitive、Effect 索引或 Worker ABI 合同。
"""

from __future__ import annotations

from typing import Mapping, assert_never

from ..model import CallRef, CompileError, DomainRef, PortRef
from ..protocol import CallInputLayout
from .analysis import ProgramAnalysis
from .logical import LogicalProgram
from .plan import (
    BroadcastEffect,
    CallInputEffect,
    FilterEffect,
    ItemEffect,
    ReduceEffect,
    RuntimePlan,
)
from .semantics import PrimitiveKind, PrimitiveSemantics, describe_origin


def verify_logical(logical: LogicalProgram) -> None:
    """验证引用闭包、Domain 关系和每种 primitive 的局部合同。"""

    # ── Graph identity and Domain tree ──────────────────────────────────
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

    # ── Call alignment and Port dependency closure ──────────────────────
    semantics = _describe_origins(logical)
    _verify_acyclic(logical, semantics)
    for call, spec in logical.calls.items():
        if spec.execution_domain not in logical.domains:
            raise CompileError(f"Call {call} has an unknown execution Domain")
        for input_ in spec.ordered_inputs:
            if input_.port not in logical.ports:
                raise CompileError(f"Call {call} references an unknown input Port")
            if logical.port(input_.port).domain != spec.execution_domain:
                raise CompileError(f"Call {call} input Domain mismatch")

    # ── Source admission contract ────────────────────────────────────────
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

    # ── Exhaustive primitive-local contracts ─────────────────────────────
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
                    raise CompileError(
                        "same-Domain Broadcast must be canonicalized by builder"
                    )
            case PrimitiveKind.FILTER:
                source, mask = (item.port for item in semantic.inputs)
                if logical.port(source).domain != spec.domain:
                    raise CompileError("Filter source Domain mismatch")
                if logical.port(mask).domain != spec.domain:
                    raise CompileError("Filter mask Domain mismatch")
            case _:
                assert_never(semantic.kind)

    # ── Call output and public output closure ────────────────────────────
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


def _describe_origins(
    logical: LogicalProgram,
) -> dict[PortRef, PrimitiveSemantics]:
    """只构造穷尽 primitive 描述，供 verifier 在 full analysis 前使用。"""

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


def verify_runtime_plan(
    logical: LogicalProgram,
    analysis: ProgramAnalysis,
    plan: RuntimePlan,
) -> None:
    """保证 lowering 没有漏掉任何 logical primitive 或 Call ABI。"""

    # ── Complete static tables ───────────────────────────────────────────
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
        # Identity is the invariant: every trigger index points at the catalog
        # Effect itself, never an equal clone that could become a second truth.
        return any(
            indexed is effect
            for indexed in plan.item_effects_by_source.get(port, ())
        )

    # ── Every primitive owns one complete, identity-shared Effect ────────
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

    # ── Worker input ABI and Call input triggers ─────────────────────────
    for call, spec in logical.calls.items():
        expected_layout = CallInputLayout(
            len(spec.args),
            tuple(name for name, _ in spec.kwargs),
        )
        if plan.input_layouts_by_call[call] != expected_layout:
            raise CompileError("RuntimePlan Worker input layout does not match CallSpec")
        for index, input_ in enumerate(spec.ordered_inputs):
            if CallInputEffect(call, index) not in plan.item_effects_by_source.get(
                input_.port, ()
            ):
                raise CompileError("RuntimePlan omitted a Call input Effect")


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


__all__ = ["verify_logical", "verify_runtime_plan"]
