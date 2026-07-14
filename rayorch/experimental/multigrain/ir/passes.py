"""Pass skeletons for the experimental multi-grain IR."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol

from .model import (
    CardinalityContract,
    IRNode,
    IRPortRef,
    IRPortSpec,
    MultigrainIR,
    NodeKind,
    OperatorRecipe,
    PhysicalHints,
    REBATCH_RECIPE,
    RelationKind,
    RelationSpec,
)


class PassKind(str, Enum):
    ANALYSIS = "analysis"
    VERIFY = "verify"
    LINT = "lint"
    TRANSFORM = "transform"


@dataclass(frozen=True)
class Diagnostic:
    message: str
    node: str | None = None
    severity: str = "error"


@dataclass(frozen=True)
class PassResult:
    graph: MultigrainIR
    diagnostics: tuple[Diagnostic, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def raise_for_errors(self) -> None:
        errors = [item for item in self.diagnostics if item.severity == "error"]
        if not errors:
            return
        details = "; ".join(
            f"{item.node}: {item.message}" if item.node else item.message
            for item in errors
        )
        raise ValueError(details)


class IRPass(Protocol):
    name: str
    kind: PassKind

    def run(self, graph: MultigrainIR) -> PassResult:
        ...


class VerifyPass:
    """Validate relation/cardinality invariants for the current IR MVP."""

    name = "verify"
    kind = PassKind.VERIFY

    def run(self, graph: MultigrainIR) -> PassResult:
        diagnostics: list[Diagnostic] = []
        known_refs = {port.ref for port in graph.inputs}
        node_names: set[str] = set()

        for node in graph.nodes:
            if node.name in node_names:
                diagnostics.append(Diagnostic("duplicate node name", node=node.name))
            node_names.add(node.name)

            for ref in node.input_refs:
                if ref not in known_refs:
                    diagnostics.append(
                        Diagnostic(
                            f"input ref {ref.node}.{ref.port} is not produced",
                            node=node.name,
                        )
                    )

            diagnostics.extend(self._check_node_contract(node))
            known_refs.update(node.output_refs)

        for ref in graph.graph_outputs:
            if ref not in known_refs:
                diagnostics.append(
                    Diagnostic(f"graph output {ref.node}.{ref.port} is not produced")
                )

        return PassResult(graph=graph, diagnostics=tuple(diagnostics))

    def _check_node_contract(self, node: IRNode) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        contract = node.contract
        if contract.kind is not node.kind:
            diagnostics.append(
                Diagnostic("contract kind must match node kind", node=node.name)
            )
        if contract.input_grains != tuple(spec.grain for spec in node.input_specs):
            diagnostics.append(
                Diagnostic(
                    "contract input grains must match input specs "
                    "before same-grain validation",
                    node=node.name,
                )
            )
        if contract.output_grains != tuple(spec.grain for spec in node.output_specs):
            diagnostics.append(
                Diagnostic(
                    "contract output grains must match output specs",
                    node=node.name,
                )
            )
        if len(contract.relations) != len(node.output_refs):
            diagnostics.append(
                Diagnostic(
                    "each output requires exactly one relation contract",
                    node=node.name,
                )
            )
        relation_outputs = tuple(relation.output for relation in contract.relations)
        if relation_outputs != node.output_refs:
            diagnostics.append(
                Diagnostic(
                    "relation outputs must match node outputs in order",
                    node=node.name,
                )
            )
        for relation in contract.relations:
            if relation.parents != node.input_refs:
                diagnostics.append(
                    Diagnostic(
                        "relation parents must match node inputs in order",
                        node=node.name,
                    )
                )

        families = {relation.relation for relation in contract.relations}
        if len(families) > 1:
            diagnostics.append(
                Diagnostic(
                    "a primitive node cannot mix relation families",
                    node=node.name,
                )
            )
            return tuple(diagnostics)
        family = next(iter(families), None)

        if family in (RelationKind.PRESERVE, RelationKind.FILTER):
            if len(set(contract.input_grains)) > 1:
                diagnostics.append(
                    Diagnostic(
                        "identity-aligned primitive requires same-grain inputs",
                        node=node.name,
                    )
                )
            if contract.input_grains:
                grain = contract.input_grains[0]
                if any(output != grain for output in contract.output_grains):
                    diagnostics.append(
                        Diagnostic(
                            "identity-aligned outputs must preserve input grain",
                            node=node.name,
                        )
                    )
        elif family is RelationKind.EXPAND:
            if node.parent_input is None:
                diagnostics.append(
                    Diagnostic("Expand requires parent_input", node=node.name)
                )
            elif node.parent_input < 0 or node.parent_input >= len(node.input_refs):
                diagnostics.append(
                    Diagnostic("Expand parent_input is out of range", node=node.name)
                )
            parent_inputs = {
                relation.parent_input for relation in node.contract.relations
            }
            if parent_inputs != {node.parent_input}:
                diagnostics.append(
                    Diagnostic(
                        "Expand relations must share node parent_input",
                        node=node.name,
                    )
                )
        elif family is RelationKind.REDUCE:
            if not node.grouped:
                diagnostics.append(
                    Diagnostic("Reduce must consume an explicit group_by", node=node.name)
                )
            if node.parent_input != 0:
                diagnostics.append(
                    Diagnostic("Reduce anchor must be input 0", node=node.name)
                )
            if node.contract.input_grains:
                anchor_grain = node.contract.input_grains[0]
                if any(grain != anchor_grain for grain in node.contract.output_grains):
                    diagnostics.append(
                        Diagnostic(
                            "Reduce outputs must return to anchor grain",
                            node=node.name,
                        )
                    )
            if any(
                relation.anchor != node.input_refs[0]
                for relation in contract.relations
            ):
                diagnostics.append(
                    Diagnostic(
                        "Reduce relation anchor must be input 0",
                        node=node.name,
                    )
                )
        elif family is RelationKind.RELATE:
            for relation in node.contract.relations:
                if relation.roles and len(relation.roles) != len(node.input_refs):
                    diagnostics.append(
                        Diagnostic(
                            "Relate roles must match input refs",
                            node=node.name,
                        )
                    )
        return tuple(diagnostics)


class RelationSummaryPass:
    """Read relation-aware IR and summarize logical shape."""

    name = "relation_summary"
    kind = PassKind.ANALYSIS

    def run(self, graph: MultigrainIR) -> PassResult:
        node_kinds: dict[str, int] = {}
        relation_kinds: dict[str, int] = {}
        expand_outputs: list[dict[str, object]] = []
        reduce_groups: list[dict[str, object]] = []

        for node in graph.nodes:
            node_kinds[node.kind.value] = node_kinds.get(node.kind.value, 0) + 1
            for relation in node.contract.relations:
                relation_kinds[relation.relation.value] = (
                    relation_kinds.get(relation.relation.value, 0) + 1
                )
                if relation.relation is RelationKind.EXPAND:
                    expand_outputs.append(
                        {
                            "node": node.name,
                            "output": relation.output.port,
                            "parent_input": relation.parent_input,
                            "grain": node.contract.output_grains[relation.output.index],
                        }
                    )
                elif relation.relation is RelationKind.REDUCE:
                    reduce_groups.append(
                        {
                            "node": node.name,
                            "anchor": relation.anchor.node if relation.anchor else None,
                            "descendants": [
                                ref.node for ref in node.input_refs[1:]
                            ],
                            "missing": relation.missing.value,
                        }
                    )
        return PassResult(
            graph=graph,
            metadata={
                "node_kinds": node_kinds,
                "relation_kinds": relation_kinds,
                "expand_outputs": expand_outputs,
                "reduce_groups": reduce_groups,
            },
        )


class RebatchCandidatePass:
    """Find child-grain ports worth physically rebatching."""

    name = "rebatch_candidates"
    kind = PassKind.ANALYSIS

    def run(self, graph: MultigrainIR) -> PassResult:
        candidates: list[dict[str, object]] = []
        for node in graph.nodes:
            if node.kind is not NodeKind.EXPAND:
                continue
            if not node.physical.prefer_rebatch:
                continue
            for ref in node.output_refs:
                consumers = [
                    consumer.name
                    for consumer in graph.nodes
                    if ref in consumer.input_refs
                ]
                candidates.append(
                    {
                        "node": node.name,
                        "port": ref.port,
                        "grain": node.outputs[ref.index].grain,
                        "consumers": consumers,
                    }
                )
        return PassResult(graph=graph, metadata={"rebatch_candidates": candidates})


class PlanReduceGroupsPass:
    """Plan how Reduce nodes should consume grouped descendants."""

    name = "plan_reduce_groups"
    kind = PassKind.ANALYSIS

    def run(self, graph: MultigrainIR) -> PassResult:
        plans: list[dict[str, object]] = []
        for node in graph.nodes:
            if node.kind is not NodeKind.REDUCE:
                continue
            relation = node.contract.relations[0]
            plans.append(
                {
                    "node": node.name,
                    "anchor": node.input_refs[0].node,
                    "descendants": [ref.node for ref in node.input_refs[1:]],
                    "order_by": "ordinal_or_stable_key",
                    "missing": relation.missing.value,
                }
            )
        return PassResult(graph=graph, metadata={"reduce_group_plans": plans})


class MarkMapFilterFusionCandidatesPass:
    """Mark canonical Map -> Filter patterns that can be fused physically."""

    name = "mark_map_filter_fusion_candidates"
    kind = PassKind.ANALYSIS

    def run(self, graph: MultigrainIR) -> PassResult:
        candidates: list[dict[str, str]] = []
        node_by_name = {node.name: node for node in graph.nodes}
        for node in graph.nodes:
            if node.kind is not NodeKind.FILTER:
                continue
            map_inputs = []
            for ref in node.input_refs:
                producer = node_by_name.get(ref.node)
                if (
                    producer is not None
                    and producer.kind is NodeKind.MAP
                    and producer.name not in map_inputs
                ):
                    map_inputs.append(producer.name)
            if map_inputs:
                candidates.append(
                    {
                        "filter": node.name,
                        "maps": tuple(map_inputs),
                        "strategy": "physical_select_fusion",
                    }
                )
        return PassResult(graph=graph, metadata={"fusion_candidates": candidates})


class InsertRebatchAfterExpandPass:
    """Insert logical REBATCH nodes after Expand outputs and rewrite consumers."""

    name = "insert_rebatch_after_expand"
    kind = PassKind.TRANSFORM

    def run(self, graph: MultigrainIR) -> PassResult:
        rewrites: dict[IRPortRef, IRPortRef] = {}
        new_nodes: list[IRNode] = []
        inserted: list[str] = []

        for node in graph.nodes:
            rewritten_node = _rewrite_node_inputs(node, rewrites)
            new_nodes.append(rewritten_node)
            if node.kind is not NodeKind.EXPAND or not node.physical.prefer_rebatch:
                continue
            for spec in node.output_specs:
                rebatch = _make_rebatch_node(spec)
                rewrites[spec.ref] = rebatch.output_specs[0].ref
                new_nodes.append(rebatch)
                inserted.append(rebatch.name)

        new_outputs = tuple(rewrites.get(ref, ref) for ref in graph.outputs)
        return PassResult(
            graph=replace(graph, nodes=tuple(new_nodes), outputs=new_outputs),
            metadata={"inserted_rebatch_nodes": inserted},
        )


class PassManager:
    def __init__(self, passes: tuple[IRPass, ...] | None = None) -> None:
        self.passes = passes or ()

    def run(self, graph: MultigrainIR) -> PassResult:
        current = graph
        diagnostics: list[Diagnostic] = []
        metadata: dict[str, object] = {}
        for ir_pass in self.passes:
            result = ir_pass.run(current)
            current = result.graph
            diagnostics.extend(result.diagnostics)
            metadata.update(result.metadata)
            if not result.ok:
                break
        return PassResult(
            graph=current,
            diagnostics=tuple(diagnostics),
            metadata=metadata,
        )


__all__ = [
    "Diagnostic",
    "InsertRebatchAfterExpandPass",
    "IRPass",
    "MarkMapFilterFusionCandidatesPass",
    "PassKind",
    "PassManager",
    "PassResult",
    "PlanReduceGroupsPass",
    "RebatchCandidatePass",
    "RelationSummaryPass",
    "VerifyPass",
]


def _make_rebatch_node(port: IRPortSpec) -> IRNode:
    """Build a passive REBATCH node that consumes ``port`` (an Expand output)."""
    node_name = f"{port.ref.node}__rebatch_{port.ref.port}"
    output_ref = IRPortRef(node=node_name, port="out", index=0)
    output_spec = IRPortSpec(ref=output_ref, grain=port.grain, name=node_name)
    relation = RelationSpec(
        output=output_ref,
        relation=RelationKind.PRESERVE,
        parents=(port.ref,),
    )
    return IRNode(
        name=node_name,
        kind=NodeKind.REBATCH,
        input_specs=(port,),
        output_specs=(output_spec,),
        contract=CardinalityContract(
            kind=NodeKind.REBATCH,
            input_grains=(port.grain,),
            output_grains=(port.grain,),
            relations=(relation,),
        ),
        op=OperatorRecipe(
            cls_ref=REBATCH_RECIPE,
            provenance={"source": f"{port.ref.node}.{port.ref.port}"},
        ),
        physical=PhysicalHints(prefer_rebatch=False),
    )


def _rewrite_node_inputs(
    node: IRNode,
    rewrites: dict[IRPortRef, IRPortRef],
) -> IRNode:
    if not rewrites:
        return node
    new_input_specs = tuple(
        replace(spec, ref=rewrites[spec.ref]) if spec.ref in rewrites else spec
        for spec in node.input_specs
    )
    if new_input_specs == node.input_specs:
        return node
    new_relations = tuple(
        replace(
            relation,
            parents=tuple(rewrites.get(ref, ref) for ref in relation.parents),
            anchor=(
                rewrites.get(relation.anchor, relation.anchor)
                if relation.anchor is not None
                else None
            ),
        )
        for relation in node.contract.relations
    )
    return replace(
        node,
        input_specs=new_input_specs,
        contract=replace(node.contract, relations=new_relations),
    )
