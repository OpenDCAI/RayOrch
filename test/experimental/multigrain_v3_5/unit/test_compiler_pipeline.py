"""v3.5 compiler boundary, exhaustive semantics, explain and canonicalization."""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import get_args

import pytest

import rayorch.experimental.multigrain_v3_5 as mg
from rayorch.experimental.multigrain_v3_5 import compiler
from rayorch.experimental.multigrain_v3_5.logical import (
    BroadcastOrigin,
    CallOutputOrigin,
    DomainSpec,
    ExpandOrigin,
    FilterOrigin,
    GroupOrigin,
    LogicalProgram,
    PortOrigin,
    PortSpec,
    SourceOrigin,
)
from rayorch.experimental.multigrain_v3_5.compiler import compile_logical
from rayorch.experimental.multigrain_v3_5.logical import freeze_mapping
from rayorch.experimental.multigrain_v3_5.model import (
    CallRef,
    CompileError,
    DomainRef,
    PortRef,
)
from rayorch.experimental.multigrain_v3_5.runtime import engine
from rayorch.experimental.multigrain_v3_5.semantics import (
    PrimitiveKind,
    describe_origin,
)


class U:
    pass


def test_logical_program_contains_only_declared_logical_facts():
    assert {field.name for field in fields(LogicalProgram)} == {
        "calls",
        "ports",
        "domains",
        "source_ports",
        "output_tree",
    }


def test_every_origin_has_one_explicit_semantic_descriptor():
    p0, p1 = PortRef(0), PortRef(1)
    origins = (
        SourceOrigin(0, "source"),
        CallOutputOrigin(CallRef(0), 0),
        ExpandOrigin(p0),
        GroupOrigin(p0, p1),
        BroadcastOrigin(p0),
        FilterOrigin(p0, p1),
    )
    assert {type(origin) for origin in origins} == set(get_args(PortOrigin))
    assert tuple(describe_origin(origin).kind for origin in origins) == tuple(
        PrimitiveKind
    )
    assert describe_origin(FilterOrigin(p0, p1)).control_demands == (p1,)
    assert describe_origin(FilterOrigin(p0, p1)).control_predecessors == (p0,)
    assert describe_origin(GroupOrigin(p0, p1)).rejects_control
    assert describe_origin(SourceOrigin(0, "source")).source_index == 0


def test_arena_module_has_no_logical_origin_interpreter():
    source = inspect.getsource(engine)
    assert "PortOrigin" not in source
    assert "FilterOrigin" not in source
    assert "BroadcastOrigin" not in source
    assert ".origin" not in source


def test_compiler_only_decodes_concrete_origins_through_semantic_descriptor():
    source = inspect.getsource(compiler)
    for origin_type in get_args(PortOrigin):
        assert origin_type.__name__ not in source


def test_chained_filter_control_is_a_fixed_point_analysis():
    class Chained(mg.Pipeline):
        def forward(self, values, bools, membership):
            selected_bools = mg.F.filter(bools, membership)
            return mg.F.filter(values, selected_bools)

    compiled = Chained().compile()
    values, bools, membership = compiled.logical.source_ports
    filters = tuple(
        port
        for port, spec in compiled.logical.ports.items()
        if isinstance(spec.origin, FilterOrigin)
    )
    selected_bools, result = filters

    assert values not in compiled.facts.control_ports
    assert {bools, membership, selected_bools}.issubset(
        compiled.facts.control_ports
    )
    assert compiled.runtime.filter_rules[selected_bools].copy_source_control
    assert not compiled.runtime.filter_rules[result].copy_source_control


def test_group_valued_mask_is_rejected_by_control_semantics():
    class Invalid(mg.Pipeline):
        def __init__(self) -> None:
            self.split = mg.RayModule(U)

        def forward(self, values):
            rows = mg.F.expand(self.split(values))
            grouped = mg.F.reduce(rows)
            return mg.F.filter(values, grouped)

    with pytest.raises(CompileError, match="group-valued Port"):
        Invalid().compile()


def test_manual_logical_port_cycle_is_rejected_before_lowering():
    domain = DomainRef(0)
    source, left, right = PortRef(0), PortRef(1), PortRef(2)
    logical = LogicalProgram(
        calls=freeze_mapping({}),
        ports=freeze_mapping(
            {
                source: PortSpec(source, domain, SourceOrigin(0, "mask")),
                left: PortSpec(left, domain, FilterOrigin(right, source)),
                right: PortSpec(right, domain, FilterOrigin(left, source)),
            }
        ),
        domains=freeze_mapping({domain: DomainSpec(domain, debug_name="root")}),
        source_ports=(source,),
        output_tree=left,
    )

    with pytest.raises(CompileError, match="dependency cycle"):
        compile_logical(logical, {}, optimize=False)


def test_unoptimized_path_and_broadcast_chain_rewrite_are_explainable():
    class BroadcastChain(mg.Pipeline):
        def __init__(self) -> None:
            self.outer = mg.RayModule(U)
            self.inner = mg.RayModule(U)

        def forward(self, values, masks):
            level_one = mg.F.expand(self.outer(values))
            level_one_masks = mg.F.broadcast(masks, like=level_one)
            level_two = mg.F.expand(self.inner(level_one))
            level_two_masks = mg.F.broadcast(level_one_masks, like=level_two)
            return mg.F.filter(level_two, level_two_masks)

    baseline = BroadcastChain().compile(optimize=False)
    optimized = BroadcastChain().compile(optimize=True)
    broadcasts = tuple(
        port
        for port, spec in optimized.logical.ports.items()
        if isinstance(spec.origin, BroadcastOrigin)
    )
    inner, outer = broadcasts
    root_mask = optimized.logical.source_ports[1]

    assert baseline.runtime.broadcast_rules[outer].source_port == inner
    assert optimized.runtime.broadcast_rules[outer].source_port == root_mask
    assert optimized.runtime.broadcast_rules[inner].source_port == root_mask
    assert len(optimized.explain.rewrites) == 1
    assert optimized.explain.rewrites[0].kind == "collapse-broadcast-chain"
    assert "rewrite collapse-broadcast-chain" in optimized.explain_text()
    assert baseline.explain_text().startswith("RuntimePlan[unoptimized]")


def test_runtime_plan_has_one_lowering_for_every_structural_port():
    class AllPrimitives(mg.Pipeline):
        def __init__(self) -> None:
            self.split = mg.RayModule(U)
            self.mask = mg.RayModule(U)

        def forward(self, values, root_mask):
            rows = mg.F.expand(self.split(values))
            masks = self.mask(rows)
            selected = mg.F.filter(rows, masks)
            copied = mg.F.broadcast(root_mask, like=rows)
            return mg.F.reduce(selected), mg.F.reduce(copied)

    compiled = AllPrimitives().compile()
    kinds = {
        port: describe_origin(spec.origin).kind
        for port, spec in compiled.logical.ports.items()
    }
    for port, kind in kinds.items():
        if kind is PrimitiveKind.EXPAND:
            assert any(
                rule.port == port
                for rules in compiled.runtime.expansions_by_source.values()
                for rule in rules
            )
        elif kind is PrimitiveKind.FILTER:
            assert port in compiled.runtime.filter_rules
        elif kind is PrimitiveKind.GROUP:
            assert port in compiled.runtime.group_rules
        elif kind is PrimitiveKind.BROADCAST:
            assert port in compiled.runtime.broadcast_rules
