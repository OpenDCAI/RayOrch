"""v3.6 compiler boundary, exhaustive semantics, explain and canonicalization."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

import rayorch.experimental.multigrain_v3_6 as mg
from rayorch.experimental.multigrain_v3_6.program import (
    analysis,
    compiler,
    lowering,
    verify,
)
from rayorch.experimental.multigrain_v3_6.program.logical import (
    BroadcastOrigin,
    CallOutputOrigin,
    DomainSpec,
    ExpandOrigin,
    FilterOrigin,
    ReduceOrigin,
    LogicalProgram,
    PortOrigin,
    PortSpec,
    SourceOrigin,
)
from rayorch.experimental.multigrain_v3_6.program.compiler import compile_logical
from rayorch.experimental.multigrain_v3_6.program.logical import freeze_mapping
from rayorch.experimental.multigrain_v3_6.model import (
    CallRef,
    CompileError,
    DomainRef,
    EntityRef,
    ItemRef,
    PortRef,
)
from rayorch.experimental.multigrain_v3_6.program.plan import (
    BroadcastEffect,
    FilterEffect,
    ReduceEffect,
)
from rayorch.experimental.multigrain_v3_6.runtime import engine
from rayorch.experimental.multigrain_v3_6.runtime.state import ExpansionRef
from rayorch.experimental.multigrain_v3_6.program.semantics import (
    PrimitiveKind,
    describe_origin,
)


class U:
    pass


def test_root_public_api_is_deliberately_small():
    assert set(mg.__all__) == {
        "CompileError",
        "CompiledProgram",
        "Executor",
        "ExecutionError",
        "F",
        "GroupFailure",
        "ItemOutcome",
        "MISSING",
        "Pipeline",
        "Port",
        "RayModule",
        "RecordFailure",
        "RecoveryPolicy",
        "RunResult",
        "function",
    }


def test_logical_program_contains_only_declared_logical_facts():
    assert {field.name for field in fields(LogicalProgram)} == {
        "calls",
        "ports",
        "domains",
        "source_ports",
        "output_tree",
    }


def test_compiled_program_snapshots_ray_module_configuration():
    class ConfiguredPipeline(mg.Pipeline):
        def __init__(self, module):
            self.module = module

        def forward(self, values):
            return self.module(values)

    module = (
        mg.RayModule(U)
        .pre_init(version="v1")
        .ray_options(replicas=2, num_cpus=1)
    )
    pipeline = ConfiguredPipeline(module)
    first = pipeline.compile()
    first_call = next(iter(first.logical.calls))

    module.init_kwargs["version"] = "v2"
    module.init_kwargs["added"] = True
    module.options["replicas"] = 3
    module.options["num_cpus"] = 2

    assert first.logical.call(first_call).udf.init_kwargs == (("version", "v1"),)
    assert first.plan.pool(first_call).replicas == 2
    assert dict(first.plan.pool(first_call).ray_options)["num_cpus"] == 1

    second = pipeline.compile()
    second_call = next(iter(second.logical.calls))
    assert second.logical.call(second_call).udf.init_kwargs == (
        ("version", "v2"),
        ("added", True),
    )
    assert second.plan.pool(second_call).replicas == 3
    assert dict(second.plan.pool(second_call).ray_options)["num_cpus"] == 2


def test_every_origin_has_one_explicit_semantic_descriptor():
    p0, p1 = PortRef(0), PortRef(1)
    origins = (
        SourceOrigin(0, "source"),
        CallOutputOrigin(CallRef(0), 0),
        ExpandOrigin(p0),
        ReduceOrigin(p0, p1),
        BroadcastOrigin(p0),
        FilterOrigin(p0, p1),
    )
    assert {type(origin) for origin in origins} == set(get_args(PortOrigin))
    assert tuple(describe_origin(origin).kind for origin in origins) == tuple(
        PrimitiveKind
    )
    assert describe_origin(FilterOrigin(p0, p1)).control_demands == (p1,)
    assert describe_origin(FilterOrigin(p0, p1)).control_predecessors == (p0,)
    assert describe_origin(ReduceOrigin(p0, p1)).rejects_control
    assert describe_origin(SourceOrigin(0, "source")).source_index == 0


def test_engine_module_has_no_logical_origin_interpreter():
    source = inspect.getsource(engine)
    assert "PortOrigin" not in source
    assert "FilterOrigin" not in source
    assert "BroadcastOrigin" not in source
    assert ".origin" not in source


def test_engine_has_one_closed_fact_propagation_entry():
    source = inspect.getsource(engine.MicrobatchEngine)

    assert set(get_args(engine._FactEvent)) == {ItemRef, ExpansionRef, EntityRef}
    assert source.count("self._fact_queue.append(") == 3
    assert "_receipts" not in source
    assert "groups_by_child_domain" not in source
    assert "broadcasts_by_target_domain" not in source


def test_dispatch_state_is_the_only_grain_execution_state_writer():
    package = Path(mg.__file__).parent
    sources = {
        path.relative_to(package): path.read_text()
        for path in package.rglob("*.py")
    }
    mutation_tokens = (
        "record.phase = grain_transition",
        "record.generation +=",
        "record.infra_failures +=",
    )
    writers = {
        path
        for path, source in sources.items()
        if any(token in source for token in mutation_tokens)
    }
    transition_users = {
        path for path, source in sources.items() if "grain_transition(" in source
    }

    assert writers == {Path("runtime/dispatch.py")}
    assert transition_users == {
        Path("runtime/dispatch.py"),
        Path("runtime/transitions.py"),
    }


def test_compiler_only_decodes_concrete_origins_through_semantic_descriptor():
    source = "\n".join(
        inspect.getsource(module)
        for module in (analysis, compiler, lowering, verify)
    )
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

    assert values not in compiled.analysis.control_ports
    assert {bools, membership, selected_bools}.issubset(
        compiled.analysis.control_ports
    )
    selected_effect = compiled.plan.structural_effects_by_target[selected_bools]
    result_effect = compiled.plan.structural_effects_by_target[result]
    assert isinstance(selected_effect, FilterEffect)
    assert isinstance(result_effect, FilterEffect)
    assert selected_effect.copy_source_control
    assert not result_effect.copy_source_control


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

    baseline_outer = baseline.plan.structural_effects_by_target[outer]
    optimized_outer = optimized.plan.structural_effects_by_target[outer]
    optimized_inner = optimized.plan.structural_effects_by_target[inner]
    assert isinstance(baseline_outer, BroadcastEffect)
    assert isinstance(optimized_outer, BroadcastEffect)
    assert isinstance(optimized_inner, BroadcastEffect)
    assert baseline_outer.source_port == inner
    assert optimized_outer.source_port == root_mask
    assert optimized_inner.source_port == root_mask
    assert len(optimized.explanation.rewrites) == 1
    assert optimized.explanation.rewrites[0].kind == "collapse-broadcast-chain"
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
                for rules in compiled.plan.expand_effects_by_source.values()
                for rule in rules
            )
        elif kind is PrimitiveKind.FILTER:
            effect = compiled.plan.structural_effects_by_target[port]
            assert isinstance(effect, FilterEffect)
            for source in (effect.source_port, effect.mask_port):
                assert any(
                    indexed is effect
                    for indexed in compiled.plan.item_effects_by_source[source]
                )
        elif kind is PrimitiveKind.REDUCE:
            effect = compiled.plan.structural_effects_by_target[port]
            assert isinstance(effect, ReduceEffect)
            for source in (effect.value_port, effect.members_port):
                assert any(
                    indexed is effect
                    for indexed in compiled.plan.item_effects_by_source[source]
                )
            assert any(
                indexed is effect
                for indexed in compiled.plan.reduce_effects_by_child_domain[
                    effect.child_domain
                ]
            )
        elif kind is PrimitiveKind.BROADCAST:
            effect = compiled.plan.structural_effects_by_target[port]
            assert isinstance(effect, BroadcastEffect)
            assert any(
                indexed is effect
                for indexed in compiled.plan.item_effects_by_source[
                    effect.source_port
                ]
            )
            assert any(
                indexed is effect
                for indexed in compiled.plan.broadcast_effects_by_target_domain[
                    effect.target_domain
                ]
            )


def test_pool_options_compile_to_one_typed_physical_contract():
    recovery = mg.RecoveryPolicy.isolate_tail(infra_retries=2)

    class Configured(mg.Pipeline):
        def __init__(self) -> None:
            self.call = mg.RayModule(U).ray_options(
                replicas=3,
                batch_size=7,
                batching_policy="single_parent",
                recovery=recovery,
                num_cpus=0.25,
            )

        def forward(self, values):
            return self.call(values)

    optimized = Configured().compile(optimize=True)
    baseline = Configured().compile(optimize=False)
    call = next(iter(optimized.logical.calls))
    pool = optimized.plan.pool(call)

    assert optimized.plan.actor_pools_by_call == baseline.plan.actor_pools_by_call
    assert set(optimized.plan.actor_pools_by_call) == {call}
    assert pool.replicas == 3
    assert pool.batch_size == 7
    assert pool.batching_policy == "single_parent"
    assert pool.recovery is recovery
    assert pool.ray_options == (("num_cpus", 0.25),)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_retries": 1}, "max_retries is ambiguous"),
        ({"recovery": "retry"}, "RecoveryPolicy"),
        ({"replicas": True}, "replicas"),
        ({"batch_size": 1.5}, "batch_size"),
        ({"batching_policy": "global"}, "batching_policy"),
        ({"batch_scope": "elastic"}, "replaced by batching_policy"),
    ],
)
def test_compiler_rejects_untyped_or_ambiguous_physical_options(
    options,
    message,
):
    class Invalid(mg.Pipeline):
        def __init__(self) -> None:
            self.call = mg.RayModule(U).ray_options(**options)

        def forward(self, values):
            return self.call(values)

    with pytest.raises(CompileError, match=message):
        Invalid().compile()
