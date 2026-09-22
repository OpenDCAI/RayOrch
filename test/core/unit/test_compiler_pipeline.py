"""Compiler boundaries, exhaustive semantics, explanations, and canonicalization."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

import rayorch as ro
from rayorch._program import (
    analysis,
    compiler,
    lowering,
    verify,
)
from rayorch._program.logical import (
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
from rayorch._program.compiler import compile_logical
from rayorch._program.logical import freeze_mapping
from rayorch._model import (
    CallRef,
    DomainRef,
    EntityRef,
    ItemRef,
    PortRef,
)
from rayorch.errors import CompileError
from rayorch._program.plan import (
    BroadcastEffect,
    FilterEffect,
    ReduceEffect,
)
from rayorch._runtime import engine
from rayorch._runtime.state import ExpansionRef
from rayorch._program.semantics import (
    PrimitiveKind,
    describe_origin,
)


class U:
    pass


@pytest.mark.parametrize("call_output", [False, True], ids=["source", "call-output"])
@pytest.mark.parametrize("check", [
    pytest.param(bool, id="bool"),
    pytest.param(lambda port: not port, id="not"),
    pytest.param(lambda port: port and True, id="and"),
    pytest.param(lambda port: port or False, id="or"),
])
def test_symbolic_truth_checks_are_rejected_during_trace(check, call_output):
    class Invalid(ro.Pipeline):
        def __init__(self):
            self.transform = ro.RayModule(U)

        def forward(self, values):
            selected = self.transform(values) if call_output else values
            check(selected)
            return selected

    with pytest.raises(TypeError, match="symbolic Port.*truth value"):
        Invalid().compile()


def test_if_on_a_port_is_rejected_inside_a_helper():
    def choose(values):
        if values:
            return values
        raise AssertionError("a symbolic Port must not choose a branch")

    class Invalid(ro.Pipeline):
        def forward(self, values):
            return choose(values)

    with pytest.raises(TypeError, match="F.filter.*UDF"):
        Invalid().compile()


@pytest.mark.parametrize("enabled", [False, True])
def test_config_boolean_can_select_a_graph_branch_before_filtering(enabled):
    class Configured(ro.Pipeline):
        def __init__(self):
            self.enabled = enabled
            self.transform = ro.RayModule(U)

        def forward(self, values, masks):
            if self.enabled:
                values = self.transform(values)
            return ro.F.filter(values, masks)

    compiled = Configured().compile()
    assert len(compiled.logical.calls) == int(enabled)
    effect = compiled.plan.structural_effects_by_target[compiled.plan.output_tree]
    assert isinstance(effect, FilterEffect)
    assert effect.mask_port == compiled.plan.source_ports[1]


def test_root_public_api_is_deliberately_small():
    assert set(ro.__all__) == {
        "CompileError",
        "Executor",
        "ExecutionError",
        "F",
        "GroupFailure",
        "ItemOutcome",
        "OutputIssue",
        "Pipeline",
        "Port",
        "RayModule",
        "RecordFailure",
        "RecoveryPolicy",
        "RunResult",
        "__version__",
        "benchmark",
        "function",
        "run",
        "version_info",
    }
    assert not hasattr(ro, "MISSING")
    assert not hasattr(ro.F, "optional")


def test_experimental_runtime_name_is_not_a_public_namespace():
    assert not hasattr(ro, "multigrain")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rayorch.multigrain")


def test_logical_program_contains_only_declared_logical_facts():
    assert {field.name for field in fields(LogicalProgram)} == {
        "calls",
        "ports",
        "domains",
        "source_ports",
        "output_tree",
    }


def test_compiled_program_snapshots_ray_module_configuration():
    class ConfiguredPipeline(ro.Pipeline):
        def __init__(self, module):
            self.module = module

        def forward(self, values):
            return self.module(values)

    module = (
        ro.RayModule(U)
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
    source = inspect.getsource(engine.InputBatchEngine)

    assert set(get_args(engine._FactEvent)) == {ItemRef, ExpansionRef, EntityRef}
    assert source.count("self._fact_queue.append(") == 3
    assert "_receipts" not in source
    assert "groups_by_child_domain" not in source
    assert "broadcasts_by_target_domain" not in source


def test_dispatch_state_is_the_only_grain_execution_state_writer():
    package = Path(ro.__file__).parent
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

    assert writers == {Path("_runtime/dispatch.py")}
    assert transition_users == {
        Path("_runtime/dispatch.py"),
        Path("_runtime/transitions.py"),
    }


def test_compiler_only_decodes_concrete_origins_through_semantic_descriptor():
    source = "\n".join(
        inspect.getsource(module)
        for module in (analysis, compiler, lowering, verify)
    )
    for origin_type in get_args(PortOrigin):
        assert origin_type.__name__ not in source


def test_chained_filter_control_is_a_fixed_point_analysis():
    class Chained(ro.Pipeline):
        def forward(self, values, bools, membership):
            selected_bools = ro.F.filter(bools, membership)
            return ro.F.filter(values, selected_bools)

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
    class Invalid(ro.Pipeline):
        def __init__(self) -> None:
            self.split = ro.RayModule(U)

        def forward(self, values):
            rows = ro.F.expand(self.split(values))
            grouped = ro.F.reduce(rows)
            return ro.F.filter(values, grouped)

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
        compile_logical(logical, {}, {}, {}, {}, optimize=False)


def test_unoptimized_path_and_broadcast_chain_rewrite_are_explainable():
    class BroadcastChain(ro.Pipeline):
        def __init__(self) -> None:
            self.outer = ro.RayModule(U)
            self.inner = ro.RayModule(U)

        def forward(self, values, masks):
            level_one = ro.F.expand(self.outer(values))
            level_one_masks = ro.F.broadcast(masks, like=level_one)
            level_two = ro.F.expand(self.inner(level_one))
            level_two_masks = ro.F.broadcast(level_one_masks, like=level_two)
            return ro.F.filter(level_two, level_two_masks)

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
    class AllPrimitives(ro.Pipeline):
        def __init__(self) -> None:
            self.split = ro.RayModule(U)
            self.mask = ro.RayModule(U)

        def forward(self, values, root_mask):
            rows = ro.F.expand(self.split(values))
            masks = self.mask(rows)
            selected = ro.F.filter(rows, masks)
            copied = ro.F.broadcast(root_mask, like=rows)
            return ro.F.reduce(selected), ro.F.reduce(copied)

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
    recovery = ro.RecoveryPolicy.isolate_tail(infra_retries=2)

    class Configured(ro.Pipeline):
        def __init__(self) -> None:
            self.call = ro.RayModule(U).ray_options(
                replicas=3,
                batch_size=7,
                recovery=recovery,
                num_cpus=0.25,
            )

        def forward(self, values):
            return self.call(values)

    optimized = Configured().compile(optimize=True)
    baseline = Configured().compile(optimize=False)
    call = next(iter(optimized.logical.calls))
    pool = optimized.plan.pool(call)
    dispatch = optimized.plan.dispatch(call)

    assert optimized.plan.actor_pools == baseline.plan.actor_pools
    assert optimized.plan.dispatch_by_call == baseline.plan.dispatch_by_call
    assert len(optimized.plan.actor_pools) == 1
    assert pool.replicas == 3
    assert dispatch.batch_size == 7
    assert dispatch.recovery is recovery
    assert pool.ray_options == (("num_cpus", 0.25),)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_retries": 1}, "max_retries is ambiguous"),
        ({"recovery": "retry"}, "RecoveryPolicy"),
        ({"replicas": True}, "replicas"),
        ({"batch_size": 1.5}, "batch_size"),
        ({"batching_policy": "any_parent"}, "not supported"),
        ({"batch_scope": "elastic"}, "not supported"),
    ],
)
def test_compiler_rejects_untyped_or_ambiguous_physical_options(
    options,
    message,
):
    class Invalid(ro.Pipeline):
        def __init__(self) -> None:
            self.call = ro.RayModule(U).ray_options(**options)

        def forward(self, values):
            return self.call(values)

    with pytest.raises(CompileError, match=message):
        Invalid().compile()
