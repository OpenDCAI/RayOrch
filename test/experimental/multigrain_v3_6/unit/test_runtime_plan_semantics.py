"""完整 RuntimePlan 驱动的 Ray-free microbatch/Worker 语义回归。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rayorch.experimental.multigrain_v3_6 as mg
from rayorch.experimental.multigrain_v3_6.executor import Executor
from rayorch.experimental.multigrain_v3_6.logical import ExpandOrigin
from rayorch.experimental.multigrain_v3_6.materialize import materialize_tree
from rayorch.experimental.multigrain_v3_6.model import (
    CallRef,
    DomainRef,
    EntityRef,
    GrainPhase,
    GrainRef,
    ItemOutcome,
    ItemRef,
    PortRef,
    ExpansionOutcome,
)
from rayorch.experimental.multigrain_v3_6.plan import BroadcastEffect, ReduceEffect
from rayorch.experimental.multigrain_v3_6.protocol import (
    BlockRef,
    GrainReport,
    DispatchFailure,
    DispatchFailureKind,
    ExpandedRows,
    CallInputLayout,
    GrainPlan,
    CallOutputLayout,
    OutputReport,
    RecordFailure,
    RowBinding,
)
from rayorch.experimental.multigrain_v3_6.runtime import CommitError, MicrobatchEngine
from rayorch.experimental.multigrain_v3_6.runtime.state import (
    EntityParent,
    GroupBinding,
    ExpansionRef,
)
from rayorch.experimental.multigrain_v3_6.worker import (
    Worker,
    WorkerContractError,
    WorkerSnapshot,
)


class MemoryStore:
    def __init__(self) -> None:
        self._next = 0
        self.blocks: dict[BlockRef, tuple[object, ...]] = {}

    def put(self, values: tuple[object, ...]) -> BlockRef:
        block = BlockRef(self._next)
        self._next += 1
        self.blocks[block] = tuple(values)
        return block

    def get(self, binding: RowBinding) -> object:
        return self.blocks[binding.block][binding.row]


def test_worker_observation_is_scalar_and_read_only():
    class Audited:
        def batch_audit(self):
            return {
                "jobs": 7,
                "mode": "batch",
                "ignored": object(),
            }

    worker = Worker(Audited, input_layout=CallInputLayout(1))
    observation = worker.observe()

    assert observation.lifetime_calls == 0
    assert observation.pid > 0
    assert observation.rss_bytes >= 0
    assert dict(observation.audit) == {"jobs": 7, "mode": "batch"}


def _execute_one(worker_target, store: MemoryStore) -> DispatchFailure:
    block = store.put((1,))
    grain_plan = GrainPlan(
        GrainRef(CallRef(0), EntityRef(DomainRef(0), 0)),
        0,
        (RowBinding(block, 0),),
    )
    result = Worker(worker_target, input_layout=CallInputLayout(1)).execute(
        (grain_plan,),
        (CallOutputLayout(PortRef(0)),),
        store,
    )
    assert isinstance(result, DispatchFailure)
    return result


def test_worker_classifies_udf_and_contract_failures_at_one_boundary():
    class UdfError:
        def run(self, values):
            raise LookupError("poison batch")

    class BadLength:
        def run(self, values):
            return []

    udf_failure = _execute_one(UdfError, MemoryStore())
    contract_failure = _execute_one(BadLength, MemoryStore())

    assert udf_failure.kind is DispatchFailureKind.UDF_ERROR
    assert udf_failure.error_type.endswith("LookupError")
    assert "poison batch" in udf_failure.message
    assert "raise LookupError" in udf_failure.traceback
    assert contract_failure.kind is DispatchFailureKind.CONTRACT_ERROR
    assert contract_failure.error_type.endswith("WorkerContractError")
    assert "expected 1 Grain rows, got 0" in contract_failure.message
    assert "PortRef(value=0)" in contract_failure.message


def test_udf_raising_contract_error_name_is_still_a_udf_failure():
    class UserCode:
        def run(self, values):
            raise WorkerContractError("a user exception, not an ABI violation")

    failure = _execute_one(UserCode, MemoryStore())

    assert failure.kind is DispatchFailureKind.UDF_ERROR
    assert "not an ABI violation" in failure.message


def test_executor_submits_all_worker_observations_before_waiting():
    events = []
    first_ref, second_ref = object(), object()
    expected = {
        first_ref: WorkerSnapshot(lifetime_calls=2, pid=1, rss_bytes=10),
        second_ref: WorkerSnapshot(lifetime_calls=3, pid=2, rss_bytes=20),
    }

    def endpoint(name, reference):
        def remote():
            events.append(f"submit:{name}")
            return reference

        return SimpleNamespace(remote=remote)

    class FakeRay:
        def wait(self, pending, *, num_returns):
            assert num_returns == 1
            assert events[:2] == ["submit:first", "submit:second"]
            events.append("wait")
            return [pending[0]], pending[1:]

        def get(self, reference):
            events.append("get")
            return expected[reference]

    executor = object.__new__(Executor)
    executor.ray = FakeRay()
    executor._actors = {
        CallRef(0): [
            SimpleNamespace(
                handle=SimpleNamespace(observe=endpoint("first", first_ref))
            ),
            SimpleNamespace(
                handle=SimpleNamespace(observe=endpoint("second", second_ref))
            ),
        ]
    }

    observations = executor._observe_workers()

    assert observations[CallRef(0)] == (
        expected[first_ref],
        expected[second_ref],
    )


@pytest.mark.parametrize(("owns_ray", "expected_shutdowns"), [(False, 0), (True, 1)])
def test_executor_close_respects_ray_runtime_ownership(
    owns_ray,
    expected_shutdowns,
):
    shutdowns = []

    class FakeRay:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def shutdown():
            shutdowns.append(True)

    executor = object.__new__(Executor)
    executor.ray = FakeRay()
    executor._owns_ray = owns_ray
    executor._closed = False
    executor._actors = {}
    executor.store = SimpleNamespace(clear_cache=lambda: None)

    executor.close()
    executor.close()

    assert len(shutdowns) == expected_shutdowns


def run_sync(pipeline: mg.Pipeline, *columns, optimize: bool = True):
    compiled = pipeline.compile(optimize=optimize)
    plan = compiled.plan
    store = MemoryStore()
    bindings = {}
    controls = {}
    for port, values in zip(plan.source_ports, columns):
        values = tuple(values)
        block = store.put(values)
        bindings[port] = tuple(RowBinding(block, row) for row in range(len(values)))
        if port in plan.control_ports:
            controls[port] = values

    engine = MicrobatchEngine(plan)
    engine.admit_sources(bindings, controls=controls)
    engine.close_admission()
    workers = {
        call: Worker(
            spec.udf.target,
            spec.udf.init_args,
            spec.udf.init_kwargs,
            input_layout=plan.input_layouts_by_call[call],
        )
        for call, spec in plan.calls.items()
    }
    turns = 0
    while engine.ready_count:
        turns += 1
        assert turns < 100
        call = next(
            call for call in plan.calls
            if engine.dispatch_priority(call) is not None
        )
        grains = engine.reserve_dispatch(call, max_size=64).grains
        grain_plans = tuple(engine.grain_plan(grain) for grain in grains)
        reports = workers[call].execute(
            grain_plans,
            plan.output_layouts_by_call[call],
            store,
        )
        for report in reports:
            engine.commit_report(report)
    assert engine.is_complete()
    return materialize_tree(plan, engine, store), compiled, engine


def test_tutorial_document_pipeline_example():
    """Keep the tutorial's full structural example executable."""

    class Split:
        def run(self, documents):
            return [
                [
                    document[index : index + 2]
                    for index in range(0, len(document), 2)
                ]
                for document in documents
            ]

    class LongEnough:
        def run(self, chunks, thresholds):
            return [
                len(chunk) >= threshold
                for chunk, threshold in zip(chunks, thresholds)
            ]

    class Summarize:
        def run(self, documents, chunk_groups):
            return [
                (document, tuple(chunks))
                for document, chunks in zip(documents, chunk_groups)
            ]

    class DocumentPipeline(mg.Pipeline):
        def __init__(self) -> None:
            self.split = mg.RayModule(Split)
            self.long_enough = mg.RayModule(LongEnough)
            self.summarize = mg.RayModule(Summarize)

        def forward(self, documents, thresholds):
            chunks = mg.F.expand(self.split(documents))
            chunk_thresholds = mg.F.broadcast(thresholds, like=chunks)
            masks = self.long_enough(chunks, chunk_thresholds)
            selected = mg.F.filter(chunks, masks)
            chunk_groups = mg.F.reduce(selected)
            return self.summarize(documents, chunk_groups)

    outputs, _compiled, _arena = run_sync(
        DocumentPipeline(),
        ["abcde", "xy"],
        [2, 2],
    )

    assert outputs == [
        ("abcde", ("ab", "cd")),
        ("xy", ("xy",)),
    ]


def test_chained_filter_executes_from_control_fixed_point():
    class Chained(mg.Pipeline):
        def forward(self, values, bools, membership):
            selected_bools = mg.F.filter(bools, membership)
            return mg.F.filter(values, selected_bools)

    outputs, _compiled, _arena = run_sync(
        Chained(),
        [10, 20, 30],
        [True, False, True],
        [True, True, False],
    )
    assert outputs == [10, ItemOutcome.DROPPED, ItemOutcome.DROPPED]


def test_keyword_only_input_preserves_default_without_identity_driver():
    class KeywordOnly:
        def run(self, values, scale=10, *, masks):
            return [
                value * scale if keep else -1
                for value, keep in zip(values, masks)
            ]

    class KeywordPipeline(mg.Pipeline):
        def __init__(self) -> None:
            self.call = mg.RayModule(KeywordOnly)

        def forward(self, values, masks):
            return self.call(values, masks=masks)

    outputs, compiled, _arena = run_sync(
        KeywordPipeline(),
        [1, 2, 3],
        [True, False, True],
    )
    call, spec = next(iter(compiled.logical.calls.items()))

    assert outputs == [10, -1, 30]
    assert not hasattr(spec, "driving_input")
    assert len(spec.args) == 1
    assert tuple(name for name, _ in spec.kwargs) == ("masks",)
    assert spec.ordered_inputs == (
        *spec.args,
        *(input_ for _, input_ in spec.kwargs),
    )
    assert not hasattr(spec, "inputs")
    assert all(not hasattr(input_, "name") for input_ in spec.ordered_inputs)
    assert all(not hasattr(input_, "keyword") for input_ in spec.ordered_inputs)
    assert compiled.plan.input_layouts_by_call[call].positional_count == 1
    assert compiled.plan.input_layouts_by_call[call].keyword_names == ("masks",)


def test_reordered_keyword_inputs_keep_python_binding_semantics():
    class Subtract:
        def run(self, *, left, right):
            return [lhs - rhs for lhs, rhs in zip(left, right)]

    class Reordered(mg.Pipeline):
        def __init__(self) -> None:
            self.call = mg.RayModule(Subtract)

        def forward(self, left, right):
            return self.call(right=right, left=left)

    outputs, compiled, _arena = run_sync(Reordered(), [10, 20], [3, 5])
    call, spec = next(iter(compiled.logical.calls.items()))

    assert outputs == [7, 15]
    assert spec.args == ()
    assert tuple(name for name, _ in spec.kwargs) == ("right", "left")
    assert compiled.plan.input_layouts_by_call[call].positional_count == 0
    assert compiled.plan.input_layouts_by_call[call].keyword_names == (
        "right",
        "left",
    )


def test_all_optional_call_runs_with_missing_values_and_no_driver():
    class MissingAware:
        def run(self, left, right):
            return [
                -1 if lhs is mg.MISSING and rhs is mg.MISSING else lhs + rhs
                for lhs, rhs in zip(left, right)
            ]

    class AllOptional(mg.Pipeline):
        def __init__(self) -> None:
            self.call = mg.RayModule(MissingAware)

        def forward(self, left, right, masks):
            selected_left = mg.F.filter(left, masks)
            selected_right = mg.F.filter(right, masks)
            return self.call(
                mg.F.optional(selected_left),
                mg.F.optional(selected_right),
            )

    outputs, _compiled, _arena = run_sync(
        AllOptional(),
        [10, 20],
        [1, 2],
        [False, True],
    )
    assert outputs == [-1, 22]


def test_required_drop_is_symmetric_across_call_inputs():
    class Add:
        def run(self, left, right):
            return [lhs + rhs for lhs, rhs in zip(left, right)]

    class SymmetricDrop(mg.Pipeline):
        def __init__(self) -> None:
            self.call = mg.RayModule(Add)

        def forward(self, left, right, left_mask, right_mask):
            return self.call(
                mg.F.filter(left, left_mask),
                mg.F.filter(right, right_mask),
            )

    outputs, _compiled, _arena = run_sync(
        SymmetricDrop(),
        [10, 20],
        [1, 2],
        [False, True],
        [True, False],
    )
    assert outputs == [ItemOutcome.DROPPED, ItemOutcome.DROPPED]


def test_call_failure_dominates_required_drop_independent_of_arrival_order():
    class Fail:
        def run(self, values):
            return [RecordFailure(f"bad {value}") for value in values]

    class Add:
        def run(self, left, right):
            return [lhs + rhs for lhs, rhs in zip(left, right)]

    class FailureAndDrop(mg.Pipeline):
        def __init__(self) -> None:
            self.fail = mg.RayModule(Fail)
            self.add = mg.RayModule(Add)

        def forward(self, values, masks):
            dropped = mg.F.filter(values, masks)
            failed = self.fail(values)
            return self.add(dropped, failed)

    outputs, _compiled, _arena = run_sync(
        FailureAndDrop(),
        [10, 20],
        [False, False],
    )
    assert outputs == [ItemOutcome.SUPPRESSED, ItemOutcome.SUPPRESSED]


def test_nested_expand_reduce_preserves_empty_groups():
    @mg.function
    def pages(documents):
        return [[0, 1] if document == 0 else [] for document in documents]

    @mg.function
    def regions(page_numbers):
        return [[] if page == 0 else [10, 11] for page in page_numbers]

    class Nested(mg.Pipeline):
        def forward(self, documents):
            page = mg.F.expand(pages(documents))
            region = mg.F.expand(regions(page))
            return mg.F.reduce(mg.F.reduce(region))

    outputs, _compiled, _arena = run_sync(Nested(), [0, 1])
    assert outputs == [[[], [10, 11]], []]


def test_broadcast_chain_optimized_and_baseline_have_exact_outcome_parity():
    @mg.function
    def outer(values):
        return [[value, value + 1] for value in values]

    @mg.function
    def inner(values):
        return [[value * 10, value * 10 + 1] for value in values]

    class BroadcastChain(mg.Pipeline):
        def forward(self, values, masks):
            level_one = mg.F.expand(outer(values))
            level_one_masks = mg.F.broadcast(masks, like=level_one)
            level_two = mg.F.expand(inner(level_one))
            level_two_masks = mg.F.broadcast(level_one_masks, like=level_two)
            return mg.F.filter(level_two, level_two_masks)

    baseline, baseline_compiled, _ = run_sync(
        BroadcastChain(), [1, 3], [True, False], optimize=False
    )
    optimized, optimized_compiled, _ = run_sync(
        BroadcastChain(), [1, 3], [True, False], optimize=True
    )
    assert optimized == baseline
    assert optimized[:4] == [10, 11, 20, 21]
    assert optimized[4:] == [ItemOutcome.DROPPED] * 4
    assert baseline_compiled.explanation.rewrites == ()
    assert len(optimized_compiled.explanation.rewrites) == 1


def test_broadcast_fact_order_reaches_the_same_fixed_point():
    class BroadcastOrder(mg.Pipeline):
        def __init__(self) -> None:
            self.split = mg.RayModule(U)
            self.label = mg.RayModule(U)

        def forward(self, values):
            rows = mg.F.expand(self.split(values))
            return mg.F.broadcast(self.label(values), like=rows)

    plan = BroadcastOrder().compile().plan
    effect = next(
        effect
        for effect in plan.structural_effects_by_target.values()
        if isinstance(effect, BroadcastEffect)
    )
    binding = RowBinding(BlockRef("label"), 0)

    def reach_fixed_point(*, source_first: bool):
        engine = MicrobatchEngine(plan)
        root = EntityRef(effect.source_domain, 0)
        child = EntityRef(effect.target_domain, 0)
        source = ItemRef(effect.source_port, root)
        target = ItemRef(effect.target_port, child)
        engine._publish_entity(root)
        engine.advance()

        if source_first:
            engine._publish_item(source, ItemOutcome.PRESENT, binding=binding)
            engine.advance()
            engine._publish_entity(child, EntityParent(root, 0))
            engine.advance()
        else:
            engine._publish_entity(child, EntityParent(root, 0))
            engine.advance()
            engine._publish_item(source, ItemOutcome.PRESENT, binding=binding)
            engine.advance()

        assert engine.item_outcome(target) is ItemOutcome.PRESENT
        assert engine.value_binding(target) == binding
        assert not engine._facts
        engine._publish_item(source, ItemOutcome.PRESENT, binding=binding)
        assert not engine._facts
        engine._publish_entity(child, EntityParent(root, 0))
        assert not engine._facts
        return engine.item_outcome(target), engine.value_binding(target)

    assert reach_fixed_point(source_first=True) == reach_fixed_point(
        source_first=False
    )


def test_group_fact_order_reaches_the_same_fixed_point():
    class GroupOrder(mg.Pipeline):
        def __init__(self) -> None:
            self.split = mg.RayModule(U)

        def forward(self, values):
            rows = mg.F.expand(self.split(values))
            return mg.F.reduce(rows)

    plan = GroupOrder().compile().plan
    effect = next(
        effect
        for effect in plan.structural_effects_by_target.values()
        if isinstance(effect, ReduceEffect)
    )
    assert effect.value_port == effect.members_port
    binding = RowBinding(BlockRef("value"), 0)

    def reach_fixed_point(*, shape_first: bool):
        engine = MicrobatchEngine(plan)
        root = EntityRef(plan.port_domain(effect.target_port), 0)
        child = EntityRef(effect.child_domain, 0)
        expansion = ExpansionRef(effect.child_domain, root)
        value = ItemRef(effect.value_port, child)
        target = ItemRef(effect.target_port, root)
        engine._publish_entity(root)
        engine._publish_entity(child, EntityParent(root, 0))
        engine.advance()

        if shape_first:
            engine._publish_expansion(
                expansion,
                ExpansionOutcome.SUCCEEDED,
                children=(child,),
            )
            engine.advance()
            engine._publish_item(value, ItemOutcome.PRESENT, binding=binding)
            engine.advance()
        else:
            engine._publish_item(value, ItemOutcome.PRESENT, binding=binding)
            engine.advance()
            engine._publish_expansion(
                expansion,
                ExpansionOutcome.SUCCEEDED,
                children=(child,),
            )
            engine.advance()

        assert engine.item_outcome(target) is ItemOutcome.PRESENT
        group = engine.value_binding(target)
        assert isinstance(group, GroupBinding)
        assert group.flat_items == (value,)
        assert not engine._facts
        engine._publish_expansion(
            expansion,
            ExpansionOutcome.SUCCEEDED,
            children=(child,),
        )
        assert not engine._facts
        return engine.item_outcome(target), group

    assert reach_fixed_point(shape_first=True) == reach_fixed_point(
        shape_first=False
    )


class U:
    pass


class ExpandReduce(mg.Pipeline):
    def __init__(self) -> None:
        self.render = mg.RayModule(U)

    def forward(self, values):
        rows = mg.F.expand(self.render(values))
        return mg.F.reduce(rows)


def _start_manual():
    compiled = ExpandReduce().compile()
    plan = compiled.plan
    store = MemoryStore()
    block = store.put(("root",))
    engine = MicrobatchEngine(plan)
    root = engine.admit_sources(
        {plan.source_ports[0]: (RowBinding(block, 0),)}
    )[0]
    call = next(iter(plan.calls))
    selection = engine.reserve_dispatch(call, max_size=1)
    grain = selection.grains[0]
    expanded = next(
        port
        for port, spec in compiled.logical.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )
    output = plan.outputs_by_call[grain.call][0]
    return compiled, engine, root, selection, output, expanded


def test_retry_keeps_grain_identity_and_generation_fences_stale_report():
    compiled, engine, root, selection, output, expanded = _start_manual()
    grain = selection.grains[0]
    assert engine.retry_infrastructure_dispatch(
        selection,
        mg.RecoveryPolicy.abort(infra_retries=1),
    )
    selection = engine.reserve_dispatch(
        grain.call,
        max_size=1,
        parent_bound=False,
    )
    assert selection.grains == (grain,)
    stale = GrainReport(
        grain,
        0,
        (OutputReport(output, expansions=(ExpandedRows(expanded, ()),)),),
    )
    with pytest.raises(CommitError, match="stale generation"):
        engine.commit_success(stale)
    engine.commit_success(
        GrainReport(
            grain,
            1,
            (OutputReport(output, expansions=(ExpandedRows(expanded, ()),)),),
        )
    )
    assert engine.grain_snapshot(grain).phase is GrainPhase.SEALED
    assert engine.item_outcome(ItemRef(output, root)) is ItemOutcome.PRESENT
    assert compiled.plan is engine.plan


def test_aligned_expand_mismatch_has_no_partial_publication():
    class Aligned(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U, num_outputs=2)

        def forward(self, values):
            left_group, right_group = self.render(values)
            return mg.F.expand_aligned(left_group, right_group)

    compiled = Aligned().compile()
    plan = compiled.plan
    store = MemoryStore()
    source_block = store.put(("root",))
    engine = MicrobatchEngine(plan)
    root = engine.admit_sources(
        {plan.source_ports[0]: (RowBinding(source_block, 0),)}
    )[0]
    call = next(iter(plan.calls))
    grain = engine.reserve_dispatch(call, max_size=1).grains[0]
    left_group, right_group = plan.outputs_by_call[grain.call]
    expanded = tuple(
        port
        for port, spec in compiled.logical.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )
    rows = store.put((1, 2, 3))

    with pytest.raises(CommitError, match="cardinality mismatch"):
        engine.commit_success(
            GrainReport(
                grain,
                0,
                (
                    OutputReport(
                        left_group,
                        expansions=(
                            ExpandedRows(
                                expanded[0],
                                (RowBinding(rows, 0), RowBinding(rows, 1)),
                            ),
                        ),
                    ),
                    OutputReport(
                        right_group,
                        expansions=(
                            ExpandedRows(
                                expanded[1],
                                tuple(RowBinding(rows, index) for index in range(3)),
                            ),
                        ),
                    ),
                ),
            )
        )

    assert engine.grain_snapshot(grain).phase is GrainPhase.IN_FLIGHT
    assert engine.expansion_count == 0
    assert engine.entities(plan.port_domain(expanded[0])) == ()
    assert root in engine.entities(root.domain)
