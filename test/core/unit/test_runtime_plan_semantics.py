"""完整 RuntimePlan 驱动的 Ray-free input batch/Worker 语义回归。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rayorch as ro
from rayorch._execution.executor import Executor
from rayorch.failures import GroupFailure, RecordFailure
from rayorch._program.logical import ExpandOrigin
from rayorch._runtime.materialize import materialize_tree
from rayorch._model import (
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
from rayorch._program.plan import BroadcastEffect, ReduceEffect
from rayorch._protocol import (
    BlockRef,
    GrainFailureReport,
    GrainReport,
    DispatchFailure,
    DispatchFailureKind,
    ExpandedRows,
    CallInputLayout,
    GrainInvocation,
    CallOutputLayout,
    PortOutputReport,
    RowBinding,
)
from rayorch._runtime import (
    CommitError,
    ExecutionMicrobatch,
    InputBatchEngine,
)
from rayorch._runtime.state import (
    EntityParent,
    NestedGroupBinding,
    ExpansionRef,
)
from rayorch._execution.worker import (
    Worker,
    WorkerContractError,
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


def _execute_one(worker_target, store: MemoryStore) -> DispatchFailure:
    block = store.put((1,))
    invocation = GrainInvocation(
        GrainRef(CallRef(0), EntityRef(DomainRef(0), 0)),
        0,
        (RowBinding(block, 0),),
    )
    worker = Worker(worker_target, input_layout=CallInputLayout(1))
    result = worker.execute(
        (invocation,),
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


def test_worker_group_failure_dominates_record_failure_per_grain():
    class MixedFailures:
        def run(self, values):
            assert values == [0, 1]
            return (
                [RecordFailure("record-0"), GroupFailure("group-1")],
                [GroupFailure("group-0"), RecordFailure("record-1")],
            )

    store = MemoryStore()
    block = store.put((0, 1))
    plans = tuple(
        GrainInvocation(
            GrainRef(CallRef(0), EntityRef(DomainRef(0), index)),
            0,
            (RowBinding(block, index),),
        )
        for index in range(2)
    )
    result = Worker(MixedFailures, input_layout=CallInputLayout(1)).execute(
        plans,
        (
            CallOutputLayout(PortRef(0)),
            CallOutputLayout(PortRef(1)),
        ),
        store,
    )

    assert not isinstance(result, DispatchFailure)
    assert all(isinstance(report, GrainFailureReport) for report in result)
    assert [report.cause for report in result] == ["group-0", "group-1"]
    assert all(report.suppress_siblings for report in result)
    assert ro.GroupFailure is GroupFailure


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


def run_sync(pipeline: ro.Pipeline, *columns, optimize: bool = True):
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

    engine = InputBatchEngine(plan)
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
        selection = engine.reserve_dispatch(call, max_size=64)
        if selection is None:
            continue
        invocations = tuple(
            engine.grain_invocation(grain) for grain in selection.grains
        )
        reports = workers[call].execute(
            invocations,
            plan.output_layouts_by_call[call],
            store,
        )
        assert not isinstance(reports, DispatchFailure)
        engine.commit_reports(selection, reports)
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

    class DocumentPipeline(ro.Pipeline):
        def __init__(self) -> None:
            self.split = ro.RayModule(Split)
            self.long_enough = ro.RayModule(LongEnough)
            self.summarize = ro.RayModule(Summarize)

        def forward(self, documents, thresholds):
            chunks = ro.F.expand(self.split(documents))
            chunk_thresholds = ro.F.broadcast(thresholds, like=chunks)
            masks = self.long_enough(chunks, chunk_thresholds)
            selected = ro.F.filter(chunks, masks)
            chunk_groups = ro.F.reduce(selected)
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
    class Chained(ro.Pipeline):
        def forward(self, values, bools, membership):
            selected_bools = ro.F.filter(bools, membership)
            return ro.F.filter(values, selected_bools)

    outputs, _compiled, _arena = run_sync(
        Chained(),
        [10, 20, 30],
        [True, False, True],
        [True, True, False],
    )
    assert outputs == [10, ro.OutputIssue(ItemOutcome.DROPPED), ro.OutputIssue(ItemOutcome.DROPPED)]


def test_keyword_only_input_preserves_default_without_identity_driver():
    class KeywordOnly:
        def run(self, values, scale=10, *, masks):
            return [
                value * scale if keep else -1
                for value, keep in zip(values, masks)
            ]

    class KeywordPipeline(ro.Pipeline):
        def __init__(self) -> None:
            self.call = ro.RayModule(KeywordOnly)

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

    class Reordered(ro.Pipeline):
        def __init__(self) -> None:
            self.call = ro.RayModule(Subtract)

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
                -1 if lhs is ro.MISSING and rhs is ro.MISSING else lhs + rhs
                for lhs, rhs in zip(left, right)
            ]

    class AllOptional(ro.Pipeline):
        def __init__(self) -> None:
            self.call = ro.RayModule(MissingAware)

        def forward(self, left, right, masks):
            selected_left = ro.F.filter(left, masks)
            selected_right = ro.F.filter(right, masks)
            return self.call(
                ro.F.optional(selected_left),
                ro.F.optional(selected_right),
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

    class SymmetricDrop(ro.Pipeline):
        def __init__(self) -> None:
            self.call = ro.RayModule(Add)

        def forward(self, left, right, left_mask, right_mask):
            return self.call(
                ro.F.filter(left, left_mask),
                ro.F.filter(right, right_mask),
            )

    outputs, _compiled, _arena = run_sync(
        SymmetricDrop(),
        [10, 20],
        [1, 2],
        [False, True],
        [True, False],
    )
    assert outputs == [ro.OutputIssue(ItemOutcome.DROPPED), ro.OutputIssue(ItemOutcome.DROPPED)]


def test_call_failure_dominates_required_drop_independent_of_arrival_order():
    class Fail:
        def run(self, values):
            return [RecordFailure(f"bad {value}") for value in values]

    class Add:
        def run(self, left, right):
            return [lhs + rhs for lhs, rhs in zip(left, right)]

    class FailureAndDrop(ro.Pipeline):
        def __init__(self) -> None:
            self.fail = ro.RayModule(Fail)
            self.add = ro.RayModule(Add)

        def forward(self, values, masks):
            dropped = ro.F.filter(values, masks)
            failed = self.fail(values)
            return self.add(dropped, failed)

    outputs, _compiled, _arena = run_sync(
        FailureAndDrop(),
        [10, 20],
        [False, False],
    )
    assert outputs == [
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad 10"),
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad 20"),
    ]


def test_record_and_group_failure_have_distinct_no_reduce_semantics():
    class RenderPages:
        def run(self, documents):
            return [
                [(document, ordinal) for ordinal in range(3)]
                for document in documents
            ]

    class FailOneRecord:
        def run(self, pages):
            return [
                RecordFailure("bad page") if page == (1, 1) else page
                for page in pages
            ]

    class FailOneGroup:
        def run(self, pages):
            return [
                GroupFailure("bad document") if page == (1, 1) else page
                for page in pages
            ]

    class PagePipeline(ro.Pipeline):
        def __init__(self, failure_type) -> None:
            self.render = ro.RayModule(RenderPages)
            self.transform = ro.RayModule(failure_type)

        def forward(self, documents):
            pages = ro.F.expand(self.render(documents))
            return self.transform(pages)

    record, _, _ = run_sync(PagePipeline(FailOneRecord), [0, 1])
    group, _, _ = run_sync(PagePipeline(FailOneGroup), [0, 1])

    assert record == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        ro.OutputIssue(ItemOutcome.FAILED, "bad page"),
        (1, 2),
    ]
    assert group == [
        (0, 0),
        (0, 1),
        (0, 2),
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad document"),
        ro.OutputIssue(ItemOutcome.FAILED, "bad document"),
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad document"),
    ]


def test_nested_expand_reduce_preserves_empty_groups():
    @ro.function
    def pages(documents):
        return [[0, 1] if document == 0 else [] for document in documents]

    @ro.function
    def regions(page_numbers):
        return [[] if page == 0 else [10, 11] for page in page_numbers]

    class Nested(ro.Pipeline):
        def forward(self, documents):
            page = ro.F.expand(pages(documents))
            region = ro.F.expand(regions(page))
            return ro.F.reduce(ro.F.reduce(region))

    outputs, _compiled, _arena = run_sync(Nested(), [0, 1])
    assert outputs == [[[], [10, 11]], []]



def _assert_reduce_layouts_match_plan(compiled, engine):
    for effect in compiled.plan.structural_effects_by_target.values():
        if not isinstance(effect, ReduceEffect):
            continue
        for item in engine.ordered_items(effect.target_port):
            if engine.item_outcome(item) is ItemOutcome.PRESENT:
                binding = engine.value_binding(item)
                assert isinstance(binding, NestedGroupBinding)
                assert binding.layout.depth == effect.value_depth + 1


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("keep_pages", [False, True])
@pytest.mark.parametrize("documents", [
    [[], [[1]]],
    [[[]], [[1], [2, 3]]],
    [[], []],
])
def test_nested_filter_preserves_group_shape(documents, keep_pages, optimize):
    @ro.function
    def identity(values):
        return values

    @ro.function
    def keep(values):
        return [keep_pages for _ in values]

    class NestedFiltered(ro.Pipeline):
        def forward(self, roots):
            documents = ro.F.expand(identity(roots))
            pages = ro.F.expand(identity(documents))
            regions = ro.F.expand(identity(pages))
            grouped = ro.F.reduce(regions)
            selected = ro.F.filter(grouped, keep(pages))
            return ro.F.reduce(ro.F.reduce(selected))

    outputs, compiled, engine = run_sync(
        NestedFiltered(), [documents], optimize=optimize,
    )
    assert outputs == [documents if keep_pages else [[] for _ in documents]]
    assert compiled.analysis.group_depth_by_port[compiled.plan.output_tree] == 3
    _assert_reduce_layouts_match_plan(compiled, engine)


@pytest.mark.parametrize("optimize", [False, True])
def test_broadcast_preserves_nested_groups_with_empty_parents(optimize):
    @ro.function
    def identity(values):
        return values

    class NestedBroadcast(ro.Pipeline):
        def forward(self, roots):
            documents = ro.F.expand(identity(roots))
            pages = ro.F.expand(identity(documents))
            regions = ro.F.expand(identity(pages))
            page_groups = ro.F.reduce(regions)
            document_groups = ro.F.reduce(page_groups)
            repeated = ro.F.broadcast(document_groups, like=pages)
            return ro.F.reduce(ro.F.reduce(repeated))

    documents = [[], [[]], [[1], [2, 3]]]
    outputs, compiled, engine = run_sync(
        NestedBroadcast(), [documents], optimize=optimize,
    )
    assert outputs == [[
        [document for _ in document] for document in documents
    ]]
    assert compiled.analysis.group_depth_by_port[compiled.plan.output_tree] == 4
    _assert_reduce_layouts_match_plan(compiled, engine)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
def test_reduce_call_output_shape_tracks_expansion_contract(expanded, optimize):
    @ro.function
    def identity(values):
        return values

    @ro.function
    def keep(values):
        return [True for _ in values]

    class CollectOutputs(ro.Pipeline):
        def forward(self, roots):
            documents = ro.F.expand(identity(roots))
            groups = identity(documents)
            if expanded:
                ro.F.expand(groups)
            selected = ro.F.filter(groups, keep(documents))
            return ro.F.reduce(selected)

    outputs, compiled, engine = run_sync(
        CollectOutputs(), [[], [[], [1, 2]]], optimize=optimize,
    )
    assert outputs == [[], [[], [1, 2]]]
    _assert_reduce_layouts_match_plan(compiled, engine)
    effect = next(
        effect for effect in compiled.plan.structural_effects_by_target.values()
        if isinstance(effect, ReduceEffect)
    )
    assert effect.value_depth == (1 if expanded else 0)


def test_broadcast_chain_optimized_and_baseline_have_exact_outcome_parity():
    @ro.function
    def outer(values):
        return [[value, value + 1] for value in values]

    @ro.function
    def inner(values):
        return [[value * 10, value * 10 + 1] for value in values]

    class BroadcastChain(ro.Pipeline):
        def forward(self, values, masks):
            level_one = ro.F.expand(outer(values))
            level_one_masks = ro.F.broadcast(masks, like=level_one)
            level_two = ro.F.expand(inner(level_one))
            level_two_masks = ro.F.broadcast(level_one_masks, like=level_two)
            return ro.F.filter(level_two, level_two_masks)

    baseline, baseline_compiled, _ = run_sync(
        BroadcastChain(), [1, 3], [True, False], optimize=False
    )
    optimized, optimized_compiled, _ = run_sync(
        BroadcastChain(), [1, 3], [True, False], optimize=True
    )
    assert optimized == baseline
    assert optimized[:4] == [10, 11, 20, 21]
    assert optimized[4:] == [ro.OutputIssue(ItemOutcome.DROPPED)] * 4
    assert baseline_compiled.explanation.rewrites == ()
    assert len(optimized_compiled.explanation.rewrites) == 1


def test_broadcast_fact_order_reaches_the_same_fixed_point():
    class BroadcastOrder(ro.Pipeline):
        def __init__(self) -> None:
            self.split = ro.RayModule(U)
            self.label = ro.RayModule(U)

        def forward(self, values):
            rows = ro.F.expand(self.split(values))
            return ro.F.broadcast(self.label(values), like=rows)

    plan = BroadcastOrder().compile().plan
    effect = next(
        effect
        for effect in plan.structural_effects_by_target.values()
        if isinstance(effect, BroadcastEffect)
    )
    binding = RowBinding(BlockRef("label"), 0)

    def reach_fixed_point(*, source_first: bool):
        engine = InputBatchEngine(plan)
        root = EntityRef(effect.source_domain, 0)
        child = EntityRef(effect.target_domain, 0)
        source = ItemRef(effect.source_port, root)
        target = ItemRef(effect.target_port, child)
        engine._publish_entity(root)
        engine.advance()

        def expand():
            # A real Worker commit publishes both lineage and Expansion facts
            # before advance() can consume their events.
            expansion = ExpansionRef(effect.target_domain, root)
            children = engine._create_children(expansion, 1)
            assert children == (child,)
            engine._publish_expansion(
                expansion, ExpansionOutcome.SUCCEEDED, children=children,
            )

        if source_first:
            engine._publish_item(source, ItemOutcome.PRESENT, binding=binding)
            engine.advance()
            expand()
            engine.advance()
        else:
            expand()
            engine.advance()
            engine._publish_item(source, ItemOutcome.PRESENT, binding=binding)
            engine.advance()

        assert engine.item_outcome(target) is ItemOutcome.PRESENT
        assert engine.value_binding(target) == binding
        assert not engine._fact_queue
        engine._publish_item(source, ItemOutcome.PRESENT, binding=binding)
        assert not engine._fact_queue
        engine._publish_entity(child, EntityParent(root, 0))
        assert not engine._fact_queue
        return engine.item_outcome(target), engine.value_binding(target)

    assert reach_fixed_point(source_first=True) == reach_fixed_point(
        source_first=False
    )


def test_group_fact_order_reaches_the_same_fixed_point():
    class GroupOrder(ro.Pipeline):
        def __init__(self) -> None:
            self.split = ro.RayModule(U)

        def forward(self, values):
            rows = ro.F.expand(self.split(values))
            return ro.F.reduce(rows)

    plan = GroupOrder().compile().plan
    effect = next(
        effect
        for effect in plan.structural_effects_by_target.values()
        if isinstance(effect, ReduceEffect)
    )
    assert effect.value_port == effect.members_port
    binding = RowBinding(BlockRef("value"), 0)

    def reach_fixed_point(*, expansion_first: bool):
        engine = InputBatchEngine(plan)
        root = EntityRef(plan.port_domain(effect.target_port), 0)
        child = EntityRef(effect.child_domain, 0)
        expansion = ExpansionRef(effect.child_domain, root)
        value = ItemRef(effect.value_port, child)
        target = ItemRef(effect.target_port, root)
        engine._publish_entity(root)
        engine._publish_entity(child, EntityParent(root, 0))
        engine.advance()

        if expansion_first:
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
        assert isinstance(group, NestedGroupBinding)
        assert group.flat_items == (value,)
        assert not engine._fact_queue
        engine._publish_expansion(
            expansion,
            ExpansionOutcome.SUCCEEDED,
            children=(child,),
        )
        assert not engine._fact_queue
        return engine.item_outcome(target), group

    assert reach_fixed_point(expansion_first=True) == reach_fixed_point(
        expansion_first=False
    )


class U:
    pass


class ExpandReduce(ro.Pipeline):
    def __init__(self) -> None:
        self.render = ro.RayModule(U)

    def forward(self, values):
        rows = ro.F.expand(self.render(values))
        return ro.F.reduce(rows)


def test_downstream_grain_becomes_dispatchable_before_upstream_stage_finishes():
    """A completed upstream Grain must release downstream work immediately."""

    class Render:
        pass

    class Transform:
        pass

    class StreamingPipeline(ro.Pipeline):
        def __init__(self) -> None:
            self.render = ro.RayModule(Render)
            self.transform = ro.RayModule(Transform)

        def forward(self, documents):
            pages = ro.F.expand(self.render(documents))
            return self.transform(pages)

    compiled = StreamingPipeline().compile()
    plan = compiled.plan
    calls = {
        spec.udf.target: call
        for call, spec in compiled.logical.calls.items()
    }
    expanded_port = next(
        port
        for port, spec in compiled.logical.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )
    store = MemoryStore()
    engine = InputBatchEngine(plan)
    documents = store.put(("doc-0", "doc-1"))
    engine.admit_sources(
        {
            plan.source_ports[0]: (
                RowBinding(documents, 0),
                RowBinding(documents, 1),
            )
        }
    )
    engine.close_admission()

    first_render = engine.reserve_dispatch(calls[Render], max_size=1)
    assert first_render is not None
    assert engine.dispatch_priority(calls[Render]) is not None
    assert engine.dispatch_priority(calls[Transform]) is None

    pages = store.put(("page-0",))
    render_output = plan.outputs_by_call[calls[Render]][0]
    engine.commit_reports(
        first_render,
        (
            GrainReport(
                first_render.grains[0],
                0,
                (
                    PortOutputReport(
                        render_output,
                        expansions=(
                            ExpandedRows(
                                expanded_port,
                                (RowBinding(pages, 0),),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    # The second Render Grain is still READY, but the first completed document
    # has already released page work to the next stage.
    assert engine.dispatch_priority(calls[Render]) is not None
    downstream = engine.reserve_dispatch(calls[Transform], max_size=1)
    assert downstream is not None
    assert len(downstream.grains) == 1
    assert store.get(engine.grain_invocation(downstream.grains[0]).inputs[0]) == "page-0"


def test_progress_summary_uses_expansion_vocabulary():
    engine = InputBatchEngine(ExpandReduce().compile().plan)

    assert engine.progress_summary() == (
        "pending=0, ready=0, grains=0, expansions=0"
    )


def test_group_barrier_linearizes_same_batch_and_late_in_flight_success():
    class RenderThree:
        pass

    class Transform:
        pass

    class Pages(ro.Pipeline):
        def __init__(self) -> None:
            self.render = ro.RayModule(RenderThree)
            self.transform = ro.RayModule(Transform)

        def forward(self, documents):
            pages = ro.F.expand(self.render(documents))
            return self.transform(pages)

    def execute(*, commit_early_success: bool):
        compiled = Pages().compile()
        plan = compiled.plan
        store = MemoryStore()
        engine = InputBatchEngine(plan)
        source = store.put(("document",))
        engine.admit_sources(
            {plan.source_ports[0]: (RowBinding(source, 0),)}
        )
        engine.close_admission()

        render_call = next(
            call for call in plan.calls
            if engine.dispatch_priority(call) is not None
        )
        render = engine.reserve_dispatch(render_call, max_size=1)
        assert render is not None
        render_grain = render.grains[0]
        render_output = plan.outputs_by_call[render_call][0]
        expanded = next(
            port
            for port, spec in compiled.logical.ports.items()
            if isinstance(spec.origin, ExpandOrigin)
        )
        page_block = store.put(("page0", "page1", "page2"))
        engine.commit_reports(
            render,
            (
                GrainReport(
                    render_grain,
                    0,
                    (
                        PortOutputReport(
                            render_output,
                            expansions=(
                                ExpandedRows(
                                    expanded,
                                    tuple(
                                        RowBinding(page_block, index)
                                        for index in range(3)
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        transform_call = next(
            call for call in plan.calls
            if engine.dispatch_priority(call) is not None
        )
        first = engine.reserve_dispatch(transform_call, max_size=1)
        second = engine.reserve_dispatch(transform_call, max_size=2)
        assert first is not None and second is not None
        transform_output = plan.outputs_by_call[transform_call][0]
        result_block = store.put(("page0", "page2"))
        first_success = GrainReport(
            first.grains[0],
            0,
            (PortOutputReport(transform_output, scalar=RowBinding(result_block, 0)),),
        )
        group_failure = GrainFailureReport(
            second.grains[0],
            0,
            "bad document",
            suppress_siblings=True,
        )
        second_success = GrainReport(
            second.grains[1],
            0,
            (PortOutputReport(transform_output, scalar=RowBinding(result_block, 1)),),
        )

        if commit_early_success:
            engine.commit_reports(first, (first_success,))
        # Deliberately reverse report order: success precedes the
        # GroupFailure in the tuple, but the whole result is pre-scanned.
        engine.commit_reports(second, (second_success, group_failure))
        if not commit_early_success:
            engine.commit_reports(
                ExecutionMicrobatch((first_success.grain,)),
                (first_success,),
            )

        assert engine.is_complete()
        return materialize_tree(plan, engine, store)

    assert execute(commit_early_success=False) == [
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad document"),
        ro.OutputIssue(ItemOutcome.FAILED, "bad document"),
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad document"),
    ]
    assert execute(commit_early_success=True) == [
        "page0",
        ro.OutputIssue(ItemOutcome.FAILED, "bad document"),
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad document"),
    ]


@pytest.mark.parametrize(
    "failure_kind, document_count, infra_retries",
    [
        pytest.param("udf", 2, 1, id="udf-mixed"),
        pytest.param("infrastructure", 2, 1, id="infra-mixed"),
        pytest.param("infrastructure", 1, 1, id="infra-all-suppressed"),
        pytest.param("infrastructure", 1, 0, id="infra-all-suppressed-no-budget"),
        pytest.param("infrastructure", 2, 0, id="infra-mixed-budget-exhausted"),
    ],
)
def test_group_barrier_partitions_late_failed_lease_before_recovery(
    failure_kind, document_count, infra_retries,
):
    class Render:
        pass

    class Transform:
        pass

    class Pages(ro.Pipeline):
        def __init__(self) -> None:
            self.render = ro.RayModule(Render)
            self.transform = ro.RayModule(Transform)

        def forward(self, documents):
            return self.transform(ro.F.expand(self.render(documents)))

    compiled = Pages().compile()
    plan = compiled.plan
    store = MemoryStore()
    engine = InputBatchEngine(plan)
    sources = store.put(tuple(f"doc{index}" for index in range(document_count)))
    engine.admit_sources(
        {
            plan.source_ports[0]: tuple(
                RowBinding(sources, index) for index in range(document_count)
            )
        }
    )
    engine.close_admission()
    calls = {spec.udf.target: call for call, spec in plan.calls.items()}

    render = engine.reserve_dispatch(calls[Render], max_size=2)
    assert render is not None
    render_output = plan.outputs_by_call[calls[Render]][0]
    expanded = next(
        port
        for port, spec in compiled.logical.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )
    pages = store.put(("a0", "a1", "b0", "b1"))
    engine.commit_reports(
        render,
        tuple(
            GrainReport(
                grain,
                0,
                (
                    PortOutputReport(
                        render_output,
                        expansions=(
                            ExpandedRows(
                                expanded,
                                (
                                    RowBinding(pages, root_index * 2),
                                    RowBinding(pages, root_index * 2 + 1),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            for root_index, grain in enumerate(render.grains)
        ),
    )

    poison = engine.reserve_dispatch(calls[Transform], max_size=1)
    late = engine.reserve_dispatch(calls[Transform], max_size=3)
    assert poison is not None and late is not None
    transform_output = plan.outputs_by_call[calls[Transform]][0]
    engine.commit_reports(
        poison,
        (
            GrainFailureReport(
                poison.grains[0],
                0,
                "bad document",
                suppress_siblings=True,
            ),
        ),
    )

    snapshots_before = tuple(engine.grain_snapshot(grain) for grain in late.grains)
    items_before = dict(engine._state.items)
    progress_before = engine.progress_summary()
    assert all(snapshot.phase is GrainPhase.IN_FLIGHT for snapshot in snapshots_before)
    assert not engine.is_complete()

    if failure_kind == "udf":
        retried = engine.apply_udf_recovery(
            late,
            ro.RecoveryPolicy.retry_tail(),
            RuntimeError("opaque UDF failure"),
        )
    else:
        retried = engine.retry_infrastructure_dispatch(
            late,
            ro.RecoveryPolicy.abort(infra_retries=infra_retries),
        )
    if failure_kind == "infrastructure" and document_count == 2 and infra_retries == 0:
        assert retried is None
        assert tuple(engine.grain_snapshot(grain) for grain in late.grains) == snapshots_before
        assert engine._state.items == items_before
        assert engine.progress_summary() == progress_before
        assert engine.dispatch_priority(calls[Transform]) is None
        assert not engine.is_complete()
        return

    blocked, *live = late.grains
    assert retried == len(live)
    blocked_item = ItemRef(transform_output, blocked.entity)
    assert engine.item_outcome(blocked_item) is ItemOutcome.SUPPRESSED
    assert engine._state.items[blocked_item].cause == "bad document"
    assert engine.grain_snapshot(blocked).phase is GrainPhase.SEALED
    assert engine.grain_snapshot(blocked).generation == 0
    if not live:
        assert retried == 0
        assert engine.grain_snapshot(blocked).infra_failures == 0
        assert engine.dispatch_priority(calls[Transform]) is None
        assert engine.is_complete()
        assert materialize_tree(plan, engine, store) == [
            ro.OutputIssue(ItemOutcome.FAILED, "bad document"),
            ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad document"),
        ]
        return

    assert {engine.grain_snapshot(grain).generation for grain in live} == {1}
    expected_infra = 1 if failure_kind == "infrastructure" else 0
    assert {
        engine.grain_snapshot(grain).infra_failures for grain in live
    } == {expected_infra}

    retry = engine.reserve_dispatch(calls[Transform], max_size=3)
    assert retry is not None
    assert retry.grains == tuple(live)
    assert retry.udf_retries == (1 if failure_kind == "udf" else 0)
    results = store.put(("B0", "B1"))
    engine.commit_reports(
        retry,
        tuple(
            GrainReport(
                grain,
                1,
                (
                    PortOutputReport(
                        transform_output,
                        scalar=RowBinding(results, index),
                    ),
                ),
            )
            for index, grain in enumerate(retry.grains)
        ),
    )

    assert engine.is_complete()
    assert materialize_tree(plan, engine, store) == [
        ro.OutputIssue(ItemOutcome.FAILED, "bad document"),
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "bad document"),
        "B0",
        "B1",
    ]


def test_group_barrier_closes_waiting_sibling_at_input_admission():
    class Render:
        pass

    class Left:
        pass

    class Right:
        pass

    class Combine:
        pass

    class WaitingSibling(ro.Pipeline):
        def __init__(self) -> None:
            self.render = ro.RayModule(Render)
            self.left = ro.RayModule(Left)
            self.right = ro.RayModule(Right)
            self.combine = ro.RayModule(Combine)

        def forward(self, documents):
            pages = ro.F.expand(self.render(documents))
            return self.combine(self.left(pages), self.right(pages))

    compiled = WaitingSibling().compile()
    plan = compiled.plan
    store = MemoryStore()
    engine = InputBatchEngine(plan)
    source = store.put(("document",))
    engine.admit_sources({plan.source_ports[0]: (RowBinding(source, 0),)})
    engine.close_admission()
    calls = {
        spec.udf.target: call for call, spec in plan.calls.items()
    }

    render = engine.reserve_dispatch(calls[Render], max_size=1)
    assert render is not None
    render_output = plan.outputs_by_call[calls[Render]][0]
    expanded = next(
        port
        for port, spec in compiled.logical.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )
    pages = store.put(("page0", "page1"))
    engine.commit_reports(
        render,
        (
            GrainReport(
                render.grains[0],
                0,
                (
                    PortOutputReport(
                        render_output,
                        expansions=(
                            ExpandedRows(
                                expanded,
                                (RowBinding(pages, 0), RowBinding(pages, 1)),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    left = engine.reserve_dispatch(calls[Left], max_size=2)
    right_first = engine.reserve_dispatch(calls[Right], max_size=1)
    right_second = engine.reserve_dispatch(calls[Right], max_size=1)
    assert left is not None and right_first is not None and right_second is not None
    left_output = plan.outputs_by_call[calls[Left]][0]
    right_output = plan.outputs_by_call[calls[Right]][0]
    left_values = store.put(("left0", "left1"))
    right_values = store.put(("right0", "right1"))
    engine.commit_reports(
        left,
        tuple(
            GrainReport(
                grain,
                0,
                (
                    PortOutputReport(
                        left_output,
                        scalar=RowBinding(left_values, index),
                    ),
                ),
            )
            for index, grain in enumerate(left.grains)
        ),
    )
    engine.commit_reports(
        right_first,
        (
            GrainReport(
                right_first.grains[0],
                0,
                (
                    PortOutputReport(
                        right_output,
                        scalar=RowBinding(right_values, 0),
                    ),
                ),
            ),
        ),
    )

    combine = engine.reserve_dispatch(calls[Combine], max_size=1)
    assert combine is not None
    engine.commit_reports(
        combine,
        (
            GrainFailureReport(
                combine.grains[0],
                0,
                "parent invalid",
                suppress_siblings=True,
            ),
        ),
    )
    engine.commit_reports(
        right_second,
        (
            GrainReport(
                right_second.grains[0],
                0,
                (
                    PortOutputReport(
                        right_output,
                        scalar=RowBinding(right_values, 1),
                    ),
                ),
            ),
        ),
    )

    assert engine.is_complete()
    assert materialize_tree(plan, engine, store) == [
        ro.OutputIssue(ItemOutcome.FAILED, "parent invalid"),
        ro.OutputIssue(ItemOutcome.SUPPRESSED, "parent invalid"),
    ]
    waiting = GrainRef(calls[Combine], right_second.grains[0].entity)
    assert engine.grain_snapshot(waiting).phase is GrainPhase.SEALED


def _start_manual():
    compiled = ExpandReduce().compile()
    plan = compiled.plan
    store = MemoryStore()
    block = store.put(("root",))
    engine = InputBatchEngine(plan)
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
        ro.RecoveryPolicy.abort(infra_retries=1),
    )
    selection = engine.reserve_dispatch(
        grain.call,
        max_size=1,
    )
    assert selection.grains == (grain,)
    stale = GrainReport(
        grain,
        0,
        (PortOutputReport(output, expansions=(ExpandedRows(expanded, ()),)),),
    )
    with pytest.raises(CommitError, match="stale generation"):
        engine.commit_reports(ExecutionMicrobatch((stale.grain,)), (stale,))
    current = GrainReport(
        grain,
        1,
        (PortOutputReport(output, expansions=(ExpandedRows(expanded, ()),)),),
    )
    engine.commit_reports(ExecutionMicrobatch((current.grain,)), (current,))
    assert engine.grain_snapshot(grain).phase is GrainPhase.SEALED
    assert engine.item_outcome(ItemRef(output, root)) is ItemOutcome.PRESENT
    assert compiled.plan is engine.plan


def test_aligned_expand_mismatch_has_no_partial_publication():
    class Aligned(ro.Pipeline):
        def __init__(self) -> None:
            self.render = ro.RayModule(U, num_outputs=2)

        def forward(self, values):
            left_group, right_group = self.render(values)
            return ro.F.expand_aligned(left_group, right_group)

    compiled = Aligned().compile()
    plan = compiled.plan
    store = MemoryStore()
    source_block = store.put(("root",))
    engine = InputBatchEngine(plan)
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

    malformed = GrainReport(
        grain,
        0,
        (
            PortOutputReport(
                left_group,
                expansions=(
                    ExpandedRows(
                        expanded[0],
                        (RowBinding(rows, 0), RowBinding(rows, 1)),
                    ),
                ),
            ),
            PortOutputReport(
                right_group,
                expansions=(
                    ExpandedRows(
                        expanded[1],
                        tuple(RowBinding(rows, index) for index in range(3)),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(CommitError, match="cardinality mismatch"):
        engine.commit_reports(ExecutionMicrobatch((malformed.grain,)), (malformed,))

    assert engine.grain_snapshot(grain).phase is GrainPhase.IN_FLIGHT
    assert engine.expansion_count == 0
    assert engine.entities(plan.port_domain(expanded[0])) == ()
    assert root in engine.entities(root.domain)


def test_group_batch_payload_preflight_installs_no_partial_barrier():
    class Identity:
        pass

    class Pair(ro.Pipeline):
        def __init__(self) -> None:
            self.identity = ro.RayModule(Identity)

        def forward(self, values):
            return self.identity(values)

    plan = Pair().compile().plan
    store = MemoryStore()
    block = store.put((1, 2))
    engine = InputBatchEngine(plan)
    engine.admit_sources(
        {
            plan.source_ports[0]: tuple(
                RowBinding(block, index) for index in range(2)
            )
        }
    )
    call = next(iter(plan.calls))
    selection = engine.reserve_dispatch(call, max_size=2)
    assert selection is not None

    malformed = GrainReport(selection.grains[0], 0, ())
    group_failure = GrainFailureReport(
        selection.grains[1],
        0,
        "must not leak",
        suppress_siblings=True,
    )
    with pytest.raises(CommitError, match="exactly match Call outputs"):
        engine.commit_reports(selection, (group_failure, malformed))

    assert {
        engine.grain_snapshot(grain).phase for grain in selection.grains
    } == {GrainPhase.IN_FLIGHT}
    assert not engine._suppression_barriers.anchors_for(call)
    assert engine.item_count == 2  # source Items only
