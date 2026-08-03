"""Focused vertical tests for the backend-neutral V3 runtime core."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rayorch.experimental.multigrain_v3.model import graph as graph_model
from rayorch.experimental.multigrain_v3.model import semantics as sem
from rayorch.experimental.multigrain_v3.model import state
from rayorch.experimental.multigrain_v3.runtime.coordinator import (
    CompiledGraphRunCoordinator,
    ExecutorSession,
    InlineTransport,
    RootContext,
    RootPhase,
    SourceRecord,
)
from rayorch.experimental.multigrain_v3.runtime.dispatch import (
    BatchSelection,
    CommitDelta,
    DispatchInvariantError,
    LocalTransaction,
    PreparedBlock,
    PreparedItemBinding,
    PreparedOutcome,
    PreparedValueNode,
    PendingDisposition,
    prepare_dispatch,
)
from rayorch.experimental.multigrain_v3.runtime.planner import (
    CreditManager,
    EventPlanner,
    ReadyScheduler,
    ReceiptStore,
    ScopeTracker,
)


@dataclass(frozen=True, slots=True)
class _Binding:
    """Minimal compiled input binding used by planner-only tests."""

    role: str
    port: sem.PortId


@dataclass(frozen=True, slots=True)
class _Port:
    """Minimal compiled output port used by planner-only tests."""

    id: sem.PortId


@dataclass(frozen=True, slots=True)
class _Node:
    """Minimal node carrying exactly the attributes consumed by EventPlanner."""

    id: sem.NodeId
    op: object
    inputs: tuple[_Binding, ...]
    outputs: tuple[_Port, ...]


@dataclass(frozen=True, slots=True)
class _Graph:
    """Small immutable graph facade for system-operation tests."""

    nodes: tuple[_Node, ...]
    consumers_by_port: dict[sem.PortId, tuple[sem.NodeId, ...]]
    scope_plans: dict[sem.ScopeDefId, graph_model.ScopePlan]


@dataclass(frozen=True, slots=True)
class _LocalGraph:
    """Graph token used by the source-level InlineTransport fallback."""

    nodes: tuple[object, ...] = ()


def _limits(**changes: int) -> state.RuntimeLimits:
    """Build generous positive hard limits with optional focused overrides."""

    values = {
        "max_active_roots": 8,
        "max_live_grains": 256,
        "max_live_occurrences": 256,
        "max_occurrences_per_root": 128,
        "max_live_structural_edges": 512,
        "max_structural_edges_per_root": 256,
        "max_scope_width": 128,
        "max_pending_expansions": 64,
        "max_pending_dispatches": 64,
        "max_local_events": 512,
        "max_gather_depth": 32,
        "max_gather_nodes_per_entry": 512,
        "max_refs_per_dispatch": 128,
        "max_manifest_bytes": 1_000_000,
        "max_error_message_bytes": 4096,
        "max_buffered_results": 8,
        "max_detached_result_nodes": 512,
        "max_detached_result_refs": 128,
    }
    values.update(changes)
    return state.RuntimeLimits(**values)


def _runtime(
    graph: _Graph,
    run: sem.RunId,
    *,
    limits: state.RuntimeLimits | None = None,
):
    """Create model stores and one wired EventPlanner for a test graph."""

    blocks = state.BlockStore()
    values = state.ValueNodeStore(blocks)
    receipts = ReceiptStore(values)
    controls = state.ControlStore()
    grains = sem.GrainStore()
    errors = sem.ErrorStore()
    scopes = ScopeTracker()
    credits = CreditManager(limits or _limits())
    scheduler = ReadyScheduler(graph, grains)
    planner = EventPlanner(
        graph,
        run,
        receipts=receipts,
        controls=controls,
        grains=grains,
        values=values,
        scopes=scopes,
        scheduler=scheduler,
        credits=credits,
        errors=errors,
        expansion_chunk_size=2,
    )
    return SimpleNamespace(
        planner=planner,
        blocks=blocks,
        values=values,
        receipts=receipts,
        controls=controls,
        grains=grains,
        errors=errors,
        scopes=scopes,
        credits=credits,
        scheduler=scheduler,
    )


def _publish(runtime, receipt: sem.Receipt) -> None:
    """Publish a receipt and append its downstream event."""

    assert runtime.planner.publish(receipt)


def _drain(runtime) -> None:
    """Drain all bounded local turns and fail if work does not converge."""

    turns = 0
    while runtime.planner.events:
        runtime.planner.drain(32)
        turns += 1
        assert turns < 100


def _present_list(
    runtime,
    *,
    run: sem.RunId,
    root: sem.RootId,
    port: sem.PortId,
    entity: sem.EntityId,
    values: tuple[object, ...],
) -> sem.Receipt:
    """Install one flat structural list and return its PRESENT receipt."""

    item = sem.ItemRef(port, entity)
    block = runtime.blocks.install(values, len(values))
    value_node = runtime.values.create_flat_list(block, 0, len(values))
    runtime.values.bind_item(root, item, value_node)
    producer = sem.GrainId.source(run, sem.NodeId(100), 0)
    return sem.Receipt(
        item,
        sem.OccurrenceContext(root),
        sem.ReceiptState.PRESENT,
        producer,
    )


def test_interleaved_reduce_settles_by_scope_and_ordinal() -> None:
    """Interleaved child arrivals must not mix independent closure slots."""

    run = sem.RunId.new()
    root = sem.RootId.for_source(run, 0)
    context = sem.OccurrenceContext(root)
    tracker = ScopeTracker()
    reduce_node = sem.NodeId(9)
    input_port = sem.PortId(3)
    parent_port = sem.PortId(0)
    definition = sem.ScopeDefId(0)
    producer = sem.GrainId.source(run, sem.NodeId(0), 0)

    closures = []
    for index in range(2):
        parent = sem.ItemRef(parent_port, sem.EntityId.source(run, index))
        scope = sem.ScopeInstanceId.for_expand(run, sem.NodeId(2), parent)
        tracker.create_pending(
            scope,
            definition,
            root,
            parent,
            context,
            producer,
        )
        lease = sem.StructuralLeaseId.derive("test-lease", scope)
        tracker.open(scope, 2, lease)
        closures.append(
            tracker.ensure_closure(reduce_node, scope, input_port)
        )

    def receipt(scope_index: int, ordinal: int, present: bool) -> sem.Receipt:
        entity = sem.EntityId.expand(
            run,
            sem.NodeId(2),
            tracker.instances[closures[scope_index].scope].parent_item,
            ordinal,
        )
        return sem.Receipt(
            sem.ItemRef(input_port, entity),
            context.push(closures[scope_index].scope, ordinal),
            (
                sem.ReceiptState.PRESENT
                if present
                else sem.ReceiptState.NORMAL_ABSENCE
            ),
            sem.GrainId.map(
                run,
                sem.NodeId(4),
                (sem.ItemRef(input_port, entity),),
            ),
        )

    assert not tracker.settle(reduce_node, receipt(0, 1, False))
    assert not tracker.settle(reduce_node, receipt(1, 0, True))
    assert tracker.settle(reduce_node, receipt(0, 0, True))
    assert tracker.settle(reduce_node, receipt(1, 1, True))
    assert [slot.context.positions[-1].ordinal for slot in closures[0].slots] == [
        0,
        1,
    ]
    assert len(closures[0].present_members()) == 1
    assert len(closures[1].present_members()) == 2


def test_filter_false_publishes_absence_tombstone_without_alias() -> None:
    """FILTER false must publish NORMAL_ABSENCE and retain no output value."""

    run = sem.RunId.new()
    root = sem.RootId.for_source(run, 0)
    context = sem.OccurrenceContext(root)
    entity = sem.EntityId.source(run, 0)
    mask_port, target_port, output_port = (
        sem.PortId(0),
        sem.PortId(1),
        sem.PortId(2),
    )
    node = _Node(
        sem.NodeId(2),
        graph_model.FilterOp(mask_port, target_port),
        (
            _Binding("mask", mask_port),
            _Binding("target", target_port),
        ),
        (_Port(output_port),),
    )
    graph = _Graph(
        (node,),
        {
            mask_port: (node.id,),
            target_port: (node.id,),
            output_port: (),
        },
        {},
    )
    runtime = _runtime(graph, run)
    producer = sem.GrainId.source(run, sem.NodeId(0), 0)

    for port, value in ((mask_port, False), (target_port, "payload")):
        item = sem.ItemRef(port, entity)
        block = runtime.blocks.install((value,), 1)
        value_node = runtime.values.create_scalar(block, 0)
        runtime.values.bind_item(root, item, value_node)
        if port == mask_port:
            runtime.controls.install_bool(root, item, False)
        _publish(
            runtime,
            sem.Receipt(
                item,
                context,
                sem.ReceiptState.PRESENT,
                producer,
            ),
        )

    _drain(runtime)
    output = sem.ItemRef(output_port, entity)
    receipt = runtime.receipts.require(output)
    assert receipt.state is sem.ReceiptState.NORMAL_ABSENCE
    assert not runtime.values.has_binding(output)
    assert len(runtime.controls) == 0


def test_empty_fanout_immediately_reduces_to_present_empty_list() -> None:
    """EXPAND N=0 must close REDUCE without waiting for child receipts."""

    run = sem.RunId.new()
    root = sem.RootId.for_source(run, 0)
    entity = sem.EntityId.source(run, 0)
    input_port, expanded_port, reduced_port = (
        sem.PortId(0),
        sem.PortId(1),
        sem.PortId(2),
    )
    scope_def = sem.ScopeDefId(0)
    expand_node = _Node(
        sem.NodeId(1),
        graph_model.ExpandOp(input_port, scope_def),
        (_Binding("input", input_port),),
        (_Port(expanded_port),),
    )
    reduce_node = _Node(
        sem.NodeId(2),
        graph_model.ReduceOp(expanded_port, scope_def),
        (_Binding("input", expanded_port),),
        (_Port(reduced_port),),
    )
    graph = _Graph(
        (expand_node, reduce_node),
        {
            input_port: (expand_node.id,),
            expanded_port: (reduce_node.id,),
            reduced_port: (),
        },
        {
            scope_def: graph_model.ScopePlan(
                scope_def,
                expand_node.id,
                (reduce_node.id,),
            )
        },
    )
    runtime = _runtime(graph, run)
    origin = _present_list(
        runtime,
        run=run,
        root=root,
        port=input_port,
        entity=entity,
        values=(),
    )
    _publish(runtime, origin)
    _drain(runtime)

    output = sem.ItemRef(reduced_port, entity)
    assert runtime.receipts.require(output).state is sem.ReceiptState.PRESENT
    final_node = runtime.values.get(runtime.values.node_for_item(output))
    assert isinstance(final_node, state.CompositeListNode)
    assert final_node.children == ()
    assert not runtime.planner.pending_expansions


@pytest.mark.parametrize(
    ("scope_terminal", "expected_receipt"),
    [
        ("absent", sem.ReceiptState.NORMAL_ABSENCE),
        ("failed", sem.ReceiptState.SUPPRESSED),
    ],
)
def test_origin_terminal_scope_closes_reduce_without_children(
    scope_terminal: str,
    expected_receipt: sem.ReceiptState,
) -> None:
    """ScopeAbsent/ScopeFailed must close REDUCE without inventing a width."""

    run = sem.RunId.new()
    root = sem.RootId.for_source(run, 0)
    context = sem.OccurrenceContext(root)
    parent = sem.ItemRef(sem.PortId(0), sem.EntityId.source(run, 0))
    input_port, output_port = sem.PortId(1), sem.PortId(2)
    scope_def = sem.ScopeDefId(0)
    reduce_node = _Node(
        sem.NodeId(2),
        graph_model.ReduceOp(input_port, scope_def),
        (_Binding("input", input_port),),
        (_Port(output_port),),
    )
    graph = _Graph(
        (reduce_node,),
        {input_port: (reduce_node.id,), output_port: ()},
        {
            scope_def: graph_model.ScopePlan(
                scope_def,
                sem.NodeId(1),
                (reduce_node.id,),
            )
        },
    )
    runtime = _runtime(graph, run)
    expand_grain = sem.GrainId.expand(run, sem.NodeId(1), parent)
    scope = sem.ScopeInstanceId.for_expand(run, sem.NodeId(1), parent)
    expand_spec = sem.GrainSpec(
        expand_grain,
        run,
        root,
        sem.NodeId(1),
        context,
        (sem.RoleBinding("input", (parent,)),),
        sem.ExpandOutputRange(input_port, parent, 0),
    )
    runtime.grains.ensure(expand_spec)
    runtime.grains.seal(
        expand_grain,
        (
            sem.Skipped((parent,))
            if scope_terminal == "absent"
            else sem.Failed(
                sem.ErrorId.for_grain(run, expand_grain, "test-failure")
            )
        ),
    )
    runtime.scopes.create_pending(
        scope,
        scope_def,
        root,
        parent,
        context,
        expand_grain,
    )
    if scope_terminal == "absent":
        origin = sem.Receipt(
            parent,
            context,
            sem.ReceiptState.NORMAL_ABSENCE,
            sem.GrainId.source(run, sem.NodeId(0), 0),
        )
        event = runtime.scopes.mark_absent(scope, origin)
    else:
        event = runtime.scopes.mark_failed(scope, expand_grain)
    runtime.planner.enqueue(event)
    _drain(runtime)

    output = sem.ItemRef(output_port, parent.entity)
    assert runtime.receipts.require(output).state is expected_receipt
    assert not runtime.values.has_binding(output)


def test_nested_expand_reduce_preserves_lifo_list_shape() -> None:
    """Inner REDUCE must restore outer ordinals instead of flattening them."""

    run = sem.RunId.new()
    root = sem.RootId.for_source(run, 0)
    entity = sem.EntityId.source(run, 0)
    ports = tuple(sem.PortId(index) for index in range(5))
    outer_def, inner_def = sem.ScopeDefId(0), sem.ScopeDefId(1)
    outer_expand = _Node(
        sem.NodeId(1),
        graph_model.ExpandOp(ports[0], outer_def),
        (_Binding("input", ports[0]),),
        (_Port(ports[1]),),
    )
    inner_expand = _Node(
        sem.NodeId(2),
        graph_model.ExpandOp(ports[1], inner_def),
        (_Binding("input", ports[1]),),
        (_Port(ports[2]),),
    )
    inner_reduce = _Node(
        sem.NodeId(3),
        graph_model.ReduceOp(ports[2], inner_def),
        (_Binding("input", ports[2]),),
        (_Port(ports[3]),),
    )
    outer_reduce = _Node(
        sem.NodeId(4),
        graph_model.ReduceOp(ports[3], outer_def),
        (_Binding("input", ports[3]),),
        (_Port(ports[4]),),
    )
    graph = _Graph(
        (outer_expand, inner_expand, inner_reduce, outer_reduce),
        {
            ports[0]: (outer_expand.id,),
            ports[1]: (inner_expand.id,),
            ports[2]: (inner_reduce.id,),
            ports[3]: (outer_reduce.id,),
            ports[4]: (),
        },
        {
            outer_def: graph_model.ScopePlan(
                outer_def,
                outer_expand.id,
                (outer_reduce.id,),
            ),
            inner_def: graph_model.ScopePlan(
                inner_def,
                inner_expand.id,
                (inner_reduce.id,),
            ),
        },
    )
    runtime = _runtime(graph, run)

    block = runtime.blocks.install(("a", "b", "c"), 3)
    first = runtime.values.create_flat_list(block, 0, 2)
    second = runtime.values.create_flat_list(block, 2, 3)
    outer = runtime.values.create_composite((first, second))
    origin_item = sem.ItemRef(ports[0], entity)
    runtime.values.bind_item(root, origin_item, outer)
    _publish(
        runtime,
        sem.Receipt(
            origin_item,
            sem.OccurrenceContext(root),
            sem.ReceiptState.PRESENT,
            sem.GrainId.source(run, sem.NodeId(0), 0),
        ),
    )
    _drain(runtime)

    final_item = sem.ItemRef(ports[4], entity)
    final = runtime.values.get(runtime.values.node_for_item(final_item))
    assert isinstance(final, state.CompositeListNode)
    inner_values = tuple(runtime.values.get(child) for child in final.children)
    assert all(isinstance(value, state.CompositeListNode) for value in inner_values)
    assert tuple(len(value.children) for value in inner_values) == (2, 1)


def test_local_transaction_applies_success_and_rejects_conflict() -> None:
    """A preflight conflict must not alter an already committed output."""

    run = sem.RunId.new()
    root = sem.RootId.for_source(run, 0)
    context = sem.OccurrenceContext(root)
    item = sem.ItemRef(sem.PortId(1), sem.EntityId.source(run, 0))
    grain_id = sem.GrainId.map(run, sem.NodeId(1), ())
    spec = sem.GrainSpec(
        grain_id,
        run,
        root,
        sem.NodeId(1),
        context,
        (),
        (item,),
    )
    blocks = state.BlockStore()
    values = state.ValueNodeStore(blocks)
    receipts = ReceiptStore(values)
    grains = sem.GrainStore()
    grains.ensure(spec)
    token = grains.reserve(grain_id)
    stores = SimpleNamespace(
        blocks=blocks,
        values=values,
        controls=state.ControlStore(),
        grains=grains,
        receipts=receipts,
        credits=None,
    )
    events: list[object] = []
    transaction = LocalTransaction(stores, event_sink=events)
    receipt = sem.Receipt(
        item,
        context,
        sem.ReceiptState.PRESENT,
        grain_id,
    )
    delta = CommitDelta(
        blocks=(PreparedBlock(sem.BlockId(0), ("value",), 1),),
        value_nodes=(
            PreparedValueNode(
                sem.ValueNodeId(0),
                state.ScalarNode(sem.BlockId(0), 0),
            ),
        ),
        item_bindings=(
            PreparedItemBinding(root, item, sem.ValueNodeId(0)),
        ),
        grain_outcomes=(PreparedOutcome(grain_id, sem.Success(), token),),
        receipts=(receipt,),
    )
    dispatch = sem.DispatchId.new()
    pending = SimpleNamespace(
        prepared=SimpleNamespace(
            id=dispatch,
            run=run,
            node=sem.NodeId(1),
            attempts=(token,),
        ),
        lease=None,
        decision=None,
        semantic_applied=False,
    )
    manifest = SimpleNamespace(
        header=SimpleNamespace(
            run=run,
            dispatch=dispatch,
            node=sem.NodeId(1),
            attempts=(token,),
            lease=None,
        ),
        semantic_delta=delta,
    )
    decision = transaction.prepare(pending, manifest)
    assert decision.kind is PendingDisposition.ACCEPTED
    result = transaction.apply_disposition(pending, decision)
    assert result.applied and result.events == (receipt,)
    assert receipts.require(item) == receipt
    assert values.has_binding(item)

    conflicting = sem.Receipt(
        item,
        context,
        sem.ReceiptState.NORMAL_ABSENCE,
        sem.GrainId.derive("other-producer", run),
    )
    with pytest.raises(DispatchInvariantError, match="conflicting receipt"):
        transaction.apply(CommitDelta(receipts=(conflicting,)))
    assert receipts.require(item) == receipt
    assert values.has_binding(item)


def test_prepared_dispatch_interns_blocks_across_roots() -> None:
    """One same-node batch must share a dispatch-wide block ref table."""

    run = sem.RunId.new()
    node = sem.NodeId(1)
    port = sem.PortId(0)
    blocks = state.BlockStore()
    values = state.ValueNodeStore(blocks)
    grains = sem.GrainStore()
    block = blocks.install(("a", "b"), 2)
    grain_ids = []

    for source_seq in range(2):
        root = sem.RootId.for_source(run, source_seq)
        item = sem.ItemRef(port, sem.EntityId.source(run, source_seq))
        value_node = values.create_scalar(block, source_seq)
        values.bind_item(root, item, value_node)
        grain = sem.GrainId.map(run, node, (item,))
        grains.ensure(
            sem.GrainSpec(
                grain,
                run,
                root,
                node,
                sem.OccurrenceContext(root),
                (sem.RoleBinding("value", (item,)),),
                (sem.ItemRef(sem.PortId(1), item.entity),),
            )
        )
        grain_ids.append(grain)

    prepared = prepare_dispatch(
        BatchSelection(node, tuple(grain_ids)),
        SimpleNamespace(grains=grains, values=values, blocks=blocks),
        run=run,
        limits=_limits(),
    )
    assert prepared.block_ids == (block,)
    assert len(prepared.attempts) == 2
    assert {
        entry.role_trees[0].ref_slot for entry in prepared.wire_entries
    } == {0}
    assert [
        entry.role_trees[0].row for entry in prepared.wire_entries
    ] == [0, 1]


def test_executor_session_allows_only_one_active_run() -> None:
    """A session must reject overlap and allow a new run after exhaustion."""

    transport = InlineTransport(
        lambda values: tuple(value * 2 for value in values)
    )
    session = ExecutorSession(
        _LocalGraph(),
        transport=transport,
        limits=_limits(max_active_roots=4, max_buffered_results=4),
        batch_size=2,
    )
    first = session.run([1, 2, 3])
    with pytest.raises(RuntimeError, match="active run"):
        session.run([4])
    assert [
        result.outputs["output"].value.get()
        for result in first.collect()
    ] == [2, 4, 6]
    second = session.run([4])
    assert second.collect()[0].outputs["output"].value.get() == 8
    session.close()


def test_run_stream_admits_source_iterable_boundedly() -> None:
    """Reading one result must not eagerly consume an unbounded source."""

    consumed: list[int] = []

    def source():
        for value in range(100):
            consumed.append(value)
            yield value

    session = ExecutorSession(
        _LocalGraph(),
        transport=InlineTransport(lambda values: values),
        limits=_limits(max_active_roots=2, max_buffered_results=1),
        batch_size=2,
    )
    stream = session.run(source())
    assert next(stream).source_seq == 0
    assert consumed == [0, 1]
    stream.close()
    session.close()


def test_root_results_remain_input_ordered_across_root_batches() -> None:
    """Cross-root batching must not change source-order result delivery."""

    transport = InlineTransport(
        lambda values: tuple({"value": value} for value in values)
    )
    session = ExecutorSession(
        _LocalGraph(),
        transport=transport,
        limits=_limits(max_active_roots=4, max_buffered_results=2),
        batch_size=2,
    )
    results = session.run(range(6)).collect()
    assert [result.source_seq for result in results] == list(range(6))
    assert [
        result.outputs["value"].value.get() for result in results
    ] == list(range(6))
    assert transport.submitted_batches == [(0, 1), (2, 3), (4, 5)]
    session.close()


def test_remove_root_reclaims_resolved_invocation_index() -> None:
    """Resolved invocation history must be reclaimed through its root index."""

    run = sem.RunId.new()
    runtime = _runtime(_Graph((), {}, {}), run)
    first = sem.RootId.for_source(run, 0)
    second = sem.RootId.for_source(run, 1)
    first_key = (sem.NodeId(1), first, ())
    second_key = (sem.NodeId(1), second, ())
    runtime.planner._resolved_invocations.update((first_key, second_key))
    runtime.planner._resolved_by_root[first] = {first_key}
    runtime.planner._resolved_by_root[second] = {second_key}

    runtime.planner.remove_root(first)

    assert first_key not in runtime.planner._resolved_invocations
    assert second_key in runtime.planner._resolved_invocations
    assert first not in runtime.planner._resolved_by_root


def test_receipt_publish_reserves_event_before_mutation() -> None:
    """Event-credit failure must not leave an unroutable published receipt."""

    run = sem.RunId.new()
    runtime = _runtime(
        _Graph((), {}, {}),
        run,
        limits=_limits(max_local_events=1),
    )
    runtime.planner.enqueue(object())
    root = sem.RootId.for_source(run, 0)
    item = sem.ItemRef(sem.PortId(0), sem.EntityId.source(run, 0))
    receipt = sem.Receipt(
        item,
        sem.OccurrenceContext(root),
        sem.ReceiptState.NORMAL_ABSENCE,
        sem.GrainId.source(run, sem.NodeId(0), 0),
    )

    with pytest.raises(Exception, match="max_local_events"):
        runtime.planner.publish(receipt)

    assert runtime.receipts.get(item) is None
    assert runtime.credits.local_events == 1


def test_transaction_reserves_all_receipt_events_atomically() -> None:
    """A multi-receipt delta must reserve every event before writing any fact."""

    run = sem.RunId.new()
    runtime = _runtime(
        _Graph((), {}, {}),
        run,
        limits=_limits(max_local_events=1),
    )
    root = sem.RootId.for_source(run, 0)
    context = sem.OccurrenceContext(root)
    producer = sem.GrainId.source(run, sem.NodeId(0), 0)
    receipts = tuple(
        sem.Receipt(
            sem.ItemRef(sem.PortId(index), sem.EntityId.source(run, 0)),
            context,
            sem.ReceiptState.NORMAL_ABSENCE,
            producer,
        )
        for index in range(2)
    )
    transaction = LocalTransaction(runtime, event_sink=runtime.planner)

    with pytest.raises(DispatchInvariantError, match="complete transaction"):
        transaction.apply(CommitDelta(receipts=receipts))

    assert all(runtime.receipts.get(receipt.item) is None for receipt in receipts)
    assert runtime.credits.local_events == 0
    assert not runtime.planner.events


def test_permanent_expand_capacity_fails_only_oversized_root() -> None:
    """Impossible fanout must fail its root while another root still completes."""

    run = sem.RunId.new()
    input_port, expanded_port, reduced_port = (
        sem.PortId(0),
        sem.PortId(1),
        sem.PortId(2),
    )
    scope_def = sem.ScopeDefId(0)
    expand_node = _Node(
        sem.NodeId(1),
        graph_model.ExpandOp(input_port, scope_def),
        (_Binding("input", input_port),),
        (_Port(expanded_port),),
    )
    reduce_node = _Node(
        sem.NodeId(2),
        graph_model.ReduceOp(expanded_port, scope_def),
        (_Binding("input", expanded_port),),
        (_Port(reduced_port),),
    )
    graph = _Graph(
        (expand_node, reduce_node),
        {
            input_port: (expand_node.id,),
            expanded_port: (reduce_node.id,),
            reduced_port: (),
        },
        {
            scope_def: graph_model.ScopePlan(
                scope_def, expand_node.id, (reduce_node.id,)
            )
        },
    )
    runtime = _runtime(graph, run, limits=_limits(max_scope_width=1))
    roots = tuple(sem.RootId.for_source(run, index) for index in range(2))
    entities = tuple(sem.EntityId.source(run, index) for index in range(2))
    oversized = _present_list(
        runtime,
        run=run,
        root=roots[0],
        port=input_port,
        entity=entities[0],
        values=("a", "b"),
    )
    healthy = _present_list(
        runtime,
        run=run,
        root=roots[1],
        port=input_port,
        entity=entities[1],
        values=(),
    )

    _publish(runtime, oversized)
    _publish(runtime, healthy)
    _drain(runtime)

    failed = runtime.receipts.require(
        sem.ItemRef(reduced_port, entities[0])
    )
    succeeded = runtime.receipts.require(
        sem.ItemRef(reduced_port, entities[1])
    )
    assert failed.state is sem.ReceiptState.SUPPRESSED
    assert succeeded.state is sem.ReceiptState.PRESENT
    assert len(runtime.errors) == 1


def test_live_grain_ledger_enforces_and_reclaims_hard_limit() -> None:
    """Executable grain credits must remain bounded and root-reclaimable."""

    run = sem.RunId.new()
    root = sem.RootId.for_source(run, 0)
    credits = CreditManager(_limits(max_live_grains=1))
    first, second = sem.GrainId.new(), sem.GrainId.new()

    assert credits.reserve_grain(root, first)
    assert not credits.reserve_grain(root, second)
    assert credits.live_grains == 1
    credits.release_root_resources(root)
    assert credits.live_grains == 0
    assert credits.reserve_grain(root, second)


def test_gather_node_limit_is_aggregated_across_roles() -> None:
    """Per-entry gather limits must include every input role together."""

    run = sem.RunId.new()
    root = sem.RootId.for_source(run, 0)
    entity = sem.EntityId.source(run, 0)
    blocks = state.BlockStore()
    values = state.ValueNodeStore(blocks)
    grains = sem.GrainStore()
    items = tuple(sem.ItemRef(sem.PortId(index), entity) for index in range(2))
    for index, item in enumerate(items):
        block = blocks.install((index,), 1)
        node = values.create_scalar(block, 0)
        values.bind_item(root, item, node)
    grain = sem.GrainId.map(run, sem.NodeId(2), items)
    grains.ensure(
        sem.GrainSpec(
            grain,
            run,
            root,
            sem.NodeId(2),
            sem.OccurrenceContext(root),
            (
                sem.RoleBinding("left", (items[0],)),
                sem.RoleBinding("right", (items[1],)),
            ),
            (sem.ItemRef(sem.PortId(3), entity),),
        )
    )

    with pytest.raises(DispatchInvariantError, match="entry roles"):
        prepare_dispatch(
            BatchSelection(sem.NodeId(2), (grain,)),
            SimpleNamespace(grains=grains, values=values, blocks=blocks),
            run=run,
            limits=_limits(max_gather_nodes_per_entry=1),
        )

    assert grains.require(grain).phase is sem.GrainPhase.READY


def test_detach_limit_becomes_root_delivery_failure() -> None:
    """A detached gather overflow must fail one leaf instead of raising."""

    run = sem.RunId.new()
    root_id = sem.RootId.for_source(run, 0)
    entity = sem.EntityId.source(run, 0)
    port = sem.PortId(0)
    item = sem.ItemRef(port, entity)
    blocks = state.BlockStore()
    values = state.ValueNodeStore(blocks)
    children = []
    for value in ("a", "b"):
        block = blocks.install((value,), 1)
        children.append(values.create_scalar(block, 0))
    composite = values.create_composite(tuple(children))
    values.bind_item(root_id, item, composite)
    producer = sem.GrainId.source(run, sem.NodeId(0), 0)
    receipt = sem.Receipt(
        item,
        sem.OccurrenceContext(root_id),
        sem.ReceiptState.PRESENT,
        producer,
    )
    root = RootContext(
        id=root_id,
        source_seq=0,
        phase=RootPhase.ACTIVE,
        source_record=SourceRecord(0, "source"),
        final_receipts={"output": receipt},
    )
    coordinator = object.__new__(CompiledGraphRunCoordinator)
    coordinator.graph = SimpleNamespace(
        outputs=(SimpleNamespace(name="output", port=port),)
    )
    coordinator.blocks = blocks
    coordinator.values = values
    coordinator.errors = sem.ErrorStore()
    coordinator.grains = sem.GrainStore()
    coordinator.limits = _limits(max_detached_result_nodes=2)

    result = coordinator._detach_root_result(root)

    assert result.status.value == "failed"
    assert isinstance(
        result.outputs["output"].failure,
        sem.DeliveryFailureSummary,
    )

