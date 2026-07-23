from __future__ import annotations

from dataclasses import replace

import pytest
import ray

from rayorch.experimental.multigrain_v2_5.executor import (
    Arena,
    BlockSlice,
    Executor,
)
from rayorch.experimental.multigrain_v2_5.graph import (
    BindingReceipt,
    UdfRecipe,
    admit_source,
    compile_graph,
    plan_expand,
    plan_filter,
    plan_map,
    plan_reduce,
)
from rayorch.experimental.multigrain_v2_5.grain import FiberBarrier, FiberId

from .reference_semantics.semantic_cases import RUN_SALT
from .support import (
    expand_node,
    filter_node,
    map_node,
    reduce_node,
    source_node,
)


pytestmark = pytest.mark.usefixtures("ray_cluster")


class ExpandPages:
    def run(self, parents):
        return [
            [f"{parent}:child:{index}" for index in range(count)]
            for parent, count in zip(parents, (2, 3))
        ]


class MapChildren:
    def run(self, children):
        return [f"mapped({child})" for child in children]


class FilterChildren:
    def run(self, children):
        return [":child:1)" not in child for child in children]


class ReduceParent:
    def run(self, anchors, members):
        return [
            f"{anchor}=>{'|'.join(group)}"
            for anchor, group in zip(anchors, members)
        ]


class BadMap:
    def run(self, children):
        return []  # contract violation: zero rows for non-empty Map dispatch


def _configured(node, target, *, batch_size, replicas=1):
    assert node.execution is not None
    return replace(
        node,
        udf_recipe=UdfRecipe(target),
        execution=replace(
            node.execution,
            batch_size=batch_size,
            replicas=replicas,
        ),
    )


def _resolve_slice(value: BlockSlice):
    return ray.get(value.block)[value.row]


def test_ray_persistent_coarse_blocks_cross_parent_rebatch_and_reduce():
    source_spec = source_node(0)
    expand_spec = _configured(
        expand_node(1, source_spec.output_ports[0]),
        ExpandPages,
        batch_size=2,
    )
    map_spec = _configured(
        map_node(2, expand_spec.output_ports[0]),
        MapChildren,
        batch_size=3,
    )
    filter_spec = _configured(
        filter_node(3, map_spec.output_ports[0]),
        FilterChildren,
        batch_size=5,
    )
    reduce_spec = _configured(
        reduce_node(
            4,
            source_spec.output_ports[0],
            filter_spec.output_ports[0],
        ),
        ReduceParent,
        batch_size=2,
    )
    graph = compile_graph(
        (source_spec, expand_spec, map_spec, filter_spec, reduce_spec)
    )
    executor = Executor(graph)
    arena = executor.new_arena(1, RUN_SALT)
    transport = executor.ray_transport(max_pending_per_actor=1)

    try:
        parent_values = ("parent-a", "parent-b")
        parent_ref = ray.put(parent_values)
        sources = tuple(
            admit_source(source_spec, RUN_SALT, position)
            for position in range(len(parent_values))
        )
        arena.admit_source_batch(sources, parent_ref)
        for source in sources:
            arena.ensure_plans(
                (
                    plan_expand(
                        expand_spec,
                        RUN_SALT,
                        (
                            BindingReceipt.present(
                                "parent",
                                source.output_slots[0],
                            ),
                        ),
                    ),
                )
            )

        expand_dispatch = arena.reserve_dispatch(1)
        assert expand_dispatch is not None
        assert transport.submit(arena, expand_dispatch)
        transport.drain()

        expand_records = tuple(
            arena.grains.get(entry.token.grain)
            for entry in expand_dispatch.entries
        )
        child_items = tuple(
            emission.item
            for record in expand_records
            if record is not None
            for emission in record.outcome.emissions_by_port[0]  # type: ignore[union-attr]
        )
        for item in child_items:
            arena.ensure_plans(
                (
                    plan_map(
                        map_spec,
                        RUN_SALT,
                        (BindingReceipt.present("primary", item),),
                    ),
                )
            )

        map_dispatches = []
        while True:
            plan = arena.reserve_dispatch(2)
            if plan is None:
                break
            map_dispatches.append(plan)
            assert transport.submit(arena, plan)
            transport.drain()
        assert [len(plan.entries) for plan in map_dispatches] == [3, 2]

        first_parent_anchors = {
            arena.expand_origins.get(
                arena.grains.get(entry.token.grain).inputs[0].items[0].entity  # type: ignore[union-attr]
            ).anchor  # type: ignore[union-attr]
            for entry in map_dispatches[0].entries
        }
        assert len(first_parent_anchors) == 2

        map_records = tuple(
            arena.grains.get(entry.token.grain)
            for plan in map_dispatches
            for entry in plan.entries
        )
        map_blocks = {
            arena.slice(record.output_slots[0]).block
            for record in map_records
            if record is not None
        }
        assert len(map_blocks) == 2  # one ObjectRef per Map dispatch/output port
        assert all(isinstance(block, ray.ObjectRef) for block in map_blocks)

        for record in map_records:
            assert record is not None
            arena.ensure_plans(
                (
                    plan_filter(
                        filter_spec,
                        RUN_SALT,
                        (
                            BindingReceipt.present(
                                "target",
                                record.output_slots[0],
                            ),
                        ),
                    ),
                )
            )
        filter_dispatch = arena.reserve_dispatch(3)
        assert filter_dispatch is not None
        assert len(filter_dispatch.entries) == 5
        assert transport.submit(arena, filter_dispatch)
        transport.drain()

        barriers = {
            source.output_slots[0]: FiberBarrier(
                FiberId(4, source.output_slots[0]),
                expand_record.id,
            )
            for source, expand_record in zip(sources, expand_records)
            if expand_record is not None
        }
        for source, expand_record in zip(sources, expand_records):
            assert expand_record is not None
            barrier = barriers[source.output_slots[0]]
            barrier.set_expected(
                len(expand_record.outcome.emissions_by_port[0])  # type: ignore[union-attr]
            )

        filter_records = tuple(
            arena.grains.get(entry.token.grain)
            for entry in filter_dispatch.entries
        )
        for record in filter_records:
            assert record is not None
            origin = arena.expand_origins.get(record.output_slots[0].entity)
            assert origin is not None
            if arena.values.contains(record.output_slots[0]):
                barriers[origin.anchor].settle_present(
                    origin.ordinal,
                    record.output_slots[0],
                )
            else:
                barriers[origin.anchor].settle_dropped(origin.ordinal)

        reduce_records = []
        for source in sources:
            reduce_records.extend(
                arena.ensure_plans(
                    (
                        plan_reduce(
                            reduce_spec,
                            RUN_SALT,
                            BindingReceipt.present(
                                "anchor",
                                source.output_slots[0],
                            ),
                            barriers[source.output_slots[0]],
                        ),
                    )
                )
            )
        reduce_dispatch = arena.reserve_dispatch(4)
        assert reduce_dispatch is not None
        assert transport.submit(arena, reduce_dispatch)
        transport.drain()

        result = arena.deliver_slices(
            tuple(record.output_slots[0] for record in reduce_records)
        )
        assert tuple(_resolve_slice(value) for value in result.outputs) == (
            "parent-a=>mapped(parent-a:child:0)",
            "parent-b=>mapped(parent-b:child:0)|mapped(parent-b:child:2)",
        )
        assert result.metrics["grains_per_rpc"] > 2.0
        assert result.metrics["pending_dispatches"] == 0.0
        assert transport.pending_dispatches == 0

        map_stats = transport.actor_stats(2)
        assert map_stats == ({"calls": 2},)
    finally:
        transport.shutdown()


def test_ray_worker_contract_violation_aborts_arena():
    source_spec = source_node(0)
    map_spec = _configured(
        map_node(1, source_spec.output_ports[0]),
        BadMap,
        batch_size=2,
    )
    graph = compile_graph((source_spec, map_spec))
    executor = Executor(graph)
    arena = executor.new_arena(2, RUN_SALT)
    transport = executor.ray_transport()
    try:
        source_ref = ray.put((1, 2))
        sources = tuple(
            admit_source(source_spec, RUN_SALT, position)
            for position in range(2)
        )
        arena.admit_source_batch(sources, source_ref)
        for source in sources:
            arena.ensure_plans(
                (
                    plan_map(
                        map_spec,
                        RUN_SALT,
                        (
                            BindingReceipt.present(
                                "primary",
                                source.output_slots[0],
                            ),
                        ),
                    ),
                )
            )
        plan = arena.reserve_dispatch(1)
        assert plan is not None
        assert transport.submit(arena, plan)
        with pytest.raises(Exception, match="contract|unknown dispatch error"):
            transport.drain()
    finally:
        transport.shutdown()


def test_ray_actor_failure_replaces_actor_and_retries_same_grain():
    source_spec = source_node(0)
    map_spec = _configured(
        map_node(1, source_spec.output_ports[0]),
        MapChildren,
        batch_size=1,
    )
    graph = compile_graph((source_spec, map_spec))
    executor = Executor(graph)
    arena = executor.new_arena(3, RUN_SALT)
    transport = executor.ray_transport()
    try:
        source = admit_source(source_spec, RUN_SALT, 0)
        arena.admit_source_batch((source,), ray.put(("child",)))
        record = arena.ensure_plans(
            (
                plan_map(
                    map_spec,
                    RUN_SALT,
                    (
                        BindingReceipt.present(
                            "primary",
                            source.output_slots[0],
                        ),
                    ),
                ),
            )
        )[0]
        first = arena.reserve_dispatch(1)
        assert first is not None
        ray.kill(transport._actors[1][0])
        assert transport.submit(arena, first)
        transport.drain()

        retry = arena.reserve_dispatch(1)
        assert retry is not None
        assert retry.entries[0].token.grain == first.entries[0].token.grain
        assert retry.entries[0].token.generation == 2
        assert transport.submit(arena, retry)
        transport.drain()

        result = arena.deliver_slices((record.output_slots[0],))
        assert _resolve_slice(result.outputs[0]) == "mapped(child)"
    finally:
        transport.shutdown()
