from __future__ import annotations

import time
from dataclasses import replace

import pytest
import ray

from rayorch.experimental.multigrain_v2_5.executor import BlockSlice, Executor
from rayorch.experimental.multigrain_v2_5.graph import (
    BindingReceipt,
    ExecutionOptions,
    InputBinding,
    KeyProjection,
    NodeSpec,
    PlanAction,
    PlannerContractError,
    Primitive,
    UdfRecipe,
    admit_source,
    compile_graph,
    plan_expand,
    plan_filter,
    plan_map,
    plan_reduce,
    plan_relate_bounded,
)
from rayorch.experimental.multigrain_v2_5.grain import (
    FiberBarrier,
    FiberId,
    ItemRef,
    PortId,
)

from .reference_semantics.semantic_cases import RUN_SALT
from .support import (
    expand_node,
    filter_node,
    map_node,
    reduce_node,
    source_node,
)


pytestmark = pytest.mark.usefixtures("ray_cluster")


def _configured(node, target, *, batch_size, replicas=1, init_args=()):
    assert node.execution is not None
    return replace(
        node,
        udf_recipe=UdfRecipe(target, init_args=tuple(init_args)),
        execution=replace(
            node.execution,
            batch_size=batch_size,
            replicas=replicas,
        ),
    )


def _resolve(value: BlockSlice):
    return ray.get(value.block)[value.row]


def _dispatch_all(arena, transport, node_id):
    plans = []
    while True:
        plan = arena.reserve_dispatch(node_id)
        if plan is None:
            break
        plans.append(plan)
        assert transport.submit(arena, plan)
        transport.drain()
    return tuple(plans)


class LeftBranch:
    def run(self, values):
        return [f"L:{value}" for value in values]


class RightBranch:
    def run(self, values):
        return [f"R:{value}" for value in values]


class JoinMultiOutput:
    def run(self, left, right):
        return (
            [f"{a}|{b}" for a, b in zip(left, right)],
            [len(a) + len(b) for a, b in zip(left, right)],
        )


def test_map_multi_role_alignment_ignores_physical_block_order():
    source_spec = source_node(0)
    left_spec = _configured(
        map_node(1, source_spec.output_ports[0]),
        LeftBranch,
        batch_size=3,
    )
    right_spec = _configured(
        map_node(2, source_spec.output_ports[0]),
        RightBranch,
        batch_size=3,
    )
    join_spec = _configured(
        map_node(
            3,
            left_spec.output_ports[0],
            ("control", right_spec.output_ports[0]),
            outputs=2,
        ),
        JoinMultiOutput,
        batch_size=3,
    )
    graph = compile_graph((source_spec, left_spec, right_spec, join_spec))
    executor = Executor(graph)
    arena = executor.new_arena(10, RUN_SALT)
    transport = executor.ray_transport()
    try:
        values = ("a", "b", "c")
        sources = tuple(
            admit_source(source_spec, RUN_SALT, position)
            for position in range(3)
        )
        arena.admit_source_batch(sources, ray.put(values))

        for source in sources:
            arena.ensure_plans(
                (
                    plan_map(
                        left_spec,
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
        for source in reversed(sources):
            arena.ensure_plans(
                (
                    plan_map(
                        right_spec,
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

        left_plan = arena.reserve_dispatch(1)
        right_plan = arena.reserve_dispatch(2)
        assert left_plan is not None and right_plan is not None
        assert transport.submit(arena, left_plan)
        assert transport.submit(arena, right_plan)
        transport.drain()

        left_by_entity = {
            record.output_slots[0].entity: record.output_slots[0]
            for record in (
                arena.grains.get(entry.token.grain)
                for entry in left_plan.entries
            )
            if record is not None
        }
        right_by_entity = {
            record.output_slots[0].entity: record.output_slots[0]
            for record in (
                arena.grains.get(entry.token.grain)
                for entry in right_plan.entries
            )
            if record is not None
        }
        join_records = []
        for source in sources:
            entity = source.output_slots[0].entity
            join_records.extend(
                arena.ensure_plans(
                    (
                        plan_map(
                            join_spec,
                            RUN_SALT,
                            (
                                BindingReceipt.present(
                                    "primary",
                                    left_by_entity[entity],
                                ),
                                BindingReceipt.present(
                                    "control",
                                    right_by_entity[entity],
                                ),
                            ),
                        ),
                    )
                )
            )

        join_plan = arena.reserve_dispatch(3)
        assert join_plan is not None
        assert transport.submit(arena, join_plan)
        transport.drain()

        first_port = tuple(
            arena.slice(record.output_slots[0]) for record in join_records
        )
        second_port = tuple(
            arena.slice(record.output_slots[1]) for record in join_records
        )
        assert tuple(_resolve(value) for value in first_port) == (
            "L:a|R:a",
            "L:b|R:b",
            "L:c|R:c",
        )
        assert tuple(_resolve(value) for value in second_port) == (6, 6, 6)
        assert len({value.block for value in first_port}) == 1
        assert len({value.block for value in second_port}) == 1
        assert first_port[0].block != second_port[0].block
    finally:
        transport.shutdown()


class MaskFromValue:
    def run(self, values):
        return [value[1] for value in values]


class ExtractId:
    def run(self, values):
        return [value[0] for value in values]


def test_filter_all_true_all_false_partial_and_absence_short_circuit():
    source_spec = source_node(0)
    filter_spec = _configured(
        filter_node(1, source_spec.output_ports[0]),
        MaskFromValue,
        batch_size=3,
    )
    downstream_spec = _configured(
        map_node(2, filter_spec.output_ports[0]),
        ExtractId,
        batch_size=9,
    )
    graph = compile_graph((source_spec, filter_spec, downstream_spec))
    executor = Executor(graph)
    arena = executor.new_arena(11, RUN_SALT)
    transport = executor.ray_transport()
    try:
        rows = (
            ("t0", True),
            ("t1", True),
            ("t2", True),
            ("f0", False),
            ("f1", False),
            ("f2", False),
            ("p0", True),
            ("p1", False),
            ("p2", True),
        )
        sources = tuple(
            admit_source(source_spec, RUN_SALT, position)
            for position in range(len(rows))
        )
        arena.admit_source_batch(sources, ray.put(rows))
        filter_records = []
        for source in sources:
            filter_records.extend(
                arena.ensure_plans(
                    (
                        plan_filter(
                            filter_spec,
                            RUN_SALT,
                            (
                                BindingReceipt.present(
                                    "target",
                                    source.output_slots[0],
                                ),
                            ),
                        ),
                    )
                )
            )
        filter_plans = _dispatch_all(arena, transport, 1)
        assert [len(plan.entries) for plan in filter_plans] == [3, 3, 3]

        downstream_decisions = []
        for record in filter_records:
            slot = record.output_slots[0]
            receipt = (
                BindingReceipt.present("primary", slot)
                if arena.values.contains(slot)
                else BindingReceipt.absent("primary", slot)
            )
            downstream_decisions.append(
                plan_map(downstream_spec, RUN_SALT, (receipt,))
            )
        assert [
            decision.action for decision in downstream_decisions
        ].count(PlanAction.NORMAL_ABSENCE) == 4
        executable = tuple(
            decision
            for decision in downstream_decisions
            if decision.action is PlanAction.ENSURE_EXECUTABLE
        )
        downstream_records = arena.ensure_plans(executable)
        plan = arena.reserve_dispatch(2)
        assert plan is not None and len(plan.entries) == 5
        assert transport.submit(arena, plan)
        transport.drain()

        assert tuple(
            _resolve(arena.slice(record.output_slots[0]))
            for record in downstream_records
        ) == ("t0", "t1", "t2", "p0", "p2")
        assert transport.actor_stats(2) == ({"calls": 1},)
    finally:
        transport.shutdown()


class WeirdExpand:
    def run(self, parents):
        counts = {"zero": 0, "all": 2, "partial": 3}
        values = [
            [f"{parent}:{ordinal}" for ordinal in range(counts[parent])]
            for parent in parents
        ]
        metadata = [
            [f"meta:{parent}:{ordinal}" for ordinal in range(counts[parent])]
            for parent in parents
        ]
        return values, metadata


class KeepPartialOnly:
    def run(self, children):
        return [
            child.startswith("partial:") and not child.endswith(":1")
            for child in children
        ]


class OrderedReduce:
    def run(self, anchors, members):
        return [
            f"{anchor}=>[{','.join(group)}]"
            for anchor, group in zip(anchors, members)
        ]


def test_expand_zero_skew_multi_output_and_reduce_empty_filtered_ordered():
    source_spec = source_node(0)
    expand_spec = _configured(
        expand_node(1, source_spec.output_ports[0], outputs=2),
        WeirdExpand,
        batch_size=3,
    )
    filter_spec = _configured(
        filter_node(2, expand_spec.output_ports[0]),
        KeepPartialOnly,
        batch_size=5,
    )
    reduce_spec = _configured(
        reduce_node(
            3,
            source_spec.output_ports[0],
            filter_spec.output_ports[0],
        ),
        OrderedReduce,
        batch_size=3,
    )
    graph = compile_graph(
        (source_spec, expand_spec, filter_spec, reduce_spec)
    )
    executor = Executor(graph)
    arena = executor.new_arena(12, RUN_SALT)
    transport = executor.ray_transport()
    try:
        parent_values = ("zero", "all", "partial")
        sources = tuple(
            admit_source(source_spec, RUN_SALT, position)
            for position in range(3)
        )
        arena.admit_source_batch(sources, ray.put(parent_values))
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
        expand_plan = arena.reserve_dispatch(1)
        assert expand_plan is not None
        assert transport.submit(arena, expand_plan)
        transport.drain()
        expand_records = tuple(
            arena.grains.get(entry.token.grain)
            for entry in expand_plan.entries
        )
        assert [
            len(record.outcome.emissions_by_port[0])  # type: ignore[union-attr]
            for record in expand_records
            if record is not None
        ] == [0, 2, 3]
        for record in expand_records:
            assert record is not None
            first, second = record.outcome.emissions_by_port  # type: ignore[union-attr]
            assert [item.item.entity for item in first] == [
                item.item.entity for item in second
            ]

        children = tuple(
            emission.item
            for record in expand_records
            if record is not None
            for emission in record.outcome.emissions_by_port[0]  # type: ignore[union-attr]
        )
        for child in children:
            arena.ensure_plans(
                (
                    plan_filter(
                        filter_spec,
                        RUN_SALT,
                        (BindingReceipt.present("target", child),),
                    ),
                )
            )
        filter_plan = arena.reserve_dispatch(2)
        assert filter_plan is not None
        assert transport.submit(arena, filter_plan)
        transport.drain()

        barriers = {
            source.output_slots[0]: FiberBarrier(
                FiberId(3, source.output_slots[0]),
                record.id,
            )
            for source, record in zip(sources, expand_records)
            if record is not None
        }
        for source, record in zip(sources, expand_records):
            assert record is not None
            barriers[source.output_slots[0]].set_expected(
                len(record.outcome.emissions_by_port[0])  # type: ignore[union-attr]
            )
        filter_records = [
            arena.grains.get(entry.token.grain)
            for entry in filter_plan.entries
        ]
        for record in reversed(filter_records):
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
        reduce_plan = arena.reserve_dispatch(3)
        assert reduce_plan is not None
        assert [len(entry.role_takes[1]) for entry in reduce_plan.entries] == [
            0,
            0,
            2,
        ]
        assert transport.submit(arena, reduce_plan)
        transport.drain()
        assert tuple(
            _resolve(arena.slice(record.output_slots[0]))
            for record in reduce_records
        ) == (
            "zero=>[]",
            "all=>[]",
            "partial=>[partial:0,partial:2]",
        )
        assert transport.actor_stats(3) == ({"calls": 1},)
    finally:
        transport.shutdown()


class PairRows:
    def run(self, left, right):
        return [f"{a}+{b}" for a, b in zip(left, right)]


def test_relate_sealing_duplicate_key_cartesian_and_unmatched():
    left_source = source_node(0)
    right_source = source_node(1)
    relate_spec = NodeSpec(
        id=2,
        kind=Primitive.RELATE,
        inputs=(
            InputBinding("left", left_source.output_ports[0]),
            InputBinding("right", right_source.output_ports[0]),
        ),
        output_ports=(PortId(2, 0),),
        udf_recipe=UdfRecipe(PairRows),
        execution=ExecutionOptions(batch_size=8),
        relate_keys=(
            KeyProjection("left", "key"),
            KeyProjection("right", "key"),
        ),
    )
    graph = compile_graph((left_source, right_source, relate_spec))
    executor = Executor(graph)
    arena = executor.new_arena(13, RUN_SALT)
    transport = executor.ray_transport()
    try:
        left_values = ("l0", "l1", "l2")
        right_values = ("r0", "r1", "r2")
        left_records = tuple(
            admit_source(left_source, RUN_SALT, position)
            for position in range(3)
        )
        right_records = tuple(
            admit_source(right_source, RUN_SALT, position)
            for position in range(3)
        )
        arena.admit_source_batch(left_records, ray.put(left_values))
        arena.admit_source_batch(right_records, ray.put(right_values))
        roles = (
            (
                "left",
                tuple(
                    zip(
                        (record.output_slots[0] for record in left_records),
                        (1, 1, 2),
                    )
                ),
            ),
            (
                "right",
                tuple(
                    zip(
                        (record.output_slots[0] for record in right_records),
                        (1, 1, 9),
                    )
                ),
            ),
        )
        open_result = plan_relate_bounded(
            relate_spec,
            RUN_SALT,
            roles,
            sealed_roles=frozenset({"left"}),
            max_cardinality=10,
        )
        assert not open_result.complete
        assert open_result.unmatched == ()

        with pytest.raises(PlannerContractError, match="max_relation"):
            plan_relate_bounded(
                relate_spec,
                RUN_SALT,
                roles,
                sealed_roles=frozenset({"left", "right"}),
                max_cardinality=3,
            )
        result = plan_relate_bounded(
            relate_spec,
            RUN_SALT,
            roles,
            sealed_roles=frozenset({"left", "right"}),
            max_cardinality=4,
        )
        assert result.complete
        assert len(result.decisions) == 4
        assert result.unmatched == (
            left_records[2].output_slots[0],
            right_records[2].output_slots[0],
        )
        records = arena.ensure_plans(result.decisions)
        plan = arena.reserve_dispatch(2)
        assert plan is not None and len(plan.entries) == 4
        assert transport.submit(arena, plan)
        transport.drain()
        assert {
            _resolve(arena.slice(record.output_slots[0]))
            for record in records
        } == {
            "l0+r0",
            "l0+r1",
            "l1+r0",
            "l1+r1",
        }
    finally:
        transport.shutdown()


class TimedStage:
    def __init__(self, stage: str, delay: float):
        self.stage = stage
        self.delay = delay

    def run(self, values):
        start = time.monotonic()
        time.sleep(self.delay)
        stop = time.monotonic()
        return [
            {
                "value": (
                    value["value"] if isinstance(value, dict) else value
                ),
                "stage": self.stage,
                "start": start,
                "stop": stop,
            }
            for value in values
        ]


def test_sleeping_stages_overlap_across_pipeline_frontier():
    source_spec = source_node(0)
    first_spec = _configured(
        map_node(1, source_spec.output_ports[0]),
        TimedStage,
        batch_size=1,
        init_args=("first", 0.35),
    )
    second_spec = _configured(
        map_node(2, first_spec.output_ports[0]),
        TimedStage,
        batch_size=1,
        init_args=("second", 0.35),
    )
    graph = compile_graph((source_spec, first_spec, second_spec))
    executor = Executor(graph)
    arena = executor.new_arena(14, RUN_SALT)
    transport = executor.ray_transport()
    try:
        transport.actor_stats(1)
        transport.actor_stats(2)
        sources = tuple(
            admit_source(source_spec, RUN_SALT, position)
            for position in range(2)
        )
        arena.admit_source_batch(sources, ray.put(("a", "b")))
        first_records = []
        for source in sources:
            first_records.extend(
                arena.ensure_plans(
                    (
                        plan_map(
                            first_spec,
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
            )

        first_a = arena.reserve_dispatch(1)
        assert first_a is not None
        assert transport.submit(arena, first_a)
        transport.drain()
        arena.ensure_plans(
            (
                plan_map(
                    second_spec,
                    RUN_SALT,
                    (
                        BindingReceipt.present(
                            "primary",
                            first_records[0].output_slots[0],
                        ),
                    ),
                ),
            )
        )

        first_b = arena.reserve_dispatch(1)
        second_a = arena.reserve_dispatch(2)
        assert first_b is not None and second_a is not None
        assert transport.submit(arena, first_b)
        assert transport.submit(arena, second_a)
        transport.drain()

        first_b_record = arena.grains.get(first_b.entries[0].token.grain)
        second_a_record = arena.grains.get(second_a.entries[0].token.grain)
        assert first_b_record is not None and second_a_record is not None
        first_interval = _resolve(
            arena.slice(first_b_record.output_slots[0])
        )
        second_interval = _resolve(
            arena.slice(second_a_record.output_slots[0])
        )
        assert max(
            first_interval["start"],
            second_interval["start"],
        ) < min(
            first_interval["stop"],
            second_interval["stop"],
        )
    finally:
        transport.shutdown()
