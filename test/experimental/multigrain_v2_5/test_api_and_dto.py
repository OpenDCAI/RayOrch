from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v2_5 as mg
from rayorch.experimental.multigrain_v2_5.api import ExecutionError
from rayorch.experimental.multigrain_v2_5.executor import (
    Executor,
    SourcePositionAllocator,
)
from rayorch.experimental.multigrain_v2_5.graph import compile_graph
from rayorch.experimental.multigrain_v2_5.grain import (
    AttemptToken,
    GrainId,
    PortId,
)
from rayorch.experimental.multigrain_v2_5.worker import (
    DispatchEntry,
    DispatchPlan,
    RowTake,
)

from .support import source_node


class Dummy:
    pass


def test_public_api_has_five_user_primitives_and_no_source_constructor():
    assert [mg.Map, mg.Filter, mg.Expand, mg.Reduce, mg.Relate]
    assert not hasattr(mg, "Source")


def test_primitive_configuration_is_raymodule_style():
    operation = (
        mg.Map(Dummy)
        .pre_init(model="tiny")
        .ray_options(replicas=3, batch_size=8, error_policy="isolate")
    )
    assert operation.init_kwargs == {"model": "tiny"}
    assert operation.options["replicas"] == 3
    assert operation.options["batch_size"] == 8


def test_keyed_port_and_bad_record_error_are_small_public_contracts():
    port = mg.Port(PortId(2, 0))
    keyed = mg.keyed(port, by=lambda value: value["id"])
    assert keyed.port == port

    error = mg.BadRecordError("bad", index=4)
    assert error.index == 4


def test_source_position_allocator_does_not_restart_per_arena():
    allocator = SourcePositionAllocator()
    source_port = PortId(0, 0)
    first_arena = [allocator.allocate(source_port) for _ in range(2)]
    second_arena = [allocator.allocate(source_port) for _ in range(3)]
    assert first_arena == [0, 1]
    assert second_arena == [2, 3, 4]


def test_dispatch_entry_remains_role_general():
    token = AttemptToken(
        arena=1,
        dispatch=2,
        grain=GrainId(bytes.fromhex("aa" * 16)),
        generation=3,
    )
    entry = DispatchEntry(
        token,
        role_takes=(
            (RowTake(0, 7),),
            (RowTake(1, 2), RowTake(2, 4)),
        ),
    )
    plan = DispatchPlan(2, 9, (entry,))
    assert len(plan.entries[0].role_takes) == 2
    assert len(plan.entries[0].role_takes[1]) == 2


def test_bare_compiled_graph_requires_explicit_arena_or_traced_pipeline():
    graph = compile_graph((source_node(0),))
    with pytest.raises(ExecutionError, match="traced Pipeline"):
        Executor(graph).run([1, 2, 3])
