"""Shared actor pools preserve Call semantics and schedule stages fairly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rayorch as ro
from rayorch._execution.executor import (
    Executor,
    _ActorSlot,
    _CallCounters,
    _InputBatchSlot,
)
from rayorch._execution.worker import Worker
from rayorch._model import CallRef, DomainRef, EntityRef, GrainRef, PortRef
from rayorch._protocol import (
    BlockRef,
    CallInputLayout,
    CallOutputLayout,
    DispatchFailure,
    GrainInvocation,
    RowBinding,
)
from rayorch._runtime.engine import InputBatchEngine
from rayorch._runtime.materialize import materialize_tree


class SharedStages:
    def run(self, values, *, stage):
        if stage == "first":
            return [value + 1 for value in values]
        if stage == "second":
            return [value * 2 for value in values]
        raise ValueError(f"unknown stage: {stage}")


class SharedLinearPipeline(ro.Pipeline):
    def __init__(self, *, replicas: int = 1) -> None:
        self.stages = ro.RayModule(SharedStages).ray_options(
            replicas=replicas,
            num_cpus=0,
            batch_size=3,
        )

    def forward(self, values):
        first = self.stages(values, stage="first")
        return self.stages(first, stage="second")


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


def _run_shared_pool_sync(pipeline: ro.Pipeline, source: tuple[object, ...]):
    """Drive one shared-pool Pipeline without Ray and record queue bounds."""

    plan = pipeline.compile().plan
    assert len(plan.actor_pools) == 1
    pool = next(iter(plan.actor_pools))
    replicas = plan.actor_pools[pool].replicas
    store = MemoryStore()
    block = store.put(source)
    engine = InputBatchEngine(plan)
    engine.admit_sources(
        {
            plan.source_ports[0]: tuple(
                RowBinding(block, row) for row in range(len(source))
            )
        }
    )
    engine.close_admission()

    worker = Worker(
        plan.actor_pools[pool].udf.target,
        plan.actor_pools[pool].udf.init_args,
        plan.actor_pools[pool].udf.init_kwargs,
    )
    reports_by_ref = {}
    next_ref = 0

    def remote(invocations, layouts, input_layout):
        nonlocal next_ref
        ref = next_ref
        next_ref += 1
        reports_by_ref[ref] = worker.execute(
            invocations,
            layouts,
            store,
            input_layout=input_layout,
        )
        return ref

    executor = object.__new__(Executor)
    executor.plan = plan
    executor._calls_by_pool = dict(plan.calls_by_pool)
    executor._pool_cursor = {pool: 0}
    executor._actors = {
        pool: [
            _ActorSlot(
                pool,
                SimpleNamespace(execute=SimpleNamespace(remote=remote)),
            )
            for _ in range(replicas)
        ]
    }
    executor._counters = {call: _CallCounters() for call in plan.calls}
    active = {0: _InputBatchSlot(0, engine)}
    pending = {}
    turns = 0
    max_ready = engine.ready_count
    max_pending = 0

    while not engine.is_complete():
        turns += 1
        assert turns <= max(100, len(source) * 16)
        dispatched = executor._dispatch_ready(active, pending)
        assert dispatched or pending, "finite dummy workload stopped making progress"
        max_pending = max(max_pending, len(pending))
        for ref, rpc in tuple(pending.items()):
            engine.commit_reports(rpc.execution_microbatch, reports_by_ref.pop(ref))
            rpc.actor.busy = False
            del pending[ref]
        max_ready = max(max_ready, engine.ready_count)

    assert not pending
    assert not reports_by_ref
    assert all(not actor.busy for actor in executor._actors[pool])
    return SimpleNamespace(
        outputs=materialize_tree(plan, engine, store),
        plan=plan,
        counters=executor._counters,
        turns=turns,
        max_ready=max_ready,
        max_pending=max_pending,
        replicas=replicas,
    )


def _driver(pipeline: ro.Pipeline):
    plan = pipeline.compile().plan
    sent = []

    def remote(invocations, layouts, input_layout):
        sent.append(
            (
                tuple(invocation.grain for invocation in invocations),
                input_layout,
                layouts,
            )
        )
        return len(sent)

    executor = object.__new__(Executor)
    executor.plan = plan
    executor._calls_by_pool = dict(plan.calls_by_pool)
    executor._pool_cursor = {pool: 0 for pool in plan.actor_pools}
    executor._actors = {
        pool: [
            _ActorSlot(
                pool,
                SimpleNamespace(execute=SimpleNamespace(remote=remote)),
            )
        ]
        for pool in plan.actor_pools
    }
    executor._counters = {call: _CallCounters() for call in plan.calls}
    return executor, sent


def test_shared_pool_compiles_to_two_calls_and_one_physical_pool():
    compiled = SharedLinearPipeline(replicas=4).compile()
    calls = tuple(compiled.plan.calls)
    first, second = calls

    assert len(compiled.plan.actor_pools) == 1
    assert compiled.plan.dispatch(first).pool == compiled.plan.dispatch(second).pool
    assert compiled.plan.pool(first) is compiled.plan.pool(second)
    assert compiled.plan.pool(first).replicas == 4
    assert dict(compiled.plan.pool(first).ray_options) == {"num_cpus": 0}
    assert compiled.plan.dispatch(first).batch_size == 3
    assert compiled.plan.dispatch(second).batch_size == 3
    assert dict(compiled.plan.input_layouts_by_call[first].static_kwargs) == {
        "stage": "first"
    }
    assert dict(compiled.plan.input_layouts_by_call[second].static_kwargs) == {
        "stage": "second"
    }


def test_static_and_dynamic_keyword_inputs_have_one_explicit_layout():
    class Combine:
        def run(self, values, *, offsets, scale, enabled):
            factor = scale if enabled else 1
            return [
                (value + offset) * factor
                for value, offset in zip(values, offsets)
            ]

    class Keywords(ro.Pipeline):
        def __init__(self):
            self.combine = ro.RayModule(Combine)

        def forward(self, values, offsets):
            return self.combine(
                values,
                offsets=offsets,
                scale=3,
                enabled=True,
            )

    plan = Keywords().compile().plan
    call = next(iter(plan.calls))
    layout = plan.input_layouts_by_call[call]

    assert layout.positional_count == 1
    assert layout.keyword_names == ("offsets",)
    assert layout.static_kwargs == (("scale", 3), ("enabled", True))


def test_ray_module_requires_a_dynamic_port_input():
    class Invalid(ro.Pipeline):
        def __init__(self):
            self.call = ro.RayModule(SharedStages)

        def forward(self, values):
            del values
            return self.call(stage="first")

    with pytest.raises(
        ro.CompileError,
        match="RayModule requires at least one Port input",
    ):
        Invalid().compile()


def test_static_keyword_arguments_cannot_hide_symbolic_ports():
    class Invalid(ro.Pipeline):
        def __init__(self):
            self.call = ro.RayModule(SharedStages)

        def forward(self, values):
            return self.call(values, config={"dependency": values})

    with pytest.raises(
        ro.CompileError,
        match="static keyword argument 'config' contains a Port",
    ):
        Invalid().compile()


def test_first_call_snapshots_one_shared_module_configuration():
    class Identity:
        def run(self, values):
            return values

    class Mutating(ro.Pipeline):
        def __init__(self):
            self.stage = ro.RayModule(Identity).ray_options(
                replicas=1,
                batch_size=4,
            )

        def forward(self, values):
            first = self.stage(values)
            self.stage.ray_options(replicas=8, batch_size=64)
            return self.stage(first)

    plan = Mutating().compile().plan
    first, second = plan.calls

    assert len(plan.actor_pools) == 1
    assert plan.pool(first).replicas == plan.pool(second).replicas == 1
    assert plan.dispatch(first).batch_size == 4
    assert plan.dispatch(second).batch_size == 4


def test_worker_forwards_static_kwargs_to_run():
    class Capture:
        def run(self, values, *, stage, flag, count):
            return [
                (value, stage, flag, count)
                for value in values
            ]

    store = MemoryStore()
    block = store.put((5, 6))
    invocations = tuple(
        GrainInvocation(
            GrainRef(CallRef(0), EntityRef(DomainRef(0), row)),
            0,
            (RowBinding(block, row),),
        )
        for row in range(2)
    )
    result = Worker(Capture).execute(
        invocations,
        (CallOutputLayout(PortRef(0)),),
        store,
        input_layout=CallInputLayout(
            1,
            static_kwargs=(("stage", "layout"), ("flag", False), ("count", 2)),
        ),
    )

    assert not isinstance(result, DispatchFailure)
    output_block = result[0].outputs[0].scalar.block
    assert store.blocks[output_block] == (
        (5, "layout", False, 2),
        (6, "layout", False, 2),
    )


def test_reusing_one_ray_module_automatically_shares_its_pool():
    class Identity:
        def run(self, values):
            return values

    class OldStyle(ro.Pipeline):
        def __init__(self):
            self.stage = ro.RayModule(Identity)

        def forward(self, values):
            return self.stage(self.stage(values))

    compiled = OldStyle().compile()

    assert len(compiled.plan.calls) == 2
    assert len(compiled.plan.actor_pools) == 1


def test_distinct_ray_module_objects_keep_independent_pools():
    class Identity:
        def run(self, values):
            return values

    class Separate(ro.Pipeline):
        def __init__(self):
            self.first = ro.RayModule(Identity)
            self.second = ro.RayModule(Identity)

        def forward(self, values):
            return self.second(self.first(values))

    compiled = Separate().compile()

    assert len(compiled.plan.calls) == 2
    assert len(compiled.plan.actor_pools) == 2


def test_pool_sharing_uses_identity_not_ray_module_equality():
    class EqualModules(ro.RayModule):
        __hash__ = None

        def __eq__(self, other):
            return isinstance(other, EqualModules)

    class Identity:
        def run(self, values):
            return values

    class ExplicitIdentity(ro.Pipeline):
        def __init__(self):
            self.first = EqualModules(Identity)
            self.second = EqualModules(Identity)

        def forward(self, values):
            first = self.first(values)
            return self.first(first), self.second(values)

    plan = ExplicitIdentity().compile().plan
    first, reused, separate = plan.calls

    assert plan.dispatch(first).pool == plan.dispatch(reused).pool
    assert plan.dispatch(separate).pool != plan.dispatch(first).pool
    assert len(plan.actor_pools) == 2


def test_pool_round_robin_selects_calls_before_call_local_queue_priority():
    pipeline = SharedLinearPipeline()
    plan = pipeline.compile().plan
    executor, _ = _driver(pipeline)
    pool = next(iter(plan.actor_pools))
    first, second = plan.calls

    # The real linear DAG cannot make both Calls runnable before first commits.
    # Use the scheduler boundary directly to prove that a selected Call's local
    # queue priority never lets it monopolize another runnable Call.
    priorities = {
        first: 0,   # immediate retry
        second: 2,  # deferred retry
    }
    slots = {
        0: _InputBatchSlot(
            0,
            SimpleNamespace(
                dispatch_priority=lambda call: priorities.get(call)
            ),
        )
    }

    selected = [
        executor._select_pool_work(pool, slots)[0]
        for _ in range(6)
    ]

    assert selected == [first, second, first, second, first, second]


def test_each_idle_replica_advances_the_same_pool_round_robin():
    pipeline = SharedLinearPipeline(replicas=4)
    plan = pipeline.compile().plan
    executor, _ = _driver(pipeline)
    pool = next(iter(plan.actor_pools))
    first, second = plan.calls
    slots = {
        0: _InputBatchSlot(
            0,
            SimpleNamespace(dispatch_priority=lambda call: 1),
        )
    }

    selected = [
        executor._select_pool_work(pool, slots)[0]
        for _ in range(8)
    ]

    assert selected == [first, second] * 4


class StressStages:
    """Small deterministic methods used to stress scheduling, not compute."""

    def run(self, values, *, stage):
        if stage == "a":
            return [value + 1 for value in values]
        if stage == "b":
            return [value * 2 for value in values]
        if stage == "c":
            return [value - 3 for value in values]
        if stage == "d":
            return [value * value for value in values]
        raise ValueError(f"unknown stage: {stage}")


class StressPipeline(ro.Pipeline):
    def __init__(self, replicas: int) -> None:
        self.stages = ro.RayModule(StressStages).ray_options(
            replicas=replicas,
            batch_size=127,
        )

    def forward(self, values):
        a = self.stages(values, stage="a")
        b = self.stages(a, stage="b")
        c = self.stages(b, stage="c")
        return self.stages(c, stage="d")


def test_100k_grain_shared_pool_stress_has_bounded_progress():
    """A finite backlog drains without duplicate enqueue or empty spin turns."""

    count = 100_000
    source = tuple(range(count))
    run = _run_shared_pool_sync(StressPipeline(replicas=4), source)

    assert run.turns <= count
    assert run.max_pending <= run.replicas
    # A linear one-to-one DAG can move backlog between Calls but cannot create
    # more runnable Grains than the admitted source count.
    assert run.max_ready <= count
    assert sum(
        counters.grain_dispatches for counters in run.counters.values()
    ) == count * len(run.plan.calls)
    assert run.outputs == [
        ((value + 1) * 2 - 3) ** 2 for value in source
    ]


class SharedNestedStages:
    def run(self, values, groups=None, *, stage):
        if stage == "split":
            return [
                [(value, ordinal) for ordinal in range(value % 5)]
                for value in values
            ]
        if stage == "transform":
            return [value * 10 + ordinal for value, ordinal in values]
        if stage == "assemble":
            return [
                (value, tuple(group))
                for value, group in zip(values, groups)
            ]
        raise ValueError(f"unknown stage: {stage}")


class SharedNestedPipeline(ro.Pipeline):
    def __init__(self) -> None:
        self.stages = ro.RayModule(SharedNestedStages).ray_options(
            replicas=3,
            batch_size=113,
        )

    def forward(self, values):
        children = ro.F.expand(self.stages(values, stage="split"))
        transformed = self.stages(children, stage="transform")
        return self.stages(
            values,
            ro.F.reduce(transformed),
            stage="assemble",
        )


def test_shared_pool_stress_preserves_expand_reduce_lineage_and_order():
    """One shared pool can cross Domain boundaries without hidden actor state."""

    source = tuple(range(10_000))
    run = _run_shared_pool_sync(SharedNestedPipeline(), source)
    child_count = sum(value % 5 for value in source)

    assert run.max_pending <= run.replicas
    assert run.max_ready <= len(source) + child_count
    assert sum(
        counters.grain_dispatches for counters in run.counters.values()
    ) == len(source) * 2 + child_count
    assert run.outputs == [
        (
            value,
            tuple(value * 10 + ordinal for ordinal in range(value % 5)),
        )
        for value in source
    ]


class SharedBranchStages:
    def run(self, values, right=None, *, stage):
        if stage == "left":
            return [value + 1 for value in values]
        if stage == "right":
            return [value * 10 for value in values]
        if stage == "merge":
            return [
                left_value + right_value
                for left_value, right_value in zip(values, right)
            ]
        raise ValueError(f"unknown stage: {stage}")


class SharedBranchPipeline(ro.Pipeline):
    def __init__(self) -> None:
        self.stages = ro.RayModule(SharedBranchStages).ray_options(
            replicas=4,
            batch_size=101,
        )

    def forward(self, values):
        left = self.stages(values, stage="left")
        right = self.stages(values, stage="right")
        return self.stages(left, right, stage="merge")


def test_shared_pool_stress_drains_branch_and_join_without_starvation():
    """Uneven batching across sibling Calls must still release their join."""

    source = tuple(range(30_000))
    run = _run_shared_pool_sync(SharedBranchPipeline(), source)

    assert run.max_pending <= run.replicas
    assert run.max_ready <= len(source) * 2
    assert sum(
        counters.grain_dispatches for counters in run.counters.values()
    ) == len(source) * 3
    assert all(counters.rpcs > 0 for counters in run.counters.values())
    assert run.outputs == [value * 11 + 1 for value in source]
