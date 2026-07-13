from __future__ import annotations

import pytest
import ray

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.graph import PhysicalHints
from rayorch.experimental.multigrain.ray_executor import MultigrainRayExecutor

from test.experimental.multigrain.dummy_ops import (
    AlwaysRetryableBadMap,
    AlwaysOpaqueFail,
    OpaquePoisonMap,
    RetryableOnceMap,
)

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("ray_cluster")]


class RecoveryMapPipe(mg.Pipeline):
    def __init__(
        self,
        op_cls,
        policy: mg.RecoveryPolicy,
        *op_args,
        replicas: int = 1,
    ) -> None:
        super().__init__()
        self.op = mg.Map(
            op_cls,
            *op_args,
            name="recover-map",
            recovery=policy,
            physical=PhysicalHints(replicas=replicas),
        )

    def forward(self, rows):
        return self.op(rows)


def test_sparse_opaque_poison_is_localized_without_losing_siblings() -> None:
    policy = mg.RecoveryPolicy(
        max_shard_retries=0,
        on_shard_exhausted="degrade",
    )
    graph = RecoveryMapPipe(
        OpaquePoisonMap,
        policy,
        "bad",
        replicas=2,
    ).compile()
    rows = mg.source(["a", "bad", "b", "c", "d", "e", "f", "g"], name="rows")
    metrics = mg.RunMetrics()
    executor = MultigrainRayExecutor(metrics=metrics)
    try:
        output = executor.execute(graph, {"rows": rows})
    finally:
        executor.shutdown()

    assert output.values == [f"ok:{value}" for value in rows.values if value != "bad"]
    assert output.record_ids == [
        record_id
        for record_id, value in zip(rows.record_ids, rows.values)
        if value != "bad"
    ]
    assert [error.logical_item for error in output.errors] == ["bad"]
    stage = metrics.by_name("recover-map")
    assert stage is not None
    assert stage.recovery_rows < 2 * 4


def test_dense_failure_stops_at_budget_and_quarantines_unresolved_leaves() -> None:
    policy = mg.RecoveryPolicy(
        max_shard_retries=0,
        on_shard_exhausted="degrade",
        isolation=mg.IsolationBudget(max_work_factor=1.0, max_calls=64),
    )
    graph = RecoveryMapPipe(AlwaysOpaqueFail, policy).compile()
    rows = mg.source([str(index) for index in range(8)], name="rows")
    metrics = mg.RunMetrics()
    executor = MultigrainRayExecutor(metrics=metrics)
    try:
        output = executor.execute(graph, {"rows": rows})
    finally:
        executor.shutdown()

    assert output.values == []
    assert {error.logical_item for error in output.errors} == set(rows.values)
    stage = metrics.by_name("recover-map")
    assert stage is not None
    assert stage.recovery_rows == 8
    assert stage.retries == 2


def test_dead_actor_is_replaced_before_retry_completes() -> None:
    graph = RecoveryMapPipe(
        OpaquePoisonMap,
        mg.RecoveryPolicy(max_shard_retries=1),
        "never",
        replicas=2,
    ).compile()
    rows = mg.source(["a", "b", "c", "d"], name="rows")
    executor = MultigrainRayExecutor()
    try:
        executor.warm_pools(graph)
        ray.kill(executor._pools["recover-map"][0])

        output = executor.execute(graph, {"rows": rows})

        assert output.values == [f"ok:{value}" for value in rows.values]
        assert ray.get(executor._pools["recover-map"][0].ping.remote()) is True
    finally:
        executor.shutdown()


def test_deferred_stage_epoch_batches_records_across_microbatches() -> None:
    policy = mg.RecoveryPolicy(
        max_record_retries=1,
        retry_timing="deferred",
    )
    graph = RecoveryMapPipe(RetryableOnceMap, policy).compile()
    microbatches = [
        {"rows": mg.source([f"good-{index}", f"flaky-{index}"], name="rows")}
        for index in range(3)
    ]
    metrics = mg.RunMetrics()
    executor = MultigrainRayExecutor(metrics=metrics)
    try:
        outputs = list(
            executor.execute_stream(
                graph,
                microbatches,
                max_inflight=3,
            )
        )
    finally:
        executor.shutdown()

    assert [output.values for output in outputs] == [
        [f"ok:good-{index}", f"ok:flaky-{index}"]
        for index in range(3)
    ]
    assert all(output.errors == [] for output in outputs)
    # At least one recovery invocation contains records from multiple physical
    # microbatches, proving this is a stage epoch rather than chunk-local retry.
    recovery_metrics = [
        metric for metric in metrics.nodes if metric.name == "recover-map"
    ]
    assert len(microbatches) < len(recovery_metrics) < 2 * len(microbatches)
    assert sum(metric.rows_in for metric in recovery_metrics) == 9


def test_deferred_retry_exhaustion_quarantines_only_the_bad_record() -> None:
    policy = mg.RecoveryPolicy(
        max_record_retries=1,
        retry_timing="deferred",
    )
    graph = RecoveryMapPipe(AlwaysRetryableBadMap, policy).compile()
    rows = mg.source(["good", "bad-0"], name="rows")
    executor = MultigrainRayExecutor()
    try:
        output = executor.execute(graph, {"rows": rows})
    finally:
        executor.shutdown()

    assert output.values == ["ok:good"]
    assert [error.logical_item for error in output.errors] == ["bad-0"]
    assert output.errors[0].action == "quarantined_deferred_exhausted"
