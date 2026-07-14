from __future__ import annotations

import pytest

import rayorch.experimental.multigrain as mg
from rayorch.experimental.multigrain.ir import (
    RecordRecoveryAction,
    ShardRecoveryAction,
)
from rayorch.runtime import BadRecordError

from test.experimental.multigrain.dummy_ops import SlowEmbed


class RecoveryPipe(mg.Pipeline):
    def __init__(
        self,
        policy: mg.RecoveryPolicy,
        op_cls=SlowEmbed,
        *op_args,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.map = mg.Map(
            op_cls,
            *op_args,
            recovery=policy,
            num_outputs=num_outputs,
        )

    def forward(self, rows):
        return self.map(rows)


def test_recovery_policy_decisions_are_scope_local() -> None:
    policy = mg.RecoveryPolicy(
        max_record_retries=2,
        retry_timing="inline",
        max_shard_retries=3,
        on_shard_exhausted="degrade",
    )

    assert policy.decide_record(
        retryable=True, attempt=0
    ) is RecordRecoveryAction.RETRY
    assert policy.decide_record(
        retryable=True, attempt=2
    ) is RecordRecoveryAction.ISOLATE
    assert policy.decide_record(
        retryable=False, attempt=0
    ) is RecordRecoveryAction.ISOLATE
    assert policy.decide_shard(attempt=2) is ShardRecoveryAction.RETRY
    assert policy.decide_shard(attempt=3) is ShardRecoveryAction.DEGRADE


def test_recovery_policy_round_trips_through_passive_ir() -> None:
    policy = mg.RecoveryPolicy(max_record_retries=1)
    graph = RecoveryPipe(policy).compile()

    assert graph.nodes[0].recovery == policy
    encoded = graph.to_dict()["nodes"][0]["recovery"]
    assert encoded["type"] == "RecoveryPolicy"
    assert encoded["max_record_retries"] == 1
    assert encoded["retry_timing"] == "inline"
    assert encoded["isolation"]["type"] == "IsolationBudget"


def test_unimplemented_policy_is_rejected_instead_of_silently_ignored() -> None:
    graph = RecoveryPipe(
        mg.RecoveryPolicy(max_record_retries=1, retry_timing="deferred")
    ).compile()
    rows = mg.source(["a"], name="rows")

    with pytest.raises(NotImplementedError, match="inline Map record retry"):
        mg.MultigrainExecutor().execute(graph, {"rows": rows})


def test_bad_record_error_classifies_transient_fault_without_internal_ids() -> None:
    error = BadRecordError("temporary decoder failure", index=3, retryable=True)

    assert error.index == 3
    assert error.retryable is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_record_retries", -1), ("max_shard_retries", -1)],
)
def test_recovery_policy_rejects_negative_retry_counts(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        mg.RecoveryPolicy(**{field: value})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_work_factor": -0.1}, "max_work_factor"),
        ({"max_work_factor": float("inf")}, "max_work_factor"),
        ({"max_calls": -1}, "max_calls"),
    ],
)
def test_isolation_budget_rejects_invalid_limits(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        mg.IsolationBudget(**kwargs)


def test_recovery_policy_accepts_serialized_isolation_mapping() -> None:
    policy = mg.RecoveryPolicy(
        isolation={
            "max_work_factor": 2,
            "max_calls": 8,
            "on_exhausted": "abort",
        }
    )

    assert policy.isolation == mg.IsolationBudget(
        max_work_factor=2.0,
        max_calls=8,
        on_exhausted="abort",
    )


class TransientMap:
    def __init__(self, failures: int) -> None:
        self.remaining = failures

    def run(self, rows):
        for index, row in enumerate(rows):
            if row == "flaky" and self.remaining > 0:
                self.remaining -= 1
                raise BadRecordError("temporary", index=index, retryable=True)
        return [row.upper() for row in rows]


class TransientMultiOutput:
    def __init__(self) -> None:
        self.failed = False

    def run(self, rows):
        for index, row in enumerate(rows):
            if row == "flaky" and not self.failed:
                self.failed = True
                raise BadRecordError("temporary", index=index, retryable=True)
        return [row.upper() for row in rows], [len(row) for row in rows]


def test_inline_record_retry_recovers_singleton_and_preserves_order() -> None:
    graph = RecoveryPipe(
        mg.RecoveryPolicy(max_record_retries=1),
        TransientMap,
        1,
    ).compile()
    rows = mg.source(["a", "flaky", "b"], name="rows")

    output = mg.MultigrainExecutor().execute(graph, {"rows": rows})

    assert output.values == ["A", "FLAKY", "B"]
    assert output.record_ids == rows.record_ids
    assert output.errors == []


def test_inline_record_retry_exhaustion_isolates_only_failed_row() -> None:
    graph = RecoveryPipe(
        mg.RecoveryPolicy(max_record_retries=2),
        TransientMap,
        5,
    ).compile()
    rows = mg.source(["a", "flaky", "b"], name="rows")

    output = mg.MultigrainExecutor().execute(graph, {"rows": rows})

    assert output.values == ["A", "B"]
    assert output.record_ids == [rows.record_ids[0], rows.record_ids[2]]
    assert [error.logical_item for error in output.errors] == ["flaky"]


def test_zero_record_budget_isolates_even_when_timing_is_deferred() -> None:
    graph = RecoveryPipe(
        mg.RecoveryPolicy(max_record_retries=0, retry_timing="deferred"),
        TransientMap,
        1,
    ).compile()
    rows = mg.source(["good", "flaky"], name="rows")

    output = mg.MultigrainExecutor().execute(graph, {"rows": rows})

    assert output.values == ["GOOD"]
    assert [error.logical_item for error in output.errors] == ["flaky"]


def test_inline_record_retry_recovers_all_output_ports_atomically() -> None:
    graph = RecoveryPipe(
        mg.RecoveryPolicy(max_record_retries=1),
        TransientMultiOutput,
        num_outputs=2,
    ).compile()
    rows = mg.source(["flaky", "ok"], name="rows")

    upper, lengths = mg.MultigrainExecutor().execute(graph, {"rows": rows})

    assert upper.values == ["FLAKY", "OK"]
    assert lengths.values == [5, 2]
    assert upper.record_ids == lengths.record_ids == rows.record_ids
