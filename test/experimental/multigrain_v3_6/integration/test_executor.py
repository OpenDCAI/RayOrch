"""Real Ray actor lifetime, retry fencing and record-level failure boundaries."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import rayorch.experimental.multigrain_v3_6 as mg
from rayorch.experimental.multigrain_v3_6.execution.executor import Executor
from rayorch.experimental.multigrain_v3_6.execution.ray_backend import _RayWorkerActor
from rayorch.experimental.multigrain_v3_6.model import ItemOutcome
from rayorch.experimental.multigrain_v3_6.protocol import RecordFailure


@pytest.fixture(scope="module", autouse=True)
def ray_runtime():
    import ray

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "local"),
            include_dashboard=False,
        )
    yield
    if started_here:
        ray.shutdown()


class PersistentIdentity:
    def __init__(self) -> None:
        self.identity = os.getpid()

    def run(self, values):
        return [(value, self.identity) for value in values]


class IdentityPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.identity = mg.RayModule(PersistentIdentity).ray_options(
            batch_size=3,
            replicas=1,
            num_cpus=0,
        )

    def forward(self, values):
        return self.identity(values)


def test_persistent_actor_is_shared_by_multiple_microbatches():
    with Executor(IdentityPipeline()) as executor:
        result = executor.run(range(15), microbatch_size=4, max_active_microbatches=3)

    assert [value for value, _ in result.outputs] == list(range(15))
    assert len({identity for _, identity in result.outputs}) == 1
    assert result.actor_count == 1
    assert result.peak_active_microbatches == 3
    assert len(result.microbatches) == 4
    assert all(microbatch.grain_count > 0 for microbatch in result.microbatches)
    assert result.released_values > 0
    assert executor.store._cache == {}
    metrics = result.calls[0]
    assert len(metrics.worker_snapshots) == 1
    assert metrics.worker_snapshots[0].lifetime_calls == metrics.rpcs


def test_empty_input_and_repeated_runs_keep_metrics_scopes_explicit():
    with Executor(IdentityPipeline()) as executor:
        empty = executor.run([])
        first = executor.run([1, 2])
        second = executor.run([3])

    assert empty.outputs == []
    assert empty.microbatches[0].entity_count == 0
    assert empty.calls[0].rpcs == 0
    assert first.outputs[0][0] == 1
    assert second.outputs[0][0] == 3
    assert first.calls[0].rpcs == 1
    assert second.calls[0].rpcs == 1
    assert second.calls[0].worker_snapshots[0].lifetime_calls == 2


def test_actor_class_is_wrapped_once_per_executor(monkeypatch):
    import ray

    wraps = 0
    original_remote = ray.remote

    def tracking_remote(*args, **kwargs):
        nonlocal wraps
        if args and args[0] is _RayWorkerActor:
            wraps += 1
        return original_remote(*args, **kwargs)

    monkeypatch.setattr(ray, "remote", tracking_remote)
    pipeline = IdentityPipeline()
    pipeline.identity = pipeline.identity.ray_options(replicas=3)
    with Executor(pipeline):
        pass

    assert wraps == 1


def test_synchronous_actor_construction_failure_cleans_partial_pool(monkeypatch):
    import ray

    created = []
    killed = []

    class ActorClass:
        def options(self, **options):
            return self

        def remote(self, *args):
            if created:
                raise RuntimeError("second actor construction failed")
            handle = object()
            created.append(handle)
            return handle

    monkeypatch.setattr(ray, "remote", lambda target: ActorClass())
    monkeypatch.setattr(
        ray,
        "kill",
        lambda handle, *, no_restart: killed.append(handle),
    )
    pipeline = IdentityPipeline()
    pipeline.identity = pipeline.identity.ray_options(replicas=2)

    with pytest.raises(RuntimeError, match="second actor construction failed"):
        Executor(pipeline)

    assert killed == created


class KeywordOnlyAdd:
    def run(self, values, *, increments):
        return [
            value + increment
            for value, increment in zip(values, increments)
        ]


class KeywordPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.call = mg.RayModule(KeywordOnlyAdd).ray_options(
            batch_size=4,
            num_cpus=0,
        )

    def forward(self, values, increments):
        return self.call(values, increments=increments)


def test_keyword_input_layout_survives_actor_boundary():
    with Executor(KeywordPipeline()) as executor:
        result = executor.run([1, 2, 3], [10, 20, 30])

    assert result.outputs == [11, 22, 33]


class CrashOnce:
    def __init__(self, marker: str) -> None:
        self.marker = Path(marker)

    def run(self, values):
        if not self.marker.exists():
            self.marker.write_text("crashed", encoding="utf-8")
            os._exit(17)
        return [value * 2 for value in values]


class CrashPipeline(mg.Pipeline):
    def __init__(self, marker: str) -> None:
        self.call = (
            mg.RayModule(CrashOnce)
            .pre_init(marker)
            .ray_options(
                batch_size=4,
                replicas=1,
                recovery=mg.RecoveryPolicy.abort(infra_retries=1),
                num_cpus=0,
                max_restarts=0,
            )
        )

    def forward(self, values):
        return self.call(values)


def test_actor_crash_replaces_actor_and_replays_same_grains(tmp_path):
    marker = tmp_path / "v36-crash-once"
    with Executor(CrashPipeline(str(marker))) as executor:
        result = executor.run(range(4), microbatch_size=4)

    metrics = result.calls[0]
    assert result.outputs == [0, 2, 4, 6]
    assert metrics.actor_instances == 2
    assert metrics.retries == 4
    assert metrics.rpcs == 2


class Render:
    def run(self, parents):
        return [[(parent, ordinal) for ordinal in range(3)] for parent in parents]


class FailOneLeaf:
    def run(self, leaves):
        return [
            RecordFailure(f"bad leaf {leaf}") if leaf == (1, 1) else leaf
            for leaf in leaves
        ]


class Assemble:
    def run(self, parents, groups):
        return list(zip(parents, groups))


class BadLeafPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.render = mg.RayModule(Render).ray_options(batch_size=8, num_cpus=0)
        self.transform = mg.RayModule(FailOneLeaf).ray_options(
            batch_size=16,
            num_cpus=0,
        )
        self.assemble = mg.RayModule(Assemble).ray_options(
            batch_size=8,
            num_cpus=0,
        )

    def forward(self, parents):
        leaves = mg.F.expand(self.render(parents))
        values = self.transform(leaves)
        return self.assemble(parents, mg.F.reduce(values))


def test_record_failure_suppresses_only_its_parent():
    with Executor(BadLeafPipeline()) as executor:
        result = executor.run([0, 1, 2])

    assert result.outputs[0] == (0, [(0, 0), (0, 1), (0, 2)])
    assert result.outputs[1] is ItemOutcome.SUPPRESSED
    assert result.outputs[2] == (2, [(2, 0), (2, 1), (2, 2)])
    transform = next(
        metrics
        for metrics in result.calls
        if metrics.udf_name.endswith("FailOneLeaf")
    )
    assert transform.grains == 9


class MultiOutputFail:
    def run(self, values):
        return (
            [value * 10 for value in values],
            [
                RecordFailure("second output failed") if value == 2 else value * 100
                for value in values
            ],
        )


class MultiOutputPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.call = mg.RayModule(MultiOutputFail, num_outputs=2).ray_options(
            batch_size=8,
            num_cpus=0,
        )

    def forward(self, values):
        return self.call(values)


def test_record_failure_makes_all_outputs_of_one_grain_failed():
    with Executor(MultiOutputPipeline()) as executor:
        left, right = executor.run([1, 2, 3]).outputs

    assert left == [10, ItemOutcome.FAILED, 30]
    assert right == [100, ItemOutcome.FAILED, 300]


class FailFirstDispatch:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, values):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient UDF failure")
        return [(value, self.calls) for value in values]


class RetryPipeline(mg.Pipeline):
    def __init__(self, recovery: mg.RecoveryPolicy) -> None:
        self.call = mg.RayModule(FailFirstDispatch).ray_options(
            batch_size=2,
            replicas=1,
            recovery=recovery,
            num_cpus=0,
        )

    def forward(self, values):
        return self.call(values)


@pytest.mark.parametrize(
    ("recovery", "expected"),
    [
        (
            mg.RecoveryPolicy.retry_batch(),
            [(0, 2), (1, 2), (2, 3), (3, 3)],
        ),
        (
            mg.RecoveryPolicy.retry_tail(),
            [(0, 3), (1, 3), (2, 2), (3, 2)],
        ),
    ],
)
def test_udf_retry_policy_controls_immediate_or_tail_order(recovery, expected):
    with Executor(RetryPipeline(recovery)) as executor:
        result = executor.run(range(4))

    metrics = result.calls[0]
    assert result.outputs == expected
    assert metrics.rpcs == 3
    assert metrics.retries == 2
    assert metrics.actor_instances == 1


class PoisonValue:
    def run(self, values):
        if 2 in values:
            raise ValueError("poison value 2")
        return [value * 10 for value in values]


class IsolationPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.call = mg.RayModule(PoisonValue).ray_options(
            batch_size=4,
            recovery=mg.RecoveryPolicy.isolate_tail(),
            num_cpus=0,
        )

    def forward(self, values):
        return self.call(values)


def test_isolate_tail_commits_only_the_poison_singleton_as_failed():
    with Executor(IsolationPipeline()) as executor:
        result = executor.run(range(4))

    metrics = result.calls[0]
    assert result.outputs == [0, 10, ItemOutcome.FAILED, 30]
    assert metrics.rpcs == 6
    assert metrics.retries == 10
    assert result.microbatches[0].grain_count == 4


class AlwaysUdfError:
    def run(self, values):
        raise ArithmeticError("deterministic UDF error")


class AbortPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.call = mg.RayModule(AlwaysUdfError).ray_options(
            batch_size=4,
            num_cpus=0,
        )

    def forward(self, values):
        return self.call(values)


def test_default_udf_policy_aborts_with_complete_dispatch_context():
    with pytest.raises(mg.ExecutionError) as captured:
        with Executor(AbortPipeline()) as executor:
            executor.run(range(4))

    message = str(captured.value)
    assert "AlwaysUdfError" in message
    assert "UDF_ERROR" in message
    assert "ArithmeticError" in message
    assert "deterministic UDF error" in message
    assert "generation=0" in message


class WrongCardinality:
    def __init__(self, marker: str) -> None:
        self.marker = Path(marker)

    def run(self, values):
        calls = int(self.marker.read_text()) if self.marker.exists() else 0
        self.marker.write_text(str(calls + 1), encoding="utf-8")
        return []


class ContractPipeline(mg.Pipeline):
    def __init__(self, marker: str) -> None:
        self.call = (
            mg.RayModule(WrongCardinality)
            .pre_init(marker)
            .ray_options(
                batch_size=4,
                recovery=mg.RecoveryPolicy.isolate_tail(),
                num_cpus=0,
            )
        )

    def forward(self, values):
        return self.call(values)


def test_worker_contract_error_is_readable_and_never_retried(tmp_path):
    marker = tmp_path / "v36-contract-calls"
    with pytest.raises(mg.ExecutionError) as captured:
        with Executor(ContractPipeline(str(marker))) as executor:
            executor.run(range(4))

    message = str(captured.value)
    assert marker.read_text(encoding="utf-8") == "1"
    assert "WrongCardinality" in message
    assert "CONTRACT_ERROR" in message
    assert "expected 4 Grain rows, got 0" in message
    assert "PortRef" in message
    assert "GrainRef" in message
    assert "generation=0" in message
    assert "worker traceback" in message


class AlwaysCrash:
    def __init__(self, marker: str) -> None:
        self.marker = Path(marker)

    def run(self, values):
        crashes = int(self.marker.read_text()) if self.marker.exists() else 0
        self.marker.write_text(str(crashes + 1), encoding="utf-8")
        os._exit(19)


class ExhaustedInfrastructurePipeline(mg.Pipeline):
    def __init__(self, marker: str) -> None:
        self.call = (
            mg.RayModule(AlwaysCrash)
            .pre_init(marker)
            .ray_options(
                batch_size=4,
                recovery=mg.RecoveryPolicy.abort(infra_retries=1),
                num_cpus=0,
                max_restarts=0,
            )
        )

    def forward(self, values):
        return self.call(values)


def test_exhausted_infrastructure_budget_aborts_without_data_failure(tmp_path):
    marker = tmp_path / "v36-always-crash"
    with pytest.raises(mg.ExecutionError) as captured:
        with Executor(ExhaustedInfrastructurePipeline(str(marker))) as executor:
            executor.run(range(4))

    message = str(captured.value)
    assert marker.read_text(encoding="utf-8") == "2"
    assert "AlwaysCrash" in message
    assert "INFRA_FAILURE" in message
    assert "generation=1" in message
