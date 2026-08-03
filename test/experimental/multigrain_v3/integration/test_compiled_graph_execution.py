"""End-to-end defect coverage for CompiledGraph coordinator behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from rayorch.experimental.multigrain_v3.api import ExecutorSession, Map, Pipeline
from rayorch.experimental.multigrain_v3.ray.worker import BadGrainError
from rayorch.experimental.multigrain_v3.runtime.coordinator import RunAborted


_RUNTIME_ENV = {
    "env_vars": {"PYTHONPATH": str(Path(__file__).parent)}
}


class _SelectiveBadMap:
    """Fail exactly the entry containing the sentinel ``bad`` value."""

    def run(self, values: list[str]) -> list[str]:
        """Return healthy values or attribute one bad batch entry."""

        for index, value in enumerate(values):
            if value == "bad":
                raise BadGrainError(index, "sentinel bad value")
        return [f"mapped:{value}" for value in values]


class _BadGrainPipeline(Pipeline[str]):
    """Expose one cross-root MAP batch with entry-local failure."""

    def __init__(self) -> None:
        """Configure one CPU actor and a three-root physical batch."""

        self.operation = Map(
            _SelectiveBadMap,
            replicas=1,
            batch_size=3,
            num_cpus=1.0,
            num_gpus=0.0,
            runtime_env=_RUNTIME_ENV,
        )

    def forward(self, source):
        """Return the MAP result directly as the named graph output."""

        return {"result": self.operation(source)}


class _GenericFailureMap:
    """Raise a generic UDF error only for the sentinel ``explode`` value."""

    def run(self, values: list[str]) -> list[str]:
        """Return healthy values or raise a non-attributable batch failure."""

        if "explode" in values:
            raise RuntimeError("generic sentinel failure")
        return [f"mapped:{value}" for value in values]


class _GenericFailurePipeline(Pipeline[str]):
    """Expose fail-fast generic failure and subsequent session reuse."""

    def __init__(self) -> None:
        """Configure one single-entry CPU MAP actor."""

        self.operation = Map(
            _GenericFailureMap,
            replicas=1,
            batch_size=1,
            num_cpus=1.0,
            num_gpus=0.0,
            runtime_env=_RUNTIME_ENV,
        )

    def forward(self, source):
        """Return the MAP result as the graph output."""

        return {"result": self.operation(source)}


@pytest.mark.cpu
@pytest.mark.usefixtures("ray_cluster")
def test_bad_grain_fails_one_root_and_retries_healthy_peers() -> None:
    """BAD_GRAIN must not abort healthy roots sharing its physical dispatch."""

    session = ExecutorSession(_BadGrainPipeline().compile())
    try:
        results = session.run(("first", "bad", "third")).collect()
        assert [result.status.value for result in results] == [
            "success",
            "failed",
            "success",
        ]
        assert results[0].outputs["result"].value.get() == "mapped:first"
        assert results[2].outputs["result"].value.get() == "mapped:third"
        failure = results[1].outputs["result"].failure
        assert failure.errors[0].kind == "BAD_GRAIN"
        assert "entry 1" in failure.errors[0].message
        assert session.transport.pending_count == 0
        assert not session.transport._submitted
    finally:
        session.close()


@pytest.mark.cpu
@pytest.mark.usefixtures("ray_cluster")
def test_generic_abort_finalizes_transport_before_session_reuse() -> None:
    """A fail-fast run must leave no pending physical owner before reuse."""

    session = ExecutorSession(_GenericFailurePipeline().compile())
    try:
        with pytest.raises(RunAborted, match="generic sentinel failure"):
            session.run(("explode",)).collect()
        assert session.transport.pending_count == 0
        assert not session.transport._submitted

        result = session.run(("healthy",)).collect()[0]
        assert result.outputs["result"].value.get() == "mapped:healthy"
        assert session.transport.pending_count == 0
    finally:
        session.close()
