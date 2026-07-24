"""Real-Ray regressions for bounded microbatch arena overlap."""

from __future__ import annotations

import time

import pytest

import rayorch.experimental.multigrain_v2_5 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class TimedStage:
    def __init__(self, stage: str, delay_s: float):
        self.stage = stage
        self.delay_s = delay_s

    def run(self, rows):
        started = time.monotonic()
        time.sleep(self.delay_s)
        finished = time.monotonic()
        outputs = []
        for row in rows:
            if isinstance(row, dict):
                value = row["value"]
                history = list(row["history"])
            else:
                value = row
                history = []
            history.append(
                {
                    "stage": self.stage,
                    "start": started,
                    "stop": finished,
                }
            )
            outputs.append({"value": value, "history": history})
        return outputs


class TwoStagePipeline(mg.Pipeline):
    def __init__(self):
        self.first = (
            mg.Map(TimedStage)
            .pre_init("first", 0.15)
            .ray_options(batch_size=1)
        )
        self.second = (
            mg.Map(TimedStage)
            .pre_init("second", 0.15)
            .ray_options(batch_size=1)
        )

    def forward(self, rows):
        return self.second(self.first(rows))


def _run(max_inflight: int):
    return mg.Executor(
        TwoStagePipeline(),
        microbatch_size=1,
        max_inflight_arenas=max_inflight,
    ).run(["a", "b", "c", "d"])


def test_multiple_arenas_overlap_pipeline_stages_and_reduce_wall_time():
    """Arena N stage-2 overlaps Arena N+1 stage-1 on shared actor pools."""

    serial = _run(1)
    overlapped = _run(3)
    assert tuple(row["value"] for row in serial.get()) == ("a", "b", "c", "d")
    assert tuple(row["value"] for row in overlapped.get()) == (
        "a",
        "b",
        "c",
        "d",
    )
    assert overlapped.metrics["active_arenas_high_watermark"] == 3.0
    assert serial.metrics["active_arenas_high_watermark"] == 1.0
    assert (
        overlapped.metrics["measured_wall_time_s"]
        < serial.metrics["measured_wall_time_s"] * 0.8
    )

    first_by_arena = {
        event.arena: event
        for event in overlapped.timeline
        if event.node == 1
    }
    second_by_arena = {
        event.arena: event
        for event in overlapped.timeline
        if event.node == 2
    }
    assert max(
        first_by_arena[1].worker_started_at,
        second_by_arena[0].worker_started_at,
    ) < min(
        first_by_arena[1].worker_finished_at,
        second_by_arena[0].worker_finished_at,
    )


def test_microbatch_outputs_are_ordered_and_source_ids_are_run_global():
    """Completion order cannot reorder outputs or restart source identities."""

    result = _run(3)
    assert tuple(row["value"] for row in result.get()) == (
        "a",
        "b",
        "c",
        "d",
    )
    assert len(result.sources) == 4
    assert len({source.grain for source in result.sources}) == 4
    assert len({source.item.entity for source in result.sources}) == 4
    assert {event.arena for event in result.timeline} == {0, 1, 2, 3}


def test_completed_arenas_reclaim_while_later_arenas_continue():
    """Run metrics expose bounded active arenas and aggregate live block peak."""

    result = _run(2)
    assert result.metrics["active_arenas_high_watermark"] == 2.0
    assert result.metrics["live_blocks_at_delivery"] > 0
    assert result.metrics["live_blocks_across_arenas_high_watermark"] >= (
        result.metrics["live_blocks_at_delivery"]
    )


class FailOneStage:
    def run(self, rows):
        if "bad" in rows:
            raise RuntimeError("microbatch failure")
        time.sleep(0.05)
        return list(rows)


class FailingPipeline(mg.Pipeline):
    def __init__(self):
        self.map = mg.Map(FailOneStage).ray_options(
            batch_size=1,
            error_policy="raise",
        )

    def forward(self, rows):
        return self.map(rows)


def test_run_control_failure_cancels_other_active_arenas_fail_fast():
    """One arena abort fails the run and reclaims/cancels sibling arenas."""

    executor = mg.Executor(
        FailingPipeline(),
        microbatch_size=1,
        max_inflight_arenas=3,
    )
    with pytest.raises(mg.ExecutionError, match="microbatch failure"):
        executor.run(["good-a", "bad", "good-b"])
