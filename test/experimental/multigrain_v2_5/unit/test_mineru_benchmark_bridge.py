"""CPU-only contract tests for the real MinerU V2.5 benchmark bridge."""

from __future__ import annotations

import json
import os

from rayorch.experimental.multigrain_v2_5.benchmark.mineru import (
    GpuDeviceSample,
    GpuSample,
    MinerUV25Pipeline,
    _configure_timeline_profiling,
    _write_observation_artifacts,
    build_parser,
)
from rayorch.experimental.multigrain_v2_5.graph import Primitive
from rayorch.experimental.multigrain_v2_5.metrics import DispatchTimeline


def test_mineru_pipeline_compiles_expand_gpu_map_and_aligned_reduce():
    """The real bridge remains a native Pipeline with ordered page side input."""

    pipeline = MinerUV25Pipeline(
        output_dir="/tmp/mineru-v2-5-test",
        mode="elastic",
        model="/tmp/model",
        replicas=4,
        batch_size=16,
        max_batch_wait_ms=5.0,
        gpu_memory_utilization=0.9,
        render_replicas=4,
        reduce_replicas=4,
        runtime_env={},
    )
    compiled = pipeline.compile()
    kinds = tuple(node.kind for node in compiled.graph.nodes)
    assert kinds == (
        Primitive.SOURCE,
        Primitive.EXPAND,
        Primitive.MAP,
        Primitive.REDUCE,
    )
    ocr = compiled.graph.nodes[2]
    assert ocr.execution is not None
    assert ocr.execution.replicas == 4
    assert dict(ocr.execution.options)["num_gpus"] == 1.0
    reduce = compiled.graph.nodes[3]
    assert tuple(binding.role for binding in reduce.inputs) == (
        "anchor",
        "members",
        "pages",
    )


def test_mineru_cli_defaults_to_safe_four_document_smoke():
    """Manual CLI defaults avoid accidentally launching the 368-PDF run."""

    args = build_parser().parse_args([])
    assert args.limit == 4
    assert args.replicas == 4
    assert args.batch_size == 16
    assert args.mode == "elastic"
    assert args.timeline_dir == ""


def test_mineru_cli_accepts_observation_only_timeline_export():
    """Timeline export is opt-in and does not alter executor configuration."""

    args = build_parser().parse_args(
        [
            "--timeline-dir",
            "/tmp/mineru-timeline",
            "--microbatch-size",
            "24",
            "--max-inflight-arenas",
            "4",
        ]
    )
    assert args.timeline_dir == "/tmp/mineru-timeline"
    assert args.microbatch_size == 24
    assert args.max_inflight_arenas == 4


def test_timeline_profiling_uses_positive_ray_report_interval(monkeypatch):
    """Ray 2.50 needs a positive interval to publish legacy timeline spans."""

    monkeypatch.delenv("RAY_PROFILING", raising=False)
    monkeypatch.setenv("RAY_task_events_report_interval_ms", "0")
    _configure_timeline_profiling()
    assert os.environ["RAY_PROFILING"] == "1"
    assert os.environ["RAY_task_events_report_interval_ms"] == "100"


def test_observation_export_writes_ray_dispatch_and_gpu_timelines(
    monkeypatch,
    tmp_path,
):
    """Benchmark-only export keeps native and V2.5-specific evidence paired."""

    import ray

    monkeypatch.setattr(
        ray,
        "timeline",
        lambda: [{"name": "run", "ph": "X"}],
    )
    dispatch = DispatchTimeline(
        arena=1,
        node=2,
        dispatch=3,
        actor_index=0,
        grains=16,
        flush_reason="full",
        submitted_at=1.0,
        manifest_received_at=2.0,
        committed_at=2.1,
        worker_started_at=1.1,
        worker_finished_at=1.9,
        worker_rss_bytes=123,
        status="accepted",
    )
    gpu = GpuSample(
        wall_time_s=10.0,
        monotonic_s=5.0,
        devices=(
            GpuDeviceSample(
                index=0,
                utilization_percent=75,
                memory_used=100,
                memory_total=200,
            ),
        ),
    )
    artifacts = _write_observation_artifacts(
        str(tmp_path),
        (dispatch,),
        (gpu,),
    )
    assert artifacts["ray_timeline_events"] == 1
    assert json.loads((tmp_path / "ray_timeline.json").read_text()) == [
        {"name": "run", "ph": "X"}
    ]
    assert json.loads(
        (tmp_path / "dispatch_timeline.jsonl").read_text()
    )["arena"] == 1
    assert json.loads(
        (tmp_path / "gpu_samples.jsonl").read_text()
    )["devices"][0]["utilization_percent"] == 75
