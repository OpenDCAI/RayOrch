"""Fast tests for workload determinism, aggregation, and manual CLI parsing."""

from __future__ import annotations

import json

from rayorch.experimental.multigrain_v2_5.benchmark import (
    BenchmarkReport,
    generate_workload,
    summarize_reports,
    write_reports,
)
from rayorch.experimental.multigrain_v2_5.benchmark.cli import build_parser


def _report(mode: str, repetition: int, wall: float) -> BenchmarkReport:
    return BenchmarkReport(
        mode=mode,
        seed=7,
        repetition=repetition,
        order_index=repetition % 2,
        fanout_mode="uniform",
        service_mode="constant",
        parent_count=10,
        total_children=40,
        kept_children=40,
        replicas=2,
        batch_size=8,
        max_batch_wait_ms=(2.0 if mode == "elastic" else 0.0),
        startup_time_s=0.1,
        measured_wall_time_s=wall,
        end_to_end_wall_time_s=wall + 0.1,
        throughput_children_s=40 / wall,
        rpc_count=8,
        grains_per_rpc=5,
        batch_fill_ratio=0.625,
        tail_rpc_fraction=0.25,
        parent_p50_s=0.2,
        parent_p95_s=0.3,
        parent_p99_s=0.35,
        expand_bubble_ratio=0.1,
        map_bubble_ratio=0.2,
        filter_bubble_ratio=0.3,
        reduce_bubble_ratio=0.4,
        flush_full=5,
        flush_timeout=1,
        flush_port_sealed=2,
        output_digest="same",
        git_commit="deadbeef",
        python_version="3.x",
        ray_version="x",
    )


def test_workload_generation_is_reproducible_and_seed_sensitive():
    """The same seed is byte-structurally stable while another seed differs."""

    kwargs = dict(
        parent_count=20,
        fanout_mode="pareto",
        fanout_scale=4,
        service_mode="lognormal",
        mean_service_s=0.001,
        drop_probability=0.2,
    )
    first = generate_workload(seed=1, **kwargs)
    retry = generate_workload(seed=1, **kwargs)
    different = generate_workload(seed=2, **kwargs)
    assert first == retry
    assert first != different


def test_summary_keeps_modes_separate_and_reports_raw_distribution():
    """Aggregation reports median/range without discarding paired raw trials."""

    reports = (
        _report("elastic", 0, 1.0),
        _report("elastic", 1, 2.0),
        _report("parent_bound", 0, 3.0),
    )
    summaries = summarize_reports(reports)
    assert [row["mode"] for row in summaries] == [
        "elastic",
        "parent_bound",
    ]
    assert summaries[0]["trials"] == 2
    assert summaries[0]["measured_wall_time_s_median"] == 1.5
    assert summaries[0]["measured_wall_time_s_min"] == 1.0
    assert summaries[0]["measured_wall_time_s_max"] == 2.0


def test_report_writer_emits_jsonl_summary_json_and_csv(tmp_path):
    """Manual runs always retain raw trials alongside aggregate reports."""

    reports = (_report("parent_bound", 0, 2.0), _report("elastic", 0, 1.0))
    raw, summary_json, summary_csv = write_reports(
        reports,
        output_dir=tmp_path,
    )
    assert raw.read_text(encoding="utf-8").count("\n") == 2
    assert len(json.loads(summary_json.read_text(encoding="utf-8"))) == 2
    assert "mode" in summary_csv.read_text(encoding="utf-8").splitlines()[0]


def test_manual_cli_parses_small_matrix_without_starting_ray():
    """CLI argument parsing is unit-testable; full matrices remain manual."""

    args = build_parser().parse_args(
        [
            "--parents",
            "12",
            "--fanout-modes",
            "uniform",
            "--actors",
            "2",
            "--seeds",
            "7,8",
            "--repetitions",
            "3",
        ]
    )
    assert args.parents == 12
    assert args.fanout_modes == "uniform"
    assert args.actors == "2"
    assert args.seeds == "7,8"
    assert args.repetitions == 3
