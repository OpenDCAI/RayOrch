"""Slow smoke test for the fair parent-bound versus elastic benchmark."""

from __future__ import annotations

import json

import pytest

from ..benchmarks import generate_workload, run_benchmark, write_reports


@pytest.mark.slow
@pytest.mark.usefixtures("ray_cluster")
def test_elastic_benchmark_preserves_results_and_emits_reports(tmp_path):
    """Both modes use identical UDFs; elastic batching must not add RPCs."""

    workload = generate_workload(
        seed=25_032_027,
        parent_count=24,
        fanout_mode="pareto",
        fanout_scale=4,
        service_mode="lognormal",
        mean_service_s=0.0005,
        drop_probability=0.15,
    )
    parent_report, parent_outputs = run_benchmark(
        workload,
        mode="parent_bound",
        batch_size=8,
        max_batch_wait_ms=0,
        replicas=2,
    )
    elastic_report, elastic_outputs = run_benchmark(
        workload,
        mode="elastic",
        batch_size=8,
        max_batch_wait_ms=5,
        replicas=2,
    )

    assert sorted(parent_outputs) == sorted(elastic_outputs)
    assert elastic_report.rpc_count <= parent_report.rpc_count
    assert elastic_report.grains_per_rpc >= parent_report.grains_per_rpc
    assert parent_report.parent_p95_s > 0
    assert elastic_report.parent_p95_s > 0

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    write_reports(
        (parent_report, elastic_report),
        json_path=json_path,
        csv_path=csv_path,
    )
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert [row["mode"] for row in rows] == [
        "parent_bound",
        "elastic",
    ]
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith(
        "mode,seed,parents,total_children"
    )
