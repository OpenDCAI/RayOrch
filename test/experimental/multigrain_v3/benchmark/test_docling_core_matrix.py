"""Docling 独立进程 matrix 顺序与波动报告测试。"""

from __future__ import annotations

import argparse

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_matrix import (
    build_commands,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_report import (
    summarize,
)


def test_matrix_rotates_each_arm_into_first_position(tmp_path) -> None:
    """四个 repeats 应让每个 arm 各有一次冷启动首位。"""

    args = argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        output_root=str(tmp_path / "out"),
        python="python",
        repeats=4,
        record_input_sha256=False,
        device="cuda",
        ocr_device="cpu",
        num_threads=4,
        stage_batch_size=4,
        native_tuned_doc_concurrency=4,
        native_tuned_doc_batch_size=24,
        parse_replicas=4,
        layout_replicas=1,
        ocr_replicas=4,
        table_replicas=3,
        reduce_replicas=4,
        parse_batch_wait_ms=2.0,
        stage_batch_wait_ms=2.0,
        layout_num_gpus=1.0,
        table_num_gpus=1.0,
        layout_actor_concurrency=1,
        ocr_actor_concurrency=1,
        table_actor_concurrency=1,
        actor_num_cpus=1.0,
        max_pending_per_actor=4,
        microbatch_size=24,
        max_inflight_arenas=3,
        ray_num_cpus=32,
        ray_num_gpus=4.0,
        four_gpu_native=True,
    )

    commands = build_commands(args)

    assert len(commands) == 16
    assert [
        row["arm"] for row in commands if row["position"] == 1
    ] == [
        "native_default",
        "native_tuned",
        "v3_parent_bound",
        "v3_elastic",
    ]
    by_repeat = {
        repeat: [
            row["arm"] for row in commands if row["repeat"] == repeat
        ]
        for repeat in range(1, 5)
    }
    assert by_repeat[2] == [
        "native_tuned",
        "v3_elastic",
        "v3_parent_bound",
        "native_default",
    ]
    assert "--table-batch-mode reference" in commands[0]["command"]
    assert "--table-batch-max-jobs 16" in commands[0]["command"]


def _row(arm: str, repeat: int, measured: float, e2e: float) -> dict:
    """构造 report 测试所需最小 arm row。"""

    metrics = {}
    if arm.startswith("v3_"):
        metrics = {"rpc_count": 10.0, "batch_fill_ratio": 0.5}
    return {
        "arm": arm,
        "matrix_repeat": repeat,
        "matrix_position": 1,
        "result": {
            "startup_s": e2e - measured,
            "measured_s": measured,
            "end_to_end_s": e2e,
            "metrics": metrics,
            "documents": [{}, {}],
        },
    }


def test_report_uses_medians_cv_and_paired_elastic_speedup() -> None:
    """报告应显式量化系统抖动，并使用 paired parent/elastic 比值。"""

    rows = tuple(
        _row(arm, repeat, measured, e2e)
        for arm, values in {
            "native_default": ((30.0, 35.0), (32.0, 37.0)),
            "native_tuned": ((20.0, 25.0), (18.0, 23.0)),
            "v3_parent_bound": ((15.0, 20.0), (18.0, 23.0)),
            "v3_elastic": ((12.0, 17.0), (15.0, 20.0)),
        }.items()
        for repeat, (measured, e2e) in enumerate(values, start=1)
    )

    report = summarize(
        rows,
        expected_repeats=2,
        expected_documents=2,
    )

    assert report["arms"]["v3_parent_bound"]["measured_median_s"] == 16.5
    assert report["speedups"]["paired_measured"] == [1.25, 1.2]
    assert report["speedups"]["paired_measured_median"] == 1.225
    assert report["arms"]["native_default"]["measured_cv"] > 0
