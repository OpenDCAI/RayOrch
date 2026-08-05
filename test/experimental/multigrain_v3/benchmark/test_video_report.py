"""Video A report 的纯函数测试；不启动 Ray 或访问 GPU。"""

from __future__ import annotations

import gzip
import json

import pytest

from rayorch.experimental.multigrain_v3.benchmark.video.report import (
    _video_output_comparison,
    correctness_report,
    load_outputs,
    summarize,
)


def test_video_correctness_tolerates_sparse_digest_only_differences() -> None:
    """不同 batch shape 的少量 ViT top-k 差异不应破坏结构 gate。"""

    baseline = [
        {
            "frames": 2,
            "source_indices": [0, 4],
            "digests": ["a", "b"],
            "mean_edge_density": 0.5,
        }
    ]
    candidate = [
        {
            "frames": 2,
            "source_indices": [0, 4],
            "digests": ["a", "c"],
            "mean_edge_density": 0.5,
        }
    ]

    comparison = _video_output_comparison(baseline, candidate)

    assert comparison["structure_exact"] is True
    assert comparison["digest_mismatch_frames"] == 1
    assert comparison["digest_mismatch_rate"] == 0.5


def _row(
    arm: str,
    repeat: int,
    wall: float,
    artifact: str | None = None,
) -> dict[str, object]:
    """构造一条最小独立进程结果。"""

    result: dict[str, object] = {
        "startup_inclusive_wall_s": wall,
        "videos": 3,
        "frames": 12,
        "metrics": {
            "rpc_count": 4,
            "batch_fill_ratio": 0.75,
            "transform_batch_size": 16,
        },
    }
    row: dict[str, object] = {
        "arm": arm,
        "matrix_repeat": repeat,
        "result": result,
    }
    if artifact is not None:
        row["outputs_output"] = artifact
    return row


def test_summarize_computes_wall_cv_paired_speedup_and_packing() -> None:
    """完整三臂矩阵应保留 wall 序列、paired speedup 与 packing 指标。"""

    rows = tuple(
        _row(arm, repeat, wall)
        for repeat in (1, 2)
        for arm, wall in (
            ("v3_parent_bound", 10.0 + repeat),
            ("v3_elastic", 5.0 + repeat),
            ("ray_data", 8.0 + repeat),
        )
    )

    report = summarize(rows, expected_repeats=2, expected_videos=3)

    assert report["arms"]["ray_data"]["wall_median_s"] == 9.5
    assert report["arms"]["ray_data"]["wall_cv"] > 0
    assert report["speedups"]["paired_parent_over_elastic"] == pytest.approx(
        [11 / 6, 12 / 7]
    )
    assert (
        report["arms"]["v3_elastic"]["packing_metrics"]["batch_fill_ratio"][
            "median"
        ]
        == 0.75
    )


def test_summarize_rejects_duplicate_or_incomplete_matrix() -> None:
    """重复 arm 或缺失 arm 都不能伪装成完整 repeat。"""

    rows = (
        _row("v3_parent_bound", 1, 10),
        _row("v3_parent_bound", 1, 11),
        _row("v3_elastic", 1, 5),
        _row("ray_data", 1, 8),
    )
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows, expected_repeats=1)


def test_correctness_reads_compressed_artifacts_and_reports_difference(
    tmp_path,
) -> None:
    """gzip artifact 完全相等时通过，差异时只报告路径而不回显大 outputs。"""

    paths = {}
    payloads = {
        "v3_parent_bound": {"outputs": [{"frames": 2, "digests": ["a"]}]},
        "v3_elastic": {"outputs": [{"frames": 2, "digests": ["a"]}]},
        "ray_data": {"outputs": [{"frames": 2, "digests": ["b"]}]},
    }
    for arm, payload in payloads.items():
        path = tmp_path / f"{arm}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as output:
            json.dump(payload, output)
        paths[arm] = str(path)

    rows = tuple(_row(arm, 1, 1.0, paths[arm]) for arm in payloads)
    report = correctness_report(rows)

    assert load_outputs(paths["v3_parent_bound"]) == payloads[
        "v3_parent_bound"
    ]["outputs"]
    assert report["all_structure_exact"] is True
    assert (
        report["repeats"]["1"]["parent_vs"]["v3_elastic"][
            "digest_mismatch_frames"
        ]
        == 0
    )
    ray_data = report["repeats"]["1"]["parent_vs"]["ray_data"]
    assert ray_data["structure_exact"] is True
    assert ray_data["digest_mismatch_frames"] == 1
