"""MinerU matrix 聚合与 correctness 工具测试。"""

from __future__ import annotations

import json

import pytest

from rayorch.experimental.multigrain_v3.benchmark.mineru_correctness import (
    compare,
)
from rayorch.experimental.multigrain_v3.benchmark.mineru_report import (
    load_rows,
    summarize,
)


def _rows(repeats: int = 2):
    """构造四模式完整 synthetic matrix。"""

    rows = []
    for engine, mode, wall in (
        ("multigrain_v3", "elastic", 5.0),
        ("multigrain_v3", "parent_bound", 8.0),
        ("ray_data", None, 7.0),
        ("flash_mineru_native_dag", None, 9.0),
    ):
        for index in range(repeats):
            row = {
                "engine": engine,
                "n_pdf": 4,
                "measured_wall_s": wall + index,
                "end_to_end_wall_s": wall + index + 2,
            }
            if mode:
                row["mode"] = mode
            rows.append(row)
    return rows


def test_report_validates_and_computes_speedups(tmp_path) -> None:
    """完整 matrix 应产生四模式中位数和 elastic speedups。"""

    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in _rows()),
        encoding="utf-8",
    )
    report = summarize(
        load_rows(str(path)),
        expected_repeats=2,
        expected_pdfs=4,
    )

    assert report["engines"]["v3_elastic"]["median_wall_s"] == 7.5
    assert report["engines"]["v3_elastic"]["median_measured_wall_s"] == 5.5
    assert report["speedups"]["elastic_vs_native"] == pytest.approx(
        11.5 / 7.5
    )


def test_report_rejects_incomplete_matrix(tmp_path) -> None:
    """缺任一 engine/repeat 时不能生成看似完整的报告。"""

    rows = _rows()
    rows.pop()
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        summarize(
            load_rows(str(path)),
            expected_repeats=2,
            expected_pdfs=4,
        )


def test_correctness_reports_missing_and_similarity(tmp_path) -> None:
    """Markdown 比较应同时报告缺失文档和 token similarity。"""

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.md").write_text("hello world", encoding="utf-8")
    (right / "a.md").write_text("hello world!", encoding="utf-8")
    (right / "b.md").write_text("missing", encoding="utf-8")

    result = compare(str(left), str(right))

    assert result["matched_documents"] == 1
    assert result["missing_from_left"] == ["b"]
    assert result["jaccard_median"] == pytest.approx(1 / 3)
