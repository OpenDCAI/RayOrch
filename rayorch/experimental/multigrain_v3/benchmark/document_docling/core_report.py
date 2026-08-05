"""聚合独立进程 Docling 四臂结果并分析系统波动与 elastic 净收益。"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .core_compare import ARM_ORDER, _compare_documents, load_documents


def load_rows(path: str) -> tuple[dict[str, Any], ...]:
    """读取 shared JSONL，并拒绝空结果。"""

    rows = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not rows:
        raise ValueError("Docling matrix results are empty")
    return rows


def _cv(values: list[float]) -> float:
    """返回样本 coefficient of variation；单样本为 0。"""

    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def summarize(
    rows: tuple[dict[str, Any], ...],
    *,
    expected_repeats: int,
    expected_documents: int,
) -> dict[str, Any]:
    """校验矩阵完整性，并计算中位数、CV 与 paired elastic speedup。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_repeat: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm = str(row["arm"])
        repeat = row.get("matrix_repeat")
        if repeat is None:
            raise ValueError("matrix result has no matrix_repeat")
        repeat = int(repeat)
        if arm in by_repeat[repeat]:
            raise ValueError(
                f"duplicate result for repeat={repeat}, arm={arm}"
            )
        grouped[arm].append(row)
        by_repeat[repeat][arm] = row
    incomplete = {
        arm: len(grouped.get(arm, ()))
        for arm in ARM_ORDER
        if len(grouped.get(arm, ())) != expected_repeats
    }
    if incomplete:
        raise ValueError(f"incomplete Docling matrix: {incomplete}")
    expected_repeat_ids = set(range(1, expected_repeats + 1))
    if set(by_repeat) != expected_repeat_ids:
        raise ValueError(
            f"unexpected repeat ids: {sorted(by_repeat)}"
        )
    for repeat, repeat_rows in by_repeat.items():
        missing = set(ARM_ORDER) - set(repeat_rows)
        if missing:
            raise ValueError(
                f"repeat {repeat} is missing arms: {sorted(missing)}"
            )

    summary = {}
    for arm in ARM_ORDER:
        arm_rows = sorted(
            grouped[arm],
            key=lambda row: int(row["matrix_repeat"]),
        )
        if any(
            len(row["result"]["documents"]) != expected_documents
            for row in arm_rows
        ):
            raise ValueError(f"{arm} has incomplete document summaries")
        measured = [
            float(row["result"]["measured_s"]) for row in arm_rows
        ]
        e2e = [
            float(row["result"]["end_to_end_s"]) for row in arm_rows
        ]
        startup = [
            float(row["result"]["startup_s"]) for row in arm_rows
        ]
        summary[arm] = {
            "runs": len(arm_rows),
            "startup_s": startup,
            "measured_s": measured,
            "end_to_end_s": e2e,
            "startup_median_s": statistics.median(startup),
            "measured_median_s": statistics.median(measured),
            "end_to_end_median_s": statistics.median(e2e),
            "measured_cv": _cv(measured),
            "end_to_end_cv": _cv(e2e),
        }
        if arm.startswith("v3_"):
            summary[arm]["rpc_count"] = [
                row["result"]["metrics"]["rpc_count"]
                for row in arm_rows
            ]
            summary[arm]["batch_fill_ratio"] = [
                row["result"]["metrics"]["batch_fill_ratio"]
                for row in arm_rows
            ]

    parent = summary["v3_parent_bound"]
    elastic = summary["v3_elastic"]
    paired_measured = []
    paired_e2e = []
    for repeat in range(1, expected_repeats + 1):
        parent_row = by_repeat[repeat]["v3_parent_bound"]["result"]
        elastic_row = by_repeat[repeat]["v3_elastic"]["result"]
        paired_measured.append(
            float(parent_row["measured_s"])
            / float(elastic_row["measured_s"])
        )
        paired_e2e.append(
            float(parent_row["end_to_end_s"])
            / float(elastic_row["end_to_end_s"])
        )
    return {
        "expected_repeats": expected_repeats,
        "expected_documents": expected_documents,
        "arms": summary,
        "speedups": {
            "elastic_vs_parent_measured_median_ratio": (
                parent["measured_median_s"]
                / elastic["measured_median_s"]
            ),
            "elastic_vs_parent_e2e_median_ratio": (
                parent["end_to_end_median_s"]
                / elastic["end_to_end_median_s"]
            ),
            "paired_measured": paired_measured,
            "paired_measured_median": statistics.median(
                paired_measured
            ),
            "paired_e2e": paired_e2e,
            "paired_e2e_median": statistics.median(paired_e2e),
        },
    }


def correctness_report(
    rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """比较每个 arm artifact 与首个 Native default artifact。"""

    baseline_rows = [
        row for row in rows if row["arm"] == "native_default"
    ]
    if not baseline_rows:
        raise ValueError("missing native_default baseline")
    baseline_path = baseline_rows[0].get("documents_output")
    if not baseline_path:
        raise ValueError("native_default has no documents artifact")
    baseline = load_documents(baseline_path)
    report = {}
    for index, row in enumerate(rows):
        path = row.get("documents_output")
        if not path:
            raise ValueError(f"{row['arm']} has no documents artifact")
        comparison = _compare_documents(
            baseline,
            load_documents(path),
        )
        jaccard = comparison["markdown_jaccard"]
        structure = comparison["structure_exact"]
        report[f"{row['arm']}#{index + 1}"] = {
            "documents": len(jaccard),
            "jaccard_min": min(jaccard) if jaccard else None,
            "jaccard_mean": (
                statistics.mean(jaccard) if jaccard else None
            ),
            "jaccard_median": (
                statistics.median(jaccard) if jaccard else None
            ),
            "structure_exact": sum(bool(value) for value in structure),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造 Docling matrix report CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--expected-repeats", type=int, default=4)
    parser.add_argument("--expected-documents", type=int, default=368)
    parser.add_argument("--output")
    parser.add_argument("--skip-correctness", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """生成统计/correctness report。"""

    args = build_parser().parse_args(argv)
    rows = load_rows(args.results_jsonl)
    report = summarize(
        rows,
        expected_repeats=args.expected_repeats,
        expected_documents=args.expected_documents,
    )
    if not args.skip_correctness:
        report["correctness"] = correctness_report(rows)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
