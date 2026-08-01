"""聚合 MinerU matrix JSONL，并校验每个 engine/repeat 是否完整。"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _engine_name(row: dict[str, Any]) -> str:
    """把各 runner 的 payload 规范化为 matrix engine 名。"""

    engine = row.get("engine")
    if engine == "multigrain_v3":
        return f"v3_{row['mode']}"
    if engine == "ray_data":
        return "ray_data"
    if engine == "flash_mineru_native_dag":
        return "native"
    raise ValueError(f"unknown MinerU engine payload: {engine!r}")


def load_rows(path: str) -> list[dict[str, Any]]:
    """读取非空 JSONL，并附加同 engine 的顺序 repeat 编号。"""

    raw = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counters: dict[str, int] = defaultdict(int)
    rows = []
    for row in raw:
        engine = _engine_name(row)
        counters[engine] += 1
        rows.append(
            {
                **row,
                "matrix_engine": engine,
                "matrix_repeat": counters[engine],
            }
        )
    return rows


def summarize(
    rows: list[dict[str, Any]],
    *,
    expected_repeats: int,
    expected_pdfs: int,
) -> dict[str, Any]:
    """校验完整性并计算每个 engine 的 wall/throughput 中位数。"""

    expected = (
        "v3_elastic",
        "v3_parent_bound",
        "ray_data",
        "native",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["matrix_engine"]].append(row)
    missing = {
        engine: expected_repeats - len(grouped.get(engine, ()))
        for engine in expected
        if len(grouped.get(engine, ())) != expected_repeats
    }
    if missing:
        raise ValueError(f"incomplete MinerU matrix: {missing}")
    for engine, engine_rows in grouped.items():
        if any(int(row["n_pdf"]) != expected_pdfs for row in engine_rows):
            raise ValueError(f"{engine} has inconsistent PDF count")
        if any(
            "docs" in row and int(row["docs"]) != expected_pdfs
            for row in engine_rows
        ):
            raise ValueError(f"{engine} has incomplete outputs")
    summary = {}
    for engine in expected:
        engine_rows = grouped[engine]
        walls = [
            float(row.get("end_to_end_wall_s", row["measured_wall_s"]))
            for row in engine_rows
        ]
        measured_walls = [
            float(row["measured_wall_s"]) for row in engine_rows
        ]
        throughput = [
            float(row["pages_per_s"])
            for row in engine_rows
            if row.get("pages_per_s") is not None
        ]
        summary[engine] = {
            "runs": len(engine_rows),
            "wall_s": walls,
            "median_wall_s": statistics.median(walls),
            "timing_scope": "end_to_end_startup_inclusive",
            "measured_wall_s": measured_walls,
            "median_measured_wall_s": statistics.median(measured_walls),
            "pages_per_s": throughput,
            "median_pages_per_s": (
                statistics.median(throughput)
                if throughput
                else None
            ),
        }
    elastic = summary["v3_elastic"]["median_wall_s"]
    parent = summary["v3_parent_bound"]["median_wall_s"]
    ray_data = summary["ray_data"]["median_wall_s"]
    native = summary["native"]["median_wall_s"]
    return {
        "expected_pdfs": expected_pdfs,
        "expected_repeats": expected_repeats,
        "engines": summary,
        "speedups": {
            "elastic_vs_parent": parent / elastic,
            "elastic_vs_ray_data": ray_data / elastic,
            "elastic_vs_native": native / elastic,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """构造 matrix report CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--expected-pdfs", type=int, default=368)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    """读取、校验、聚合并打印或写入 JSON report。"""

    args = build_parser().parse_args(argv)
    report = summarize(
        load_rows(args.results_jsonl),
        expected_repeats=args.expected_repeats,
        expected_pdfs=args.expected_pdfs,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
