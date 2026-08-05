"""聚合 Video A 三臂独立进程结果，并验证压缩 outputs artifact。"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ARM_ORDER = ("v3_parent_bound", "v3_elastic", "ray_data")
_WALL_KEYS = (
    "startup_inclusive_wall_s",
    "end_to_end_wall_s",
    "wall_s",
)
_PACKING_KEYS = (
    "rpc_count",
    "grains_per_rpc",
    "batch_fill_ratio",
    "transform_actors",
    "transform_batch_size",
    "transform_num_gpus",
    "decode_replicas",
    "model_repeats",
)


def load_rows(path: str) -> tuple[dict[str, Any], ...]:
    """读取共享 JSONL，拒绝空文件与非 object 行。"""

    rows = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Video A results JSONL is empty or invalid")
    return rows


def _result(row: dict[str, Any]) -> dict[str, Any]:
    """兼容新 runner 的 ``result`` 包装和早期平铺结果。"""

    value = row.get("result", row)
    if not isinstance(value, dict):
        raise ValueError("matrix result must contain an object result")
    return value


def _number(row: dict[str, Any], keys: tuple[str, ...]) -> float:
    """在 result/top-level 中取第一个存在的数值指标。"""

    result = _result(row)
    for key in keys:
        value = result.get(key, row.get(key))
        if value is not None:
            return float(value)
    raise ValueError(f"matrix result misses one of {keys}")


def _cv(values: list[float]) -> float:
    """返回样本 CV；单样本或零均值时按 0 处理。"""

    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _metrics(row: dict[str, Any]) -> dict[str, float]:
    """提取 V3/Ray Data 可比较的 packing 参数和运行时指标。"""

    result = _result(row)
    source = result.get("metrics", row.get("metrics", {}))
    if not isinstance(source, dict):
        return {}
    return {
        key: float(source[key])
        for key in _PACKING_KEYS
        if source.get(key) is not None
    }


def summarize(
    rows: tuple[dict[str, Any], ...],
    *,
    expected_repeats: int,
    expected_videos: int | None = None,
) -> dict[str, Any]:
    """校验三臂 repeat 矩阵，计算 wall median/CV、paired speedup 和 packing。"""

    if expected_repeats <= 0:
        raise ValueError("expected_repeats must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_repeat: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm = str(row.get("arm", ""))
        if arm not in ARM_ORDER:
            continue
        repeat = row.get("matrix_repeat")
        if repeat is None:
            raise ValueError(f"{arm} result has no matrix_repeat")
        repeat = int(repeat)
        if arm in by_repeat[repeat]:
            raise ValueError(f"duplicate result for repeat={repeat}, arm={arm}")
        grouped[arm].append(row)
        by_repeat[repeat][arm] = row

    incomplete = {
        arm: len(grouped.get(arm, ()))
        for arm in ARM_ORDER
        if len(grouped.get(arm, ())) != expected_repeats
    }
    if incomplete:
        raise ValueError(f"incomplete Video A matrix: {incomplete}")
    expected_ids = set(range(1, expected_repeats + 1))
    if set(by_repeat) != expected_ids:
        raise ValueError(f"unexpected repeat ids: {sorted(by_repeat)}")
    for repeat, values in by_repeat.items():
        missing = set(ARM_ORDER) - set(values)
        if missing:
            raise ValueError(f"repeat {repeat} is missing arms: {sorted(missing)}")

    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        arm_rows = sorted(grouped[arm], key=lambda row: int(row["matrix_repeat"]))
        walls = [_number(row, _WALL_KEYS) for row in arm_rows]
        videos = [
            int(
                _result(row).get(
                    "videos",
                    _result(row).get("documents", row.get("videos", 0)),
                )
            )
            for row in arm_rows
        ]
        frames = [
            int(_result(row).get("frames", row.get("frames", 0)))
            for row in arm_rows
        ]
        if expected_videos is not None and any(
            value != expected_videos for value in videos
        ):
            raise ValueError(f"{arm} has unexpected video count: {videos}")
        metric_values: dict[str, list[float]] = defaultdict(list)
        for row in arm_rows:
            for key, value in _metrics(row).items():
                metric_values[key].append(value)
        arms[arm] = {
            "runs": len(arm_rows),
            "startup_inclusive_wall_s": walls,
            "wall_median_s": statistics.median(walls),
            "wall_cv": _cv(walls),
            "videos": videos,
            "frames": frames,
            "packing_metrics": {
                key: {
                    "values": values,
                    "median": statistics.median(values),
                }
                for key, values in sorted(metric_values.items())
            },
        }

    paired = []
    for repeat in range(1, expected_repeats + 1):
        parent = _number(by_repeat[repeat]["v3_parent_bound"], _WALL_KEYS)
        elastic = _number(by_repeat[repeat]["v3_elastic"], _WALL_KEYS)
        if elastic <= 0:
            raise ValueError("v3_elastic wall time must be positive")
        paired.append(parent / elastic)
    return {
        "expected_repeats": expected_repeats,
        "expected_videos": expected_videos,
        "arms": arms,
        "speedups": {
            "paired_parent_over_elastic": paired,
            "paired_parent_over_elastic_median": statistics.median(paired),
            "parent_over_elastic_median_wall_ratio": (
                arms["v3_parent_bound"]["wall_median_s"]
                / arms["v3_elastic"]["wall_median_s"]
            ),
        },
    }


def _artifact_path(row: dict[str, Any]) -> str:
    """从不同 runner 命名中找到 correctness artifact 路径。"""

    result = _result(row)
    for key in (
        "outputs_output",
        "outputs_artifact",
        "correctness_output",
    ):
        value = row.get(key, result.get(key))
        if value:
            return str(value)
    raise ValueError(f"{row.get('arm')} has no outputs artifact")


def load_outputs(path: str) -> Any:
    """读取 gzip/plain JSON artifact，并返回其中稳定 outputs payload。"""

    artifact = Path(path)
    opener = gzip.open if artifact.suffix == ".gz" else open
    with opener(artifact, "rt", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if isinstance(value, dict) and "outputs" in value:
        return value["outputs"]
    return value


def _first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    """定位第一个 JSON 语义差异，避免在 report 中打印大型 outputs。"""

    if type(left) is not type(right):
        return {
            "path": path,
            "reason": "type",
            "left_type": type(left).__name__,
            "right_type": type(right).__name__,
        }
    if isinstance(left, dict):
        if set(left) != set(right):
            return {
                "path": path,
                "reason": "keys",
                "left_only": sorted(set(left) - set(right)),
                "right_only": sorted(set(right) - set(left)),
            }
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {
                "path": path,
                "reason": "length",
                "left_length": len(left),
                "right_length": len(right),
            }
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return {"path": path, "reason": "value"}
    return None


def _video_output_comparison(
    baseline: Any,
    candidate: Any,
) -> dict[str, Any]:
    """比较 Video A 结构，并量化 batch-shape 浮点导致的 digest 差异。

    ViT 在不同物理 batch shape 下可能产生末位浮点差异，进而改变少量 top-k
    embedding index signature。正式 gate 要求 video/frame/source order 和
    mean-edge 等业务结构完全一致，同时报告 digest mismatch，不要求 bitwise JSON。
    """

    if not isinstance(baseline, list) or not isinstance(candidate, list):
        return {
            "structure_exact": False,
            "reason": "outputs are not lists",
        }
    if len(baseline) != len(candidate):
        return {
            "structure_exact": False,
            "reason": "video count",
            "baseline_videos": len(baseline),
            "candidate_videos": len(candidate),
        }
    structure_fields = ("frames", "source_indices", "mean_edge_density")
    structure_mismatch_videos = 0
    digest_mismatch_videos = 0
    digest_mismatch_frames = 0
    total_frames = 0
    first_structure_difference = None
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        if any(left.get(field) != right.get(field) for field in structure_fields):
            structure_mismatch_videos += 1
            if first_structure_difference is None:
                first_structure_difference = {
                    "video_index": index,
                    "fields": [
                        field
                        for field in structure_fields
                        if left.get(field) != right.get(field)
                    ],
                }
        left_digests = list(left.get("digests", ()))
        right_digests = list(right.get("digests", ()))
        total_frames += max(len(left_digests), len(right_digests))
        mismatches = abs(len(left_digests) - len(right_digests))
        mismatches += sum(
            lhs != rhs for lhs, rhs in zip(left_digests, right_digests)
        )
        if mismatches:
            digest_mismatch_videos += 1
            digest_mismatch_frames += mismatches
    return {
        "structure_exact": structure_mismatch_videos == 0,
        "structure_mismatch_videos": structure_mismatch_videos,
        "first_structure_difference": first_structure_difference,
        "digest_mismatch_videos": digest_mismatch_videos,
        "digest_mismatch_frames": digest_mismatch_frames,
        "total_frames": total_frames,
        "digest_mismatch_rate": (
            digest_mismatch_frames / total_frames
            if total_frames
            else 0.0
        ),
    }


def correctness_report(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """按 repeat 验证结构一致，并报告 top-k digest mismatch 率。"""

    by_repeat: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm = str(row.get("arm", ""))
        if arm in ARM_ORDER and row.get("matrix_repeat") is not None:
            by_repeat[int(row["matrix_repeat"])][arm] = row
    reports: dict[str, Any] = {}
    all_structure_exact = True
    for repeat, values in sorted(by_repeat.items()):
        if set(values) != set(ARM_ORDER):
            continue
        baseline = load_outputs(_artifact_path(values["v3_parent_bound"]))
        comparisons = {}
        for arm in ("v3_elastic", "ray_data"):
            candidate = load_outputs(_artifact_path(values[arm]))
            comparison = _video_output_comparison(baseline, candidate)
            comparisons[arm] = comparison
            all_structure_exact = (
                all_structure_exact and comparison["structure_exact"]
            )
        reports[str(repeat)] = {
            "parent_vs": comparisons,
            "all_structure_exact": all(
                item["structure_exact"] for item in comparisons.values()
            ),
        }
    return {
        "all_structure_exact": all_structure_exact,
        "repeats": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造 Video A 聚合报告 CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--expected-repeats", type=int, default=4)
    parser.add_argument("--expected-videos", type=int)
    parser.add_argument("--output")
    parser.add_argument("--skip-correctness", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """写出小型聚合报告；artifact 内容不会直接打印。"""

    args = build_parser().parse_args(argv)
    rows = load_rows(args.results_jsonl)
    report = summarize(
        rows,
        expected_repeats=args.expected_repeats,
        expected_videos=args.expected_videos,
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
