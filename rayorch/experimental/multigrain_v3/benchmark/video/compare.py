"""V3 与裸 Ray Data 视频 workload 的手工对比 CLI。"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from .data import (
    download_tiny_video,
    download_ucf101_samples,
    probe_video,
)
from .ray_data import run_ray_data
from .v3 import run_v3
from .workload import prepare_resnet18_weights


def _paths(args: argparse.Namespace) -> list[str]:
    """根据 CLI source 选择本地或 Hugging Face videos。"""

    if args.paths:
        base = [str(Path(path).resolve()) for path in args.paths]
    else:
        cache = args.cache_dir or tempfile.mkdtemp(
            prefix="multigrain-v3-video-",
            dir="/tmp",
        )
        if args.dataset == "tiny":
            base = [download_tiny_video(cache)]
        else:
            base = list(download_ucf101_samples(cache))
    if args.repeat_inputs <= 0:
        raise ValueError("repeat_inputs must be positive")
    return base * args.repeat_inputs


def _run_once(
    args: argparse.Namespace,
    paths: list[str],
    common: dict[str, Any],
    *,
    order: str,
) -> dict[str, Any]:
    """运行一次 V3/Ray Data paired trial 并验证 semantic parity。"""

    if order not in {"v3_first", "ray_data_first"}:
        raise ValueError(f"unknown paired order: {order}")
    results: dict[str, Any] = {}
    sequence = (
        ("v3", "ray_data")
        if order == "v3_first"
        else ("ray_data", "v3")
    )
    v3_result = None
    for engine in sequence:
        started = time.perf_counter()
        if engine == "v3":
            v3_result = run_v3(
                paths,
                **common,
                batch_scope=args.batch_scope,
                microbatch_size=args.microbatch_size,
                max_inflight_arenas=args.max_inflight_arenas,
            )
            results["v3_outputs"] = v3_result.get()
        else:
            results["ray_data_outputs"] = run_ray_data(paths, **common)
        results[f"{engine}_wall"] = time.perf_counter() - started
    assert v3_result is not None
    v3_outputs = results["v3_outputs"]
    ray_data_outputs = results["ray_data_outputs"]
    if v3_outputs != ray_data_outputs:
        raise ValueError("V3 and Ray Data video outputs differ")
    return {
        "order": order,
        "v3_wall_s": results["v3_wall"],
        "ray_data_wall_s": results["ray_data_wall"],
        "sampled_frames": [
            output["frames"] for output in v3_outputs
        ],
        "v3_rpc_count": v3_result.metrics["rpc_count"],
        "v3_grains_per_rpc": v3_result.metrics["grains_per_rpc"],
        "v3_batch_fill_ratio": v3_result.metrics["batch_fill_ratio"],
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    """在同一 Ray session 中运行 warmup 与多次 paired trials。"""

    import ray

    paths = _paths(args)
    metadata = [probe_video(path) for path in paths]
    if args.transform_backend == "resnet18":
        prepare_resnet18_weights()
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=args.num_cpus,
            include_dashboard=False,
        )
    common = {
        "stride": args.stride,
        "max_frames": args.max_frames,
        "decode_replicas": args.decode_replicas,
        "transform_replicas": args.transform_replicas,
        "transform_batch_size": args.transform_batch_size,
        "transform_backend": args.transform_backend,
        "torch_num_threads": args.torch_num_threads,
    }
    warmups = [
        _run_once(
            args,
            paths,
            common,
            order=(
                "v3_first"
                if index % 2 == 0
                else "ray_data_first"
            ),
        )
        for index in range(args.warmup)
    ]
    trials = [
        _run_once(
            args,
            paths,
            common,
            order=(
                "v3_first"
                if (args.warmup + index) % 2 == 0
                else "ray_data_first"
            ),
        )
        for index in range(args.repeats)
    ]
    v3_walls = [trial["v3_wall_s"] for trial in trials]
    ray_data_walls = [trial["ray_data_wall_s"] for trial in trials]
    representative = trials[len(trials) // 2]
    return {
        "dataset": args.dataset,
        "transform_backend": args.transform_backend,
        "videos": len(paths),
        "unique_videos": len(set(paths)),
        "repeat_inputs": args.repeat_inputs,
        "metadata": metadata,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sampled_frames": representative["sampled_frames"],
        "v3_wall_s": [round(value, 6) for value in v3_walls],
        "ray_data_wall_s": [
            round(value, 6) for value in ray_data_walls
        ],
        "v3_median_wall_s": round(statistics.median(v3_walls), 6),
        "ray_data_median_wall_s": round(
            statistics.median(ray_data_walls),
            6,
        ),
        "paired_speedup_median": round(
            statistics.median(
                ray_data / v3
                for v3, ray_data in zip(v3_walls, ray_data_walls)
            ),
            6,
        ),
        "trial_order": [trial["order"] for trial in trials],
        "timing_scope": "startup_inclusive",
        "v3_rpc_count": representative["v3_rpc_count"],
        "v3_grains_per_rpc": representative["v3_grains_per_rpc"],
        "v3_batch_fill_ratio": representative["v3_batch_fill_ratio"],
        "warmup_trials": len(warmups),
        "outputs_match": True,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造视频 compare CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tiny", "ucf101"), default="tiny")
    parser.add_argument("--paths", nargs="*")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--repeat-inputs",
        type=int,
        default=1,
        help="repeat the small public sample set as independent logical videos",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--decode-replicas", type=int, default=2)
    parser.add_argument("--transform-replicas", type=int, default=4)
    parser.add_argument("--transform-batch-size", type=int, default=16)
    parser.add_argument(
        "--transform-backend",
        choices=("opencv", "resnet18"),
        default="opencv",
    )
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument(
        "--batch-scope",
        choices=("elastic", "parent_bound"),
        default="elastic",
    )
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--max-inflight-arenas", type=int, default=2)
    parser.add_argument("--num-cpus", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行 compare 并打印 JSON。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_compare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
