"""Paired V3↔V3.5 gates for caption and audio/frame model workloads."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, cast

from ....multigrain_v3.benchmark.video.caption_matrix import (
    caption_difference_summary,
)
from ....multigrain_v3.benchmark.video.caption_v3 import run_v3 as run_caption_v3
from ....multigrain_v3.benchmark.video.multimodal_v3 import (
    run_v3 as run_multimodal_v3,
)
from .caption_v35 import run_caption_v35
from .multimodal_v35 import run_multimodal_v35


def _read_paths(manifest: str, limit: int) -> list[str]:
    raw = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("manifest must be a JSON list")
    rows = raw[:limit] if limit else raw
    paths = [
        str(item["path"] if isinstance(item, dict) else item)
        for item in rows
    ]
    if not paths or any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("manifest contains no inputs or missing videos")
    return paths


def _caption_trial(
    paths: list[str],
    options: dict[str, Any],
    *,
    arena_size: int,
    max_in_flight: int,
    order: str,
    maximum_normalized_mismatch: float,
) -> dict[str, Any]:
    sequence = ("v3", "v35") if order == "v3_first" else ("v35", "v3")
    outputs: dict[str, list[Any]] = {}
    walls: dict[str, float] = {}
    rpcs: dict[str, int] = {}
    for engine in sequence:
        started = time.perf_counter()
        if engine == "v3":
            result = run_caption_v3(
                paths,
                microbatch_size=arena_size,
                max_inflight_arenas=max_in_flight,
                max_pending_per_actor=1,
                **options,
            )
            outputs[engine] = list(result.get())
            rpcs[engine] = int(result.metrics["rpc_count"])
        else:
            result = run_caption_v35(
                paths,
                arena_size=arena_size,
                max_in_flight=max_in_flight,
                **options,
            )
            outputs[engine] = list(cast(Iterable[Any], result.outputs))
            rpcs[engine] = result.rpc_count
        walls[engine] = time.perf_counter() - started

    difference = caption_difference_summary(outputs["v3"], outputs["v35"])
    if not difference.get("structure_exact"):
        raise ValueError(f"caption structure differs: {difference}")
    if difference["normalized_mismatch_rate"] > maximum_normalized_mismatch:
        raise ValueError(
            "caption normalized mismatch rate "
            f"{difference['normalized_mismatch_rate']:.6f} exceeds "
            f"{maximum_normalized_mismatch:.6f}"
        )
    return {
        "order": order,
        "v3_wall_s": walls["v3"],
        "v35_wall_s": walls["v35"],
        "v3_rpc_count": rpcs["v3"],
        "v35_rpc_count": rpcs["v35"],
        "correctness": difference,
    }


def _multimodal_trial(
    paths: list[str],
    options: dict[str, Any],
    *,
    arena_size: int,
    max_in_flight: int,
    order: str,
) -> dict[str, Any]:
    sequence = ("v3", "v35") if order == "v3_first" else ("v35", "v3")
    outputs: dict[str, list[Any]] = {}
    walls: dict[str, float] = {}
    rpcs: dict[str, int] = {}
    for engine in sequence:
        started = time.perf_counter()
        if engine == "v3":
            result = run_multimodal_v3(
                paths,
                microbatch_size=arena_size,
                max_inflight_arenas=max_in_flight,
                **options,
            )
            outputs[engine] = list(result.get())
            rpcs[engine] = int(result.metrics["rpc_count"])
        else:
            result = run_multimodal_v35(
                paths,
                arena_size=arena_size,
                max_in_flight=max_in_flight,
                **options,
            )
            outputs[engine] = list(cast(Iterable[Any], result.outputs))
            rpcs[engine] = result.rpc_count
        walls[engine] = time.perf_counter() - started
    if outputs["v3"] != outputs["v35"]:
        raise ValueError("multimodal V3 and V3.5 outputs differ")
    return {
        "order": order,
        "v3_wall_s": walls["v3"],
        "v35_wall_s": walls["v35"],
        "v3_rpc_count": rpcs["v3"],
        "v35_rpc_count": rpcs["v35"],
        "outputs_exact": True,
    }


def run_paired(args: argparse.Namespace) -> dict[str, Any]:
    import ray  # pyright: ignore[reportMissingImports]

    if args.limit < 0 or args.repeats <= 0:
        raise ValueError("limit must be non-negative and repeats positive")
    paths = _read_paths(args.manifest, args.limit)
    if args.workload == "caption":
        options = {
            "model_path": args.model_path,
            "stride": args.stride,
            "max_frames": args.max_frames,
            "batch_scope": args.batch_scope,
            "caption_batch_size": args.batch_size,
            "caption_replicas": args.model_replicas,
            "decode_replicas": args.decode_replicas,
            "reduce_replicas": args.reduce_replicas,
            "max_new_tokens": args.max_new_tokens,
        }
    else:
        options = {
            "whisper_model_path": args.whisper_model_path,
            "vit_model_path": args.vit_model_path,
            "batch_scope": args.batch_scope,
        }

    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        ray.init(
            address="local",
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
            include_dashboard=False,
        )
    try:
        trials = []
        for index in range(args.repeats):
            order = "v3_first" if index % 2 == 0 else "v35_first"
            if args.workload == "caption":
                trial = _caption_trial(
                    paths,
                    options,
                    arena_size=args.arena_size,
                    max_in_flight=args.max_in_flight,
                    order=order,
                    maximum_normalized_mismatch=args.maximum_normalized_mismatch,
                )
            else:
                trial = _multimodal_trial(
                    paths,
                    options,
                    arena_size=args.arena_size,
                    max_in_flight=args.max_in_flight,
                    order=order,
                )
            trials.append(trial)
    finally:
        if started_ray_here:
            ray.shutdown()

    old = [trial["v3_wall_s"] for trial in trials]
    new = [trial["v35_wall_s"] for trial in trials]
    payload = {
        "schema_version": 1,
        "workload": args.workload,
        "manifest": str(Path(args.manifest).resolve()),
        "videos": len(paths),
        "options": options,
        "timing_scope": "startup_materialization_and_teardown_inclusive",
        "trials": trials,
        "v3_median_wall_s": statistics.median(old),
        "v35_median_wall_s": statistics.median(new),
        "v35_speedup_median": statistics.median(
            left / right for left, right in zip(old, new)
        ),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=("caption", "multimodal"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arena-size", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int, default=2)
    parser.add_argument("--num-cpus", type=int, default=16)
    parser.add_argument("--num-gpus", type=float, default=4)
    parser.add_argument(
        "--batch-scope",
        choices=("elastic", "parent_bound"),
        default="elastic",
    )
    parser.add_argument("--model-path")
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-replicas", type=int, default=4)
    parser.add_argument("--decode-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--maximum-normalized-mismatch", type=float, default=0.02)
    parser.add_argument("--whisper-model-path")
    parser.add_argument("--vit-model-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workload == "caption" and not args.model_path:
        raise ValueError("caption requires --model-path")
    if args.workload == "multimodal" and (
        not args.whisper_model_path or not args.vit_model_path
    ):
        raise ValueError("multimodal requires both model paths")
    print(json.dumps(run_paired(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["build_parser", "run_paired"]
