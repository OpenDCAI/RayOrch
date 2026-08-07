"""Paired V3↔V3.6 gates for caption and audio/frame model workloads."""

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
from ..paired_stats import paired_timing_summary
from .caption_v36 import run_caption_v36
from .multimodal_v36 import run_multimodal_v36


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
    microbatch_size: int,
    max_active_microbatches: int,
    order: str,
    maximum_normalized_mismatch: float,
) -> dict[str, Any]:
    sequence = ("v3", "v36") if order == "v3_first" else ("v36", "v3")
    outputs: dict[str, list[Any]] = {}
    walls: dict[str, float] = {}
    rpcs: dict[str, int] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for engine in sequence:
        started = time.perf_counter()
        if engine == "v3":
            result = run_caption_v3(
                paths,
                microbatch_size=microbatch_size,
                max_inflight_arenas=max_active_microbatches,
                max_pending_per_actor=1,
                **options,
            )
            outputs[engine] = list(result.get())
            rpcs[engine] = int(result.metrics["rpc_count"])
            diagnostics[engine] = {
                key: value
                for key, value in result.metrics.items()
                if key
                in {
                    "startup_time_s",
                    "measured_wall_time_s",
                    "end_to_end_wall_time_s",
                }
            }
        else:
            result = run_caption_v36(
                paths,
                microbatch_size=microbatch_size,
                max_active_microbatches=max_active_microbatches,
                **options,
            )
            outputs[engine] = list(cast(Iterable[Any], result.outputs))
            rpcs[engine] = result.rpc_count
            diagnostics[engine] = {
                "runtime_elapsed_s": result.elapsed_s,
                "peak_active_microbatches": result.peak_active_microbatches,
                "released_values": result.released_values,
                "calls": {
                    f"call_{metric.call_index}": {
                        "udf_name": metric.udf_name,
                        "actor_instances": metric.actor_instances,
                        "rpcs": metric.rpcs,
                        "grains": metric.grains,
                        "average_batch": metric.average_batch,
                        "retries": metric.retries,
                    }
                    for metric in result.calls
                },
            }
        walls[engine] = time.perf_counter() - started

    difference = caption_difference_summary(outputs["v3"], outputs["v36"])
    if not difference.get("structure_exact"):
        raise ValueError(f"caption structure differs: {difference}")
    if difference["normalized_mismatch_rate"] > maximum_normalized_mismatch:
        raise ValueError(
            "caption normalized mismatch rate "
            f"{difference['normalized_mismatch_rate']:.6f} exceeds "
            f"{maximum_normalized_mismatch:.6f}: {difference}"
        )
    return {
        "order": order,
        "v3_wall_s": walls["v3"],
        "v36_wall_s": walls["v36"],
        "v3_rpc_count": rpcs["v3"],
        "v36_rpc_count": rpcs["v36"],
        "v3_diagnostics": diagnostics["v3"],
        "v36_diagnostics": diagnostics["v36"],
        "correctness": difference,
    }


def multimodal_difference_summary(
    baseline: list[Any],
    candidate: list[Any],
) -> dict[str, Any]:
    """Separate lineage/merge errors from deterministic model text drift."""

    if len(baseline) != len(candidate):
        return {
            "outputs_exact": False,
            "structure_exact": False,
            "reason": "video count",
            "baseline_videos": len(baseline),
            "candidate_videos": len(candidate),
        }

    def normalize(text: Any) -> str:
        return " ".join(str(text).casefold().split()).strip(" .,!?:;")

    structure_mismatch_videos = 0
    frame_digest_mismatch_videos = 0
    transcript_mismatch_videos = 0
    normalized_transcript_mismatch_videos = 0
    first_difference = None
    for video_index, (left, right) in enumerate(zip(baseline, candidate)):
        if not isinstance(left, dict) or not isinstance(right, dict):
            structure_mismatch_videos += 1
        elif (
            left.get("audio_chunks") != right.get("audio_chunks")
            or left.get("frames") != right.get("frames")
        ):
            structure_mismatch_videos += 1
        if (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.get("frame_digest") != right.get("frame_digest")
        ):
            frame_digest_mismatch_videos += 1
        left_text = left.get("transcript") if isinstance(left, dict) else None
        right_text = right.get("transcript") if isinstance(right, dict) else None
        if left_text != right_text:
            transcript_mismatch_videos += 1
            if normalize(left_text) != normalize(right_text):
                normalized_transcript_mismatch_videos += 1
        if left != right and first_difference is None:
            first_difference = {
                "video_index": video_index,
                "baseline": left,
                "candidate": right,
            }

    videos = len(baseline)
    return {
        "outputs_exact": baseline == candidate,
        "structure_exact": structure_mismatch_videos == 0,
        "structure_mismatch_videos": structure_mismatch_videos,
        "frame_digests_exact": frame_digest_mismatch_videos == 0,
        "frame_digest_mismatch_videos": frame_digest_mismatch_videos,
        "transcript_mismatch_videos": transcript_mismatch_videos,
        "normalized_transcript_mismatch_videos": (
            normalized_transcript_mismatch_videos
        ),
        "normalized_transcript_mismatch_rate": (
            normalized_transcript_mismatch_videos / videos if videos else 0.0
        ),
        "first_difference": first_difference,
    }


def _multimodal_trial(
    paths: list[str],
    options: dict[str, Any],
    *,
    microbatch_size: int,
    max_active_microbatches: int,
    order: str,
    maximum_normalized_mismatch: float = 0.02,
) -> dict[str, Any]:
    sequence = ("v3", "v36") if order == "v3_first" else ("v36", "v3")
    outputs: dict[str, list[Any]] = {}
    walls: dict[str, float] = {}
    rpcs: dict[str, int] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for engine in sequence:
        started = time.perf_counter()
        if engine == "v3":
            result = run_multimodal_v3(
                paths,
                microbatch_size=microbatch_size,
                max_inflight_arenas=max_active_microbatches,
                **options,
            )
            outputs[engine] = list(result.get())
            rpcs[engine] = int(result.metrics["rpc_count"])
            diagnostics[engine] = {
                key: value
                for key, value in result.metrics.items()
                if key
                in {
                    "startup_time_s",
                    "measured_wall_time_s",
                    "end_to_end_wall_time_s",
                }
            }
        else:
            result = run_multimodal_v36(
                paths,
                microbatch_size=microbatch_size,
                max_active_microbatches=max_active_microbatches,
                **options,
            )
            outputs[engine] = list(cast(Iterable[Any], result.outputs))
            rpcs[engine] = result.rpc_count
            diagnostics[engine] = {
                "runtime_elapsed_s": result.elapsed_s,
                "peak_active_microbatches": result.peak_active_microbatches,
                "released_values": result.released_values,
                "calls": {
                    f"call_{metric.call_index}": {
                        "udf_name": metric.udf_name,
                        "actor_instances": metric.actor_instances,
                        "rpcs": metric.rpcs,
                        "grains": metric.grains,
                        "average_batch": metric.average_batch,
                        "retries": metric.retries,
                    }
                    for metric in result.calls
                },
            }
        walls[engine] = time.perf_counter() - started
    difference = multimodal_difference_summary(outputs["v3"], outputs["v36"])
    if not difference.get("structure_exact"):
        raise ValueError(f"multimodal structure differs: {difference}")
    if not difference.get("frame_digests_exact"):
        raise ValueError(f"multimodal frame digests differ: {difference}")
    if (
        difference["normalized_transcript_mismatch_rate"]
        > maximum_normalized_mismatch
    ):
        raise ValueError(
            "multimodal normalized transcript mismatch rate "
            f"{difference['normalized_transcript_mismatch_rate']:.6f} exceeds "
            f"{maximum_normalized_mismatch:.6f}: {difference}"
        )
    return {
        "order": order,
        "v3_wall_s": walls["v3"],
        "v36_wall_s": walls["v36"],
        "v3_rpc_count": rpcs["v3"],
        "v36_rpc_count": rpcs["v36"],
        "v3_diagnostics": diagnostics["v3"],
        "v36_diagnostics": diagnostics["v36"],
        "correctness": difference,
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
            order = "v3_first" if index % 2 == 0 else "v36_first"
            if args.workload == "caption":
                trial = _caption_trial(
                    paths,
                    options,
                    microbatch_size=args.microbatch_size,
                    max_active_microbatches=args.max_active_microbatches,
                    order=order,
                    maximum_normalized_mismatch=args.maximum_normalized_mismatch,
                )
            else:
                trial = _multimodal_trial(
                    paths,
                    options,
                    microbatch_size=args.microbatch_size,
                    max_active_microbatches=args.max_active_microbatches,
                    order=order,
                    maximum_normalized_mismatch=(
                        args.maximum_normalized_mismatch
                    ),
                )
            trials.append(trial)
    finally:
        if started_ray_here:
            ray.shutdown()

    old = [trial["v3_wall_s"] for trial in trials]
    new = [trial["v36_wall_s"] for trial in trials]
    payload = {
        "schema_version": 1,
        "workload": args.workload,
        "manifest": str(Path(args.manifest).resolve()),
        "videos": len(paths),
        "options": options,
        "timing_scope": "startup_materialization_and_teardown_inclusive",
        "trials": trials,
        "v3_median_wall_s": statistics.median(old),
        "v36_median_wall_s": statistics.median(new),
        "v36_speedup_median": statistics.median(
            left / right for left, right in zip(old, new)
        ),
        "timing_statistics": paired_timing_summary(
            old,
            new,
            (str(trial["order"]) for trial in trials),
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
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--max-active-microbatches", type=int, default=2)
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
