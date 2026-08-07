"""Paired V3↔V3.5 video correctness and performance gate.

Both arms consume the same ordered manifest and unchanged workload UDFs.  A
trial alternates arm order, materializes the old V3 ``BlockSlice`` results via
its public ``get()`` API, and rejects any business-output difference before it
reports timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, cast

from ....multigrain_v3.benchmark.video.data import load_video_manifest
from ....multigrain_v3.benchmark.video.v3 import run_v3
from ....multigrain_v3.benchmark.video.workload import prepare_resnet18_weights
from ..paired_stats import paired_timing_summary
from .v35 import run_v35


def _digest(outputs: list[Any]) -> str:
    payload = json.dumps(
        outputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _first_difference(left: list[Any], right: list[Any]) -> str:
    if len(left) != len(right):
        return f"output length differs: V3={len(left)}, V3.5={len(right)}"
    for index, (old, new) in enumerate(zip(left, right)):
        if old != new:
            return f"first output difference at video index {index}: {old!r} != {new!r}"
    return "outputs differ but no differing row was found"


def _compare_outputs(left: list[Any], right: list[Any]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError(_first_difference(left, right))
    digest_mismatches = 0
    digest_count = 0
    for index, (old, new) in enumerate(zip(left, right)):
        old_structure = {key: value for key, value in old.items() if key != "digests"}
        new_structure = {key: value for key, value in new.items() if key != "digests"}
        if old_structure != new_structure:
            raise ValueError(
                f"first structural difference at video index {index}: "
                f"{old_structure!r} != {new_structure!r}"
            )
        old_digests = tuple(old.get("digests", ()))
        new_digests = tuple(new.get("digests", ()))
        if len(old_digests) != len(new_digests):
            raise ValueError(
                f"digest length differs at video index {index}: "
                f"{len(old_digests)} != {len(new_digests)}"
            )
        digest_count += len(old_digests)
        digest_mismatches += sum(
            old_digest != new_digest
            for old_digest, new_digest in zip(old_digests, new_digests)
        )
    return {
        "structure_exact": True,
        "digest_count": digest_count,
        "digest_mismatches": digest_mismatches,
        "digest_mismatch_rate": (
            digest_mismatches / digest_count if digest_count else 0.0
        ),
        "outputs_exact": left == right,
    }


def _run_trial(
    paths: list[str],
    pipeline_options: dict[str, Any],
    *,
    arena_size: int,
    max_in_flight: int,
    order: str,
    maximum_digest_mismatch: float = 0.0,
) -> dict[str, Any]:
    if order not in {"v3_first", "v35_first"}:
        raise ValueError(f"unknown trial order: {order}")

    sequence = ("v3", "v35") if order == "v3_first" else ("v35", "v3")
    outputs: dict[str, list[Any]] = {}
    walls: dict[str, float] = {}
    rpc_counts: dict[str, int] = {}
    grains_per_rpc: dict[str, float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for engine in sequence:
        started = time.perf_counter()
        if engine == "v3":
            result = run_v3(
                paths,
                microbatch_size=arena_size,
                max_inflight_arenas=max_in_flight,
                **pipeline_options,
            )
            # V3 deliberately returns detached BlockSlices; get() is the public
            # materialization boundary and must run before Ray is shut down.
            outputs[engine] = list(result.get())
            rpc_counts[engine] = int(result.metrics["rpc_count"])
            grains_per_rpc[engine] = float(result.metrics["grains_per_rpc"])
            diagnostics[engine] = {
                key: value
                for key, value in result.metrics.items()
                if key
                in {
                    "startup_time_s",
                    "measured_wall_time_s",
                    "end_to_end_wall_time_s",
                    "active_arenas_high_watermark",
                    "live_blocks_across_arenas_high_watermark",
                }
            }
        else:
            result = run_v35(
                paths,
                arena_size=arena_size,
                max_in_flight=max_in_flight,
                **pipeline_options,
            )
            outputs[engine] = list(cast(Iterable[Any], result.outputs))
            rpc_counts[engine] = result.rpc_count
            grains = sum(metric.grains for metric in result.calls.values())
            grains_per_rpc[engine] = grains / result.rpc_count if result.rpc_count else 0.0
            worker_observations = [
                observation
                for observations in result.workers.values()
                for observation in observations
            ]
            diagnostics[engine] = {
                "runtime_elapsed_s": result.elapsed_s,
                "max_active_arenas": result.max_active_arenas,
                "released_values": result.released_values,
                "worker_observation_errors": [
                    observation.error
                    for observation in worker_observations
                    if observation.error is not None
                ],
                "worker_rss_bytes_max": max(
                    (observation.rss_bytes for observation in worker_observations),
                    default=0,
                ),
                "calls": {
                    f"call_{call.value}": {
                        "actor_starts": metric.actor_starts,
                        "rpcs": metric.rpcs,
                        "grains": metric.grains,
                        "average_batch": metric.average_batch,
                        "batch_histogram": dict(
                            sorted(Counter(metric.batch_sizes).items())
                        ),
                        "retries": metric.retries,
                    }
                    for call, metric in sorted(result.calls.items())
                },
            }
        walls[engine] = time.perf_counter() - started

    correctness = _compare_outputs(outputs["v3"], outputs["v35"])
    if correctness["digest_mismatch_rate"] > maximum_digest_mismatch:
        raise ValueError(
            "frame digest mismatch rate "
            f"{correctness['digest_mismatch_rate']:.6f} exceeds "
            f"{maximum_digest_mismatch:.6f}"
        )
    sampled_frames = [int(output["frames"]) for output in outputs["v3"]]
    trial = {
        "order": order,
        "v3_wall_s": walls["v3"],
        "v35_wall_s": walls["v35"],
        "v3_rpc_count": rpc_counts["v3"],
        "v35_rpc_count": rpc_counts["v35"],
        "v3_grains_per_rpc": grains_per_rpc["v3"],
        "v35_grains_per_rpc": grains_per_rpc["v35"],
        "v3_output_digest": _digest(outputs["v3"]),
        "v35_output_digest": _digest(outputs["v35"]),
        "correctness": correctness,
        "v3_diagnostics": diagnostics["v3"],
        "v35_diagnostics": diagnostics["v35"],
        "sampled_frame_count": sum(sampled_frames),
        "sampled_frame_histogram": dict(
            sorted(Counter(sampled_frames).items())
        ),
    }
    if len(sampled_frames) <= 256:
        trial["sampled_frames"] = sampled_frames
    return trial


def run_paired(args: argparse.Namespace) -> dict[str, Any]:
    """Run warmups and alternating paired trials in one owned Ray session."""

    import ray  # pyright: ignore[reportMissingImports]

    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if args.num_gpus < 0 or args.transform_num_gpus < 0:
        raise ValueError("GPU counts must be non-negative")
    if not 0.0 <= args.maximum_digest_mismatch <= 1.0:
        raise ValueError("maximum digest mismatch must be between zero and one")
    if args.transform_num_gpus * args.transform_replicas > args.num_gpus:
        raise ValueError("transform actors request more GPUs than Ray exposes")

    entries = load_video_manifest(
        args.manifest,
        limit=args.limit or None,
    )
    paths = [entry.path for entry in entries]
    if not paths:
        raise ValueError("selected video manifest is empty")
    if args.transform_backend == "resnet18":
        prepare_resnet18_weights()

    pipeline_options = {
        "stride": args.stride,
        "max_frames": args.max_frames,
        "decode_replicas": args.decode_replicas,
        "reduce_replicas": args.reduce_replicas,
        "transform_replicas": args.transform_replicas,
        "transform_batch_size": args.transform_batch_size,
        "batch_scope": args.batch_scope,
        "transform_backend": args.transform_backend,
        "torch_num_threads": args.torch_num_threads,
        "model_path": args.model_path,
        "transform_num_gpus": args.transform_num_gpus,
        "model_repeats": args.model_repeats,
    }

    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        init_options: dict[str, Any] = {
            "address": "local",
            "num_cpus": args.num_cpus,
            "num_gpus": args.num_gpus,
            "include_dashboard": False,
        }
        if args.object_store_gb is not None:
            init_options["object_store_memory"] = int(
                args.object_store_gb * 1024**3
            )
        ray.init(**init_options)

    try:
        all_trials = [
            _run_trial(
                paths,
                pipeline_options,
                arena_size=args.arena_size,
                max_in_flight=args.max_in_flight,
                order="v3_first" if index % 2 == 0 else "v35_first",
                maximum_digest_mismatch=(
                    0.0
                    if args.transform_backend == "opencv"
                    else args.maximum_digest_mismatch
                ),
            )
            for index in range(args.warmup + args.repeats)
        ]
    finally:
        if started_ray_here:
            ray.shutdown()

    trials = all_trials[args.warmup :]
    v3_walls = [float(trial["v3_wall_s"]) for trial in trials]
    v35_walls = [float(trial["v35_wall_s"]) for trial in trials]
    payload = {
        "schema_version": 1,
        "workload": "video_frame_feature",
        "manifest": str(Path(args.manifest).resolve()),
        "datasets": sorted({entry.dataset for entry in entries}),
        "videos": len(entries),
        "unique_source_ids": len(
            {(entry.dataset, entry.split, entry.source_id) for entry in entries}
        ),
        "input_bytes": sum(entry.bytes or 0 for entry in entries),
        "input_duration_s": sum(entry.duration_s or 0.0 for entry in entries),
        "pipeline_options": pipeline_options,
        "arena_size": args.arena_size,
        "max_in_flight": args.max_in_flight,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_scope": (
            "actor_startup_materialization_and_executor_teardown_inclusive;"
            "ray_cluster_startup_and_shutdown_excluded"
        ),
        "trial_order": [trial["order"] for trial in trials],
        "v3_wall_s": [round(value, 6) for value in v3_walls],
        "v35_wall_s": [round(value, 6) for value in v35_walls],
        "v3_median_wall_s": round(statistics.median(v3_walls), 6),
        "v35_median_wall_s": round(statistics.median(v35_walls), 6),
        "v35_speedup_median": round(
            statistics.median(
                old / new for old, new in zip(v3_walls, v35_walls)
            ),
            6,
        ),
        "timing_statistics": paired_timing_summary(
            v3_walls,
            v35_walls,
            (str(trial["order"]) for trial in trials),
        ),
        "outputs_equivalent": True,
        "outputs_exact": all(
            trial["correctness"]["outputs_exact"] for trial in trials
        ),
        "maximum_digest_mismatch": (
            0.0
            if args.transform_backend == "opencv"
            else args.maximum_digest_mismatch
        ),
        "v3_output_digest": trials[-1]["v3_output_digest"],
        "v35_output_digest": trials[-1]["v35_output_digest"],
        "trials": trials,
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--decode-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=2)
    parser.add_argument("--transform-replicas", type=int, default=4)
    parser.add_argument("--transform-batch-size", type=int, default=64)
    parser.add_argument(
        "--batch-scope",
        choices=("elastic", "parent_bound"),
        default="elastic",
    )
    parser.add_argument(
        "--transform-backend",
        choices=("opencv", "resnet18", "vit"),
        default="opencv",
    )
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--model-path")
    parser.add_argument("--transform-num-gpus", type=float, default=0.0)
    parser.add_argument("--model-repeats", type=int, default=1)
    parser.add_argument(
        "--maximum-digest-mismatch",
        type=float,
        default=0.0001,
        help="maximum model top-k digest mismatch rate; OpenCV always requires zero",
    )
    parser.add_argument("--arena-size", type=int, default=24)
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--object-store-gb", type=float)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_paired(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["build_parser", "run_paired"]
