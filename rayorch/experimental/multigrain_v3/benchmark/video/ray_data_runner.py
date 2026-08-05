"""Video A 正式 Ray Data 独立 arm runner。

该 runner 只运行 Ray Data 基线，刻意不复用 V3 进程中的 Ray runtime。计时从创建
独立 Ray session 前开始，到 ``take_all`` 收到所有 video summary 为止，因此是没有
ready barrier 的 startup-inclusive wall time。
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ..document_docling.gpu_monitor import GpuMonitor, GpuSample
from .data import load_video_manifest
from .ray_data import run_ray_data


ARM_NAME = "ray_data"
FORMAL_TRANSFORM_REPLICAS = 4
FORMAL_TRANSFORM_BATCH_SIZE = 16
FORMAL_TRANSFORM_NUM_GPUS = 1.0


def summarize_gpu_samples(
    samples: tuple[GpuSample, ...],
    *,
    monitor: GpuMonitor,
) -> dict[str, Any]:
    """把 NVML 原始采样压缩为四卡可写入 summary 的统计值。"""

    devices: dict[int, list[Any]] = {
        index: [] for index in monitor.device_indices
    }
    for sample in samples:
        for device in sample.devices:
            devices.setdefault(device.index, []).append(device)

    summaries = []
    for index in monitor.device_indices:
        values = devices.get(index, [])
        utilizations = [
            value.utilization_percent
            for value in values
            if value.utilization_percent is not None
        ]
        memories = [
            value.memory_used_bytes
            for value in values
            if value.memory_used_bytes is not None
        ]
        powers = [
            value.power_watts
            for value in values
            if value.power_watts is not None
        ]
        summaries.append(
            {
                "index": index,
                "samples": len(values),
                "utilization_mean_percent": (
                    round(statistics.mean(utilizations), 4)
                    if utilizations
                    else None
                ),
                "utilization_max_percent": max(utilizations, default=None),
                "memory_max_bytes": max(memories, default=None),
                "power_mean_watts": (
                    round(statistics.mean(powers), 4) if powers else None
                ),
                "power_max_watts": max(powers, default=None),
            }
        )
    return {
        "available": bool(monitor.available),
        "unavailable_reason": monitor.unavailable_reason,
        "sample_count": len(samples),
        "devices": summaries,
    }


def _write_gzip_json(path: Path, value: Any) -> None:
    """原子性要求由调用方目录保证；这里写紧凑 correctness artifact。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """追加一条独立 arm 结果，不重写其他 worker 已写入的内容。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        output.write("\n")


def _validate_formal_arguments(args: argparse.Namespace) -> None:
    """拒绝会破坏 Video A 四卡可比性的 Ray Data 配置。"""

    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if args.decode_replicas <= 0:
        raise ValueError("decode_replicas must be positive")
    if args.transform_replicas != FORMAL_TRANSFORM_REPLICAS:
        raise ValueError("formal Video A requires exactly 4 transform actors")
    if args.transform_batch_size != FORMAL_TRANSFORM_BATCH_SIZE:
        raise ValueError("formal Video A requires transform batch size 16")
    if args.transform_num_gpus != FORMAL_TRANSFORM_NUM_GPUS:
        raise ValueError("formal Video A requires 1 GPU per transform actor")
    if args.ray_num_gpus < FORMAL_TRANSFORM_REPLICAS:
        raise ValueError("ray-num-gpus must expose at least four GPUs")
    if args.model_repeats <= 0:
        raise ValueError("model-repeats must be positive")
    if args.transform_backend == "vit" and not args.model_path:
        raise ValueError("ViT backend requires --model-path")


def run_formal_ray_data(args: argparse.Namespace) -> dict[str, Any]:
    """读取正式 manifest，在新 Ray session 中执行一次 Ray Data arm。"""

    _validate_formal_arguments(args)
    import ray

    if ray.is_initialized():
        raise RuntimeError(
            "ray_data_runner requires an independent Ray session; "
            "run it from a fresh Python process"
        )

    entries = load_video_manifest(args.manifest)
    paths = [entry.path for entry in entries]
    output_path = Path(args.output).resolve()
    artifact_path = (
        Path(args.outputs_output).resolve()
        if args.outputs_output
        else output_path.with_name("outputs.json.gz")
    )
    monitor_path = (
        Path(args.gpu_monitor_output).resolve()
        if args.gpu_monitor_output
        else None
    )
    monitor = GpuMonitor(
        interval_s=args.gpu_monitor_interval_s,
        device_indices=(0, 1, 2, 3),
        jsonl_path=monitor_path,
    )
    started = time.perf_counter()
    monitor.start()
    samples: tuple[GpuSample, ...] = ()
    try:
        ray.init(
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus,
            include_dashboard=False,
            ignore_reinit_error=False,
        )
        outputs = run_ray_data(
            paths,
            stride=args.stride,
            max_frames=args.max_frames,
            decode_replicas=args.decode_replicas,
            transform_replicas=args.transform_replicas,
            transform_batch_size=args.transform_batch_size,
            transform_backend=args.transform_backend,
            torch_num_threads=args.torch_num_threads,
            model_path=args.model_path,
            transform_num_gpus=args.transform_num_gpus,
            model_repeats=args.model_repeats,
        )
    finally:
        samples = monitor.stop()
        if ray.is_initialized():
            ray.shutdown()
    wall_s = time.perf_counter() - started

    artifact = {
        "schema_version": 1,
        "kind": "video_outputs",
        "manifest": str(Path(args.manifest).resolve()),
        "sources": [
            {
                "dataset": entry.dataset,
                "split": entry.split,
                "source_id": entry.source_id,
            }
            for entry in entries
        ],
        "outputs": outputs,
    }
    _write_gzip_json(artifact_path, artifact)
    frames = sum(int(output["frames"]) for output in outputs)
    gpu_summary = summarize_gpu_samples(samples, monitor=monitor)
    result = {
        "schema_version": 1,
        "arm": ARM_NAME,
        "matrix_repeat": args.matrix_repeat,
        "matrix_position": args.matrix_position,
        "manifest": str(Path(args.manifest).resolve()),
        "outputs_output": str(artifact_path),
        "result": {
            "timing_scope": "startup_inclusive_no_ready_barrier",
            "startup_inclusive_wall_s": round(wall_s, 6),
            "documents": len(outputs),
            "videos": len(outputs),
            "frames": frames,
            "frames_per_s": round(frames / wall_s, 6) if wall_s else None,
            "gpu": gpu_summary,
            "metrics": {
                "transform_actors": args.transform_replicas,
                "transform_batch_size": args.transform_batch_size,
                "transform_num_gpus": args.transform_num_gpus,
                "decode_replicas": args.decode_replicas,
                "model_repeats": args.model_repeats,
            },
        },
        "config": {
            "stride": args.stride,
            "max_frames": args.max_frames,
            "decode_replicas": args.decode_replicas,
            "transform_replicas": args.transform_replicas,
            "transform_batch_size": args.transform_batch_size,
            "transform_backend": args.transform_backend,
            "model_path": args.model_path,
            "model_repeats": args.model_repeats,
            "transform_num_gpus": args.transform_num_gpus,
            "ray_num_cpus": args.ray_num_cpus,
            "ray_num_gpus": args.ray_num_gpus,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.append_jsonl:
        _append_jsonl(Path(args.append_jsonl).resolve(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """构造 Video A Ray Data 独立 arm CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--outputs-output")
    parser.add_argument("--append-jsonl")
    parser.add_argument("--matrix-repeat", type=int)
    parser.add_argument("--matrix-position", type=int)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--decode-replicas", type=int, default=2)
    parser.add_argument("--transform-replicas", type=int, default=4)
    parser.add_argument("--transform-batch-size", type=int, default=16)
    parser.add_argument(
        "--transform-backend",
        choices=("opencv", "resnet18", "vit"),
        default="vit",
    )
    parser.add_argument("--model-path")
    parser.add_argument("--model-repeats", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--transform-num-gpus", type=float, default=1.0)
    parser.add_argument("--ray-num-cpus", type=int, default=32)
    parser.add_argument("--ray-num-gpus", type=float, default=4.0)
    parser.add_argument("--gpu-monitor-output")
    parser.add_argument("--gpu-monitor-interval-s", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行 arm，仅打印小型路径/摘要，不打印 correctness outputs。"""

    result = run_formal_ray_data(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "arm": result["arm"],
                "matrix_repeat": result["matrix_repeat"],
                "summary": result["result"],
                "outputs_output": result["outputs_output"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
