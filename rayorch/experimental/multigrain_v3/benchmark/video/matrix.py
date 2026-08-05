"""Video A 正式 V3 parent/elastic 独立 arm runner。

每次 CLI 调用只运行一个 V3 arm，并独立创建和关闭本地 Ray session。两个 arm 共享
完全相同的输入、GPU、batch cap、microbatch 和模型配置，唯一差异是 transform stage 的
``batch_scope``。完整 video outputs 只写入 gzip artifact，stdout 和 summary JSON 仅保留
可审计的紧凑摘要。
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping

from .data import VideoManifestEntry, load_video_manifest
from .v3 import VideoV3Pipeline, run_v3


ARM_ORDER = ("v3_parent_bound", "v3_elastic")


@dataclass(frozen=True, slots=True)
class VideoMatrixConfig:
    """Video A 两臂共享的 V3 编译和执行合同。

    默认值是正式四卡配置：四个 transform actor，每个 actor 独占一张 GPU。decode 和
    reduce replicas 可单独调节，但 parent/elastic 两臂不能据此产生任何差异。
    """

    stride: int = 1
    max_frames: int | None = None
    decode_replicas: int = 4
    reduce_replicas: int = 4
    transform_replicas: int = 4
    transform_batch_size: int = 16
    transform_backend: str = "vit"
    torch_num_threads: int = 1
    model_path: str | None = None
    transform_num_gpus: float = 1.0
    model_repeats: int = 1
    microbatch_size: int = 2
    max_inflight_arenas: int = 2
    max_pending_per_actor: int = 4
    actor_max_concurrency: int = 1

    def __post_init__(self) -> None:
        """拒绝会破坏正式四卡或 batch-cap 合同的配置。"""

        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.max_frames is not None and self.max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        if min(
            self.decode_replicas,
            self.reduce_replicas,
            self.transform_replicas,
            self.torch_num_threads,
            self.model_repeats,
            self.microbatch_size,
            self.max_inflight_arenas,
            self.max_pending_per_actor,
            self.actor_max_concurrency,
        ) <= 0:
            raise ValueError("replicas and execution limits must be positive")
        if not 1 <= self.transform_batch_size <= 16:
            raise ValueError("transform_batch_size must be in [1, 16]")
        if self.transform_num_gpus < 0:
            raise ValueError("transform_num_gpus must be non-negative")
        if self.transform_backend not in {"opencv", "resnet18", "vit"}:
            raise ValueError("unsupported transform_backend")
        if self.transform_backend == "vit" and not self.model_path:
            raise ValueError("ViT backend requires a local model_path")


@dataclass(frozen=True, slots=True)
class MatrixArm:
    """单个 V3 arm 的纯配置，不创建 Ray runtime。"""

    name: str
    options: Mapping[str, Any]


def build_matrix_plan(config: VideoMatrixConfig) -> tuple[MatrixArm, ...]:
    """构建两臂计划；除 ``batch_scope`` 外 options 必须字节级等价。"""

    common = {
        "stride": config.stride,
        "max_frames": config.max_frames,
        "decode_replicas": config.decode_replicas,
        "reduce_replicas": config.reduce_replicas,
        "transform_replicas": config.transform_replicas,
        "transform_batch_size": config.transform_batch_size,
        "transform_backend": config.transform_backend,
        "torch_num_threads": config.torch_num_threads,
        "model_path": config.model_path,
        "transform_num_gpus": config.transform_num_gpus,
        "model_repeats": config.model_repeats,
        "microbatch_size": config.microbatch_size,
        "max_inflight_arenas": config.max_inflight_arenas,
        "max_pending_per_actor": config.max_pending_per_actor,
        "actor_max_concurrency": config.actor_max_concurrency,
    }
    return (
        MatrixArm(
            "v3_parent_bound",
            {**common, "batch_scope": "parent_bound"},
        ),
        MatrixArm("v3_elastic", {**common, "batch_scope": "elastic"}),
    )


def compile_arm(arm: MatrixArm):
    """在不导入 Ray 的条件下编译 arm，供配置审计和 Ray-free 单测使用。"""

    options = dict(arm.options)
    for key in (
        "microbatch_size",
        "max_inflight_arenas",
        "max_pending_per_actor",
        "actor_max_concurrency",
    ):
        options.pop(key)
    return VideoV3Pipeline(**options).compile()


def _timeline_summary(timeline: Iterable[Any]) -> dict[str, Any]:
    """按 decode/transform/reduce stage 汇总 RPC、batch、busy/span 和峰值。"""

    names = {1: "decode", 2: "transform", 3: "reduce"}
    grouped: dict[int, list[Any]] = defaultdict(list)
    for event in timeline:
        grouped[int(event.stage)].append(event)
    summary: dict[str, Any] = {}
    for stage, events in sorted(grouped.items()):
        worker_events = [
            event
            for event in events
            if event.worker_started_at is not None
            and event.worker_finished_at is not None
        ]
        transitions = [
            (event.worker_started_at, 1) for event in worker_events
        ] + [
            (event.worker_finished_at, -1) for event in worker_events
        ]
        active = peak = 0
        for _, delta in sorted(transitions, key=lambda row: (row[0], row[1])):
            active += delta
            peak = max(peak, active)
        grains = [int(event.grains) for event in events]
        busy = sum(
            event.worker_finished_at - event.worker_started_at
            for event in worker_events
        )
        summary[names.get(stage, f"stage_{stage}")] = {
            "stage_id": stage,
            "rpc_count": len(events),
            "grains": sum(grains),
            "grains_per_rpc": sum(grains) / len(grains) if grains else 0.0,
            "batch_histogram": dict(sorted(Counter(grains).items())),
            "flush_reasons": dict(
                sorted(Counter(event.flush_reason for event in events).items())
            ),
            "worker_busy_sum_s": busy,
            "worker_span_s": (
                max(event.worker_finished_at for event in worker_events)
                - min(event.worker_started_at for event in worker_events)
                if worker_events
                else 0.0
            ),
            "peak_concurrency": peak,
        }
    return summary


class _UnavailableGpuMonitor:
    """NVML helper 不可导入时的兼容降级，避免绑定 Docling benchmark 语义。"""

    available = False
    unavailable_reason = "GpuMonitor unavailable"

    def start(self) -> "_UnavailableGpuMonitor":
        """保持与共享监控器相同的 start 形状。"""

        return self

    def stop(self) -> tuple[Any, ...]:
        """没有 NVML 时返回空 samples。"""

        return ()


def _new_gpu_monitor(path: str | None):
    """延迟复用通用 NVML 采样器；模块不可用不影响 Video runner。"""

    try:
        module = importlib.import_module(
            "rayorch.experimental.multigrain_v3.benchmark."
            "document_docling.gpu_monitor"
        )
        return module.GpuMonitor(
            device_indices=(0, 1, 2, 3),
            jsonl_path=path,
        )
    except Exception as error:
        monitor = _UnavailableGpuMonitor()
        monitor.unavailable_reason = f"{type(error).__name__}: {error}"
        return monitor


def _gpu_summary(monitor: Any, samples: Iterable[Any]) -> dict[str, Any]:
    """将四卡 NVML 原始 samples 归约为稳定的 summary。"""

    rows = tuple(samples)
    per_device: dict[int, list[Any]] = defaultdict(list)
    for sample in rows:
        for device in sample.devices:
            per_device[int(device.index)].append(device)
    devices = {}
    for index in range(4):
        values = per_device[index]
        utilization = [
            value.utilization_percent
            for value in values
            if value.utilization_percent is not None
        ]
        memory = [
            value.memory_used_bytes
            for value in values
            if value.memory_used_bytes is not None
        ]
        power = [
            value.power_watts
            for value in values
            if value.power_watts is not None
        ]
        devices[str(index)] = {
            "samples": len(values),
            "nonzero_utilization_samples": sum(value > 0 for value in utilization),
            "utilization_mean_percent": (
                sum(utilization) / len(utilization) if utilization else None
            ),
            "utilization_max_percent": max(utilization) if utilization else None,
            "memory_peak_bytes": max(memory) if memory else None,
            "power_mean_watts": sum(power) / len(power) if power else None,
        }
    return {
        "available": monitor.available,
        "unavailable_reason": monitor.unavailable_reason,
        "sample_count": len(rows),
        "devices": devices,
    }


def _json_default(value: Any) -> Any:
    """序列化 tuple、dataclass 等实验输出中的轻量 Python 值。"""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_outputs(path: str, outputs: Iterable[Any]) -> None:
    """压缩保存完整 outputs artifact，避免 summary 和终端泄漏大结果。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        json.dump(
            list(outputs),
            stream,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )


def _package_version(name: str) -> str | None:
    """读取环境 package version，缺失时保留 null。"""

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _input_manifest(entries: Iterable[VideoManifestEntry]) -> list[dict[str, Any]]:
    """将已校验 manifest 保留在 summary 中，而非重复写任何视频内容。"""

    return [entry.to_json() for entry in entries]


def run_single_arm(
    entries: tuple[VideoManifestEntry, ...],
    *,
    arm_name: str,
    config: VideoMatrixConfig,
    ray_num_cpus: int,
    ray_num_gpus: float,
    outputs_path: str | None = None,
    gpu_samples_path: str | None = None,
) -> dict[str, Any]:
    """在独立本地 Ray session 中执行一个 Video V3 arm。"""

    if ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be positive")
    if ray_num_gpus < 0:
        raise ValueError("ray_num_gpus must be non-negative")
    matches = [arm for arm in build_matrix_plan(config) if arm.name == arm_name]
    if len(matches) != 1:
        raise ValueError(f"unknown arm: {arm_name}")
    arm = matches[0]
    import ray

    if ray.is_initialized():
        raise RuntimeError("single-arm runner requires an independent Ray session")
    paths = [entry.path for entry in entries]
    ray.init(
        address="local",
        num_cpus=ray_num_cpus,
        num_gpus=ray_num_gpus,
        include_dashboard=False,
    )
    monitor = _new_gpu_monitor(gpu_samples_path).start()
    try:
        result = run_v3(paths, **dict(arm.options))
        outputs = result.get()
    finally:
        gpu_samples = monitor.stop()
        ray.shutdown()
    if outputs_path:
        _write_outputs(outputs_path, outputs)
    return {
        "schema_version": 1,
        "created_unix_s": time.time(),
        "arm": arm.name,
        "options": dict(arm.options),
        "input_manifest": _input_manifest(entries),
        "outputs_artifact": (
            str(Path(outputs_path).resolve()) if outputs_path else None
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "ray": _package_version("ray"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "result": {
            "startup_s": result.metrics["startup_time_s"],
            "measured_s": result.metrics["measured_wall_time_s"],
            "end_to_end_s": result.metrics["end_to_end_wall_time_s"],
            "metrics": dict(result.metrics),
            "outputs_count": len(outputs),
            "failures_count": len(result.failures),
            "suppressions_count": len(result.suppressions),
            "timeline_summary": _timeline_summary(result.timeline),
            "gpu_monitor": _gpu_summary(monitor, gpu_samples),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """构造 Video A 单 arm runner CLI。"""

    parser = argparse.ArgumentParser(
        description="Run one independent Video A V3 parent/elastic arm."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm", required=True, choices=ARM_ORDER)
    parser.add_argument("--output", required=True)
    parser.add_argument("--append-jsonl")
    parser.add_argument("--outputs-artifact")
    parser.add_argument("--gpu-samples-output")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--decode-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--transform-replicas", type=int, default=4)
    parser.add_argument("--transform-batch-size", type=int, default=16)
    parser.add_argument(
        "--transform-backend",
        choices=("opencv", "resnet18", "vit"),
        default="vit",
    )
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--model-path")
    parser.add_argument("--transform-num-gpus", type=float, default=1.0)
    parser.add_argument("--model-repeats", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--max-inflight-arenas", type=int, default=2)
    parser.add_argument("--max-pending-per-actor", type=int, default=4)
    parser.add_argument("--actor-max-concurrency", type=int, default=1)
    parser.add_argument("--ray-num-cpus", type=int, default=16)
    parser.add_argument("--ray-num-gpus", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """读取现有 manifest、执行单 arm，并只打印紧凑 summary JSON。"""

    args = build_parser().parse_args(argv)
    entries = load_video_manifest(args.manifest)
    config = VideoMatrixConfig(
        stride=args.stride,
        max_frames=args.max_frames,
        decode_replicas=args.decode_replicas,
        reduce_replicas=args.reduce_replicas,
        transform_replicas=args.transform_replicas,
        transform_batch_size=args.transform_batch_size,
        transform_backend=args.transform_backend,
        torch_num_threads=args.torch_num_threads,
        model_path=args.model_path,
        transform_num_gpus=args.transform_num_gpus,
        model_repeats=args.model_repeats,
        microbatch_size=args.microbatch_size,
        max_inflight_arenas=args.max_inflight_arenas,
        max_pending_per_actor=args.max_pending_per_actor,
        actor_max_concurrency=args.actor_max_concurrency,
    )
    result = run_single_arm(
        entries,
        arm_name=args.arm,
        config=config,
        ray_num_cpus=args.ray_num_cpus,
        ray_num_gpus=args.ray_num_gpus,
        outputs_path=args.outputs_artifact,
        gpu_samples_path=args.gpu_samples_output,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
    output.write_text(encoded + "\n", encoding="utf-8")
    if args.append_jsonl:
        append = Path(args.append_jsonl)
        append.parent.mkdir(parents=True, exist_ok=True)
        with append.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, default=_json_default))
            stream.write("\n")
    print(
        json.dumps(
            {
                "arm": result["arm"],
                "output": str(output.resolve()),
                "outputs_artifact": result["outputs_artifact"],
                "startup_s": result["result"]["startup_s"],
                "measured_s": result["result"]["measured_s"],
                "end_to_end_s": result["result"]["end_to_end_s"],
                "outputs_count": result["result"]["outputs_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
