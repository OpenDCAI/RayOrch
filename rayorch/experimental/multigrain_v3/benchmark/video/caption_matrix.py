"""SmolVLM caption 正式四卡 parent/elastic 单臂 runner。

每次命令只运行一个 arm，并在该命令内独立创建、关闭 Ray session。parent 与
elastic 的唯一差别是 caption Map 的 ``batch_scope``；其余模型、GPU、batching
和 backpressure 参数完全相同。完整 caption outputs 仅写 gzip artifact。
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

from .caption_v3 import VideoCaptionV3Pipeline, run_v3
from .data import VideoManifestEntry, load_video_manifest


ARM_ORDER = ("caption_parent_bound", "caption_elastic")


@dataclass(frozen=True, slots=True)
class CaptionMatrixConfig:
    """Caption 两臂共享的正式四卡编译与执行配置。"""

    model_path: str
    stride: int = 12
    max_frames: int | None = 4
    caption_batch_size: int = 16
    caption_replicas: int = 4
    decode_replicas: int = 4
    reduce_replicas: int = 4
    max_new_tokens: int = 12
    microbatch_size: int = 32
    max_inflight_arenas: int = 4
    max_pending_per_actor: int = 4
    actor_max_concurrency: int = 1

    def __post_init__(self) -> None:
        """拒绝破坏四卡 caption 或执行上限合同的配置。"""

        if not self.model_path:
            raise ValueError("model_path is required")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.max_frames is not None and self.max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        if not 1 <= self.caption_batch_size <= 16:
            raise ValueError("caption_batch_size must be in [1, 16]")
        if min(
            self.caption_replicas,
            self.decode_replicas,
            self.reduce_replicas,
            self.max_new_tokens,
            self.microbatch_size,
            self.max_inflight_arenas,
            self.max_pending_per_actor,
            self.actor_max_concurrency,
        ) <= 0:
            raise ValueError("replicas and execution limits must be positive")


@dataclass(frozen=True, slots=True)
class CaptionMatrixArm:
    """不创建 Ray runtime 的单臂纯配置。"""

    name: str
    options: Mapping[str, Any]


def build_matrix_plan(config: CaptionMatrixConfig) -> tuple[CaptionMatrixArm, ...]:
    """构建两臂；除 ``batch_scope`` 外 options 必须相等。"""

    common = {
        "model_path": config.model_path,
        "stride": config.stride,
        "max_frames": config.max_frames,
        "caption_batch_size": config.caption_batch_size,
        "caption_replicas": config.caption_replicas,
        "decode_replicas": config.decode_replicas,
        "reduce_replicas": config.reduce_replicas,
        "max_new_tokens": config.max_new_tokens,
        "microbatch_size": config.microbatch_size,
        "max_inflight_arenas": config.max_inflight_arenas,
        "max_pending_per_actor": config.max_pending_per_actor,
        "actor_max_concurrency": config.actor_max_concurrency,
    }
    return (
        CaptionMatrixArm(
            "caption_parent_bound",
            {**common, "batch_scope": "parent_bound"},
        ),
        CaptionMatrixArm(
            "caption_elastic",
            {**common, "batch_scope": "elastic"},
        ),
    )


def compile_arm(arm: CaptionMatrixArm):
    """编译单臂而不导入 Ray，供配置审计和 Ray-free 单测使用。"""

    options = dict(arm.options)
    for key in (
        "microbatch_size",
        "max_inflight_arenas",
        "max_pending_per_actor",
        "actor_max_concurrency",
    ):
        options.pop(key)
    return VideoCaptionV3Pipeline(**options).compile()


def _timeline_summary(timeline: Iterable[Any]) -> dict[str, Any]:
    """按 decode/caption/reduce 汇总 RPC、batch、busy/span 与并发峰值。"""

    names = {1: "decode", 2: "caption", 3: "reduce"}
    grouped: dict[int, list[Any]] = defaultdict(list)
    for event in timeline:
        grouped[int(event.stage)].append(event)
    result: dict[str, Any] = {}
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
        result[names.get(stage, f"stage_{stage}")] = {
            "stage_id": stage,
            "rpc_count": len(events),
            "grains": sum(grains),
            "grains_per_rpc": sum(grains) / len(grains) if grains else 0.0,
            "batch_histogram": dict(sorted(Counter(grains).items())),
            "flush_reasons": dict(
                sorted(Counter(event.flush_reason for event in events).items())
            ),
            "worker_busy_sum_s": sum(
                event.worker_finished_at - event.worker_started_at
                for event in worker_events
            ),
            "worker_span_s": (
                max(event.worker_finished_at for event in worker_events)
                - min(event.worker_started_at for event in worker_events)
                if worker_events
                else 0.0
            ),
            "peak_concurrency": peak,
        }
    return result


class _UnavailableGpuMonitor:
    """NVML 不可用时的无副作用兼容监控器。"""

    available = False
    unavailable_reason = "GpuMonitor unavailable"

    def start(self) -> "_UnavailableGpuMonitor":
        """返回自身以匹配真实监控器接口。"""

        return self

    def stop(self) -> tuple[Any, ...]:
        """没有 NVML 样本。"""

        return ()


def _new_gpu_monitor(path: str | None):
    """延迟创建四卡 NVML 监控器，缺失依赖时保留错误摘要。"""

    try:
        module = importlib.import_module(
            "rayorch.experimental.multigrain_v3.benchmark."
            "document_docling.gpu_monitor"
        )
        return module.GpuMonitor(device_indices=(0, 1, 2, 3), jsonl_path=path)
    except Exception as error:
        monitor = _UnavailableGpuMonitor()
        monitor.unavailable_reason = f"{type(error).__name__}: {error}"
        return monitor


def _gpu_summary(monitor: Any, samples: Iterable[Any]) -> dict[str, Any]:
    """归约 NVML 四卡 samples 为稳定 summary。"""

    per_device: dict[int, list[Any]] = defaultdict(list)
    rows = tuple(samples)
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
        devices[str(index)] = {
            "samples": len(values),
            "nonzero_utilization_samples": sum(value > 0 for value in utilization),
            "utilization_mean_percent": (
                sum(utilization) / len(utilization) if utilization else None
            ),
            "utilization_max_percent": max(utilization) if utilization else None,
            "memory_peak_bytes": max(memory) if memory else None,
        }
    return {
        "available": monitor.available,
        "unavailable_reason": monitor.unavailable_reason,
        "sample_count": len(rows),
        "devices": devices,
    }


def _json_default(value: Any) -> Any:
    """序列化 dataclass、路径和 tuple 等轻量结果值。"""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_outputs(path: str, outputs: Iterable[Any]) -> None:
    """将完整 captions 写为 gzip JSON artifact。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        json.dump(list(outputs), stream, ensure_ascii=False, default=_json_default)


def _package_version(name: str) -> str | None:
    """读取可选环境包版本。"""

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def compare_outputs(baseline: Any, candidate: Any) -> dict[str, Any]:
    """严格比较 caption 的 frames、source_indices、captions，并给出首差异。

    Greedy caption 的正式正确性 gate 要求三个业务字段均 exact；比较不会打印
    大型 outputs，只保留第一个不一致视频和字段。
    """

    fields = ("frames", "source_indices", "captions")
    if not isinstance(baseline, list) or not isinstance(candidate, list):
        return {
            "exact": False,
            "reason": "outputs are not lists",
            "first_difference": {"path": "$", "reason": "type"},
        }
    if len(baseline) != len(candidate):
        return {
            "exact": False,
            "reason": "video count",
            "baseline_videos": len(baseline),
            "candidate_videos": len(candidate),
            "first_difference": {
                "path": "$",
                "reason": "length",
                "left_length": len(baseline),
                "right_length": len(candidate),
            },
        }
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return {
                "exact": False,
                "reason": "video output is not an object",
                "first_difference": {"path": f"$[{index}]", "reason": "type"},
            }
        for field in fields:
            if left.get(field) != right.get(field):
                return {
                    "exact": False,
                    "reason": field,
                    "first_difference": {
                        "video_index": index,
                        "field": field,
                        "baseline": left.get(field),
                        "candidate": right.get(field),
                    },
                }
    return {
        "exact": True,
        "videos": len(baseline),
        "fields": list(fields),
        "first_difference": None,
    }


def caption_difference_summary(
    baseline: Any,
    candidate: Any,
) -> dict[str, Any]:
    """量化 greedy caption 的 batch-shape 文本漂移。

    结构 gate 要求 video 数、frame 数和 source indices 完全一致；caption 另外报告 exact
    和简单归一化后的 mismatch 率。不同 batch padding/GEMM shape 即使 ``do_sample=False``
    也可能改变少量 token，因此不能只用第一个差异代表整体正确性。
    """

    if not isinstance(baseline, list) or not isinstance(candidate, list):
        return {"structure_exact": False, "reason": "outputs are not lists"}
    if len(baseline) != len(candidate):
        return {
            "structure_exact": False,
            "reason": "video count",
            "baseline_videos": len(baseline),
            "candidate_videos": len(candidate),
        }

    def normalize(text: Any) -> str:
        """做大小写、空白和末尾标点归一化，不进行语义改写。"""

        return " ".join(str(text).casefold().split()).strip(" .,!?:;")

    structure_mismatch_videos = 0
    caption_mismatch_videos = 0
    caption_mismatch_frames = 0
    normalized_mismatch_frames = 0
    total_frames = 0
    first_difference = None
    for video_index, (left, right) in enumerate(zip(baseline, candidate)):
        if (
            left.get("frames") != right.get("frames")
            or left.get("source_indices") != right.get("source_indices")
        ):
            structure_mismatch_videos += 1
        lhs_values = list(left.get("captions", ()))
        rhs_values = list(right.get("captions", ()))
        total_frames += max(len(lhs_values), len(rhs_values))
        raw_mismatches = abs(len(lhs_values) - len(rhs_values))
        normalized_mismatches = abs(len(lhs_values) - len(rhs_values))
        for frame_index, (lhs, rhs) in enumerate(zip(lhs_values, rhs_values)):
            if lhs != rhs:
                raw_mismatches += 1
                if first_difference is None:
                    first_difference = {
                        "video_index": video_index,
                        "frame_index": frame_index,
                        "baseline": lhs,
                        "candidate": rhs,
                    }
            normalized_mismatches += normalize(lhs) != normalize(rhs)
        if raw_mismatches:
            caption_mismatch_videos += 1
            caption_mismatch_frames += raw_mismatches
            normalized_mismatch_frames += normalized_mismatches
    return {
        "structure_exact": structure_mismatch_videos == 0,
        "structure_mismatch_videos": structure_mismatch_videos,
        "caption_mismatch_videos": caption_mismatch_videos,
        "caption_mismatch_frames": caption_mismatch_frames,
        "normalized_mismatch_frames": normalized_mismatch_frames,
        "total_frames": total_frames,
        "caption_mismatch_rate": (
            caption_mismatch_frames / total_frames if total_frames else 0.0
        ),
        "normalized_mismatch_rate": (
            normalized_mismatch_frames / total_frames
            if total_frames
            else 0.0
        ),
        "first_difference": first_difference,
    }


def run_single_arm(
    entries: tuple[VideoManifestEntry, ...],
    *,
    arm_name: str,
    config: CaptionMatrixConfig,
    ray_num_cpus: int,
    ray_num_gpus: float,
    outputs_path: str | None = None,
    gpu_samples_path: str | None = None,
) -> dict[str, Any]:
    """在独立本地 Ray session 中运行一个 caption arm。"""

    if ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be positive")
    if ray_num_gpus < 0:
        raise ValueError("ray_num_gpus must be non-negative")
    arms = [arm for arm in build_matrix_plan(config) if arm.name == arm_name]
    if len(arms) != 1:
        raise ValueError(f"unknown arm: {arm_name}")
    import ray

    if ray.is_initialized():
        raise RuntimeError("single-arm runner requires an independent Ray session")
    arm = arms[0]
    ray.init(
        address="local",
        num_cpus=ray_num_cpus,
        num_gpus=ray_num_gpus,
        include_dashboard=False,
    )
    monitor = _new_gpu_monitor(gpu_samples_path).start()
    try:
        result = run_v3([entry.path for entry in entries], **dict(arm.options))
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
        "input_manifest": [entry.to_json() for entry in entries],
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
    """构造 SmolVLM caption 单臂 CLI。"""

    parser = argparse.ArgumentParser(
        description="Run one independent SmolVLM caption parent/elastic arm."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--arm", required=True, choices=ARM_ORDER)
    parser.add_argument("--output", required=True)
    parser.add_argument("--append-jsonl")
    parser.add_argument("--outputs-artifact")
    parser.add_argument("--gpu-samples-output")
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--caption-batch-size", type=int, default=16)
    parser.add_argument("--caption-replicas", type=int, default=4)
    parser.add_argument("--decode-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--max-inflight-arenas", type=int, default=4)
    parser.add_argument("--max-pending-per-actor", type=int, default=4)
    parser.add_argument("--actor-max-concurrency", type=int, default=1)
    parser.add_argument("--ray-num-cpus", type=int, default=16)
    parser.add_argument("--ray-num-gpus", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """读取已有 manifest、执行单臂、写 summary 和可选 JSONL。"""

    args = build_parser().parse_args(argv)
    config = CaptionMatrixConfig(
        model_path=args.model_path,
        stride=args.stride,
        max_frames=args.max_frames,
        caption_batch_size=args.caption_batch_size,
        caption_replicas=args.caption_replicas,
        decode_replicas=args.decode_replicas,
        reduce_replicas=args.reduce_replicas,
        max_new_tokens=args.max_new_tokens,
        microbatch_size=args.microbatch_size,
        max_inflight_arenas=args.max_inflight_arenas,
        max_pending_per_actor=args.max_pending_per_actor,
        actor_max_concurrency=args.actor_max_concurrency,
    )
    result = run_single_arm(
        load_video_manifest(args.manifest),
        arm_name=args.arm,
        config=config,
        ray_num_cpus=args.ray_num_cpus,
        ray_num_gpus=args.ray_num_gpus,
        outputs_path=args.outputs_artifact,
        gpu_samples_path=args.gpu_samples_output,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
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
