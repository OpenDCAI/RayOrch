"""Docling core-stage 四臂矩阵的可复现实验入口。

矩阵固定比较：

``native_default → native_tuned → v3_parent_bound → v3_elastic``。

四臂使用相同 PDF source order、模型 device、heavy-stage batch cap 与 correctness gate。
Native tuned 只增加 Docling 的 document-level concurrency；V3 的两个臂只切换
``batch_scope``。本 runner 不写模型输出、Ray timeline 或大 ObjectRef 到仓库。
``--timeline-output`` 可把 timeline 写入仓库外的实验目录，用于核验 arena/stage overlap。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from .core_native import NativeCoreConfig, run_native_core
from .core_native_multi import run_native_core_multi
from .gpu_monitor import GpuMonitor
from .core_v3 import run_v3


ARM_ORDER = (
    "native_default",
    "native_tuned",
    "v3_parent_bound",
    "v3_elastic",
)

BALANCED_ARM_ORDERS = (
    ARM_ORDER,
    (
        "native_tuned",
        "v3_elastic",
        "v3_parent_bound",
        "native_default",
    ),
    (
        "v3_parent_bound",
        "native_default",
        "v3_elastic",
        "native_tuned",
    ),
    (
        "v3_elastic",
        "v3_parent_bound",
        "native_tuned",
        "native_default",
    ),
)


@dataclass(frozen=True, slots=True)
class CoreMatrixConfig:
    """四臂 Docling core-stage 试验共享的资源和 batching 参数。

    ``native_tuned_doc_batch_size=0`` 与 ``microbatch_size=0`` 分别表示使用全部输入
    documents；这让默认配置在固定 manifest 上只有一个 document batch / 一个 V3 arena，
    便于比较 Native document concurrency 与 V3 page-stage scheduling。
    """

    device: str = "cpu"
    ocr_device: str = "cpu"
    num_threads: int = 4
    stage_batch_size: int = 4
    table_core_batch_size: int = 0
    native_tuned_doc_concurrency: int = 4
    native_tuned_doc_batch_size: int = 0
    parse_replicas: int = 4
    layout_replicas: int = 1
    ocr_replicas: int = 1
    table_replicas: int = 1
    reduce_replicas: int = 1
    parse_batch_wait_ms: float = 2.0
    stage_batch_wait_ms: float = 2.0
    layout_num_gpus: float = 0.0
    table_num_gpus: float = 0.0
    layout_actor_concurrency: int = 1
    ocr_actor_concurrency: int = 1
    table_actor_concurrency: int = 1
    ocr_batch_mode: str = "reference"
    ocr_recognition_batch_size: int = 6
    table_batch_mode: str = "reference"
    table_batch_max_jobs: int = 16
    actor_num_cpus: float = 1.0
    max_pending_per_actor: int = 4
    microbatch_size: int = 0
    max_inflight_arenas: int = 1
    four_gpu_native: bool = False

    def __post_init__(self) -> None:
        """校验矩阵参数，避免悄悄比较不同 batch/resource 合同。"""

        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.stage_batch_size <= 0:
            raise ValueError("stage_batch_size must be positive")
        if self.table_core_batch_size < 0:
            raise ValueError("table_core_batch_size must be non-negative")
        if self.native_tuned_doc_concurrency <= 0:
            raise ValueError(
                "native_tuned_doc_concurrency must be positive"
            )
        if self.native_tuned_doc_batch_size < 0:
            raise ValueError(
                "native_tuned_doc_batch_size must be non-negative"
            )
        if min(
            self.parse_replicas,
            self.layout_replicas,
            self.ocr_replicas,
            self.table_replicas,
            self.reduce_replicas,
        ) <= 0:
            raise ValueError("all stage replica counts must be positive")
        if self.parse_batch_wait_ms < 0:
            raise ValueError("parse_batch_wait_ms must be non-negative")
        if self.stage_batch_wait_ms < 0:
            raise ValueError("stage_batch_wait_ms must be non-negative")
        if self.actor_num_cpus <= 0:
            raise ValueError("actor_num_cpus must be positive")
        if min(
            self.layout_actor_concurrency,
            self.ocr_actor_concurrency,
            self.table_actor_concurrency,
        ) <= 0:
            raise ValueError("all actor concurrency values must be positive")
        if self.max_pending_per_actor <= 0:
            raise ValueError("max_pending_per_actor must be positive")
        if self.microbatch_size < 0:
            raise ValueError("microbatch_size must be non-negative")
        if self.max_inflight_arenas <= 0:
            raise ValueError("max_inflight_arenas must be positive")
        if self.ocr_batch_mode not in {
            "reference",
            "recognition_shadow",
            "recognition_accelerated",
        }:
            raise ValueError("unsupported ocr_batch_mode")
        if self.ocr_recognition_batch_size <= 0:
            raise ValueError("ocr_recognition_batch_size must be positive")
        if self.table_batch_mode not in {
            "reference",
            "encoder_shadow",
            "encoder_accelerated",
            "decoder_accelerated",
            "v1_batch",
            "v2_batch",
        }:
            raise ValueError("unsupported table_batch_mode")
        if self.table_batch_max_jobs <= 0:
            raise ValueError("table_batch_max_jobs must be positive")
        if (
            self.table_batch_mode != "reference"
            and self.table_actor_concurrency != 1
        ):
            raise ValueError(
                "table_actor_concurrency must be 1 for table batching"
            )

    def resolved_native_tuned_batch_size(self, sources: int) -> int:
        """返回 tuned Native 一个 document batch 接收的 source 数。"""

        batch_size = self.native_tuned_doc_batch_size or sources
        if batch_size < self.native_tuned_doc_concurrency:
            raise ValueError(
                "resolved native tuned doc batch is smaller than concurrency"
            )
        return batch_size

    def resolved_microbatch_size(self, sources: int) -> int:
        """返回 V3 一个 Arena 接收的 source 数。"""

        return self.microbatch_size or sources


@dataclass(frozen=True, slots=True)
class MatrixArm:
    """一条矩阵 arm 的纯配置描述；不持有 Docling 或 Ray runtime。"""

    name: str
    kind: str
    options: Mapping[str, Any]


def counterbalanced_arm_order(
    plan: tuple[MatrixArm, ...],
    trial_index: int,
) -> tuple[MatrixArm, ...]:
    """轮转四臂执行顺序，削弱 GPU/文件缓存与温度的固定顺序偏差。

    这个函数只改变物理执行顺序，不改变每个 arm 的配置或 JSON key。论文汇总始终按
    ``ARM_ORDER`` 呈现。四个 repeats 恰好让每个 arm 各出现一次首位。
    """

    if not plan:
        raise ValueError("plan must not be empty")
    by_name = {arm.name: arm for arm in plan}
    if set(by_name) != set(ARM_ORDER):
        raise ValueError("plan does not contain the canonical four arms")
    order = BALANCED_ARM_ORDERS[
        trial_index % len(BALANCED_ARM_ORDERS)
    ]
    return tuple(by_name[name] for name in order)


def build_matrix_plan(
    source_count: int,
    config: CoreMatrixConfig,
) -> tuple[MatrixArm, ...]:
    """生成四臂计划，并冻结唯一允许变化的参数。"""

    if source_count <= 0:
        raise ValueError("source_count must be positive")

    native_common = {
        "device": config.device,
        "num_threads": config.num_threads,
        "layout_batch_size": config.stage_batch_size,
        "ocr_batch_size": config.stage_batch_size,
        "table_batch_size": config.stage_batch_size,
    }
    native_default = {
        **native_common,
        "doc_batch_size": 1,
        "doc_batch_concurrency": 1,
    }
    native_tuned = {
        **native_common,
        "doc_batch_size": config.resolved_native_tuned_batch_size(
            source_count
        ),
        "doc_batch_concurrency": config.native_tuned_doc_concurrency,
    }
    v3_common = {
        "parse_replicas": config.parse_replicas,
        "parse_batch_size": 1,
        "parse_batch_wait_ms": config.parse_batch_wait_ms,
        "parse_actor_concurrency": 1,
        "image_scales": (1.0,),
        "layout_replicas": config.layout_replicas,
        "layout_batch_size": config.stage_batch_size,
        "layout_batch_wait_ms": config.stage_batch_wait_ms,
        "layout_num_gpus": config.layout_num_gpus,
        "layout_actor_concurrency": config.layout_actor_concurrency,
        "ocr_replicas": config.ocr_replicas,
        "ocr_batch_size": config.stage_batch_size,
        "ocr_batch_wait_ms": config.stage_batch_wait_ms,
        "ocr_actor_concurrency": config.ocr_actor_concurrency,
        "ocr_batch_mode": config.ocr_batch_mode,
        "ocr_recognition_batch_size": config.ocr_recognition_batch_size,
        "table_replicas": config.table_replicas,
        "table_batch_size": config.stage_batch_size,
        "table_core_batch_size": (
            config.table_core_batch_size
            or config.stage_batch_size
        ),
        "table_batch_wait_ms": config.stage_batch_wait_ms,
        "table_num_gpus": config.table_num_gpus,
        "table_actor_concurrency": config.table_actor_concurrency,
        "table_batch_mode": config.table_batch_mode,
        "table_batch_max_jobs": config.table_batch_max_jobs,
        "reduce_replicas": config.reduce_replicas,
        "layout_device": config.device,
        "ocr_device": config.ocr_device,
        "table_device": config.device,
        "num_threads": config.num_threads,
        "actor_num_cpus": config.actor_num_cpus,
        "microbatch_size": config.resolved_microbatch_size(source_count),
        "max_inflight_arenas": config.max_inflight_arenas,
        "max_pending_per_actor": config.max_pending_per_actor,
        "actor_max_concurrency": 1,
    }
    return (
        MatrixArm(
            "native_default",
            "native_multi" if config.four_gpu_native else "native",
            native_default,
        ),
        MatrixArm(
            "native_tuned",
            "native_multi" if config.four_gpu_native else "native",
            native_tuned,
        ),
        MatrixArm(
            "v3_parent_bound",
            "v3",
            {**v3_common, "batch_scope": "parent_bound"},
        ),
        MatrixArm(
            "v3_elastic",
            "v3",
            {**v3_common, "batch_scope": "elastic"},
        ),
    )


def _token_jaccard(left: str, right: str) -> float:
    """计算 Markdown whitespace-token Jaccard，避免输出大文本。"""

    lhs = set(left.lower().split())
    rhs = set(right.lower().split())
    union = lhs | rhs
    return len(lhs & rhs) / len(union) if union else 1.0


def _document_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """移除大 Markdown，只保留 correctness 与结果溯源所需字段。"""

    return {
        key: document[key]
        for key in (
            "pages",
            "markdown_chars",
            "texts",
            "tables",
            "pictures",
        )
    }


def _timeline_summary(timeline: Iterable[Any]) -> dict[str, Any]:
    """按 Stage 汇总 V3 dispatch 数、batch shape、busy/span 与峰值并发。

    全 DAG ``rpc_count`` 会被 parse/reduce 固定调用稀释；Docling elastic 归因必须单独观察
    Layout/OCR/Table 三个 page stages。
    """

    stage_names = {
        1: "parse",
        2: "layout",
        3: "ocr",
        4: "postprocess",
        5: "table_expand",
        6: "table_core",
        7: "page_reduce",
        8: "document_reduce",
    }
    grouped: dict[int, list[Any]] = defaultdict(list)
    for event in timeline:
        grouped[int(event.stage)].append(event)
    result = {}
    for stage, events in sorted(grouped.items()):
        worker_events = [
            event
            for event in events
            if event.worker_started_at is not None
            and event.worker_finished_at is not None
        ]
        concurrency_events = [
            (event.worker_started_at, 1)
            for event in worker_events
        ] + [
            (event.worker_finished_at, -1)
            for event in worker_events
        ]
        current = peak = 0
        for _, delta in sorted(
            concurrency_events,
            key=lambda item: (item[0], item[1]),
        ):
            current += delta
            peak = max(peak, current)
        grains = [int(event.grains) for event in events]
        busy = sum(
            event.worker_finished_at - event.worker_started_at
            for event in worker_events
        )
        result[stage_names.get(stage, f"stage_{stage}")] = {
            "stage_id": stage,
            "rpc_count": len(events),
            "grains": sum(grains),
            "grains_per_rpc": (
                sum(grains) / len(grains) if grains else 0.0
            ),
            "batch_histogram": dict(sorted(Counter(grains).items())),
            "flush_reasons": dict(
                Counter(event.flush_reason for event in events)
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
    return result


def _compare_documents(
    baseline: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """按 source order 比较 Markdown 与结构计数。"""

    left = tuple(baseline)
    right = tuple(candidate)
    if len(left) != len(right):
        return {
            "document_count_matches": False,
            "markdown_jaccard": (),
            "structure_exact": (),
        }
    structure_fields = ("pages", "texts", "tables", "pictures")
    return {
        "document_count_matches": True,
        "markdown_jaccard": tuple(
            round(
                _token_jaccard(
                    str(native["markdown"]),
                    str(other["markdown"]),
                ),
                8,
            )
            for native, other in zip(left, right)
        ),
        "structure_exact": tuple(
            all(
                native.get(field) == other.get(field)
                for field in structure_fields
            )
            for native, other in zip(left, right)
        ),
    }


def _run_arm(
    arm: MatrixArm,
    paths: list[str],
    *,
    page_counts: tuple[int, ...] | None = None,
    sizes: tuple[int, ...] | None = None,
    gpu_monitor_path: str | None = None,
    timeline_output: str | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """执行一个 arm，并返回 JSON 摘要和进程内 correctness 文档。"""

    if arm.kind == "native":
        result = run_native_core(
            paths,
            config=NativeCoreConfig(**dict(arm.options)),
        )
        return (
            {
                "kind": arm.kind,
                "startup_s": result.startup_s,
                "measured_s": result.measured_s,
                "end_to_end_s": result.startup_s + result.measured_s,
                "metrics": {},
                "documents": tuple(
                    _document_summary(document)
                    for document in result.documents
                ),
            },
            result.documents,
        )

    if arm.kind == "native_multi":
        if page_counts is None or sizes is None:
            raise ValueError(
                "four-GPU native arm requires manifest pages and size_bytes"
            )
        monitor = GpuMonitor(jsonl_path=gpu_monitor_path).start()
        try:
            result = run_native_core_multi(
                paths,
                page_counts,
                sizes,
                NativeCoreConfig(**dict(arm.options)),
            )
        finally:
            gpu_samples = monitor.stop()
        return (
            {
                "kind": arm.kind,
                "startup_s": result.startup_s,
                "measured_s": result.measured_s,
                "end_to_end_s": result.e2e_s,
                "metrics": {},
                "worker_records": tuple(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "documents"
                    }
                    for record in result.worker_records
                ),
                "gpu_monitor": _gpu_monitor_summary(
                    monitor,
                    gpu_samples,
                ),
                "documents": tuple(
                    _document_summary(document)
                    for document in result.documents
                ),
            },
            result.documents,
        )

    if arm.kind == "v3":
        monitor = GpuMonitor(jsonl_path=gpu_monitor_path).start()
        try:
            result = run_v3(paths, **dict(arm.options))
        finally:
            gpu_samples = monitor.stop()
        if timeline_output:
            _write_timeline(timeline_output, result.timeline)
        documents = tuple(result.get())
        return (
            {
                "kind": arm.kind,
                "startup_s": result.metrics["startup_time_s"],
                "measured_s": result.metrics["measured_wall_time_s"],
                "end_to_end_s": result.metrics["end_to_end_wall_time_s"],
                "metrics": {
                    key: value
                    for key, value in result.metrics.items()
                    if key
                    in {
                        "rpc_count",
                        "grains_per_rpc",
                        "batch_fill_ratio",
                        "tail_or_recovery_rpc_fraction",
                        "live_blocks_at_delivery",
                        "reduce_slots",
                        "active_arenas_high_watermark",
                        "live_blocks_across_arenas_high_watermark",
                    }
                    or key.startswith("actor_count_stage_")
                    or key.startswith("actor_calls_stage_")
                    or key.startswith("actor_audit_stage_")
                },
                "timeline_summary": _timeline_summary(result.timeline),
                "gpu_monitor": _gpu_monitor_summary(
                    monitor,
                    gpu_samples,
                ),
                "documents": tuple(
                    _document_summary(document)
                    for document in documents
                ),
            },
            documents,
        )
    raise ValueError(f"unknown matrix arm kind: {arm.kind}")


def _gpu_monitor_summary(
    monitor: GpuMonitor,
    samples: Iterable[Any],
) -> dict[str, Any]:
    """把原始 NVML samples 归约为每卡 utilization/memory/power 证据。"""

    rows = tuple(samples)
    by_gpu: dict[int, list[Any]] = defaultdict(list)
    for sample in rows:
        for device in sample.devices:
            by_gpu[int(device.index)].append(device)
    devices = {}
    for index in sorted(by_gpu):
        values = by_gpu[index]
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
            "nonzero_utilization_samples": sum(
                value > 0 for value in utilization
            ),
            "utilization_mean_percent": (
                sum(utilization) / len(utilization)
                if utilization
                else None
            ),
            "utilization_max_percent": (
                max(utilization) if utilization else None
            ),
            "memory_peak_bytes": max(memory) if memory else None,
            "power_mean_watts": (
                sum(power) / len(power) if power else None
            ),
        }
    return {
        "available": monitor.available,
        "unavailable_reason": monitor.unavailable_reason,
        "sample_count": len(rows),
        "devices": devices,
    }


def _package_version(name: str) -> str | None:
    """返回已安装 package version；缺失时用 null 保留环境事实。"""

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _input_identity(
    paths: list[str],
    *,
    include_sha256: bool,
) -> tuple[dict[str, Any], ...]:
    """记录有序输入的 path/stat，可选记录完整 SHA-256。"""

    identities = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        stat = path.stat()
        item: dict[str, Any] = {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if include_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            item["sha256"] = digest.hexdigest()
        identities.append(item)
    return tuple(identities)


def _write_documents(
    path: str,
    documents: Iterable[Mapping[str, Any]],
) -> None:
    """把 correctness 所需完整文档输出压缩写入实验目录。

    Markdown 不进入 summary JSON 或仓库；它只作为跨独立进程比较的临时实验 artifact。
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        json.dump(
            list(documents),
            stream,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _write_timeline(path: str, timeline: Iterable[Any]) -> None:
    """把 V3 原始 dispatch timeline 写成逐行 JSON 实验 artifact。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for event in timeline:
            stream.write(
                json.dumps(asdict(event), ensure_ascii=False) + "\n"
            )


def load_documents(path: str) -> tuple[dict[str, Any], ...]:
    """读取 ``_write_documents`` 生成的压缩 correctness artifact。"""

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, list):
        raise ValueError("documents artifact must contain a JSON list")
    return tuple(dict(item) for item in raw)


def run_single_arm(
    paths: list[str],
    *,
    arm_name: str,
    config: CoreMatrixConfig,
    ray_num_cpus: int,
    ray_num_gpus: float,
    include_input_sha256: bool = False,
    documents_output: str | None = None,
    page_counts: tuple[int, ...] | None = None,
    sizes: tuple[int, ...] | None = None,
    gpu_monitor_path: str | None = None,
    timeline_output: str | None = None,
    matrix_repeat: int | None = None,
    matrix_position: int | None = None,
) -> dict[str, Any]:
    """在独立进程语义下运行一个四臂 arm。

    Native arm 不启动 Ray；V3 arm 才创建本地 Ray session。这样 native pipeline 不会与
    Ray GPU context/actors 共存，长时 368-PDF 矩阵可逐臂失败恢复，也不会在一臂失败时丢失
    之前数小时的结果。
    """

    if not paths:
        raise ValueError("paths must not be empty")
    if ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be positive")
    if ray_num_gpus < 0:
        raise ValueError("ray_num_gpus must be non-negative")
    if matrix_repeat is not None and matrix_repeat <= 0:
        raise ValueError("matrix_repeat must be positive")
    if matrix_position is not None and matrix_position <= 0:
        raise ValueError("matrix_position must be positive")

    ordered_paths = [str(Path(path).resolve()) for path in paths]
    plan = build_matrix_plan(len(ordered_paths), config)
    matches = [arm for arm in plan if arm.name == arm_name]
    if len(matches) != 1:
        raise ValueError(f"unknown matrix arm: {arm_name}")
    arm = matches[0]
    if timeline_output and arm.kind != "v3":
        raise ValueError(
            "timeline_output is supported only for a V3 arm"
        )

    started_ray_here = False
    if arm.kind == "v3":
        import ray

        if not ray.is_initialized():
            ray.init(
                address="local",
                num_cpus=ray_num_cpus,
                num_gpus=ray_num_gpus,
                include_dashboard=False,
            )
            started_ray_here = True
    try:
        result, documents = _run_arm(
            arm,
            ordered_paths,
            page_counts=page_counts,
            sizes=sizes,
            gpu_monitor_path=gpu_monitor_path,
            timeline_output=timeline_output,
        )
    finally:
        if started_ray_here:
            import ray

            ray.shutdown()

    if documents_output:
        _write_documents(documents_output, documents)
    return {
        "schema_version": 1,
        "created_unix_s": time.time(),
        "arm": arm.name,
        "kind": arm.kind,
        "matrix_repeat": matrix_repeat,
        "matrix_position": matrix_position,
        "options": dict(arm.options),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "docling": _package_version("docling"),
            "docling_core": _package_version("docling-core"),
            "docling_ibm_models": _package_version("docling-ibm-models"),
            "ray": _package_version("ray"),
            "torch": _package_version("torch"),
        },
        "input_manifest": _input_identity(
            ordered_paths,
            include_sha256=include_input_sha256,
        ),
        "documents_output": (
            str(Path(documents_output).resolve())
            if documents_output
            else None
        ),
        "timeline_output": (
            str(Path(timeline_output).resolve())
            if timeline_output
            else None
        ),
        "result": result,
    }


def run_matrix(
    paths: list[str],
    *,
    config: CoreMatrixConfig,
    ray_num_cpus: int,
    ray_num_gpus: float,
    warmup: int = 0,
    repeats: int = 1,
    include_input_sha256: bool = False,
) -> dict[str, Any]:
    """按固定 arm 顺序运行 warmup/repeats，并生成可写入论文记录的 JSON。

    每个 arm 都是 cold actor/converter run；因此同时报告 startup、measured 与 E2E。Native
    的 global perf settings 在 arm 结束后恢复。warmup 按 canonical order 运行一次或多次；
    measured repeats 按循环移位 counterbalance arm 顺序。若本函数负责启动 Ray，也会在结束后
    关闭。
    """

    if not paths:
        raise ValueError("paths must not be empty")
    if ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be positive")
    if ray_num_gpus < 0:
        raise ValueError("ray_num_gpus must be non-negative")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")

    ordered_paths = [str(Path(path).resolve()) for path in paths]
    plan = build_matrix_plan(len(ordered_paths), config)

    import ray

    started_ray_here = False
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=ray_num_cpus,
            num_gpus=ray_num_gpus,
            include_dashboard=False,
        )
        started_ray_here = True
    try:
        for _ in range(warmup):
            for arm in plan:
                _run_arm(arm, ordered_paths)

        trials: list[dict[str, Any]] = []
        for repeat_index in range(repeats):
            runs: dict[str, dict[str, Any]] = {}
            documents: dict[str, tuple[dict[str, Any], ...]] = {}
            execution_order = counterbalanced_arm_order(plan, repeat_index)
            for arm in execution_order:
                summary, outputs = _run_arm(arm, ordered_paths)
                runs[arm.name] = summary
                documents[arm.name] = outputs

            baseline = documents["native_default"]
            for arm in plan:
                runs[arm.name]["correctness_vs_native_default"] = (
                    _compare_documents(baseline, documents[arm.name])
                )
            trials.append(
                {
                    "repeat": repeat_index,
                    "execution_order": tuple(
                        arm.name for arm in execution_order
                    ),
                    "arms": runs,
                }
            )

        arm_summary = {}
        for arm in plan:
            rows = [trial["arms"][arm.name] for trial in trials]
            arm_summary[arm.name] = {
                "startup_median_s": median(
                    row["startup_s"] for row in rows
                ),
                "measured_median_s": median(
                    row["measured_s"] for row in rows
                ),
                "end_to_end_median_s": median(
                    row["end_to_end_s"] for row in rows
                ),
            }
        return {
            "schema_version": 1,
            "created_unix_s": time.time(),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "docling": _package_version("docling"),
                "docling_core": _package_version("docling-core"),
                "ray": _package_version("ray"),
                "torch": _package_version("torch"),
            },
            "input_manifest": _input_identity(
                ordered_paths,
                include_sha256=include_input_sha256,
            ),
            "config": asdict(config),
            "arms": [
                {
                    "name": arm.name,
                    "kind": arm.kind,
                    "options": dict(arm.options),
                }
                for arm in plan
            ],
            "warmup_runs": warmup,
            "measured_repeats": repeats,
            "trials": trials,
            "summary": arm_summary,
        }
    finally:
        if started_ray_here:
            ray.shutdown()


def _read_paths(args: argparse.Namespace) -> list[str]:
    """从显式 paths、JSON manifest 或目录读取稳定 source order。"""

    if args.manifest:
        raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("manifest must be a JSON list")
        paths = [
            str(item["path"] if isinstance(item, dict) else item)
            for item in raw
        ]
    elif args.paths:
        paths = list(args.paths)
    else:
        paths = sorted(
            str(path)
            for path in Path(args.pdf_dir).glob("*.pdf")
        )
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError("no PDF inputs")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing input PDFs: {missing[:3]}")
    return paths


def _read_manifest_metadata(
    args: argparse.Namespace,
    paths: list[str],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """读取四卡 Native 分片所需的 pages/size_bytes；普通 paths 模式返回空。"""

    if not args.manifest:
        return None, None
    raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not all(isinstance(item, dict) for item in raw):
        return None, None
    if not all("pages" in item for item in raw):
        return None, None
    selected = raw[: len(paths)]
    page_counts = tuple(int(item["pages"]) for item in selected)
    sizes = tuple(
        int(item.get("size_bytes", Path(path).stat().st_size))
        for item, path in zip(selected, paths)
    )
    return page_counts, sizes


def build_parser() -> argparse.ArgumentParser:
    """构造四臂 core-stage compare CLI。"""

    parser = argparse.ArgumentParser(
        description=(
            "Run Native default/tuned and V3 parent-bound/elastic "
            "Docling core-stage matrix."
        )
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--manifest")
    sources.add_argument("--paths", nargs="+")
    parser.add_argument("--pdf-dir", default=".")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--append-jsonl")
    parser.add_argument("--documents-output")
    parser.add_argument("--gpu-monitor-output")
    parser.add_argument("--timeline-output")
    parser.add_argument("--arm", choices=ARM_ORDER)
    parser.add_argument("--matrix-repeat", type=int)
    parser.add_argument("--matrix-position", type=int)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--record-input-sha256", action="store_true")
    parser.add_argument("--device", default=os.environ.get("DOCLING_DEVICE", "cpu"))
    parser.add_argument("--ocr-device", default=None)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--stage-batch-size", type=int, default=4)
    parser.add_argument("--table-core-batch-size", type=int, default=0)
    parser.add_argument("--native-tuned-doc-concurrency", type=int, default=4)
    parser.add_argument("--native-tuned-doc-batch-size", type=int, default=0)
    parser.add_argument("--parse-replicas", type=int, default=4)
    parser.add_argument("--layout-replicas", type=int, default=1)
    parser.add_argument("--ocr-replicas", type=int, default=1)
    parser.add_argument("--table-replicas", type=int, default=1)
    parser.add_argument("--reduce-replicas", type=int, default=1)
    parser.add_argument("--parse-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--stage-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--layout-num-gpus", type=float, default=0.0)
    parser.add_argument("--table-num-gpus", type=float, default=0.0)
    parser.add_argument("--layout-actor-concurrency", type=int, default=1)
    parser.add_argument("--ocr-actor-concurrency", type=int, default=1)
    parser.add_argument(
        "--ocr-batch-mode",
        choices=("reference", "recognition_shadow", "recognition_accelerated"),
        default="reference",
    )
    parser.add_argument("--ocr-recognition-batch-size", type=int, default=6)
    parser.add_argument("--table-actor-concurrency", type=int, default=1)
    parser.add_argument(
        "--table-batch-mode",
        choices=(
            "reference",
            "encoder_shadow",
            "encoder_accelerated",
            "decoder_accelerated",
            "v1_batch",
            "v2_batch",
        ),
        default="reference",
    )
    parser.add_argument("--table-batch-max-jobs", type=int, default=16)
    parser.add_argument("--actor-num-cpus", type=float, default=1.0)
    parser.add_argument("--max-pending-per-actor", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=0)
    parser.add_argument("--max-inflight-arenas", type=int, default=1)
    parser.add_argument("--ray-num-cpus", type=int, default=16)
    parser.add_argument("--ray-num-gpus", type=float, default=0.0)
    parser.add_argument("--four-gpu-native", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行矩阵并将 JSON 打印到 stdout/可选 output 文件。"""

    args = build_parser().parse_args(argv)
    paths = _read_paths(args)
    page_counts, sizes = _read_manifest_metadata(args, paths)
    config = CoreMatrixConfig(
        device=args.device,
        ocr_device=args.ocr_device or args.device,
        num_threads=args.num_threads,
        stage_batch_size=args.stage_batch_size,
        table_core_batch_size=args.table_core_batch_size,
        native_tuned_doc_concurrency=args.native_tuned_doc_concurrency,
        native_tuned_doc_batch_size=args.native_tuned_doc_batch_size,
        parse_replicas=args.parse_replicas,
        layout_replicas=args.layout_replicas,
        ocr_replicas=args.ocr_replicas,
        table_replicas=args.table_replicas,
        reduce_replicas=args.reduce_replicas,
        parse_batch_wait_ms=args.parse_batch_wait_ms,
        stage_batch_wait_ms=args.stage_batch_wait_ms,
        layout_num_gpus=args.layout_num_gpus,
        table_num_gpus=args.table_num_gpus,
        layout_actor_concurrency=args.layout_actor_concurrency,
        ocr_actor_concurrency=args.ocr_actor_concurrency,
        ocr_batch_mode=args.ocr_batch_mode,
        ocr_recognition_batch_size=args.ocr_recognition_batch_size,
        table_actor_concurrency=args.table_actor_concurrency,
        table_batch_mode=args.table_batch_mode,
        table_batch_max_jobs=args.table_batch_max_jobs,
        actor_num_cpus=args.actor_num_cpus,
        max_pending_per_actor=args.max_pending_per_actor,
        microbatch_size=args.microbatch_size,
        max_inflight_arenas=args.max_inflight_arenas,
        four_gpu_native=args.four_gpu_native,
    )
    if args.arm:
        if args.warmup or args.repeats != 1:
            raise ValueError(
                "--arm is one cold run; repeat/order is owned by core_matrix"
            )
        result = run_single_arm(
            paths,
            arm_name=args.arm,
            config=config,
            ray_num_cpus=args.ray_num_cpus,
            ray_num_gpus=args.ray_num_gpus,
            include_input_sha256=args.record_input_sha256,
            documents_output=args.documents_output,
            page_counts=page_counts,
            sizes=sizes,
            gpu_monitor_path=args.gpu_monitor_output,
            timeline_output=args.timeline_output,
            matrix_repeat=args.matrix_repeat,
            matrix_position=args.matrix_position,
        )
    else:
        if (
            args.documents_output
            or args.append_jsonl
            or args.gpu_monitor_output
            or args.timeline_output
            or args.matrix_repeat
            or args.matrix_position
        ):
            raise ValueError(
                "single-arm artifact/matrix metadata options require --arm"
            )
        result = run_matrix(
            paths,
            config=config,
            ray_num_cpus=args.ray_num_cpus,
            ray_num_gpus=args.ray_num_gpus,
            warmup=args.warmup,
            repeats=args.repeats,
            include_input_sha256=args.record_input_sha256,
        )
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    if args.append_jsonl:
        jsonl = Path(args.append_jsonl)
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        with jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
