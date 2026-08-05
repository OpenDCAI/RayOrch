"""Docling Native 四 GPU 整文档并行基线。

每个 worker 都是独立的 ``spawn`` 进程，并且会在导入 Docling 或 torch 之前将
``CUDA_VISIBLE_DEVICES`` 收窄为一张物理卡。这样每个进程持有一个独立的
``DocumentConverter``，避免 CUDA context、Docling pipeline cache 和全局 perf setting
在进程间相互污染。

本模块按已知页数做确定性的 LPT 分片。worker 的完整 Markdown 不经由
``multiprocessing.Queue`` 回传，而是写入临时 gzip JSON，以避免大文档堵塞 IPC pipe。
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from .core_native import NativeCoreConfig


_WORKER_COUNT = 4


@dataclass(frozen=True, slots=True)
class MultiGpuNativeRun:
    """一次四 GPU native Docling 运行的计时、worker 记录和原顺序业务输出。

    ``startup_s`` 是从 parent 启动首个 worker 到全部 worker pipeline ready 的 wall
    time。``measured_s`` 是 parent 统一释放 start event 后，到最慢 worker 完成
    conversion/normalisation 的 makespan。``e2e_s`` 覆盖进程创建、startup、conversion
    和结果聚合。``worker_records`` 逐 slot 保留各 worker 的完整规范化文档。
    """

    startup_s: float
    measured_s: float
    e2e_s: float
    documents: tuple[dict[str, Any], ...]
    worker_records: tuple[dict[str, Any], ...]


def _validate_inputs(
    paths: Sequence[str],
    page_counts: Sequence[int],
    sizes: Sequence[int],
    config: NativeCoreConfig,
    gpu_ids: Sequence[int],
) -> None:
    """校验四 GPU runner 的不可变实验前提。"""

    if not paths:
        raise ValueError("paths must not be empty")
    if len(page_counts) != len(paths):
        raise ValueError("page_counts must have the same length as paths")
    if len(sizes) != len(paths):
        raise ValueError("sizes must have the same length as paths")
    if len(gpu_ids) != _WORKER_COUNT:
        raise ValueError("gpu_ids must contain exactly four GPU ids")
    if len(set(gpu_ids)) != _WORKER_COUNT:
        raise ValueError("gpu_ids must be unique")
    if any(not isinstance(value, int) or value < 0 for value in page_counts):
        raise ValueError("page_counts must contain non-negative integers")
    if any(not isinstance(value, int) or value < 0 for value in sizes):
        raise ValueError("sizes must contain non-negative integers")
    if not config.device.lower().startswith("cuda"):
        raise ValueError("multi-GPU native baseline requires a CUDA config.device")


def _partition_indices_lpt(
    page_counts: Sequence[int],
    sizes: Sequence[int],
    worker_count: int = _WORKER_COUNT,
) -> tuple[tuple[int, ...], ...]:
    """按页数优先的确定性 LPT 算法将文档分给 worker。

    页数是主负载，字节数仅用于在相同页数下稳定地降低 I/O 偏斜。选择当前总页数最小
    的 worker；若相同，依次按总字节数和 slot 编号打破平局。每个 shard 在分配完成后
    重新按输入下标排序，因此 worker 内的 ``convert_all`` 输入也保持 source order。
    """

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if len(page_counts) != len(sizes):
        raise ValueError("page_counts and sizes must have equal lengths")

    loads = [[0, 0] for _ in range(worker_count)]
    partitions: list[list[int]] = [[] for _ in range(worker_count)]
    descending_indices = sorted(
        range(len(page_counts)),
        key=lambda index: (-page_counts[index], -sizes[index], index),
    )
    for index in descending_indices:
        slot = min(
            range(worker_count),
            key=lambda candidate: (
                loads[candidate][0],
                loads[candidate][1],
                candidate,
            ),
        )
        partitions[slot].append(index)
        loads[slot][0] += page_counts[index]
        loads[slot][1] += sizes[index]

    return tuple(tuple(sorted(partition)) for partition in partitions)


def _write_payload(path: str, payload: dict[str, Any]) -> None:
    """以 gzip JSON 写入完整 worker 结果，绕开大 Markdown 的 Queue 传输。"""

    temporary = f"{path}.tmp"
    with gzip.open(temporary, "wt", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def _read_payload(path: str) -> dict[str, Any]:
    """读取一个 worker 的 gzip JSON 结果。"""

    with gzip.open(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise RuntimeError("worker result payload must be a JSON object")
    return value


def _native_multi_worker(
    slot: int,
    gpu_id: int,
    paths: tuple[str, ...],
    page_counts: tuple[int, ...],
    sizes: tuple[int, ...],
    indices: tuple[int, ...],
    config: NativeCoreConfig,
    start_event: Any,
    status_queue: Any,
    result_path: str,
) -> None:
    """在单卡可见的子进程中初始化一个 converter 并执行一个 document shard。"""

    try:
        # 必须在任何可能触碰 torch/Docling 的 import 之前执行。
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        started = time.perf_counter()
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        from .core_native import _docling_perf_settings, _normalise_document

        # 可见设备已被掩码为一张卡，因此任何 parent 传入的 cuda:N 都须转为本地 cuda。
        local_config = replace(config, device="cuda")
        with _docling_perf_settings(local_config):
            options = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(
                    device=local_config.device,
                    num_threads=local_config.num_threads,
                ),
                layout_batch_size=local_config.layout_batch_size,
                ocr_batch_size=local_config.ocr_batch_size,
                table_batch_size=local_config.table_batch_size,
            )
            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                },
            )
            converter._get_pipeline(InputFormat.PDF)
            startup_s = time.perf_counter() - started
            status_queue.put(("ready", slot, startup_s))

            start_event.wait()
            conversion_started = time.perf_counter()
            shard_paths = [paths[index] for index in indices]
            conversions = (
                tuple(converter.convert_all(shard_paths)) if shard_paths else ()
            )
            conversion_wall_s = time.perf_counter() - conversion_started
            if len(conversions) != len(indices):
                raise RuntimeError("Docling native changed document cardinality")
            documents = [
                {"source_index": index, "document": _normalise_document(conversion)}
                for index, conversion in zip(indices, conversions, strict=True)
            ]

        finished_at = time.perf_counter()
        payload = {
            "gpu_slot": slot,
            "gpu_id": gpu_id,
            "source_indices": list(indices),
            "doc_count": len(indices),
            "page_count": sum(page_counts[index] for index in indices),
            "byte_count": sum(sizes[index] for index in indices),
            "startup_s": startup_s,
            "conversion_wall_s": conversion_wall_s,
            "finished_at": finished_at,
            "documents": documents,
        }
        _write_payload(result_path, payload)
        status_queue.put(("done", slot, result_path))
    except BaseException as error:
        status_queue.put(
            (
                "error",
                slot,
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        )


def _aggregate_worker_payloads(
    paths: Sequence[str],
    payloads: Iterable[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """校验 worker 输出并按最初 source index 聚合完整文档。"""

    records = tuple(sorted(payloads, key=lambda payload: payload["gpu_slot"]))
    indexed_documents: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        documents = record.get("documents")
        indices = record.get("source_indices")
        if not isinstance(documents, list) or not isinstance(indices, list):
            raise RuntimeError("worker result is missing document metadata")
        if record.get("doc_count") != len(documents) or len(indices) != len(documents):
            raise RuntimeError("worker document count does not match payload")
        for item in documents:
            if not isinstance(item, dict) or "source_index" not in item:
                raise RuntimeError("worker document is missing source_index")
            indexed_documents.append((item["source_index"], item["document"]))

    indexed_documents.sort(key=lambda item: item[0])
    expected_indices = list(range(len(paths)))
    if [index for index, _ in indexed_documents] != expected_indices:
        raise RuntimeError("worker results do not cover each input source exactly once")
    return (
        tuple(document for _, document in indexed_documents),
        records,
    )


def _terminate_processes(processes: Iterable[Any]) -> None:
    """异常路径上停止尚未退出的 worker，避免遗留子进程。"""

    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


def run_native_core_multi(
    paths: list[str],
    page_counts: Sequence[int],
    sizes: Sequence[int],
    config: NativeCoreConfig,
    gpu_ids: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> MultiGpuNativeRun:
    """运行四个单卡 Docling native worker，并恢复为初始 source 顺序。

    Parent 仅在所有 worker 对 converter pipeline 的初始化完成后释放同一个 start event，
    因而 ``measured_s`` 不包含单卡模型加载的 startup。任何 worker 的初始化、转换或
    序列化异常都会带 traceback 回传，并会终止剩余 worker。
    """

    _validate_inputs(paths, page_counts, sizes, config, gpu_ids)
    path_tuple = tuple(paths)
    page_count_tuple = tuple(page_counts)
    size_tuple = tuple(sizes)
    partitions = _partition_indices_lpt(page_count_tuple, size_tuple)
    context = mp.get_context("spawn")
    start_event = context.Event()
    status_queue = context.Queue()
    e2e_started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="rayorch-docling-native-multi-") as directory:
        processes = [
            context.Process(
                target=_native_multi_worker,
                args=(
                    slot,
                    gpu_id,
                    path_tuple,
                    page_count_tuple,
                    size_tuple,
                    partitions[slot],
                    config,
                    start_event,
                    status_queue,
                    str(Path(directory) / f"worker-{slot}.json.gz"),
                ),
                name=f"docling-native-gpu-{gpu_id}",
            )
            for slot, gpu_id in enumerate(gpu_ids)
        ]
        for process in processes:
            process.start()

        ready_slots: set[int] = set()
        done_paths: dict[int, str] = {}
        try:
            while len(ready_slots) < _WORKER_COUNT:
                try:
                    status, slot, value = status_queue.get(timeout=0.1)
                except queue.Empty:
                    if any(
                        process.exitcode is not None and process.exitcode != 0
                        for process in processes
                    ):
                        raise RuntimeError("a Docling native worker exited before ready")
                    continue
                if status == "error":
                    raise RuntimeError(
                        f"Docling native worker slot {slot} failed:\n"
                        f"{value['type']}: {value['message']}\n{value['traceback']}"
                    )
                if status == "ready":
                    ready_slots.add(slot)
                elif status == "done":
                    done_paths[slot] = value
                else:
                    raise RuntimeError(f"unknown worker status: {status!r}")

            startup_s = time.perf_counter() - e2e_started
            measured_started = time.perf_counter()
            start_event.set()

            while len(done_paths) < _WORKER_COUNT:
                try:
                    status, slot, value = status_queue.get(timeout=0.1)
                except queue.Empty:
                    if any(
                        process.exitcode is not None and process.exitcode != 0
                        for process in processes
                    ):
                        raise RuntimeError("a Docling native worker exited before completion")
                    continue
                if status == "error":
                    raise RuntimeError(
                        f"Docling native worker slot {slot} failed:\n"
                        f"{value['type']}: {value['message']}\n{value['traceback']}"
                    )
                if status == "done":
                    done_paths[slot] = value
                elif status != "ready":
                    raise RuntimeError(f"unknown worker status: {status!r}")

            payloads = [_read_payload(done_paths[slot]) for slot in range(_WORKER_COUNT)]
            documents, worker_records = _aggregate_worker_payloads(path_tuple, payloads)
            # 收到所有 done 即所有 worker 已完成 conversion、normalisation 和结果落盘；
            # 用 parent wall time 表示统一 start 后的真实 makespan。
            measured_s = time.perf_counter() - measured_started
            for process in processes:
                process.join()
            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError("a Docling native worker exited unsuccessfully")
        except BaseException:
            _terminate_processes(processes)
            raise

    return MultiGpuNativeRun(
        startup_s=startup_s,
        measured_s=measured_s,
        e2e_s=time.perf_counter() - e2e_started,
        documents=documents,
        worker_records=worker_records,
    )
