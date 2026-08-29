"""在 TaiJi Ray 集群运行多节点 H20 的 MinerU v3.6 workload。

PDF 输入由 MinerU actor 直接从 HDFS 读取；vLLM 模型缓存到 GPU 节点本地盘。
Reduce 将完整 batch 原子投递到节点本地 spool，每个 GPU 节点一个独立 uploader
进程异步保持原目录格式提交 HDFS；driver 仅在所有 backlog drain 后写全局成功标记。
所有 ``pyarrow`` 导入都留在 HDFS helper 的调用路径中，使本地纯单元测试不依赖
``libhdfs``。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlsplit

from ...multigrain_v3.benchmark.mineru import DEFAULT_FLASH_REPO
from ...multigrain_v3.benchmark.profile_events import (
    ProfileEventWriter,
    merge_profile_events,
    write_gpu_samples,
)
from .mineru import run_benchmark


EXPECTED_PDF_FILES = (2_000, 1_690)
EXPECTED_PDF_BYTES_BY_INPUT = (4_060_449_252, 1_099_397_628_624)
EXPECTED_PDFS = sum(EXPECTED_PDF_FILES)
EXPECTED_MODEL_FILES = 15
EXPECTED_MODEL_BYTES = 2_323_649_236
EXPECTED_GPU_WORKERS = int(os.environ.get("RAYORCH_EXPECTED_GPU_WORKERS", "8"))
GPUS_PER_WORKER = int(os.environ.get("RAYORCH_GPUS_PER_WORKER", "8"))
GPU_REPLICAS = EXPECTED_GPU_WORKERS * GPUS_PER_WORKER
TOPOLOGY_STABLE_SAMPLES = 2
DEFAULT_LOCAL_ROOT = "/tmp/rayorch-mineru-v3-6-taiji"
HDFS_DIRECT_READY_MARKER = "RAYORCH_MINERU_HDFS_DIRECT_READY "


@dataclass(frozen=True, slots=True)
class ClusterNode:
    """运行一次 node-affinity 操作所需的稳定 Ray 节点信息。"""

    node_id: str
    address: str
    gpus: float


class _GpuNodeSampler:
    """Node-local NVML sampler used to isolate one benchmark's utilization."""

    def __init__(self, expected_node_id: str, interval_s: float = 1.0) -> None:
        self.node_id = _assert_current_node(expected_node_id)
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        try:
            import pynvml  # pyright: ignore[reportMissingImports]

            pynvml.nvmlInit()
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            while not self._stop.is_set():
                sample = {
                    "time": time.time(),
                    "epoch_s": time.time(),
                    "monotonic_s": time.perf_counter(),
                    "utilization": [
                        int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                        for handle in handles
                    ],
                    "memory_used": [
                        int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                        for handle in handles
                    ],
                    "power_w": [
                        round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 3)
                        for handle in handles
                    ],
                }
                self.samples.append(sample)
                report_samples = max(1, round(30.0 / self.interval_s))
                if len(self.samples) % report_samples == 0:
                    recent = self.samples[-report_samples:]
                    values = [
                        int(value)
                        for item in recent
                        for value in item["utilization"]
                    ]
                    print(
                        "RAYORCH_MINERU_GPU_ROLLING "
                        + json.dumps(
                            {
                                "node_id": self.node_id,
                                "window_s": round(
                                    len(recent) * self.interval_s, 3
                                ),
                                "mean_gpu_utilization": round(
                                    sum(values) / len(values), 3
                                ),
                                "min_gpu_utilization": min(values),
                                "max_gpu_utilization": max(values),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                self._stop.wait(self.interval_s)
            pynvml.nvmlShutdown()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def ping(self) -> str:
        return self.node_id

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(10.0, self.interval_s * 3))
        return {
            "node_id": self.node_id,
            "samples": self.samples,
            "error": self.error,
        }


def _hdfs_document_committed(
    filesystem: Any,
    document_root: str,
    stem: str,
) -> bool:
    """Return whether one loose-file HDFS document is atomically committed."""

    markdown_path = posixpath.join(document_root, "vlm", f"{stem}.md")
    layout_path = posixpath.join(document_root, "vlm", "layout.json")
    success_path = posixpath.join(document_root, "_SUCCESS")
    try:
        infos = filesystem.get_file_info(
            [markdown_path, layout_path, success_path]
        )
        if not all(bool(getattr(info, "is_file", False)) for info in infos):
            return False
        with filesystem.open_input_file(
            markdown_path
        ) as source:
            source.read(1)
        with filesystem.open_input_file(
            layout_path
        ) as source:
            layout = json.loads(source.read().decode("utf-8"))
        with filesystem.open_input_file(
            success_path
        ) as source:
            commit = json.loads(source.read().decode("utf-8"))
    except Exception:
        return False
    return (
        isinstance(layout, dict)
        and isinstance(layout.get("pdf_info"), list)
        and isinstance(commit, dict)
        and commit.get("pdf") == stem
    )


def _upload_spool_batch_hdfs(
    batch_dir: str,
    hdfs_documents_uri: str,
) -> dict[str, Any]:
    """Upload one ready local batch and atomically publish each loose document."""

    local_batch = Path(batch_dir)
    manifest = json.loads(
        (local_batch / "manifest.json").read_text(encoding="utf-8")
    )
    batch_id = _safe_component(str(manifest["batch_id"]))
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"spool batch has no documents: {batch_dir}")
    _, filesystem, documents_root = _hdfs_filesystem(hdfs_documents_uri)
    run_root = posixpath.dirname(documents_root)
    filesystem.create_dir(documents_root, recursive=True)
    uploaded = 0
    skipped = 0
    copied_files = 0
    copied_bytes = 0
    for document in documents:
        stem = str(document["pdf"])
        if PurePosixPath(stem).name != stem or stem in {"", ".", ".."}:
            raise ValueError(f"unsafe PDF stem in spool manifest: {stem!r}")
        local_document = local_batch / "docs" / stem
        if not local_document.is_dir():
            raise FileNotFoundError(f"spooled document is missing: {local_document}")
        final_root = posixpath.join(documents_root, stem)
        if _hdfs_document_committed(filesystem, final_root, stem):
            skipped += 1
            continue
        staging_root = posixpath.join(
            run_root,
            "_upload_staging",
            f"{stem}.{batch_id}.{uuid.uuid4().hex}",
        )
        filesystem.create_dir(staging_root, recursive=True)
        try:
            local_files = tuple(
                path
                for path in sorted(local_document.rglob("*"))
                if path.is_file()
            )
            for local_path in local_files:
                relative = local_path.relative_to(local_document).as_posix()
                remote_path = posixpath.join(staging_root, relative)
                filesystem.create_dir(
                    posixpath.dirname(remote_path), recursive=True
                )
                with local_path.open("rb") as source:
                    with filesystem.open_output_stream(remote_path) as output:
                        _copy_stream(source, output)
                copied_files += 1
                copied_bytes += local_path.stat().st_size
            try:
                filesystem.move(staging_root, final_root)
            except Exception:
                if not _hdfs_document_committed(filesystem, final_root, stem):
                    raise
                filesystem.delete_dir(staging_root)
            uploaded += 1
        except BaseException:
            try:
                filesystem.delete_dir(staging_root)
            except Exception:
                pass
            raise
    batches_root = posixpath.join(run_root, "_batches")
    filesystem.create_dir(batches_root, recursive=True)
    manifest_path = posixpath.join(batches_root, f"{batch_id}.json")
    temporary_path = manifest_path + f".{uuid.uuid4().hex}.tmp"
    with filesystem.open_output_stream(temporary_path) as output:
        output.write(
            (
                json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
        )
    try:
        filesystem.move(temporary_path, manifest_path)
    except Exception:
        try:
            filesystem.delete_file(temporary_path)
        except Exception:
            pass
    return {
        "batch_id": batch_id,
        "documents": len(documents),
        "uploaded": uploaded,
        "skipped": skipped,
        "files": copied_files,
        "bytes": copied_bytes,
    }


def _ceph_document_committed(document_root: Path, stem: str) -> bool:
    try:
        layout = json.loads(
            (document_root / "vlm" / "layout.json").read_text(encoding="utf-8")
        )
        commit = json.loads(
            (document_root / "_SUCCESS").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        (document_root / "vlm" / f"{stem}.md").is_file()
        and isinstance(layout, dict)
        and isinstance(layout.get("pdf_info"), list)
        and isinstance(commit, dict)
        and commit.get("pdf") == stem
    )


def _upload_spool_batch_ceph(
    batch_dir: str,
    ceph_documents_dir: str,
) -> dict[str, Any]:
    """Upload one ready batch into a shared Ceph mount using atomic renames."""

    local_batch = Path(batch_dir)
    manifest = json.loads(
        (local_batch / "manifest.json").read_text(encoding="utf-8")
    )
    batch_id = _safe_component(str(manifest["batch_id"]))
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"spool batch has no documents: {batch_dir}")
    documents_root = Path(ceph_documents_dir)
    run_root = documents_root.parent
    documents_root.mkdir(parents=True, exist_ok=True)
    uploaded = 0
    skipped = 0
    copied_files = 0
    copied_bytes = 0
    for document in documents:
        stem = str(document["pdf"])
        if PurePosixPath(stem).name != stem or stem in {"", ".", ".."}:
            raise ValueError(f"unsafe PDF stem in spool manifest: {stem!r}")
        local_document = local_batch / "docs" / stem
        if not local_document.is_dir():
            raise FileNotFoundError(f"spooled document is missing: {local_document}")
        final_root = documents_root / stem
        if _ceph_document_committed(final_root, stem):
            skipped += 1
            continue
        staging_root = (
            run_root
            / "_upload_staging"
            / f"{stem}.{batch_id}.{uuid.uuid4().hex}"
        )
        staging_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(local_document, staging_root)
            local_files = tuple(
                path for path in staging_root.rglob("*") if path.is_file()
            )
            copied_files += len(local_files)
            copied_bytes += sum(path.stat().st_size for path in local_files)
            try:
                staging_root.replace(final_root)
            except OSError:
                if not _ceph_document_committed(final_root, stem):
                    raise
                shutil.rmtree(staging_root, ignore_errors=True)
            uploaded += 1
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
    batches_root = run_root / "_batches"
    batches_root.mkdir(parents=True, exist_ok=True)
    manifest_path = batches_root / f"{batch_id}.json"
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "batch_id": batch_id,
        "documents": len(documents),
        "uploaded": uploaded,
        "skipped": skipped,
        "files": copied_files,
        "bytes": copied_bytes,
    }


def _upload_spool_batch(
    batch_dir: str,
    documents_target: str,
) -> dict[str, Any]:
    if urlsplit(documents_target).scheme == "hdfs":
        return _upload_spool_batch_hdfs(batch_dir, documents_target)
    if Path(documents_target).is_absolute():
        return _upload_spool_batch_ceph(batch_dir, documents_target)
    raise ValueError(f"unsupported spool output target: {documents_target}")


class _NodeSpoolUploader:
    """One node-local process supervising a bounded pool of HDFS uploads."""

    def __init__(
        self,
        expected_node_id: str,
        spool_root: str,
        documents_target: str,
        workers: int = 4,
        poll_s: float = 0.25,
        profile_dir: str | None = None,
        profile_system: str = "rayorch",
    ) -> None:
        init_epoch_s = time.time()
        init_monotonic_s = time.perf_counter()
        self.node_id = _assert_current_node(expected_node_id)
        self._profile = ProfileEventWriter(
            profile_dir,
            system=profile_system,
            stage="async_upload",
            role="write",
        )
        self.spool_root = Path(spool_root)
        self.ready = self.spool_root / "ready"
        self.uploading = self.spool_root / "uploading"
        self.failed = self.spool_root / "failed"
        for path in (self.ready, self.uploading, self.failed):
            path.mkdir(parents=True, exist_ok=True)
        for path in tuple(self.uploading.iterdir()):
            if path.is_dir():
                target = self.ready / path.name
                if not target.exists():
                    path.replace(target)
        self.documents_target = documents_target
        self.workers = workers
        self.poll_s = poll_s
        self._attempts: dict[str, int] = {}
        self._completed: list[dict[str, Any]] = []
        self._active: dict[
            concurrent.futures.Future[dict[str, Any]], Path
        ] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="rayorch-hdfs-upload",
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._profile.actor_ready(
            started_epoch_s=init_epoch_s,
            started_monotonic_s=init_monotonic_s,
        )

    def _upload_profiled(self, claimed: Path) -> dict[str, Any]:
        started_epoch_s = time.time()
        started_monotonic_s = time.perf_counter()
        result = _upload_spool_batch(str(claimed), self.documents_target)
        self._profile.stage_batch(
            started_epoch_s=started_epoch_s,
            started_monotonic_s=started_monotonic_s,
            items=int(result.get("documents") or 0),
            batch_size=int(result.get("documents") or 0),
            batch_id=str(result.get("batch_id") or claimed.name),
        )
        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                completed = [
                    future for future in self._active if future.done()
                ]
                for future in completed:
                    claimed = self._active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        attempts = self._attempts.get(claimed.name, 0) + 1
                        self._attempts[claimed.name] = attempts
                        destination = (
                            self.failed if attempts >= 3 else self.ready
                        ) / claimed.name
                        if not destination.exists():
                            claimed.replace(destination)
                        if attempts >= 3:
                            (destination / "upload-error.txt").write_text(
                                f"{type(error).__name__}: {error}\n",
                                encoding="utf-8",
                            )
                    else:
                        self._completed.append(result)
                        shutil.rmtree(claimed, ignore_errors=True)
                capacity = self.workers - len(self._active)
                for candidate in sorted(self.ready.iterdir())[:capacity]:
                    if not candidate.is_dir():
                        continue
                    claimed = self.uploading / candidate.name
                    try:
                        candidate.replace(claimed)
                    except FileNotFoundError:
                        continue
                    future = self._pool.submit(self._upload_profiled, claimed)
                    self._active[future] = claimed
            self._stop.wait(self.poll_s)

    def ping(self) -> str:
        return self.node_id

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.node_id,
                "ready": sum(path.is_dir() for path in self.ready.iterdir()),
                "uploading": len(self._active),
                "failed": sum(path.is_dir() for path in self.failed.iterdir()),
                "staging": sum(
                    path.is_dir()
                    for path in (self.spool_root / "staging").iterdir()
                ) if (self.spool_root / "staging").is_dir() else 0,
                "completed_batches": len(self._completed),
                "completed_documents": sum(
                    int(item["documents"]) for item in self._completed
                ),
            }

    def drain(self, timeout_s: float = 7200) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.status()
            if status["failed"]:
                raise RuntimeError(f"node spool upload failed: {status}")
            if not any(
                status[key] for key in ("ready", "uploading", "staging")
            ):
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"node spool drain timed out: {status}")
            time.sleep(self.poll_s)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=10)
        self._pool.shutdown(wait=False, cancel_futures=False)
        return self.status()


def _load_pyarrow_fs():
    """延迟导入 PyArrow filesystem，避免模块导入触发 ``libhdfs``。"""

    from pyarrow import fs as pyarrow_fs  # pyright: ignore[reportMissingImports]

    return pyarrow_fs


def _hdfs_filesystem(uri: str):
    """返回 HDFS URI 对应的 PyArrow filesystem 和无 scheme 路径。"""

    if urlsplit(uri).scheme != "hdfs":
        raise ValueError(f"expected an hdfs:// URI, got: {uri}")
    pyarrow_fs = _load_pyarrow_fs()
    filesystem, path = pyarrow_fs.FileSystem.from_uri(uri)
    return pyarrow_fs, filesystem, path.rstrip("/") or "/"


def _remote_file_infos(uri: str):
    """递归列出一个 HDFS 文件或目录下的普通文件。"""

    pyarrow_fs, filesystem, root = _hdfs_filesystem(uri)
    root_info = filesystem.get_file_info(root)
    if root_info.type == pyarrow_fs.FileType.NotFound:
        raise FileNotFoundError(f"HDFS path does not exist: {uri}")
    if root_info.type == pyarrow_fs.FileType.File:
        return pyarrow_fs, filesystem, root, (root_info,)
    if root_info.type != pyarrow_fs.FileType.Directory:
        raise ValueError(f"HDFS path is not a file or directory: {uri}")
    selector = pyarrow_fs.FileSelector(root, recursive=True)
    infos = tuple(
        info
        for info in filesystem.get_file_info(selector)
        if info.type == pyarrow_fs.FileType.File
    )
    return pyarrow_fs, filesystem, root, infos


def _copy_stream(source: Any, destination: Any) -> None:
    """以固定内存窗口复制本地或 HDFS 文件流。"""

    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)


def _reset_tmp_directory(path: str) -> Path:
    """安全地清空一个明确位于 ``/tmp`` 下的节点本地目录。"""

    target = Path(path).absolute()
    resolved = target.resolve(strict=False)
    tmp_root = Path("/tmp").resolve()
    if resolved == tmp_root or tmp_root not in resolved.parents:
        raise ValueError(f"node-local directory must be below /tmp: {path}")
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target


def _download_hdfs_tree(hdfs_uri: str, local_dir: str) -> dict[str, Any]:
    """保持相对路径，把一个非空 HDFS 文件树下载到节点本地。"""

    _, filesystem, root, infos = _remote_file_infos(hdfs_uri)
    if not infos:
        raise FileNotFoundError(f"no model files found under {hdfs_uri}")
    destination = Path(local_dir)
    copied_bytes = 0
    root_is_file = len(infos) == 1 and infos[0].path == root
    for info in sorted(infos, key=lambda item: item.path):
        relative = (
            PurePosixPath(info.path).name
            if root_is_file
            else posixpath.relpath(info.path, root)
        )
        if relative == ".." or relative.startswith("../"):
            raise ValueError(f"HDFS entry escapes source root: {info.path}")
        local_path = destination / Path(relative)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with filesystem.open_input_file(info.path) as source:
            with local_path.open("wb") as output:
                _copy_stream(source, output)
        copied_bytes += local_path.stat().st_size
    return {"files": len(infos), "bytes": copied_bytes}


def _upload_tree(local_dir: str, hdfs_uri: str) -> dict[str, Any]:
    """把一个节点本地目录递归上传到指定 HDFS 唯一目录。"""

    source_root = Path(local_dir)
    if not source_root.is_dir():
        raise NotADirectoryError(f"local upload directory does not exist: {local_dir}")
    _, filesystem, remote_root = _hdfs_filesystem(hdfs_uri)
    filesystem.create_dir(remote_root, recursive=True)
    files = tuple(
        path for path in sorted(source_root.rglob("*")) if path.is_file()
    )
    copied_bytes = 0
    for local_path in files:
        relative = local_path.relative_to(source_root).as_posix()
        remote_path = posixpath.join(remote_root, relative)
        filesystem.create_dir(posixpath.dirname(remote_path), recursive=True)
        with local_path.open("rb") as source:
            with filesystem.open_output_stream(remote_path) as output:
                _copy_stream(source, output)
        copied_bytes += local_path.stat().st_size
    return {"files": len(files), "bytes": copied_bytes, "uri": hdfs_uri}


def _node_id_text(value: Any) -> str:
    """兼容字符串、bytes 和 Ray NodeID 对象的稳定文本表示。"""

    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.hex()
    hex_method = getattr(value, "hex", None)
    if callable(hex_method):
        return str(hex_method())
    return str(value)


def _assert_current_node(expected_node_id: str) -> str:
    """确认 hard node-affinity task 的实际执行节点。"""

    import ray  # pyright: ignore[reportMissingImports]

    actual = _node_id_text(ray.get_runtime_context().get_node_id())
    if actual != expected_node_id:
        raise RuntimeError(
            f"node-affinity violation: expected {expected_node_id}, got {actual}"
        )
    return actual


def _mount_ceph_node(
    expected_node_id: str,
    token: str,
    app_group: str,
    location: str,
    output_base: str,
    run_dir: str,
) -> dict[str, Any]:
    """Mount the requested Ceph region without exposing the PAT in logs/state."""

    node_id = _assert_current_node(expected_node_id)
    if not token:
        raise RuntimeError("TaiJi PAT token is empty")

    def run(command: list[str]) -> None:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode:
            details = (result.stderr or result.stdout or "").replace(
                token, "***REDACTED***"
            )
            raise RuntimeError(
                f"Ceph mount command failed with exit {result.returncode}: "
                f"{details[-1000:]}"
            )

    run(["sudo", "taiji_client", "update"])
    run(
        [
            "sudo",
            "taiji_client",
            "mount",
            "-tk",
            token,
            "-bf",
            app_group,
            "-l",
            location,
        ]
    )
    root = Path(output_base)
    if not root.is_dir():
        raise RuntimeError(f"mounted Ceph output base is unavailable: {output_base}")
    run(["sudo", "mkdir", "-p", run_dir])
    run(["sudo", "chown", f"{os.getuid()}:{os.getgid()}", run_dir])
    run(["sudo", "chmod", "0775", run_dir])
    writable_root = Path(run_dir)
    probe = writable_root / f".rayorch-mount-probe-{node_id}-{uuid.uuid4().hex}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)
    return {
        "node_id": node_id,
        "output_base": output_base,
        "run_dir": run_dir,
        "app_group": app_group,
        "location": location,
        "mounted": True,
    }


def _prepare_ceph_run_root(run_dir: str) -> None:
    root = Path(run_dir)
    if (root / "_SUCCESS").is_file():
        raise FileExistsError(f"Ceph run is already complete: {run_dir}")
    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root / "_upload_staging", ignore_errors=True)
    for name in ("docs", "_batches"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _validate_ceph_outputs(
    run_dir: str,
    *,
    expected_documents: int,
) -> dict[str, Any]:
    root = Path(run_dir)
    docs = root / "docs"
    markdown = tuple(docs.glob("*/vlm/*.md"))
    layouts = tuple(docs.glob("*/vlm/layout.json"))
    success = tuple(docs.glob("*/_SUCCESS"))
    markdown_stems = {path.stem for path in markdown}
    layout_stems = {path.parent.parent.name for path in layouts}
    success_stems = {path.parent.name for path in success}
    errors = []
    for label, values in (
        ("Markdown", markdown_stems),
        ("layout", layout_stems),
        ("document success", success_stems),
    ):
        if len(values) != expected_documents:
            errors.append(f"{label} count={len(values)}")
    if not (markdown_stems == layout_stems == success_stems):
        errors.append("Markdown/layout/success stem sets differ")
    staging = tuple((root / "_upload_staging").rglob("*")) if (
        root / "_upload_staging"
    ).exists() else ()
    if any(path.is_file() for path in staging):
        errors.append("Ceph upload staging is not empty")
    if errors:
        raise RuntimeError("Ceph result validation failed: " + "; ".join(errors))
    semantic_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(success)
    ]
    output_digest = hashlib.sha256(
        json.dumps(
            semantic_payloads,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "markdown": len(markdown_stems),
        "layout": len(layout_stems),
        "document_success": len(success_stems),
        "unique_documents": len(markdown_stems),
        "staging_files": sum(path.is_file() for path in staging),
        "output_semantic_digest": output_digest,
    }


def _write_ceph_success(run_dir: str, payload: dict[str, Any]) -> str:
    root = Path(run_dir)
    destination = root / "_SUCCESS"
    temporary = root / f"._SUCCESS.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def _prepare_ceph_run_root_node(
    expected_node_id: str,
    run_dir: str,
) -> dict[str, Any]:
    node_id = _assert_current_node(expected_node_id)
    _prepare_ceph_run_root(run_dir)
    return {"node_id": node_id, "run_dir": run_dir, "prepared": True}


def _validate_ceph_outputs_node(
    expected_node_id: str,
    run_dir: str,
    expected_documents: int,
) -> dict[str, Any]:
    node_id = _assert_current_node(expected_node_id)
    return {
        "node_id": node_id,
        **_validate_ceph_outputs(
            run_dir,
            expected_documents=expected_documents,
        ),
    }


def _commit_ceph_driver_artifacts_node(
    expected_node_id: str,
    run_dir: str,
    artifact_files: dict[str, bytes],
    wrapper_payload: dict[str, Any],
) -> dict[str, Any]:
    node_id = _assert_current_node(expected_node_id)
    artifact_root = Path(run_dir) / "driver" / "artifacts"
    copied_bytes = 0
    for relative, content in artifact_files.items():
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe artifact path: {relative}")
        target = artifact_root / Path(relative_path.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        copied_bytes += len(content)
    driver_upload = {
        "files": len(artifact_files),
        "bytes": copied_bytes,
        "uri": str(artifact_root),
    }
    payload = {**wrapper_payload, "driver_upload": driver_upload}
    return {
        "node_id": node_id,
        "driver_upload": driver_upload,
        "success_uri": _write_ceph_success(run_dir, payload),
    }


def _finalize_ceph_profile_node(
    expected_node_id: str,
    profile_dir: str,
    driver_event_files: dict[str, bytes],
    raw_gpu_records: list[dict[str, Any]],
    clock: dict[str, Any],
    fairness: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Merge a profile on a mounted GPU node, not the unmounted Job driver."""

    node_id = _assert_current_node(expected_node_id)
    root = Path(profile_dir)
    raw_root = root / "raw_events"
    raw_root.mkdir(parents=True, exist_ok=True)
    for name, content in driver_event_files.items():
        safe_name = PurePosixPath(name)
        if safe_name.name != name or name in {"", ".", ".."}:
            raise ValueError(f"unsafe driver profile event name: {name}")
        (raw_root / f"driver-{name}").write_bytes(content)
    merged = merge_profile_events(
        profile_dir,
        root / "profile_events.jsonl",
    )
    gpu_artifact = write_gpu_samples(
        raw_gpu_records,
        root / "gpu_samples.jsonl",
    )
    for name, payload in (
        ("clock.json", clock),
        ("fairness.json", fairness),
        ("config.json", config),
        ("metrics.json", metrics),
    ):
        (root / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return {
        "node_id": node_id,
        "directory": profile_dir,
        "events": merged,
        "gpu_samples": gpu_artifact,
        "clock": clock,
        "fairness": fairness,
        "metrics": metrics,
    }


def _prepare_hdfs_pdf_inputs_node(
    expected_node_id: str,
    hdfs_pdf_uris: Sequence[str],
    expected_pdf_files: Sequence[int],
    expected_pdf_bytes: Sequence[int],
    spool_root: str | None = None,
) -> dict[str, Any]:
    """确认一个 GPU 节点可直读全部 HDFS 输入。"""

    node_id = _assert_current_node(expected_node_id)
    if spool_root:
        root = Path(spool_root).absolute()
        tmp_root = Path("/tmp").resolve()
        if root.resolve(strict=False) == tmp_root or tmp_root not in root.resolve(
            strict=False
        ).parents:
            raise ValueError(f"node spool must be below /tmp: {spool_root}")
        shutil.rmtree(root / "staging", ignore_errors=True)
        for name in ("staging", "ready", "uploading", "failed"):
            (root / name).mkdir(parents=True, exist_ok=True)
    if not (
        len(hdfs_pdf_uris)
        == len(expected_pdf_files)
        == len(expected_pdf_bytes)
        > 0
    ):
        raise ValueError("PDF input URI/file/byte contracts must have equal lengths")

    manifests = []
    stems = []
    for uri, expected_files, expected_bytes in zip(
        hdfs_pdf_uris,
        expected_pdf_files,
        expected_pdf_bytes,
        strict=True,
    ):
        _, filesystem, _, infos = _remote_file_infos(uri)
        pdf_infos = tuple(
            info
            for info in infos
            if PurePosixPath(info.path).suffix.lower() == ".pdf"
        )
        manifest = {
            "uri": uri,
            "files": len(pdf_infos),
            "bytes": sum(int(info.size) for info in pdf_infos),
        }
        if manifest["files"] != expected_files:
            raise RuntimeError(
                f"expected {expected_files} directly readable HDFS PDFs under "
                f"{uri}, got {manifest['files']} on node {node_id}"
            )
        if manifest["bytes"] != expected_bytes:
            raise RuntimeError(
                f"expected {expected_bytes} directly readable HDFS PDF bytes "
                f"under {uri}, got {manifest['bytes']} on node {node_id}"
            )
        first_pdf = min(pdf_infos, key=lambda info: info.path)
        with filesystem.open_input_file(first_pdf.path) as source:
            header = source.read(5)
        if header != b"%PDF-":
            raise RuntimeError(
                "HDFS direct-read probe returned an invalid PDF header on node "
                f"{node_id}: {header!r}"
            )
        manifest["sample_pdf"] = first_pdf.path
        manifest["sample_header"] = header.decode("ascii")
        manifests.append(manifest)
        stems.extend(PurePosixPath(info.path).stem for info in pdf_infos)

    duplicate_stems = sorted(
        stem for stem in set(stems) if stems.count(stem) > 1
    )
    if duplicate_stems:
        raise ValueError(
            "combined HDFS PDF inputs contain duplicate output stems: "
            + ", ".join(duplicate_stems[:5])
        )
    return {
        "node_id": node_id,
        "mode": "direct_hdfs",
        "files": sum(item["files"] for item in manifests),
        "bytes": sum(item["bytes"] for item in manifests),
        "inputs": manifests,
    }


def _stage_model_node(
    expected_node_id: str,
    hdfs_model_uri: str,
    local_model_dir: str,
    expected_model_files: int,
    expected_model_bytes: int,
) -> dict[str, Any]:
    """只在一个固定 GPU 节点准备 MinerU 模型。"""

    node_id = _assert_current_node(expected_node_id)
    cached_root = Path(local_model_dir)
    cached_files = (
        tuple(path for path in cached_root.rglob("*") if path.is_file())
        if cached_root.is_dir()
        else ()
    )
    cached_bytes = sum(path.stat().st_size for path in cached_files)
    if (
        len(cached_files) == expected_model_files
        and cached_bytes == expected_model_bytes
    ):
        return {
            "node_id": node_id,
            "files": len(cached_files),
            "bytes": cached_bytes,
            "cached": True,
        }
    _reset_tmp_directory(local_model_dir)
    manifest = _download_hdfs_tree(hdfs_model_uri, local_model_dir)
    if manifest["files"] != expected_model_files:
        raise RuntimeError(
            f"expected {expected_model_files} staged model files, got "
            f"{manifest['files']} on node {node_id}"
        )
    if manifest["bytes"] != expected_model_bytes:
        raise RuntimeError(
            f"expected {expected_model_bytes} staged model bytes, got "
            f"{manifest['bytes']} on node {node_id}"
        )
    return {"node_id": node_id, **manifest, "cached": False}


def _cluster_nodes(records: Iterable[dict[str, Any]]) -> tuple[ClusterNode, ...]:
    """从 ``ray.nodes()`` 选择 alive 且拥有可调度资源的节点。"""

    nodes: list[ClusterNode] = []
    for record in records:
        if not bool(record.get("Alive")):
            continue
        resources = record.get("Resources") or {}
        if not any(float(value) > 0 for value in resources.values()):
            continue
        node_id = _node_id_text(record.get("NodeID", "")).strip()
        if not node_id:
            raise RuntimeError("alive Ray node is missing NodeID")
        nodes.append(
            ClusterNode(
                node_id=node_id,
                address=str(record.get("NodeManagerAddress", "")),
                gpus=float(resources.get("GPU", 0.0)),
            )
        )
    if not nodes:
        raise RuntimeError("Ray cluster has no alive schedulable nodes")
    return tuple(sorted(nodes, key=lambda node: node.node_id))


def _validate_gpu_topology(
    nodes: Sequence[ClusterNode],
    *,
    expected_workers: int = EXPECTED_GPU_WORKERS,
    expected_gpus_per_worker: int = GPUS_PER_WORKER,
) -> tuple[ClusterNode, ...]:
    """强制集群恰好包含两个各 8 GPU 的 worker。"""

    gpu_nodes = tuple(node for node in nodes if node.gpus > 0)
    observed = tuple(node.gpus for node in gpu_nodes)
    if len(gpu_nodes) != expected_workers or any(
        gpu_count != float(expected_gpus_per_worker)
        for gpu_count in observed
    ):
        raise RuntimeError(
            "expected exactly "
            f"{expected_workers} GPU workers × {expected_gpus_per_worker} GPUs, "
            f"observed {len(gpu_nodes)} GPU workers with GPU resources {observed}"
        )
    return gpu_nodes


def _wait_for_gpu_topology(
    ray_module: Any,
    *,
    timeout_s: float,
    poll_s: float,
    stable_samples: int = TOPOLOGY_STABLE_SAMPLES,
) -> tuple[tuple[ClusterNode, ...], tuple[ClusterNode, ...]]:
    """等待同一双节点 GPU 拓扑连续出现至少两次，避开 worker 注册竞态。"""

    if timeout_s < 0:
        raise ValueError("topology timeout must be non-negative")
    if poll_s < 0:
        raise ValueError("topology poll interval must be non-negative")
    if stable_samples < 2:
        raise ValueError("topology stable samples must be at least 2")

    deadline = time.monotonic() + timeout_s
    previous_signature: tuple[tuple[str, float], ...] | None = None
    consecutive = 0
    last_error = "no topology observation"
    while True:
        try:
            nodes = _cluster_nodes(ray_module.nodes())
            gpu_nodes = _validate_gpu_topology(nodes)
        except RuntimeError as error:
            previous_signature = None
            consecutive = 0
            last_error = str(error)
        else:
            signature = tuple((node.node_id, node.gpus) for node in gpu_nodes)
            if signature == previous_signature:
                consecutive += 1
            else:
                previous_signature = signature
                consecutive = 1
            if consecutive >= stable_samples:
                return nodes, gpu_nodes
            last_error = (
                f"valid topology observed {consecutive}/{stable_samples} "
                "consecutive times"
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "timed out waiting for stable GPU topology: " + last_error
            )
        time.sleep(min(poll_s, remaining))


def _node_affinity_strategy():
    """延迟导入 Ray 的 hard node-affinity scheduling strategy。"""

    # Ray is an optional dependency for import-only local unit tests.
    from ray.util.scheduling_strategies import (
        NodeAffinitySchedulingStrategy,
    )

    return NodeAffinitySchedulingStrategy


def _run_pinned_tasks(
    ray_module: Any,
    task: Callable[..., dict[str, Any]],
    calls: Sequence[tuple[ClusterNode, tuple[Any, ...]]],
) -> list[dict[str, Any]]:
    """并行运行一组 hard-affinity node task 并等待全部成功。"""

    strategy = _node_affinity_strategy()
    remote_task = ray_module.remote(num_cpus=0)(task)
    refs = [
        remote_task.options(
            scheduling_strategy=strategy(node_id=node.node_id, soft=False)
        ).remote(node.node_id, *arguments)
        for node, arguments in calls
    ]
    return list(ray_module.get(refs))


def _start_gpu_samplers(
    ray_module: Any,
    gpu_nodes: Sequence[ClusterNode],
    *,
    interval_s: float,
) -> list[Any]:
    strategy = _node_affinity_strategy()
    actor_class = ray_module.remote(num_cpus=0)(_GpuNodeSampler)
    actors = [
        actor_class.options(
            scheduling_strategy=strategy(node_id=node.node_id, soft=False)
        ).remote(node.node_id, interval_s)
        for node in gpu_nodes
    ]
    actual = ray_module.get([actor.ping.remote() for actor in actors])
    if actual != [node.node_id for node in gpu_nodes]:
        raise RuntimeError(f"GPU sampler node-affinity mismatch: {actual}")
    return actors


def _start_spool_uploaders(
    ray_module: Any,
    gpu_nodes: Sequence[ClusterNode],
    *,
    spool_root: str,
    documents_target: str,
    workers_per_node: int,
    profile_dir: str | None = None,
    profile_system: str = "rayorch",
) -> list[Any]:
    """Start exactly one node-affine uploader process on every GPU worker."""

    strategy = _node_affinity_strategy()
    actor_class = ray_module.remote(num_cpus=1)(_NodeSpoolUploader)
    actors = [
        actor_class.options(
            scheduling_strategy=strategy(node_id=node.node_id, soft=False)
        ).remote(
            node.node_id,
            spool_root,
            documents_target,
            workers_per_node,
            profile_dir,
            profile_system,
        )
        for node in gpu_nodes
    ]
    actual = ray_module.get([actor.ping.remote() for actor in actors])
    if actual != [node.node_id for node in gpu_nodes]:
        raise RuntimeError(f"spool uploader node-affinity mismatch: {actual}")
    return actors


def _drain_spool_uploaders(
    ray_module: Any,
    actors: Sequence[Any],
    *,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Wait until every node has no staging, ready, uploading, or failed batch."""

    references = [actor.drain.remote(timeout_s) for actor in actors]
    return list(ray_module.get(references, timeout=timeout_s + 60))


def _stop_spool_uploaders(ray_module: Any, actors: Sequence[Any]) -> None:
    for actor in actors:
        try:
            ray_module.get(actor.stop.remote(), timeout=30)
        except Exception:
            pass
        try:
            ray_module.kill(actor, no_restart=True)
        except Exception:
            pass


def _summarize_gpu_samples(
    records: Sequence[dict[str, Any]],
    *,
    expected_gpus: int,
    measured_start_time: float | None = None,
    measured_end_time: float | None = None,
) -> dict[str, Any]:
    buckets: dict[int, list[int]] = {}
    memory_peak = [0] * expected_gpus
    errors = []
    for record in records:
        if record.get("error"):
            errors.append(
                {"node_id": record.get("node_id"), "error": record["error"]}
            )
        for sample in record.get("samples") or []:
            bucket = int(float(sample["time"]))
            buckets.setdefault(bucket, []).extend(
                int(value) for value in sample["utilization"]
            )
            for index, value in enumerate(sample["memory_used"]):
                if index < expected_gpus:
                    memory_peak[index] = max(memory_peak[index], int(value))
    minimum_values = max(1, int(expected_gpus * 0.75))
    timed_means = [
        (timestamp, sum(values) / len(values))
        for timestamp, values in sorted(buckets.items())
        if len(values) >= minimum_values
    ]
    means = [value for _, value in timed_means]
    if not means:
        return {
            "sample_buckets": 0,
            "overall_mean_gpu_utilization": 0.0,
            "active_mean_gpu_utilization": 0.0,
            "errors": errors or ["no complete GPU sample buckets"],
        }
    active_indexes = [index for index, value in enumerate(means) if value >= 5.0]
    active = (
        means[active_indexes[0] : active_indexes[-1] + 1]
        if active_indexes
        else []
    )
    ordered = sorted(active or means)
    percentile = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    measured = [
        value
        for timestamp, value in timed_means
        if measured_start_time is not None
        and measured_end_time is not None
        and measured_start_time <= timestamp <= measured_end_time
    ]
    measured_ordered = sorted(measured)
    measured_percentile = lambda fraction: measured_ordered[
        round((len(measured_ordered) - 1) * fraction)
    ]
    measured_quartiles = [
        measured[index * len(measured) // 4 : (index + 1) * len(measured) // 4]
        for index in range(4)
    ]
    trim = len(measured) // 10
    measured_central = measured[trim : len(measured) - trim] if trim else measured
    return {
        "sample_buckets": len(means),
        "active_sample_buckets": len(active),
        "overall_mean_gpu_utilization": round(sum(means) / len(means), 3),
        "active_mean_gpu_utilization": round(
            sum(active) / len(active) if active else 0.0, 3
        ),
        "active_p50_gpu_utilization": round(percentile(0.5), 3),
        "active_p90_gpu_utilization": round(percentile(0.9), 3),
        "active_peak_gpu_utilization": round(max(ordered), 3),
        "measured_sample_buckets": len(measured),
        "measured_mean_gpu_utilization": round(
            sum(measured) / len(measured) if measured else 0.0, 3
        ),
        "measured_p50_gpu_utilization": round(
            measured_percentile(0.5) if measured else 0.0, 3
        ),
        "measured_p90_gpu_utilization": round(
            measured_percentile(0.9) if measured else 0.0, 3
        ),
        "measured_peak_gpu_utilization": round(
            max(measured_ordered) if measured else 0.0, 3
        ),
        "measured_fraction_at_least_90": round(
            sum(value >= 90.0 for value in measured) / len(measured)
            if measured
            else 0.0,
            6,
        ),
        "measured_quartile_mean_gpu_utilization": [
            round(sum(values) / len(values), 3) if values else 0.0
            for values in measured_quartiles
        ],
        "measured_central_80_mean_gpu_utilization": round(
            sum(measured_central) / len(measured_central)
            if measured_central
            else 0.0,
            3,
        ),
        "memory_peak_bytes": memory_peak,
        "errors": errors,
    }


def _stop_gpu_samplers(
    ray_module: Any,
    actors: Sequence[Any],
    *,
    expected_gpus: int,
    measured_start_time: float | None = None,
    measured_end_time: float | None = None,
    raw_records_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = []
    for actor in actors:
        try:
            records.append(ray_module.get(actor.stop.remote(), timeout=30))
        except Exception as error:
            records.append(
                {
                    "node_id": None,
                    "samples": [],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        try:
            ray_module.kill(actor, no_restart=True)
        except Exception:
            pass
    if raw_records_out is not None:
        raw_records_out.extend(records)
    return _summarize_gpu_samples(
        records,
        expected_gpus=expected_gpus,
        measured_start_time=measured_start_time,
        measured_end_time=measured_end_time,
    )


def _hdfs_join(root: str, *parts: str) -> str:
    """在保留 HDFS authority 的前提下连接 URI path。"""

    return root.rstrip("/") + "/" + "/".join(
        part.strip("/") for part in parts if part.strip("/")
    )


def _safe_component(value: str) -> str:
    """把 job/node 标识收敛为一个安全的 HDFS path component。"""

    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not component:
        raise ValueError(f"empty path component after sanitizing: {value!r}")
    return component


def _prepare_hdfs_run_root(run_uri: str) -> None:
    """Create a new run or safely resume one without a global success marker."""

    pyarrow_fs, filesystem, path = _hdfs_filesystem(run_uri)
    info = filesystem.get_file_info(path)
    if info.type == pyarrow_fs.FileType.NotFound:
        filesystem.create_dir(path, recursive=True)
        return
    if info.type != pyarrow_fs.FileType.Directory:
        raise FileExistsError(f"HDFS run path is not a directory: {run_uri}")
    success = filesystem.get_file_info(posixpath.join(path, "_SUCCESS"))
    if success.type == pyarrow_fs.FileType.File:
        raise FileExistsError(f"HDFS run is already complete: {run_uri}")
    for temporary_name in ("_staging", "_upload_staging"):
        temporary = posixpath.join(path, temporary_name)
        temporary_info = filesystem.get_file_info(temporary)
        if temporary_info.type == pyarrow_fs.FileType.Directory:
            filesystem.delete_dir(temporary)


def _validate_hdfs_outputs(
    run_uri: str,
    *,
    expected_documents: int = EXPECTED_PDFS,
) -> dict[str, Any]:
    """校验 HDFS Markdown/layout 一一配对且没有未完成事务标记。"""

    _, filesystem, _, infos = _remote_file_infos(run_uri)
    paths = tuple(PurePosixPath(info.path) for info in infos)
    markdown = tuple(path for path in paths if path.suffix == ".md")
    layouts = tuple(path for path in paths if path.name == "layout.json")
    document_success = tuple(
        path
        for path in paths
        if path.name == "_SUCCESS"
        and len(path.parts) >= 3
        and path.parent.parent.name == "docs"
    )
    incomplete = tuple(path for path in paths if path.name == ".rayorch-incomplete")
    staging = tuple(path for path in paths if "_staging" in path.parts)
    markdown_stems = {path.stem for path in markdown}
    layout_stems = {path.parent.parent.name for path in layouts}
    errors = []
    if len(markdown) != expected_documents:
        errors.append(f"Markdown count={len(markdown)}")
    if len(layouts) != expected_documents:
        errors.append(f"layout count={len(layouts)}")
    if len(markdown_stems) != expected_documents:
        errors.append(f"unique Markdown stems={len(markdown_stems)}")
    if len(layout_stems) != expected_documents:
        errors.append(f"unique layout stems={len(layout_stems)}")
    if len(document_success) != expected_documents:
        errors.append(f"document success markers={len(document_success)}")
    if markdown_stems != layout_stems:
        errors.append("Markdown/layout stem sets differ")
    if incomplete:
        errors.append(f"incomplete markers={len(incomplete)}")
    if staging:
        errors.append(f"staging files={len(staging)}")
    if errors:
        raise RuntimeError("HDFS result validation failed: " + "; ".join(errors))
    result = {
        "markdown": len(markdown),
        "layout": len(layouts),
        "unique_documents": len(markdown_stems),
        "document_success": len(document_success),
        "incomplete": len(incomplete),
        "staging_files": len(staging),
    }
    if filesystem is not None:
        semantic_payloads = []
        for success_path in sorted(document_success):
            with filesystem.open_input_file(success_path.as_posix()) as source:
                semantic_payloads.append(json.loads(source.read().decode("utf-8")))
        result["output_semantic_digest"] = hashlib.sha256(
            json.dumps(
                semantic_payloads,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    return result


def _write_success(run_uri: str, payload: dict[str, Any]) -> str:
    """在所有验证通过后写 Hadoop 风格的 ``_SUCCESS`` marker。"""

    _, filesystem, root = _hdfs_filesystem(run_uri)
    success_path = posixpath.join(root, "_SUCCESS")
    temporary_path = posixpath.join(
        root,
        f"._SUCCESS.{uuid.uuid4().hex}.tmp",
    )
    with filesystem.open_output_stream(temporary_path) as output:
        output.write(
            (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
    filesystem.move(temporary_path, success_path)
    return _hdfs_join(run_uri, "_SUCCESS")


def _default_run_id(ray_module: Any) -> str:
    """组合 Ray Job ID、UTC 时间和随机后缀，避免跨集群目录碰撞。"""

    try:
        job_id = _node_id_text(ray_module.get_runtime_context().get_job_id())
    except Exception:
        job_id = ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = _safe_component(job_id) if job_id else "ray-job"
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _pdf_input_contract(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """返回经过长度与总数校验的多 HDFS 输入契约。"""

    uris = tuple(args.hdfs_pdf_uri)
    files = tuple(args.expected_pdf_files)
    sizes = tuple(args.expected_pdf_bytes)
    if not uris or len(uris) != len(files) or len(uris) != len(sizes):
        raise ValueError(
            "--hdfs-pdf-uri, --expected-pdf-files and --expected-pdf-bytes "
            "must be repeated the same non-zero number of times"
        )
    if any(value <= 0 for value in files) or any(value <= 0 for value in sizes):
        raise ValueError("expected PDF file and byte counts must be positive")
    return uris, files, sizes


def _benchmark_args(
    args: argparse.Namespace,
    local_root: Path,
    *,
    output_dir: str | None = None,
    remote_output_dir: str | None = None,
    spool_batches: bool = False,
    profile_dir: str | None = None,
    profile_driver_dir: str | None = None,
    profile_system: str = "rayorch",
    cold_e2e_start_epoch_s: float | None = None,
    cold_e2e_start_monotonic_s: float | None = None,
) -> argparse.Namespace:
    """构造固定 8×8 GPU/联合 PDF 输入的 MinerU runner 参数。"""

    artifacts = local_root / "artifacts"
    pdf_uris, pdf_files, _ = _pdf_input_contract(args)
    pdf_limits = tuple(args.pdf_limit or pdf_files)
    if len(pdf_limits) != len(pdf_files) or any(
        limit <= 0 or limit > available
        for limit, available in zip(pdf_limits, pdf_files, strict=True)
    ):
        raise ValueError("PDF sample limits must match inputs and frozen manifests")
    return argparse.Namespace(
        mode=args.mode,
        limit=sum(pdf_limits),
        replicas=args.ocr_replicas,
        gpus_per_ocr_actor=args.gpus_per_ocr_actor,
        microbatch_size=args.microbatch_size,
        max_active_microbatches=args.max_active_microbatches,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        render_replicas=args.render_replicas,
        reduce_replicas=args.reduce_replicas,
        num_cpus=args.num_cpus,
        object_store_gb=args.object_store_gb,
        rss_interval_s=args.rss_interval_s,
        ray_address="auto",
        flash_repo=args.flash_repo,
        pdf_dirs=list(pdf_uris),
        pdf_limits=list(pdf_limits),
        skip_existing=bool(remote_output_dir),
        completion_dir=remote_output_dir,
        spool_batches=spool_batches,
        remote_output_dir=remote_output_dir,
        model=args.local_model_dir or str(local_root / "model"),
        output_dir=output_dir or str(local_root / "output"),
        artifact_dir=str(artifacts),
        result_jsonl=str(artifacts / "results.jsonl"),
        actor_resources={"accelerator_type:H20": 0.001},
        actor_scheduling_strategy="SPREAD",
        profile_dir=profile_dir,
        profile_driver_dir=profile_driver_dir,
        profile_system=profile_system,
        cold_e2e_start_epoch_s=cold_e2e_start_epoch_s,
        cold_e2e_start_monotonic_s=cold_e2e_start_monotonic_s,
    )


def run_taiji(
    args: argparse.Namespace,
    *,
    ray_module: Any | None = None,
    benchmark_runner: Callable[[argparse.Namespace], dict[str, Any]] = run_benchmark,
) -> dict[str, Any]:
    """执行 HDFS 直读、64-GPU benchmark、直接写 HDFS、校验和提交。"""

    cold_start_epoch_s = float(
        getattr(args, "cold_e2e_start_epoch_s", 0.0) or time.time()
    )
    cold_start_monotonic_s = float(
        getattr(args, "cold_e2e_start_monotonic_s", 0.0)
        or time.perf_counter()
    )
    profile_system = str(getattr(args, "profile_system", "rayorch"))
    if ray_module is None:
        import ray as ray_module  # pyright: ignore[reportMissingImports,no-redef]

    if not ray_module.is_initialized():
        ray_module.init(address="auto")

    nodes, gpu_nodes = _wait_for_gpu_topology(
        ray_module,
        timeout_s=args.topology_timeout_s,
        poll_s=args.topology_poll_s,
    )
    # Start distributed sampling as soon as the 8x8 topology is available so
    # mount, input preparation, model staging, actor initialization, and the
    # measured pipeline are all represented in the cold-run GPU trace.
    gpu_samplers = _start_gpu_samplers(
        ray_module,
        gpu_nodes,
        interval_s=args.gpu_sample_interval_s,
    )
    raw_gpu_records: list[dict[str, Any]] = []
    local_root = Path(args.local_root).absolute()
    if local_root == Path("/tmp") or Path("/tmp") not in local_root.parents:
        raise ValueError("--local-root must be an absolute directory below /tmp")
    pdf_uris, expected_pdf_files, expected_pdf_bytes = _pdf_input_contract(args)
    pdf_limits = tuple(args.pdf_limit or expected_pdf_files)
    if len(pdf_limits) != len(expected_pdf_files) or any(
        limit <= 0 or limit > available
        for limit, available in zip(
            pdf_limits, expected_pdf_files, strict=True
        )
    ):
        raise ValueError("PDF sample limits must match inputs and frozen manifests")
    expected_documents = sum(pdf_limits)

    run_id = (
        _safe_component(args.run_id)
        if args.run_id
        else _default_run_id(ray_module)
    )
    if args.ceph_output_dir and args.hdfs_output_uri:
        raise ValueError("choose exactly one of Ceph or HDFS output")
    ceph_output = bool(args.ceph_output_dir)
    mount_results: list[dict[str, Any]] = []
    if ceph_output:
        output_base = str(Path(args.ceph_output_dir).absolute())
        run_uri = str(Path(output_base) / run_id)
        token_path = Path(args.ceph_token_file)
        token = token_path.read_text(encoding="utf-8").strip()
        mount_results = _run_pinned_tasks(
            ray_module,
            _mount_ceph_node,
            [
                (
                    node,
                    (
                        token,
                        args.ceph_app_group,
                        args.ceph_location,
                        output_base,
                        run_uri,
                    ),
                )
                for node in gpu_nodes
            ],
        )
        _run_pinned_tasks(
            ray_module,
            _prepare_ceph_run_root_node,
            [(gpu_nodes[0], (run_uri,))],
        )
    else:
        output_base = (
            args.hdfs_output_uri
            or pdf_uris[0].rstrip("/") + "_multigrain_v3_6_results"
        )
        run_uri = _hdfs_join(output_base, run_id)
        _prepare_hdfs_run_root(run_uri)

    profile_dir = str(Path(run_uri) / "profile") if ceph_output else None
    driver_profile_dir = str(local_root / "profile_driver")
    _reset_tmp_directory(driver_profile_dir)
    profile = ProfileEventWriter(
        driver_profile_dir,
        system=profile_system,
        stage="driver",
        role="collect",
    )
    profile.emit(
        {
            "type": "milestone",
            "name": "cold_e2e_started",
            "epoch_s": cold_start_epoch_s,
            "monotonic_s": cold_start_monotonic_s,
        }
    )
    profile.emit(
        {
            "type": "milestone",
            "name": "ray_initialized",
            "epoch_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
    )

    local_model_dir = args.local_model_dir or str(local_root / "model")
    local_spool_root = str(local_root / "spool")
    local_artifact_dir = str(local_root / "artifacts")
    _reset_tmp_directory(local_artifact_dir)
    pdf_input = _run_pinned_tasks(
        ray_module,
        _prepare_hdfs_pdf_inputs_node,
        [
            (
                node,
                (
                    pdf_uris,
                    expected_pdf_files,
                    expected_pdf_bytes,
                    None if args.benchmark_only else local_spool_root,
                ),
            )
            for node in gpu_nodes
        ],
    )
    model_stage = _run_pinned_tasks(
        ray_module,
        _stage_model_node,
        [
            (
                node,
                (
                    args.hdfs_model_uri,
                    local_model_dir,
                    args.expected_model_files,
                    args.expected_model_bytes,
                ),
            )
            for node in gpu_nodes
        ],
    )
    document_output_target = (
        str(Path(run_uri) / "docs")
        if ceph_output
        else _hdfs_join(run_uri, "docs")
    )
    spool_uploaders: list[Any] = []
    initial_spool_drain: list[dict[str, Any]] = []
    if not args.benchmark_only:
        spool_uploaders = _start_spool_uploaders(
            ray_module,
            gpu_nodes,
            spool_root=local_spool_root,
            documents_target=document_output_target,
            workers_per_node=args.spool_upload_workers,
            profile_dir=profile_dir,
            profile_system=profile_system,
        )
        initial_spool_drain = _drain_spool_uploaders(
            ray_module,
            spool_uploaders,
            timeout_s=args.spool_drain_timeout_s,
        )
    print(
        HDFS_DIRECT_READY_MARKER
        + json.dumps(
            {
                "run_id": run_id,
                "mode": (
                    "benchmark_only"
                    if args.benchmark_only
                    else (
                        "local_spool_async_ceph"
                        if ceph_output
                        else "local_spool_async_hdfs"
                    )
                ),
                "pdf_uris": pdf_uris,
                "gpu_workers": len(gpu_nodes),
                "pdf_manifests": pdf_input,
                "mount_results": mount_results,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    benchmark_started_at = time.time()
    document_output_dir = (
        str(local_root / "output")
        if args.benchmark_only
        else local_spool_root
    )
    try:
        benchmark = benchmark_runner(
            _benchmark_args(
                args,
                local_root,
                output_dir=document_output_dir,
                remote_output_dir=(
                    None if args.benchmark_only else document_output_target
                ),
                spool_batches=not args.benchmark_only,
                profile_dir=profile_dir,
                profile_driver_dir=driver_profile_dir,
                profile_system=profile_system,
                cold_e2e_start_epoch_s=cold_start_epoch_s,
                cold_e2e_start_monotonic_s=cold_start_monotonic_s,
            )
        )
    except BaseException:
        _stop_gpu_samplers(
            ray_module,
            gpu_samplers,
            expected_gpus=GPU_REPLICAS,
            raw_records_out=raw_gpu_records,
        )
        _stop_spool_uploaders(ray_module, spool_uploaders)
        raise
    measured_start_time = benchmark_started_at + float(benchmark["startup_s"])
    profile.emit(
        {
            "type": "milestone",
            "name": "benchmark_materialized",
            "epoch_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
    )
    if benchmark.get("status") != "completed":
        raise RuntimeError(f"MinerU benchmark did not complete: {benchmark}")
    discovered = int(benchmark.get("discovered_pdfs") or 0)
    skipped = int(benchmark.get("skipped_existing") or 0)
    processed = int(benchmark.get("n_pdf") or 0)
    documents = int(benchmark.get("docs") or 0)
    if discovered != expected_documents or skipped + processed != expected_documents or documents != processed:
        raise RuntimeError(
            "MinerU benchmark document count mismatch: "
            f"discovered={discovered}, skipped={skipped}, "
            f"n_pdf={processed}, docs={documents}"
        )
    if (
        args.expected_pages is not None
        and skipped == 0
        and benchmark.get("pages") != args.expected_pages
    ):
        raise RuntimeError(
            "MinerU benchmark page count mismatch: "
            f"expected {args.expected_pages}, got {benchmark.get('pages')}"
        )

    if args.benchmark_only:
        output_uploads = []
        output_mode = "benchmark_only_local"
        validation = {
            "skipped": True,
            "reason": "benchmark_only",
            "expected_documents": expected_documents,
        }
        local_cleanup = [
            {
                "skipped": True,
                "reason": "node ids may change after raylet recovery",
            }
        ]
    else:
        output_uploads = _drain_spool_uploaders(
            ray_module,
            spool_uploaders,
            timeout_s=args.spool_drain_timeout_s,
        )
        _stop_spool_uploaders(ray_module, spool_uploaders)
        output_mode = (
            "local_spool_async_ceph"
            if ceph_output
            else "local_spool_async_hdfs"
        )
        validation = (
            _run_pinned_tasks(
                ray_module,
                _validate_ceph_outputs_node,
                [(gpu_nodes[0], (run_uri, expected_documents))],
            )[0]
            if ceph_output
            else _validate_hdfs_outputs(
                run_uri, expected_documents=expected_documents
            )
        )
        profile.emit(
            {
                "type": "milestone",
                "name": "terminal_outputs_validated",
                "epoch_s": time.time(),
                "monotonic_s": time.perf_counter(),
            }
        )
        local_cleanup = []
    gpu_sampling = _stop_gpu_samplers(
        ray_module,
        gpu_samplers,
        expected_gpus=GPU_REPLICAS,
        measured_start_time=measured_start_time,
        measured_end_time=(
            measured_start_time + float(benchmark["measured_wall_s"])
        ),
        raw_records_out=raw_gpu_records,
    )
    wrapper_payload = {
        "status": "completed",
        "run_id": run_id,
        "run_uri": run_uri,
        "cluster_nodes": [asdict(node) for node in nodes],
        "gpu_workers": [asdict(node) for node in gpu_nodes],
        "pdf_input": pdf_input,
        "model_stage": model_stage,
        "mount_results": mount_results,
        "output_uploads": output_uploads,
        "initial_spool_drain": initial_spool_drain,
        "output_mode": output_mode,
        "document_output_uri": (
            None if args.benchmark_only else document_output_target
        ),
        "validation": validation,
        "benchmark": benchmark,
        "gpu_sampling": gpu_sampling,
        "benchmark_only": args.benchmark_only,
        "local_cleanup": local_cleanup,
    }
    artifact_root = Path(local_artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "taiji-wrapper.json").write_text(
        json.dumps(wrapper_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if ceph_output:
        local_artifacts = Path(local_artifact_dir)
        artifact_files = {
            path.relative_to(local_artifacts).as_posix(): path.read_bytes()
            for path in local_artifacts.rglob("*")
            if path.is_file()
        }
        commit_result = _run_pinned_tasks(
            ray_module,
            _commit_ceph_driver_artifacts_node,
            [(gpu_nodes[0], (run_uri, artifact_files, wrapper_payload))],
        )[0]
        driver_upload = commit_result["driver_upload"]
    else:
        driver_upload = _upload_tree(
            local_artifact_dir,
            _hdfs_join(run_uri, "driver", "artifacts"),
        )
    wrapper_payload["driver_upload"] = driver_upload
    wrapper_payload["success_uri"] = (
        commit_result["success_uri"]
        if ceph_output
        else _write_success(run_uri, wrapper_payload)
    )
    profile.emit(
        {
            "type": "milestone",
            "name": "collect_finished",
            "epoch_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
    )
    cold_end_epoch_s = time.time()
    cold_end_monotonic_s = time.perf_counter()
    cold_wall_s = cold_end_monotonic_s - cold_start_monotonic_s
    if profile_dir is not None:
        input_digest = hashlib.sha256(
            json.dumps(
                [
                    {
                        "uri": uri,
                        "files": files,
                        "bytes": size,
                        "limit": limit,
                    }
                    for uri, files, size, limit in zip(
                        pdf_uris,
                        expected_pdf_files,
                        expected_pdf_bytes,
                        pdf_limits,
                        strict=True,
                    )
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        fairness = {
            "input_digest": input_digest,
            "output_digest": validation.get("output_semantic_digest"),
            "model": args.hdfs_model_uri,
            "model_manifest": {
                "files": args.expected_model_files,
                "bytes": args.expected_model_bytes,
            },
            "pdf_count": expected_documents,
            "page_count": int(benchmark.get("pages") or 0),
            "gpu_type": "NVIDIA H20",
            "gpu_count": GPU_REPLICAS,
            "model_replicas": args.ocr_replicas,
            "effective_request_batch_size": args.batch_size,
            "render_concurrency": args.render_replicas,
            "reduce_concurrency": args.reduce_replicas,
            "image_dpi": 200,
            "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
            "framework_version": getattr(ray_module, "__version__", None),
            "code_revision": os.environ.get("RAYORCH_SOURCE_DIGEST"),
        }
        config = {
            "engine": profile_system,
            "microbatch_size": args.microbatch_size,
            "max_active_microbatches": args.max_active_microbatches,
            "gpus_per_ocr_actor": args.gpus_per_ocr_actor,
            "spool_upload_workers": args.spool_upload_workers,
            "output_mode": output_mode,
        }
        metrics = {
            "cold_e2e_wall_s": round(cold_wall_s, 6),
            "pages_per_s": round(
                int(benchmark.get("pages") or 0) / max(cold_wall_s, 1e-9), 6
            ),
            "benchmark_measured_wall_s": benchmark.get("measured_wall_s"),
            "benchmark_pages_per_s": benchmark.get("pages_per_s"),
            "gpu_sampling_summary": gpu_sampling,
        }
        clock = {
            "origin": "cold_e2e",
            "e2e_start_epoch_s": cold_start_epoch_s,
            "e2e_end_epoch_s": cold_end_epoch_s,
            "e2e_start_monotonic_s": cold_start_monotonic_s,
            "e2e_end_monotonic_s": cold_end_monotonic_s,
            "e2e_wall_s": cold_wall_s,
            "absolute_anchor_estimated": False,
        }
        driver_events_root = Path(driver_profile_dir) / "raw_events"
        driver_event_files = {
            path.name: path.read_bytes()
            for path in sorted(driver_events_root.glob("*.jsonl"))
            if path.is_file()
        }
        wrapper_payload["profile"] = _run_pinned_tasks(
            ray_module,
            _finalize_ceph_profile_node,
            [
                (
                    gpu_nodes[0],
                    (
                        profile_dir,
                        driver_event_files,
                        raw_gpu_records,
                        clock,
                        fairness,
                        config,
                        metrics,
                    ),
                )
            ],
        )[0]
    return wrapper_payload


def build_parser() -> argparse.ArgumentParser:
    """构造供 WeData Ray Job 调用的 TaiJi benchmark CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdfs-pdf-uri",
        action="append",
        required=True,
        help="repeat once per HDFS PDF input root",
    )
    parser.add_argument("--hdfs-model-uri", required=True)
    parser.add_argument(
        "--hdfs-output-uri",
        default=None,
        help="run 目录的 HDFS 父目录；默认是 PDF URI 加结果后缀",
    )
    parser.add_argument("--ceph-output-dir", default=None)
    parser.add_argument(
        "--ceph-token-file",
        default="taiji_multinode_mineru/taijiPATToken.runtime",
    )
    parser.add_argument(
        "--ceph-app-group",
        default="TaiJi_HYAide_LLM_Pretrain_Data",
    )
    parser.add_argument("--ceph-location", default="gy")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--local-model-dir", default=None)
    parser.add_argument(
        "--expected-pdf-files",
        action="append",
        type=int,
        required=True,
        help="repeat in the same order as --hdfs-pdf-uri",
    )
    parser.add_argument(
        "--expected-pdf-bytes",
        action="append",
        type=int,
        required=True,
        help="repeat in the same order as --hdfs-pdf-uri",
    )
    parser.add_argument("--expected-pages", type=int, default=None)
    parser.add_argument(
        "--pdf-limit",
        action="append",
        type=int,
        default=None,
        help="optional per-input deterministic tuning sample size",
    )
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument(
        "--expected-model-files",
        type=int,
        default=EXPECTED_MODEL_FILES,
        help="每个 GPU 节点 staging 后必须匹配的模型文件数",
    )
    parser.add_argument(
        "--expected-model-bytes",
        type=int,
        default=EXPECTED_MODEL_BYTES,
        help="每个 GPU 节点 staging 后必须匹配的模型总字节数",
    )
    parser.add_argument("--topology-timeout-s", type=float, default=1800)
    parser.add_argument("--topology-poll-s", type=float, default=5)
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument(
        "--mode", choices=("elastic", "parent_bound"), default="elastic"
    )
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-active-microbatches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--ocr-replicas", type=int, default=GPU_REPLICAS)
    parser.add_argument("--gpus-per-ocr-actor", type=float, default=1.0)
    parser.add_argument("--render-replicas", type=int, default=120)
    parser.add_argument("--reduce-replicas", type=int, default=64)
    parser.add_argument("--num-cpus", type=int, default=256)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--gpu-sample-interval-s", type=float, default=1)
    parser.add_argument("--spool-upload-workers", type=int, default=4)
    parser.add_argument("--spool-drain-timeout-s", type=float, default=7200)
    return parser


def main(argv: list[str] | None = None) -> int:
    """连接 TaiJi Ray 集群并打印最终 HDFS 提交摘要。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_taiji(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())


__all__ = [
    "ClusterNode",
    "build_parser",
    "run_taiji",
]
