"""通过 V3 public Pipeline API 运行真实 Flash-MinerU 回归。

本模块是手工 4×H20 benchmark 入口，只负责构造真实 UDF、采集 observation 数据和写
复现实验产物，不参与 V3 runtime correctness。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import posixpath
import shutil
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from ..api import Expand, Map, Pipeline, Reduce
from ..executor import Executor
from .profile_events import ProfileEventWriter


DEFAULT_FLASH_REPO = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/workspace/Flash-mineru"
)
DEFAULT_MODEL = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/model/MinerU2.5-2509-1.2B"
)
V25_ELASTIC_MEDIAN_S = 581.509


def _is_hdfs_uri(path: str) -> bool:
    """Return whether ``path`` is an HDFS URI without importing PyArrow."""

    return urlsplit(path).scheme.lower() == "hdfs"


def _pdf_stem(path: str) -> str:
    """Return a PDF stem for either a local path or an HDFS URI."""

    if _is_hdfs_uri(path):
        return PurePosixPath(urlsplit(path).path).stem
    return Path(path).stem


def _vllm_port_for_pid(pid: int) -> int:
    """Allocate a process-stable eight-port slot for node-local vLLM actors."""

    return 10_000 + (pid % 6_000) * 8


def _available_vllm_port_for_pid(pid: int) -> int:
    """Find an unused eight-port slot while the caller holds the init lock.

    PID hashing alone can collide once hundreds of actors share a node, and a
    vLLM process keeps its TCPStore port for its lifetime. Probe the hashed
    slot first, then every other slot, binding all eight ports temporarily so
    adjacent vLLM ports cannot overlap either.
    """

    slot_count = 6_000
    first_slot = pid % slot_count
    for step in range(slot_count):
        port = 10_000 + ((first_slot + step) % slot_count) * 8
        probes: list[socket.socket] = []
        try:
            for candidate in range(port, port + 8):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probes.append(probe)
                probe.bind(("127.0.0.1", candidate))
        except OSError:
            continue
        finally:
            for probe in probes:
                probe.close()
        return port
    raise RuntimeError("no free eight-port slot is available for node-local vLLM")


def _read_binary_path(
    path: str,
    *,
    hdfs_filesystems: dict[str, Any] | None = None,
) -> bytes:
    """Read a local file or HDFS URI, lazily importing PyArrow for HDFS."""

    if not _is_hdfs_uri(path):
        with open(path, "rb") as handle:
            return handle.read()

    from pyarrow import fs as pyarrow_fs

    parsed = urlsplit(path)
    authority = f"{parsed.scheme.lower()}://{parsed.netloc}"
    filesystem = (
        hdfs_filesystems.get(authority)
        if hdfs_filesystems is not None
        else None
    )
    if filesystem is None:
        filesystem, normalized_path = pyarrow_fs.FileSystem.from_uri(path)
        if hdfs_filesystems is not None:
            hdfs_filesystems[authority] = filesystem
    else:
        normalized_path = filesystem.normalize_path(parsed.path)
    with filesystem.open_input_file(normalized_path) as handle:
        return handle.read()


def _atomic_write_text(path: Path, content: str) -> None:
    """原子提交文本产物，避免恢复逻辑看见截断的最终文件。"""

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hdfs_filesystem_and_path(
    uri: str,
    filesystems: dict[str, Any],
) -> tuple[Any, str]:
    """Resolve an HDFS URI while reusing one filesystem per authority."""

    from pyarrow import fs as pyarrow_fs

    parsed = urlsplit(uri)
    authority = f"{parsed.scheme.lower()}://{parsed.netloc}"
    filesystem = filesystems.get(authority)
    if filesystem is None:
        filesystem, path = pyarrow_fs.FileSystem.from_uri(uri)
        filesystems[authority] = filesystem
    else:
        path = filesystem.normalize_path(parsed.path)
    return filesystem, path.rstrip("/") or "/"


def _hdfs_uri_join(root: str, *parts: str) -> str:
    """Join path components without losing an HDFS authority."""

    return root.rstrip("/") + "/" + "/".join(
        part.strip("/") for part in parts if part.strip("/")
    )


class _HdfsDataWriter:
    """Minimal MinerU DataWriter implementation backed by PyArrow HDFS."""

    def __init__(self, filesystem: Any, parent_path: str) -> None:
        self.filesystem = filesystem
        self.parent_path = parent_path.rstrip("/")

    def write(self, path: str, data: bytes) -> None:
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"HDFS writer path escapes its parent: {path}")
        remote_path = posixpath.join(self.parent_path, relative.as_posix())
        self.filesystem.create_dir(
            posixpath.dirname(remote_path), recursive=True
        )
        with self.filesystem.open_output_stream(remote_path) as output:
            output.write(data)


def _hdfs_write_bytes(filesystem: Any, path: str, data: bytes) -> None:
    filesystem.create_dir(posixpath.dirname(path), recursive=True)
    with filesystem.open_output_stream(path) as output:
        output.write(data)


def _hdfs_committed_document(
    filesystem: Any,
    document_root: str,
    stem: str,
    parse_method: str,
) -> dict[str, Any] | None:
    """Read a committed document, returning ``None`` for an incomplete path."""

    markdown_path = posixpath.join(
        document_root, parse_method, f"{stem}.md"
    )
    layout_path = posixpath.join(document_root, parse_method, "layout.json")
    success_path = posixpath.join(document_root, "_SUCCESS")
    try:
        with filesystem.open_input_file(markdown_path) as source:
            markdown = source.read().decode("utf-8")
        with filesystem.open_input_file(layout_path) as source:
            layout = json.loads(source.read().decode("utf-8"))
        with filesystem.open_input_file(success_path) as source:
            commit = json.loads(source.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(layout, dict) or not isinstance(
        layout.get("pdf_info"), list
    ):
        return None
    if not isinstance(commit, dict) or commit.get("pdf") != stem:
        return None
    return {
        "chars": len(markdown),
        "pages": int(commit.get("pages", 0)),
        "status": str(commit.get("status") or "completed"),
    }


class MinerUPdfToPages:
    """在 CPU actor 中把每个 PDF 渲染为 first-class page records。"""

    def __init__(
        self,
        dpi: int = 200,
        profile_dir: str | None = None,
        profile_system: str = "unknown",
    ) -> None:
        """保存 PDF 渲染分辨率。"""

        init_epoch_s = time.time()
        init_monotonic_s = time.perf_counter()
        self._profile = ProfileEventWriter(
            profile_dir,
            system=profile_system,
            stage="render",
            role="render",
        )
        self.dpi = dpi
        self._hdfs_filesystems: dict[str, Any] = {}
        self._profile.actor_ready(
            started_epoch_s=init_epoch_s,
            started_monotonic_s=init_monotonic_s,
        )

    def run(self, pdf_paths: list[str]) -> list[list[dict[str, Any]]]:
        """逐 PDF 读取字节并返回按 page ordinal 排列的页面记录。"""

        started_epoch_s = time.time()
        started_monotonic_s = time.perf_counter()
        from flash_mineru.mineru_core.utils.pdf_image_tools import (
            load_images_from_pdf,
        )

        groups = []
        for path in pdf_paths:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    pdf_bytes = _read_binary_path(
                        path,
                        hdfs_filesystems=self._hdfs_filesystems,
                    )
                    images, pdf_doc = load_images_from_pdf(
                        pdf_bytes, dpi=self.dpi
                    )
                    break
                except Exception as error:
                    last_error = error
                    parsed = urlsplit(path)
                    self._hdfs_filesystems.pop(
                        f"{parsed.scheme.lower()}://{parsed.netloc}", None
                    )
                    if attempt < 2:
                        time.sleep(0.5 * (2**attempt))
            else:
                print(
                    "RAYORCH_MINERU_RENDER_FAILED "
                    + json.dumps(
                        {
                            "pdf_path": path,
                            "error": (
                                f"{type(last_error).__name__}: {last_error}"
                            ),
                            "attempts": 3,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                groups.append([])
                continue
            pages = []
            for page_id, image in enumerate(images):
                width, height = map(int, pdf_doc[page_id].get_size())
                pages.append(
                    {
                        "pdf_path": path,
                        "page_id": page_id,
                        "img_pil": image["img_pil"],
                        "scale": image.get("scale"),
                        "page_width": width,
                        "page_height": height,
                        "pdf_len": len(images),
                    }
                )
            pdf_doc.close()
            groups.append(pages)
        self._profile.stage_batch(
            started_epoch_s=started_epoch_s,
            started_monotonic_s=started_monotonic_s,
            items=len(pdf_paths),
            batch_size=len(pdf_paths),
            output_items=sum(len(group) for group in groups),
        )
        return groups


class MinerUVlmOcrPage:
    """在单个 GPU persistent actor 中运行真实 MinerU vLLM 页面抽取。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.8,
        profile_dir: str | None = None,
        profile_system: str = "unknown",
    ) -> None:
        """加载 vLLM 模型并创建 MinerUClient；每个 actor 只初始化一次。"""

        init_epoch_s = time.time()
        init_monotonic_s = time.perf_counter()
        self._profile = ProfileEventWriter(
            profile_dir,
            system=profile_system,
            stage="ocr",
            role="model",
        )
        import fcntl
        from mineru_vl_utils import MinerUClient
        from vllm import LLM

        with open("/tmp/rayorch-vllm-init.lock", "a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                os.environ["VLLM_PORT"] = str(
                    _available_vllm_port_for_pid(os.getpid())
                )
                self.llm = LLM(
                    model=model,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        self.client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=self.llm,
        )
        self._profile.actor_ready(
            started_epoch_s=init_epoch_s,
            started_monotonic_s=init_monotonic_s,
        )

    def run(self, pages: list[dict[str, Any]]) -> list[Any]:
        """对一个 logical page batch 执行两阶段 VLM extraction。"""

        started_epoch_s = time.time()
        started_monotonic_s = time.perf_counter()
        result = list(
            self.client.batch_two_step_extract(
                images=[page["img_pil"] for page in pages]
            )
        )
        self._profile.stage_batch(
            started_epoch_s=started_epoch_s,
            started_monotonic_s=started_monotonic_s,
            items=len(pages),
            batch_size=len(pages),
        )
        return result


class PdfMetadata:
    """生成 Reduce UDF 所需的轻量 parent context，避免传输 PDF anchor payload。"""

    def __init__(
        self,
        profile_dir: str | None = None,
        profile_system: str = "unknown",
    ) -> None:
        init_epoch_s = time.time()
        init_monotonic_s = time.perf_counter()
        self._profile = ProfileEventWriter(
            profile_dir,
            system=profile_system,
            stage="metadata",
            role="select",
        )
        self._profile.actor_ready(
            started_epoch_s=init_epoch_s,
            started_monotonic_s=init_monotonic_s,
        )

    def run(self, paths: list[str]) -> list[str]:
        """把 PDF path 转为用于输出目录命名的 stem。"""

        started_epoch_s = time.time()
        started_monotonic_s = time.perf_counter()
        result = [_pdf_stem(path) for path in paths]
        self._profile.stage_batch(
            started_epoch_s=started_epoch_s,
            started_monotonic_s=started_monotonic_s,
            items=len(paths),
            batch_size=len(paths),
        )
        return result


class MinerUAssembleDoc:
    """在不接收 PDF anchor payload 的情况下组装有序页面结果。"""

    def __init__(
        self,
        output_dir: str,
        parse_method: str = "vlm",
        spool_batches: bool = False,
        remote_output_dir: str | None = None,
        profile_dir: str | None = None,
        profile_system: str = "unknown",
    ) -> None:
        """保存输出根目录和 MinerU parse method。"""

        init_epoch_s = time.time()
        init_monotonic_s = time.perf_counter()
        self._profile = ProfileEventWriter(
            profile_dir,
            system=profile_system,
            stage="assemble_write",
            role="assemble",
        )
        self.output_dir = output_dir
        self.parse_method = parse_method
        self.spool_batches = spool_batches
        self.remote_output_dir = remote_output_dir
        self._hdfs_filesystems: dict[str, Any] = {}
        self._profile.actor_ready(
            started_epoch_s=init_epoch_s,
            started_monotonic_s=init_monotonic_s,
        )

    def run(
        self,
        grouped_contents: list[list[Any]],
        grouped_pages: list[list[dict[str, Any]]],
        stems: list[str],
    ) -> list[dict[str, Any]]:
        """把 ordered OCR/page GROUPs 写成 Markdown、layout JSON 和摘要。"""

        started_epoch_s = time.time()
        started_monotonic_s = time.perf_counter()
        from flash_mineru.mineru_core.data.data_reader_writer import (
            FileBasedDataWriter,
        )
        from flash_mineru.mineru_core.engine.model_output_to_middle_json import (
            result_to_middle_json,
        )
        from flash_mineru.mineru_core.engine.vlm_middle_json_mkcontent import (
            union_make as vlm_union_make,
        )
        from flash_mineru.mineru_core.utils.enum_class import MakeMode

        if self.spool_batches:
            outputs = self._spool_local_batch(
                grouped_contents,
                grouped_pages,
                stems,
                file_writer=FileBasedDataWriter,
                result_to_middle_json=result_to_middle_json,
                vlm_union_make=vlm_union_make,
                make_mode=MakeMode,
            )
            self._profile.stage_batch(
                started_epoch_s=started_epoch_s,
                started_monotonic_s=started_monotonic_s,
                items=len(stems),
                batch_size=len(stems),
                output_published=True,
            )
            return outputs

        outputs = []
        for contents, pages, stem in zip(
            grouped_contents,
            grouped_pages,
            stems,
        ):
            if _is_hdfs_uri(self.output_dir):
                outputs.append(
                    self._write_hdfs_document(
                        contents,
                        pages,
                        stem,
                        result_to_middle_json=result_to_middle_json,
                        vlm_union_make=vlm_union_make,
                        make_mode=MakeMode,
                    )
                )
                continue
            markdown_dir = Path(self.output_dir) / stem / self.parse_method
            image_dir = markdown_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            incomplete = markdown_dir / ".rayorch-incomplete"
            _atomic_write_text(incomplete, "document assembly in progress\n")
            image_writer = FileBasedDataWriter(str(image_dir))
            middle = result_to_middle_json(
                list(contents),
                list(pages),
                image_writer,
            )
            markdown = vlm_union_make(
                middle["pdf_info"],
                MakeMode.MM_MD,
                "images",
            )
            markdown_path = markdown_dir / f"{stem}.md"
            _atomic_write_text(markdown_path, markdown)
            _atomic_write_text(
                markdown_dir / "layout.json",
                json.dumps(middle, indent=2),
            )
            incomplete.unlink()
            outputs.append(
                {
                    "pdf": stem,
                    "md_path": str(markdown_path.resolve()),
                    "chars": len(markdown),
                    "pages": len(pages),
                    "status": "completed" if pages else "render_failed",
                }
            )
        self._profile.stage_batch(
            started_epoch_s=started_epoch_s,
            started_monotonic_s=started_monotonic_s,
            items=len(stems),
            batch_size=len(stems),
            output_published=True,
        )
        return outputs

    def _spool_local_batch(
        self,
        grouped_contents: list[list[Any]],
        grouped_pages: list[list[dict[str, Any]]],
        stems: list[str],
        *,
        file_writer: Any,
        result_to_middle_json: Any,
        vlm_union_make: Any,
        make_mode: Any,
    ) -> list[dict[str, Any]]:
        """Atomically publish one locally assembled batch for an uploader."""

        if _is_hdfs_uri(self.output_dir):
            raise ValueError("batch spool root must be node-local, not HDFS")
        spool_root = Path(self.output_dir)
        batch_id = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}"
        staging_root = spool_root / "staging" / batch_id
        ready_root = spool_root / "ready" / batch_id
        documents_root = staging_root / "docs"
        outputs: list[dict[str, Any]] = []
        manifest_documents = []
        try:
            for contents, pages, stem in zip(
                grouped_contents,
                grouped_pages,
                stems,
                strict=True,
            ):
                document_root = documents_root / stem
                markdown_dir = document_root / self.parse_method
                image_dir = markdown_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
                middle = result_to_middle_json(
                    list(contents),
                    list(pages),
                    file_writer(str(image_dir)),
                )
                markdown = vlm_union_make(
                    middle["pdf_info"],
                    make_mode.MM_MD,
                    "images",
                )
                _atomic_write_text(markdown_dir / f"{stem}.md", markdown)
                _atomic_write_text(
                    markdown_dir / "layout.json",
                    json.dumps(middle, indent=2),
                )
                status = "completed" if pages else "render_failed"
                commit = {
                    "pdf": stem,
                    "status": status,
                    "pages": len(pages),
                    "chars": len(markdown),
                }
                _atomic_write_text(
                    document_root / "_SUCCESS",
                    json.dumps(commit, sort_keys=True) + "\n",
                )
                manifest_documents.append(commit)
                output_root = self.remote_output_dir or self.output_dir
                outputs.append(
                    {
                        "pdf": stem,
                        "md_path": _hdfs_uri_join(
                            output_root,
                            stem,
                            self.parse_method,
                            f"{stem}.md",
                        ),
                        "chars": len(markdown),
                        "pages": len(pages),
                        "status": status,
                    }
                )
            _atomic_write_text(
                staging_root / "manifest.json",
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "documents": manifest_documents,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
            )
            ready_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_root, ready_root)
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        return outputs

    def _write_hdfs_document(
        self,
        contents: list[Any],
        pages: list[dict[str, Any]],
        stem: str,
        *,
        result_to_middle_json: Any,
        vlm_union_make: Any,
        make_mode: Any,
    ) -> dict[str, Any]:
        """Write and atomically commit one complete document directly to HDFS."""

        filesystem, output_root = _hdfs_filesystem_and_path(
            self.output_dir,
            self._hdfs_filesystems,
        )
        final_root = posixpath.join(output_root, stem)
        committed = _hdfs_committed_document(
            filesystem,
            final_root,
            stem,
            self.parse_method,
        )
        if committed is not None:
            return {
                "pdf": stem,
                "md_path": _hdfs_uri_join(
                    self.output_dir,
                    stem,
                    self.parse_method,
                    f"{stem}.md",
                ),
                **committed,
            }
        staging_root = posixpath.join(
            posixpath.dirname(output_root),
            "_staging",
            f"{stem}.{uuid.uuid4().hex}",
        )
        markdown_dir = posixpath.join(staging_root, self.parse_method)
        image_dir = posixpath.join(markdown_dir, "images")
        filesystem.create_dir(image_dir, recursive=True)
        try:
            middle = result_to_middle_json(
                list(contents),
                list(pages),
                _HdfsDataWriter(filesystem, image_dir),
            )
            markdown = vlm_union_make(
                middle["pdf_info"],
                make_mode.MM_MD,
                "images",
            )
            markdown_name = f"{stem}.md"
            _hdfs_write_bytes(
                filesystem,
                posixpath.join(markdown_dir, markdown_name),
                markdown.encode("utf-8"),
            )
            _hdfs_write_bytes(
                filesystem,
                posixpath.join(markdown_dir, "layout.json"),
                json.dumps(middle, indent=2).encode("utf-8"),
            )
            status = "completed" if pages else "render_failed"
            _hdfs_write_bytes(
                filesystem,
                posixpath.join(staging_root, "_SUCCESS"),
                (
                    json.dumps(
                        {
                            "pdf": stem,
                            "status": status,
                            "pages": len(pages),
                            "chars": len(markdown),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
            filesystem.create_dir(posixpath.dirname(final_root), recursive=True)
            try:
                filesystem.move(staging_root, final_root)
            except Exception:
                committed = _hdfs_committed_document(
                    filesystem,
                    final_root,
                    stem,
                    self.parse_method,
                )
                if committed is None:
                    raise
                try:
                    filesystem.delete_dir(staging_root)
                except Exception:
                    pass
        except BaseException:
            try:
                filesystem.delete_dir(staging_root)
            except Exception:
                pass
            raise
        return {
            "pdf": stem,
            "md_path": _hdfs_uri_join(
                self.output_dir, stem, self.parse_method, markdown_name
            ),
            "chars": len(markdown),
            "pages": len(pages),
            "status": status,
        }


class MinerUV3Pipeline(Pipeline):
    """用于 V3 性能回归的真实 PDF→Page→OCR→Document DAG。"""
    def __init__(
        self,
        *,
        output_dir: str,
        mode: str,
        model: str,
        replicas: int,
        batch_size: int,
        max_batch_wait_ms: float,
        gpu_memory_utilization: float,
        render_replicas: int,
        reduce_replicas: int,
        runtime_env: dict[str, Any],
    ) -> None:
        """按 benchmark 参数配置 render、metadata、OCR 和 assemble Stages。"""

        scope = "elastic" if mode == "elastic" else "parent_bound"
        self.render = (
            Expand(MinerUPdfToPages)
            .pre_init(dpi=200)
            .ray_options(
                replicas=render_replicas,
                batch_size=1,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )
        self.metadata = Map(PdfMetadata).ray_options(
            replicas=1,
            batch_size=32,
            num_cpus=1,
            runtime_env=runtime_env,
        )
        self.ocr = (
            Map(MinerUVlmOcrPage)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            .ray_options(
                replicas=replicas,
                batch_size=batch_size,
                max_batch_wait_ms=max_batch_wait_ms,
                batch_scope=scope,
                num_gpus=1.0,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )
        self.assemble = (
            Reduce(MinerUAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(
                replicas=reduce_replicas,
                batch_size=4,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )

    def forward(self, pdfs):
        """声明 semantic-only PDF anchor 与 page-level elastic OCR DAG。"""

        pages = self.render(pdfs)
        contents = self.ocr(pages)
        metadata = self.metadata(pdfs)
        return self.assemble(
            anchor=pdfs,
            members=contents,
            pages=pages,
            context=metadata,
        )


@dataclass(frozen=True, slots=True)
class GpuSample:
    """只用于观测的 GPU utilization 与 memory sample。"""
    monotonic_s: float
    utilization: tuple[int | None, ...]
    memory_used: tuple[int, ...]
    epoch_s: float | None = None


class ResourceSampler:
    """后台采集 driver RSS/GPU 指标，不拥有任何调度 authority。"""
    def __init__(self, interval_s: float) -> None:
        """初始化采样间隔、driver process 和后台线程。"""

        import psutil

        self.interval_s = interval_s
        self.process = psutil.Process()
        self.driver_start = int(self.process.memory_info().rss)
        self.driver_peak = self.driver_start
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """启动 daemon observation thread。"""

        self._thread.start()

    def stop(self) -> tuple[int, int, tuple[GpuSample, ...]]:
        """停止采样并返回 driver RSS 起点/峰值和 GPU samples。"""

        self._stop.set()
        self._thread.join(timeout=max(2, self.interval_s * 2))
        return (
            self.driver_start,
            self.driver_peak,
            tuple(self.samples),
        )

    def _run(self) -> None:
        """按固定间隔采集 driver RSS 与 GPU 状态，失败时跳过本次样本。"""

        while not self._stop.wait(self.interval_s):
            try:
                self.driver_peak = max(
                    self.driver_peak,
                    int(self.process.memory_info().rss),
                )
                self.samples.append(_gpu_sample())
            except Exception:
                continue


def _gpu_sample() -> GpuSample:
    """best-effort 采集所有可见 GPU 的 utilization 和 used memory。"""

    try:
        import pynvml

        pynvml.nvmlInit()
        utilization = []
        memory = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                utilization.append(
                    int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                )
            except Exception:
                utilization.append(None)
            memory.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
        pynvml.nvmlShutdown()
        return GpuSample(
            time.monotonic(),
            tuple(utilization),
            tuple(memory),
            epoch_s=time.time(),
        )
    except Exception:
        return GpuSample(time.monotonic(), (), (), epoch_s=time.time())


def _runtime_env(flash_repo: str) -> dict[str, Any]:
    """构造让 Ray actors 能导入 Flash-MinerU 与当前仓库的 runtime_env。"""

    current = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        path for path in (flash_repo, os.getcwd(), current) if path
    )
    return {
        "env_vars": {
            "PYTHONPATH": pythonpath,
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "LOGURU_LEVEL": os.environ.get("LOGURU_LEVEL", "WARNING"),
            "VLLM_LOGGING_LEVEL": os.environ.get(
                "VLLM_LOGGING_LEVEL", "WARNING"
            ),
        }
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """运行一次真实 MinerU benchmark，并写 timeline/GPU/result artifacts。"""

    import ray

    flash_repo = os.path.abspath(args.flash_repo)
    pdfs = sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {flash_repo}")
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=args.num_cpus,
            num_gpus=args.replicas,
            object_store_memory=int(args.object_store_gb * 1024**3),
            include_dashboard=False,
        )

    pipeline = MinerUV3Pipeline(
        output_dir=os.path.abspath(args.output_dir),
        mode=args.mode,
        model=args.model,
        replicas=args.replicas,
        batch_size=args.batch_size,
        max_batch_wait_ms=args.max_batch_wait_ms,
        gpu_memory_utilization=args.gpu_memory_utilization,
        render_replicas=args.render_replicas,
        reduce_replicas=args.reduce_replicas,
        runtime_env=_runtime_env(flash_repo),
    )
    ocr_stage = next(
        stage.id
        for stage in pipeline.compile().dag.stages
        if stage.udf is not None and stage.udf.target is MinerUVlmOcrPage
    )
    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    try:
        result = Executor(
            pipeline,
            microbatch_size=args.microbatch_size,
            max_inflight_arenas=args.max_inflight_arenas,
        ).run(pdfs)
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
    outputs = result.get()
    end_to_end = time.perf_counter() - started
    pages = sum(int(output["pages"]) for output in outputs)
    ocr_events = [
        event
        for event in result.timeline
        if event.stage == ocr_stage
        and event.worker_started_at is not None
        and event.worker_finished_at is not None
    ]
    busy = sum(
        event.worker_finished_at - event.worker_started_at
        for event in ocr_events
    )
    if ocr_events:
        start = min(event.worker_started_at for event in ocr_events)
        stop = max(event.worker_finished_at for event in ocr_events)
        capacity = max(stop - start, 1e-9) * args.replicas
        bubble = max(0.0, 1.0 - busy / capacity)
    else:
        bubble = 0.0
    gpu_peak = tuple(
        max((sample.memory_used[index] for sample in gpu_samples), default=0)
        for index in range(args.replicas)
    )
    payload = {
        "engine": "multigrain_v3",
        "mode": args.mode,
        "n_pdf": len(pdfs),
        "pages": pages,
        "docs": len(outputs),
        "batch_size": args.batch_size,
        "replicas": args.replicas,
        "microbatch_size": args.microbatch_size,
        "max_inflight_arenas": args.max_inflight_arenas,
        "startup_s": round(result.metrics["startup_time_s"], 3),
        "measured_wall_s": round(result.metrics["measured_wall_time_s"], 3),
        "end_to_end_wall_s": round(end_to_end, 3),
        "pages_per_s": round(
            pages / result.metrics["measured_wall_time_s"],
            4,
        ),
        "rpc_count": result.metrics["rpc_count"],
        "grains_per_rpc": result.metrics["grains_per_rpc"],
        "batch_fill_ratio": result.metrics["batch_fill_ratio"],
        "tail_or_recovery_rpc_fraction": result.metrics[
            "tail_or_recovery_rpc_fraction"
        ],
        "active_arenas_high_watermark": result.metrics[
            "active_arenas_high_watermark"
        ],
        "ocr_bubble_ratio": round(bubble, 4),
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "speedup_vs_v25_elastic": (
            V25_ELASTIC_MEDIAN_S / result.metrics["measured_wall_time_s"]
            if len(pdfs) == 368
            else None
        ),
        "output_dir": os.path.abspath(args.output_dir),
    }
    root = Path(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "dispatch_timeline.jsonl").open("w") as handle:
        for event in result.timeline:
            handle.write(json.dumps(asdict(event)) + "\n")
    with (root / "gpu_samples.jsonl").open("w") as handle:
        for sample in gpu_samples:
            handle.write(json.dumps(asdict(sample)) + "\n")
    with Path(args.result_jsonl).open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造 V3 MinerU 手工 benchmark CLI 参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("elastic", "parent_bound"), default="elastic")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-inflight-arenas", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batch-wait-ms", type=float, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析 CLI、执行 benchmark，并把摘要 JSON 打印到 stdout。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
