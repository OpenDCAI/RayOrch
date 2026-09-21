"""Scale-oriented MinerU UDFs extracted from the completed 64-GPU run.

The heavy optional dependencies are imported only inside actor constructors or
``run`` methods.  Importing and compiling the benchmark therefore requires only
RayOrch itself.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import json
import os
import posixpath
import socket
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


DEFAULT_MODEL = os.environ.get(
    "RAYORCH_MINERU_MODEL",
    "./models/MinerU2.5-2509-1.2B",
)


def is_hdfs_uri(path: str) -> bool:
    """Return whether *path* is an HDFS URI without importing PyArrow."""

    return urlsplit(path).scheme.lower() == "hdfs"


def pdf_stem(path: str) -> str:
    """Return a safe output stem for either a local path or an HDFS URI."""

    stem = (
        PurePosixPath(urlsplit(path).path).stem
        if is_hdfs_uri(path)
        else Path(path).stem
    )
    if (
        stem in {"", ".", ".."}
        or stem.startswith(".")
        or "/" in stem
        or "\\" in stem
    ):
        raise ValueError(f"unsafe PDF stem derived from {path!r}")
    return stem


def _available_vllm_port_for_pid(pid: int) -> int:
    """Find a free process-stable block of eight node-local vLLM ports."""

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

    if not is_hdfs_uri(path):
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
    """Atomically publish one local text artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
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
    """Read a committed HDFS document or return ``None`` if incomplete."""

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


def _local_committed_document(
    document_root: Path,
    stem: str,
    parse_method: str,
) -> dict[str, Any] | None:
    """Read a committed shared-filesystem document if all artifacts are valid."""

    markdown_path = document_root / parse_method / f"{stem}.md"
    layout_path = document_root / parse_method / "layout.json"
    success_path = document_root / "_SUCCESS"
    if (document_root / ".rayorch-incomplete").exists():
        return None
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        commit = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
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


class MinerUScalePdfToPages:
    """Render local or HDFS PDFs into ordered first-class page records."""

    def __init__(self, dpi: int = 200, read_attempts: int = 3) -> None:
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        if read_attempts <= 0:
            raise ValueError("read_attempts must be positive")
        self.dpi = dpi
        self.read_attempts = read_attempts
        self._hdfs_filesystems: dict[str, Any] = {}

    def run(self, pdf_paths: list[str]) -> list[list[dict[str, Any]]]:
        """Read each PDF with retry and return page records in document order."""

        from flash_mineru.mineru_core.utils.pdf_image_tools import (
            load_images_from_pdf,
        )

        groups: list[list[dict[str, Any]]] = []
        for path in pdf_paths:
            last_error: Exception | None = None
            for attempt in range(self.read_attempts):
                try:
                    pdf_bytes = _read_binary_path(
                        path,
                        hdfs_filesystems=self._hdfs_filesystems,
                    )
                    images, pdf_doc = load_images_from_pdf(
                        pdf_bytes,
                        dpi=self.dpi,
                    )
                    try:
                        pages = []
                        for page_id, image in enumerate(images):
                            width, height = map(
                                int, pdf_doc[page_id].get_size()
                            )
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
                    finally:
                        pdf_doc.close()
                    groups.append(pages)
                    break
                except Exception as error:
                    last_error = error
                    parsed = urlsplit(path)
                    self._hdfs_filesystems.pop(
                        f"{parsed.scheme.lower()}://{parsed.netloc}", None
                    )
                    if attempt + 1 < self.read_attempts:
                        time.sleep(0.5 * (2**attempt))
            else:
                print(
                    "RAYORCH_MINERU_RENDER_FAILED "
                    + json.dumps(
                        {
                            "pdf_path": path,
                            "error": f"{type(last_error).__name__}: {last_error}",
                            "attempts": self.read_attempts,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                groups.append([])
        return groups


class MinerUScaleVlmOcrPage:
    """Run MinerU extraction in one persistent, node-safe vLLM actor."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.32,
    ) -> None:
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

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

    def run(self, pages: list[dict[str, Any]]) -> list[Any]:
        """Execute MinerU's two-step extraction for one logical page batch."""

        return list(
            self.client.batch_two_step_extract(
                images=[page["img_pil"] for page in pages]
            )
        )


class MinerUScalePdfMetadata:
    """Create lightweight document context without forwarding PDF payloads."""

    def run(self, paths: list[str]) -> list[str]:
        return [pdf_stem(path) for path in paths]


class MinerUScaleAssembleDoc:
    """Atomically assemble ordered page results on shared POSIX or HDFS output."""

    def __init__(self, output_dir: str, parse_method: str = "vlm") -> None:
        self.output_dir = output_dir
        self.parse_method = parse_method
        self._hdfs_filesystems: dict[str, Any] = {}

    def run(
        self,
        grouped_contents: list[list[Any]],
        grouped_pages: list[list[dict[str, Any]]],
        stems: list[str],
    ) -> list[dict[str, Any]]:
        """Write Markdown, layout JSON, images, and a document commit marker."""

        from flash_mineru.mineru_core.data.data_reader_writer import (
            FileBasedDataWriter,
        )
        from flash_mineru.mineru_core.engine.model_output_to_middle_json import (
            result_to_middle_json,
        )
        from flash_mineru.mineru_core.engine.vlm_middle_json_mkcontent import (
            union_make as vlm_union_make,
        )
        from flash_mineru.mineru_core.utils.enum_class import (
            MakeMode,
        )

        outputs = []
        for contents, pages, stem in zip(
            grouped_contents,
            grouped_pages,
            stems,
            strict=True,
        ):
            if len(contents) != len(pages):
                raise ValueError("MinerU content and page groups must align")
            if is_hdfs_uri(self.output_dir):
                output = self._write_hdfs_document(
                    contents,
                    pages,
                    stem,
                    result_to_middle_json=result_to_middle_json,
                    vlm_union_make=vlm_union_make,
                    make_mode=MakeMode,
                )
            else:
                output = self._write_local_document(
                    contents,
                    pages,
                    stem,
                    file_writer=FileBasedDataWriter,
                    result_to_middle_json=result_to_middle_json,
                    vlm_union_make=vlm_union_make,
                    make_mode=MakeMode,
                )
            outputs.append(output)
        return outputs

    def _write_local_document(
        self,
        contents: list[Any],
        pages: list[dict[str, Any]],
        stem: str,
        *,
        file_writer: Any,
        result_to_middle_json: Any,
        vlm_union_make: Any,
        make_mode: Any,
    ) -> dict[str, Any]:
        document_root = Path(self.output_dir) / stem
        markdown_dir = document_root / self.parse_method
        markdown_path = markdown_dir / f"{stem}.md"
        committed = _local_committed_document(
            document_root,
            stem,
            self.parse_method,
        )
        if committed is not None:
            return {
                "pdf": stem,
                "md_path": str(markdown_path.resolve()),
                **committed,
            }

        image_dir = markdown_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        incomplete = document_root / ".rayorch-incomplete"
        _atomic_write_text(incomplete, "document assembly in progress\n")
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
        status = "completed" if pages else "render_failed"
        commit = {
            "pdf": stem,
            "status": status,
            "pages": len(pages),
            "chars": len(markdown),
        }
        _atomic_write_text(markdown_path, markdown)
        _atomic_write_text(
            markdown_dir / "layout.json",
            json.dumps(middle, ensure_ascii=False, indent=2),
        )
        _atomic_write_text(
            document_root / "_SUCCESS",
            json.dumps(commit, ensure_ascii=False, sort_keys=True) + "\n",
        )
        incomplete.unlink(missing_ok=True)
        return {
            "pdf": stem,
            "md_path": str(markdown_path.resolve()),
            **commit,
        }

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
        markdown = ""
        status = "render_failed"
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
            status = "completed" if pages else "render_failed"
            commit = {
                "pdf": stem,
                "status": status,
                "pages": len(pages),
                "chars": len(markdown),
            }
            _hdfs_write_bytes(
                filesystem,
                posixpath.join(markdown_dir, f"{stem}.md"),
                markdown.encode("utf-8"),
            )
            _hdfs_write_bytes(
                filesystem,
                posixpath.join(markdown_dir, "layout.json"),
                json.dumps(
                    middle,
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
            )
            _hdfs_write_bytes(
                filesystem,
                posixpath.join(staging_root, "_SUCCESS"),
                (
                    json.dumps(commit, ensure_ascii=False, sort_keys=True)
                    + "\n"
                ).encode("utf-8"),
            )
            filesystem.create_dir(
                posixpath.dirname(final_root), recursive=True
            )
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
                self.output_dir,
                stem,
                self.parse_method,
                f"{stem}.md",
            ),
            "chars": len(markdown),
            "pages": len(pages),
            "status": status,
        }


__all__ = [
    "DEFAULT_MODEL",
    "MinerUScaleAssembleDoc",
    "MinerUScalePdfMetadata",
    "MinerUScalePdfToPages",
    "MinerUScaleVlmOcrPage",
    "is_hdfs_uri",
    "pdf_stem",
]
