"""MinerU-shaped workloads for the Multigrain V3 public API.

The real workload keeps Flash-MinerU and vLLM behind actor-time lazy imports.
The dummy workload has the same MAP/EXPAND/MAP/REDUCE/MAP shape and exercises
variable page fan-out entirely on CPUs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..api import Map, Pipeline, expand, reduce


DEFAULT_FLASH_REPO = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/workspace/Flash-mineru"
)
DEFAULT_MODEL = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/model/MinerU2.5-2509-1.2B"
)
DEFAULT_DUMMY_PAGE_COUNTS = (3, 1, 4, 2, 0)


def artifact_key_for_pdf(pdf_path: str) -> str:
    """Return a readable, full-path-derived key for one PDF's artifacts."""

    if not isinstance(pdf_path, str) or not pdf_path:
        raise ValueError("pdf_path must be a non-empty string")
    normalized = os.path.abspath(
        os.path.normpath(os.path.expanduser(pdf_path))
    )
    digest = hashlib.blake2b(
        normalized.encode("utf-8"),
        digest_size=8,
        person=b"mgv3-mineru",
    ).hexdigest()
    stem = Path(pdf_path).stem or "document"
    return f"{stem}-{digest}"


class MinerUPdfToPages:
    """Render batches of PDF paths into structural lists of page records."""

    def __init__(self, dpi: int = 200) -> None:
        """Store rendering configuration without importing Flash-MinerU."""

        self.dpi = dpi

    def run(
        self,
        pdfs: list[str],
    ) -> list[list[dict[str, Any]]]:
        """Render every input PDF and preserve its page ordinal metadata."""

        from flash_mineru.mineru_core.utils.pdf_image_tools import (
            load_images_from_pdf,
        )

        groups: list[list[dict[str, Any]]] = []
        for pdf_path in pdfs:
            with open(pdf_path, "rb") as handle:
                pdf_bytes = handle.read()
            images, pdf_doc = load_images_from_pdf(pdf_bytes, dpi=self.dpi)
            try:
                pages: list[dict[str, Any]] = []
                for page_id, image in enumerate(images):
                    width, height = map(int, pdf_doc[page_id].get_size())
                    pages.append(
                        {
                            "pdf_path": pdf_path,
                            "page_id": page_id,
                            "img_pil": image["img_pil"],
                            "scale": image.get("scale"),
                            "page_width": width,
                            "page_height": height,
                            "pdf_len": len(images),
                        }
                    )
                groups.append(pages)
            finally:
                pdf_doc.close()
        return groups


class MinerUVlmOcrPage:
    """Run real MinerU two-step page extraction inside a GPU MAP actor."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        """Lazily import and construct vLLM only when an actor is created."""

        from mineru_vl_utils import MinerUClient
        from vllm import LLM

        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=self.llm,
        )

    def run(self, pages: list[dict[str, Any]]) -> list[Any]:
        """Extract one OCR content value for every page in the batch."""

        return list(
            self.client.batch_two_step_extract(
                images=[page["img_pil"] for page in pages]
            )
        )


class MinerUAssembleDoc:
    """Assemble reduced OCR contents into one output record per PDF."""

    def __init__(
        self,
        output_dir: str,
        parse_method: str = "vlm",
    ) -> None:
        """Store output configuration without importing Flash-MinerU."""

        self.output_dir = output_dir
        self.parse_method = parse_method

    def run(
        self,
        pdfs: list[str],
        contents: list[list[Any]],
        pages: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Write Markdown/layout artifacts from ordinal-preserving page groups."""

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

        if not (len(pdfs) == len(contents) == len(pages)):
            raise ValueError("assemble batch columns must have equal lengths")

        outputs: list[dict[str, Any]] = []
        for pdf_path, page_contents, page_records in zip(
            pdfs,
            contents,
            pages,
        ):
            if len(page_contents) != len(page_records):
                raise ValueError(
                    f"OCR/page count mismatch for {pdf_path!r}: "
                    f"{len(page_contents)} != {len(page_records)}"
                )
            stem = Path(pdf_path).stem
            artifact_key = artifact_key_for_pdf(pdf_path)
            markdown_dir = (
                Path(self.output_dir) / artifact_key / self.parse_method
            )
            image_dir = markdown_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_writer = FileBasedDataWriter(str(image_dir))
            markdown_writer = FileBasedDataWriter(str(markdown_dir))
            middle = result_to_middle_json(
                list(page_contents),
                list(page_records),
                image_writer,
            )
            markdown = vlm_union_make(
                middle["pdf_info"],
                MakeMode.MM_MD,
                "images",
            )
            markdown_path = markdown_dir / f"{artifact_key}.md"
            markdown_writer.write_string(markdown_path.name, markdown)
            layout_path = markdown_dir / "layout.json"
            layout_path.write_text(
                json.dumps(middle, indent=2),
                encoding="utf-8",
            )
            outputs.append(
                {
                    "pdf": stem,
                    "md_path": str(markdown_path.resolve()),
                    "layout_path": str(layout_path.resolve()),
                    "chars": len(markdown),
                    "pages": len(page_records),
                }
            )
        return outputs


class MinerUV3Pipeline(Pipeline):
    """Real MinerU V3 graph using only MAP UDF execution primitives."""

    def __init__(
        self,
        *,
        output_dir: str,
        model: str = DEFAULT_MODEL,
        replicas: int = 4,
        batch_size: int = 16,
        max_batch_wait_ms: float = 5.0,
        gpu_memory_utilization: float = 0.9,
        render_replicas: int = 4,
        render_batch_size: int = 1,
        assemble_replicas: int = 4,
        assemble_batch_size: int = 4,
        runtime_env: Mapping[str, object] | None = None,
        dpi: int = 200,
    ) -> None:
        """Declare actor recipes without importing or loading model packages."""

        actor_env = dict(runtime_env or {})
        self.render = Map(
            MinerUPdfToPages,
            replicas=render_replicas,
            batch_size=render_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            num_cpus=1.0,
            num_gpus=0.0,
            runtime_env=actor_env,
        ).pre_init(dpi=dpi)
        self.ocr = Map(
            MinerUVlmOcrPage,
            replicas=replicas,
            batch_size=batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            num_cpus=1.0,
            num_gpus=1.0,
            runtime_env=actor_env,
        ).pre_init(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.assemble = Map(
            MinerUAssembleDoc,
            replicas=assemble_replicas,
            batch_size=assemble_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            num_cpus=1.0,
            num_gpus=0.0,
            runtime_env=actor_env,
        ).pre_init(output_dir=output_dir)

    def forward(self, pdfs):
        """Trace render, page expansion, OCR, ordered reduction, and assembly."""

        page_groups = self.render(pdfs)
        pages = expand(page_groups)
        page_contents = self.ocr(pages)
        content_groups = reduce(page_contents)
        documents = self.assemble(
            pdfs=pdfs,
            contents=content_groups,
            pages=page_groups,
        )
        return {"document": documents}


@dataclass(frozen=True, slots=True)
class DummyPdf:
    """A dependency-free source document with a requested page fan-out."""

    document_id: str
    page_count: int


@dataclass(frozen=True, slots=True)
class DummyPage:
    """A synthetic rendered page carrying a stable document ordinal."""

    document_id: str
    page_id: int
    payload: str


@dataclass(frozen=True, slots=True)
class DummyOcrContent:
    """A synthetic OCR result that retains its source page identity."""

    document_id: str
    page_id: int
    text: str


@dataclass(frozen=True, slots=True)
class DummyDocument:
    """A synthetic final document used to assert ordered page reduction."""

    document_id: str
    page_ids: tuple[int, ...]
    markdown: str


class DummyPdfToPages:
    """Expand synthetic PDF records into variable-length structural page lists."""

    def __init__(self, payload_prefix: str = "rendered") -> None:
        """Store a deterministic payload prefix for generated pages."""

        self.payload_prefix = payload_prefix

    def run(self, pdfs: list[DummyPdf]) -> list[list[DummyPage]]:
        """Create pages in source ordinal order for each synthetic document."""

        groups: list[list[DummyPage]] = []
        for pdf in pdfs:
            if pdf.page_count < 0:
                raise ValueError("dummy page_count must be non-negative")
            groups.append(
                [
                    DummyPage(
                        document_id=pdf.document_id,
                        page_id=page_id,
                        payload=(
                            f"{self.payload_prefix}:"
                            f"{pdf.document_id}:{page_id}"
                        ),
                    )
                    for page_id in range(pdf.page_count)
                ]
            )
        return groups


class DummyOcrPage:
    """Produce deterministic OCR values without Flash-MinerU or a GPU."""

    def __init__(self, delay_scale_s: float = 0.0) -> None:
        """Configure an optional deterministic delay for completion reordering."""

        if delay_scale_s < 0:
            raise ValueError("delay_scale_s must be non-negative")
        self.delay_scale_s = delay_scale_s

    def run(self, pages: list[DummyPage]) -> list[DummyOcrContent]:
        """Return one identity-preserving OCR record per input page."""

        if self.delay_scale_s and pages:
            delay_bucket = sum(
                page.page_id + len(page.document_id) for page in pages
            ) % 3
            time.sleep(self.delay_scale_s * delay_bucket)
        return [
            DummyOcrContent(
                document_id=page.document_id,
                page_id=page.page_id,
                text=f"text({page.payload})",
            )
            for page in pages
        ]


class DummyAssembleDoc:
    """Validate aligned ordered groups and assemble synthetic documents."""

    def __init__(self, separator: str = "\n") -> None:
        """Store the text separator used by the final assembly MAP."""

        self.separator = separator

    def run(
        self,
        pdfs: list[DummyPdf],
        contents: list[list[DummyOcrContent]],
        pages: list[list[DummyPage]],
    ) -> list[DummyDocument]:
        """Assemble documents, rejecting missing, duplicated, or unordered pages."""

        if not (len(pdfs) == len(contents) == len(pages)):
            raise ValueError("assemble batch columns must have equal lengths")

        documents: list[DummyDocument] = []
        for pdf, page_contents, page_records in zip(pdfs, contents, pages):
            expected_ids = list(range(pdf.page_count))
            page_ids = [page.page_id for page in page_records]
            content_ids = [content.page_id for content in page_contents]
            if page_ids != expected_ids:
                raise ValueError(
                    f"rendered pages for {pdf.document_id!r} are unordered: "
                    f"{page_ids!r}"
                )
            if content_ids != expected_ids:
                raise ValueError(
                    f"reduced OCR for {pdf.document_id!r} is unordered: "
                    f"{content_ids!r}"
                )
            if any(
                page.document_id != pdf.document_id for page in page_records
            ):
                raise ValueError("rendered page crossed document scope")
            if any(
                content.document_id != pdf.document_id
                for content in page_contents
            ):
                raise ValueError("OCR content crossed document scope")
            documents.append(
                DummyDocument(
                    document_id=pdf.document_id,
                    page_ids=tuple(content_ids),
                    markdown=self.separator.join(
                        content.text for content in page_contents
                    ),
                )
            )
        return documents


class MinerUDummyPipeline(Pipeline):
    """CPU-only MinerU-shaped graph for end-to-end V3 validation."""

    def __init__(
        self,
        *,
        replicas: int = 2,
        batch_size: int = 3,
        max_batch_wait_ms: float = 2.0,
        render_replicas: int = 1,
        render_batch_size: int = 2,
        assemble_replicas: int = 1,
        assemble_batch_size: int = 2,
        runtime_env: Mapping[str, object] | None = None,
        delay_scale_s: float = 0.0,
    ) -> None:
        """Declare a dependency-free actor graph with CPU-only resources."""

        actor_env = dict(runtime_env or {})
        self.render = Map(
            DummyPdfToPages,
            replicas=render_replicas,
            batch_size=render_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            num_cpus=1.0,
            num_gpus=0.0,
            runtime_env=actor_env,
        ).pre_init()
        self.ocr = Map(
            DummyOcrPage,
            replicas=replicas,
            batch_size=batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            num_cpus=1.0,
            num_gpus=0.0,
            runtime_env=actor_env,
        ).pre_init(delay_scale_s=delay_scale_s)
        self.assemble = Map(
            DummyAssembleDoc,
            replicas=assemble_replicas,
            batch_size=assemble_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            num_cpus=1.0,
            num_gpus=0.0,
            runtime_env=actor_env,
        ).pre_init()

    def forward(self, pdfs):
        """Trace the same five-stage topology used by the real workload."""

        page_groups = self.render(pdfs)
        pages = expand(page_groups)
        page_contents = self.ocr(pages)
        content_groups = reduce(page_contents)
        documents = self.assemble(
            pdfs=pdfs,
            contents=content_groups,
            pages=page_groups,
        )
        return {"document": documents}


def make_dummy_pdfs(
    limit: int,
    page_counts: Sequence[int] = DEFAULT_DUMMY_PAGE_COUNTS,
) -> list[DummyPdf]:
    """Build deterministic multi-document input with variable page fan-out."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not page_counts:
        raise ValueError("page_counts must not be empty")
    if any(count < 0 for count in page_counts):
        raise ValueError("page_counts must be non-negative")
    return [
        DummyPdf(
            document_id=f"dummy-{index:03d}",
            page_count=page_counts[index % len(page_counts)],
        )
        for index in range(limit)
    ]


def discover_pdfs(input_dir: str, limit: int) -> list[str]:
    """Return a stable, bounded list of PDFs without importing MinerU."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    pdfs = sorted(
        str(path)
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    selected = pdfs[:limit]
    if not selected:
        raise FileNotFoundError(f"no PDFs found under {root}")
    return selected


def real_runtime_env(
    flash_repo: str,
    base: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Add the Flash-MinerU repository to a copied Ray runtime environment."""

    runtime_env = dict(base or {})
    raw_env_vars = runtime_env.get("env_vars", {})
    if not isinstance(raw_env_vars, Mapping):
        raise TypeError("runtime_env['env_vars'] must be a mapping")
    env_vars = {str(key): str(value) for key, value in raw_env_vars.items()}
    inherited = env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    pythonpath = os.pathsep.join(
        path
        for path in (
            str(Path(flash_repo).expanduser().resolve()),
            os.getcwd(),
            inherited,
        )
        if path
    )
    env_vars["PYTHONPATH"] = pythonpath
    runtime_env["env_vars"] = env_vars
    return runtime_env


def compile_pipeline(pipeline: Pipeline):
    """Compile a V3 pipeline through its public authoring facade."""

    compiler = getattr(pipeline, "compile", None)
    if callable(compiler):
        return compiler()
    from .. import api as public_api

    compile_function = getattr(public_api, "compile", None)
    if callable(compile_function):
        return compile_function(pipeline)
    from ..model.graph import compile as compile_graph

    return compile_graph(pipeline)


def execute_pipeline(
    pipeline: Pipeline,
    source: Iterable[object],
    *,
    output_name: str = "document",
) -> list[object]:
    """Run a compiled V3 graph and materialize one named output per root."""

    try:
        from ..api import ExecutorSession
    except ImportError:
        from ..runtime import ExecutorSession

    graph = compile_pipeline(pipeline)
    session = ExecutorSession(graph)
    stream = None
    try:
        stream = session.run(source)
        root_results = stream.collect(order="input")
        return [
            _materialize_output(root_result, output_name)
            for root_result in root_results
        ]
    finally:
        if stream is not None:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
        close_session = getattr(session, "close", None)
        if callable(close_session):
            close_session()


def _materialize_output(root_result: object, output_name: str) -> object:
    """Materialize one successful detached output and release its refs."""

    status = getattr(root_result, "status", None)
    status_name = getattr(status, "value", status)
    if status_name != "success":
        raise RuntimeError(f"root did not succeed: {status_name!r}")
    outputs = getattr(root_result, "outputs", None)
    if outputs is None or output_name not in outputs:
        raise RuntimeError(f"root result has no {output_name!r} output")
    leaf = outputs[output_name]
    leaf_state = getattr(getattr(leaf, "state", None), "value", None)
    if leaf_state != "present":
        failure = getattr(leaf, "failure", None)
        raise RuntimeError(
            f"output {output_name!r} is {leaf_state!r}: {failure!r}"
        )
    detached = getattr(leaf, "value", None)
    if detached is None:
        raise RuntimeError(f"output {output_name!r} has no detached value")
    try:
        return detached.get()
    finally:
        close_value = getattr(detached, "close", None)
        if callable(close_value):
            close_value()


def jsonable(value: object) -> object:
    """Convert benchmark outputs into recursively JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


__all__ = [
    "DEFAULT_DUMMY_PAGE_COUNTS",
    "DEFAULT_FLASH_REPO",
    "DEFAULT_MODEL",
    "DummyAssembleDoc",
    "DummyDocument",
    "DummyOcrContent",
    "DummyOcrPage",
    "DummyPage",
    "DummyPdf",
    "DummyPdfToPages",
    "MinerUAssembleDoc",
    "MinerUDummyPipeline",
    "MinerUPdfToPages",
    "MinerUV3Pipeline",
    "MinerUVlmOcrPage",
    "compile_pipeline",
    "discover_pdfs",
    "execute_pipeline",
    "jsonable",
    "make_dummy_pdfs",
    "real_runtime_env",
]
