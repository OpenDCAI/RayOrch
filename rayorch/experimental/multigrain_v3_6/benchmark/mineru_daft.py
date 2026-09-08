"""Daft 0.7.x 上的公平 MinerU PDF→Page→OCR→PDF 竞品基线。

本 runner 使用 Daft 自身最自然的表达：class UDF + explode + batched GPU UDF +
groupby/map_groups。它与 V3.6 复用完全相同的 render、OCR 和 assemble 业务实现，显式携带
parent/ordinal，并保存 Daft 的优化前、优化后和物理计划。

Daft 0.7.21 会把 Python ``dict`` 推断为 Struct，而其 Struct→Python cast 对当前 class
UDF 路径尚未实现。因此复杂 page/content 使用只含一个字段的 opaque Python envelope。
默认 ``full_value`` arm 不改变 payload，也不引入 RayOrch 的引用协议；它会在 groupby 中
shuffle 完整 page/content 值。``reference_only`` 仅用于控制变量实验：render、OCR、identity、
batching 和 Reduce 均保持不变，只让轻量 manifest 进入 hash regroup，完整 payload 由显式
owner actor 管理。该专家控制臂故意暴露 acquire/release 成本，不是推荐的 Daft production
pattern。两臂的物理计划和资源事实都会写入 artifact。

Daft 是可选论文依赖，不应加入 RayOrch 的 runtime requirements。推荐在隔离环境运行：

    python -m venv --system-site-packages /tmp/rayorch-vldb-daft
    /tmp/rayorch-vldb-daft/bin/python -m pip install daft==0.7.21
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import daft  # pyright: ignore[reportMissingImports]
from daft import DataType, Series  # pyright: ignore[reportMissingImports]

from ...multigrain_v3.benchmark.mineru import (
    DEFAULT_FLASH_REPO,
    DEFAULT_MODEL,
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    ResourceSampler,
    _runtime_env,
)
from .mineru_poison import (
    DEFAULT_POISON_SEED,
    PoisonPage,
    gpu_memory_peaks,
    load_pdf_manifest,
    pdf_page_counts,
    select_poison_pages,
    worker_resource_options,
)


@dataclass(frozen=True, slots=True)
class _PagePayload:
    """Force a rendered page to remain one opaque Daft Python value."""

    value: dict[str, Any]
    render_started_s: float = 0.0
    render_finished_s: float = 0.0


@dataclass(frozen=True, slots=True)
class _ContentPayload:
    """Keep OCR output together with non-invasive batch observations."""

    value: Any
    batch_token: str
    batch_size: int
    actor_init_started_s: float = 0.0
    actor_ready_s: float = 0.0
    batch_started_s: float = 0.0
    batch_finished_s: float = 0.0
    payload_publish_started_s: float = 0.0
    payload_publish_finished_s: float = 0.0
    payload_block_bytes: int = 0
    poisoned: bool = False


@dataclass(frozen=True, slots=True)
class _ReferenceManifest:
    """Describe one row while keeping its page/content payload out of regroup."""

    page_ordinal: int
    store_slot: int
    block_id: int
    payload_row: int
    batch_token: str
    batch_size: int
    actor_init_started_s: float = 0.0
    actor_ready_s: float = 0.0
    batch_started_s: float = 0.0
    batch_finished_s: float = 0.0
    payload_publish_started_s: float = 0.0
    payload_publish_finished_s: float = 0.0
    payload_block_bytes: int = 0
    poisoned: bool = False


@dataclass(frozen=True, slots=True)
class _DocumentPayload:
    """Return one business output plus enough facts to audit batching."""

    value: dict[str, Any]
    page_ids: tuple[int, ...]
    batch_tokens: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    render_started_s: tuple[float, ...]
    render_finished_s: tuple[float, ...]
    actor_init_started_s: tuple[float, ...]
    actor_ready_s: tuple[float, ...]
    batch_started_s: tuple[float, ...]
    batch_finished_s: tuple[float, ...]
    payload_publish_started_s: tuple[float, ...]
    payload_publish_finished_s: tuple[float, ...]
    payload_block_bytes: tuple[int, ...]
    assemble_started_s: float
    assemble_finished_s: float


_PAYLOAD_STORE_CLASS = None


def _get_payload_store_class():
    """Create the explicit owner used by the reference-only control arm."""

    global _PAYLOAD_STORE_CLASS
    if _PAYLOAD_STORE_CLASS is not None:
        return _PAYLOAD_STORE_CLASS

    import ray  # pyright: ignore[reportMissingImports]

    @ray.remote(max_concurrency=1, max_restarts=0)
    class PayloadStore:
        """Own coarse OCR blocks until every parent group has consumed them."""

        def __init__(self) -> None:
            self.next_id = 0
            self.refs: dict[int, Any] = {}
            self.remaining: dict[int, int] = {}
            self.rows: dict[int, int] = {}
            self.sizes: dict[int, int] = {}
            self.total_blocks = 0
            self.total_rows = 0
            self.total_payload_bytes = 0
            self.peak_blocks = 0
            self.peak_rows = 0
            self.peak_payload_bytes = 0

        def store(
            self,
            block: tuple[Any, ...],
            consumers: int,
        ) -> tuple[int, int]:
            if consumers <= 0:
                raise ValueError("payload block must have consumers")
            block_id = self.next_id
            self.next_id += 1
            ref = ray.put(block)
            location = ray.experimental.get_object_locations([ref]).get(ref, {})
            size = int(location.get("object_size") or 0)
            self.refs[block_id] = ref
            self.remaining[block_id] = consumers
            self.rows[block_id] = len(block)
            self.sizes[block_id] = size
            self.total_blocks += 1
            self.total_rows += len(block)
            self.total_payload_bytes += size
            self.peak_blocks = max(self.peak_blocks, len(self.refs))
            self.peak_rows = max(self.peak_rows, sum(self.rows.values()))
            self.peak_payload_bytes = max(
                self.peak_payload_bytes,
                sum(self.sizes.values()),
            )
            return block_id, size

        def acquire(self, block_id: int):
            return self.refs[block_id]

        def release(self, block_id: int) -> None:
            self.remaining[block_id] -= 1
            if self.remaining[block_id] == 0:
                self.remaining.pop(block_id)
                self.rows.pop(block_id)
                self.sizes.pop(block_id)
                self.refs.pop(block_id)

        def stats(self) -> dict[str, int]:
            return {
                "live_blocks": len(self.refs),
                "live_rows": sum(self.rows.values()),
                "live_payload_bytes": sum(self.sizes.values()),
                "peak_blocks": self.peak_blocks,
                "peak_rows": self.peak_rows,
                "peak_payload_bytes": self.peak_payload_bytes,
                "total_blocks": self.total_blocks,
                "total_rows": self.total_rows,
                "total_payload_bytes": self.total_payload_bytes,
            }

    _PAYLOAD_STORE_CLASS = PayloadStore
    return PayloadStore


def _effective_source_partitions(
    args: argparse.Namespace,
    source_count: int,
) -> int:
    """Resolve Daft's independent, bounded source partition count once."""

    requested = (
        args.partitions
        if args.partitions is not None
        else args.render_replicas
    )
    if requested <= 0:
        raise ValueError("partitions must be positive")
    return max(1, min(source_count, requested))


def _package_ocr_batch(
    *,
    pages: list[_PagePayload],
    contents: list[Any],
    parent_ids: list[int],
    stores: tuple[Any, ...],
    batch_token: str,
    actor_init_started_s: float,
    actor_ready_s: float,
    batch_started_s: float,
    batch_finished_s: float,
    poisoned: list[bool] | None = None,
) -> list[_ContentPayload] | list[_ReferenceManifest]:
    """Apply the single controlled variable at the regroup boundary."""

    if not (len(pages) == len(contents) == len(parent_ids)):
        raise ValueError("OCR pages, contents, and parent ids must align")
    size = len(pages)
    poison_flags = poisoned if poisoned is not None else [False] * size
    if len(poison_flags) != size:
        raise ValueError("OCR poison flags must align with input rows")
    if not stores:
        return [
            _ContentPayload(
                value=content,
                batch_token=batch_token,
                batch_size=size,
                actor_init_started_s=actor_init_started_s,
                actor_ready_s=actor_ready_s,
                batch_started_s=batch_started_s,
                batch_finished_s=batch_finished_s,
                poisoned=poison_flags[row],
            )
            for row, content in enumerate(contents)
        ]

    import ray  # pyright: ignore[reportMissingImports]

    store_slot = min(parent_ids) % len(stores)
    publish_started_s = time.time()
    block_id, block_bytes = ray.get(
        stores[store_slot].store.remote(
            tuple(zip(pages, contents)),
            len(set(parent_ids)),
        )
    )
    publish_finished_s = time.time()
    return [
        _ReferenceManifest(
            page_ordinal=int(page.value["page_id"]),
            store_slot=store_slot,
            block_id=int(block_id),
            payload_row=row,
            batch_token=batch_token,
            batch_size=size,
            actor_init_started_s=actor_init_started_s,
            actor_ready_s=actor_ready_s,
            batch_started_s=batch_started_s,
            batch_finished_s=batch_finished_s,
            payload_publish_started_s=publish_started_s,
            payload_publish_finished_s=publish_finished_s,
            payload_block_bytes=int(block_bytes),
            poisoned=poison_flags[row],
        )
        for row, page in enumerate(pages)
    ]


def _build_dataframe(
    args: argparse.Namespace,
    pdfs: list[str],
    stores: tuple[Any, ...] = (),
    poison_manifest: tuple[PoisonPage, ...] = (),
):
    """Build one Daft plan whose only variable is regroup representation."""

    reference_only = args.regroup_mode == "reference_only"
    if reference_only != bool(stores):
        raise ValueError("reference_only mode requires payload stores")
    poison_pages = {
        (os.path.abspath(entry.pdf_path), entry.page_id)
        for entry in poison_manifest
    }

    @daft.cls(
        cpus=1,
        max_concurrency=args.render_replicas,
        ray_options=worker_resource_options(args.cpu_worker_resource),
    )
    class RenderPdf:
        def __init__(self, dpi: int) -> None:
            self.renderer = MinerUPdfToPages(dpi=dpi)

        @daft.method(return_dtype=DataType.list(DataType.python()))
        def run(self, path: str) -> list[_PagePayload]:
            started = time.time()
            pages = self.renderer.run([path])[0]
            finished = time.time()
            return [
                _PagePayload(page, started, finished)
                for page in pages
            ]

    if args.smoke_no_model:

        @daft.cls(
            cpus=1,
            max_concurrency=args.replicas,
            ray_options=worker_resource_options(args.cpu_worker_resource),
        )
        class SmokeOcrPages:
            def __init__(self) -> None:
                self.actor_init_started_s = time.time()
                self.actor_token = f"{os.getpid()}-{uuid.uuid4().hex}"
                self.calls = 0
                self.actor_ready_s = time.time()

            @daft.method.batch(
                return_dtype=DataType.python(),
                batch_size=args.batch_size,
            )
            def run(
                self,
                pages: Series,
                parent_ids: Series,
            ) -> list[Any]:
                values = pages.to_pylist()
                parents = [int(value) for value in parent_ids.to_pylist()]
                poisoned = [
                    (
                        os.path.abspath(str(page.value["pdf_path"])),
                        int(page.value["page_id"]),
                    )
                    in poison_pages
                    for page in values
                ]
                if args.poison_policy == "raise" and any(poisoned):
                    raise ValueError("injected deterministic poison page")
                token = f"{self.actor_token}-{self.calls}"
                self.calls += 1
                started = time.time()
                finished = time.time()
                return _package_ocr_batch(
                    pages=values,
                    contents=[
                        None if bad else int(page.value["page_id"])
                        for page, bad in zip(values, poisoned, strict=True)
                    ],
                    parent_ids=parents,
                    stores=stores,
                    batch_token=token,
                    actor_init_started_s=self.actor_init_started_s,
                    actor_ready_s=self.actor_ready_s,
                    batch_started_s=started,
                    batch_finished_s=finished,
                    poisoned=poisoned,
                )

        ocr = SmokeOcrPages()
    else:

        @daft.cls(
            cpus=1,
            gpus=1,
            max_concurrency=args.replicas,
            ray_options=worker_resource_options(args.gpu_worker_resource),
        )
        class GpuOcrPages:
            def __init__(
                self,
                model: str,
                gpu_memory_utilization: float,
            ) -> None:
                self.actor_init_started_s = time.time()
                self.ocr = MinerUVlmOcrPage(
                    model=model,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
                self.actor_token = f"{os.getpid()}-{uuid.uuid4().hex}"
                self.calls = 0
                self.actor_ready_s = time.time()

            @daft.method.batch(
                return_dtype=DataType.python(),
                batch_size=args.batch_size,
            )
            def run(
                self,
                pages: Series,
                parent_ids: Series,
            ) -> list[Any]:
                values = pages.to_pylist()
                parents = [int(value) for value in parent_ids.to_pylist()]
                poisoned = [
                    (
                        os.path.abspath(str(page.value["pdf_path"])),
                        int(page.value["page_id"]),
                    )
                    in poison_pages
                    for page in values
                ]
                if args.poison_policy == "raise" and any(poisoned):
                    raise ValueError("injected deterministic poison page")
                started = time.time()
                live_indices = [
                    index for index, bad in enumerate(poisoned) if not bad
                ]
                live_contents = (
                    self.ocr.run(
                        [values[index].value for index in live_indices]
                    )
                    if live_indices
                    else []
                )
                by_index = dict(
                    zip(live_indices, live_contents, strict=True)
                )
                contents = [
                    None if bad else by_index[index]
                    for index, bad in enumerate(poisoned)
                ]
                finished = time.time()
                if len(contents) != len(values):
                    raise ValueError(
                        "MinerU OCR output count does not match Daft input batch"
                    )
                token = f"{self.actor_token}-{self.calls}"
                self.calls += 1
                return _package_ocr_batch(
                    pages=values,
                    contents=contents,
                    parent_ids=parents,
                    stores=stores,
                    batch_token=token,
                    actor_init_started_s=self.actor_init_started_s,
                    actor_ready_s=self.actor_ready_s,
                    batch_started_s=started,
                    batch_finished_s=finished,
                    poisoned=poisoned,
                )

        ocr = GpuOcrPages(args.model, args.gpu_memory_utilization)

    @daft.cls(
        cpus=1,
        max_concurrency=args.reduce_replicas,
        ray_options=worker_resource_options(args.cpu_worker_resource),
    )
    class AssemblePdf:
        def __init__(
            self,
            output_dir: str,
            metadata_only: bool,
            poison_policy: str,
        ) -> None:
            self.metadata_only = metadata_only
            self.poison_policy = poison_policy
            self.assembler = (
                None
                if metadata_only
                else MinerUAssembleDoc(output_dir=output_dir)
            )

        @daft.method.batch(return_dtype=DataType.python())
        def run(
            self,
            contents: Series,
            pages: Series,
            paths: Series,
        ) -> list[_DocumentPayload]:
            assemble_started_s = time.time()
            rows = sorted(
                zip(
                    pages.to_pylist(),
                    contents.to_pylist(),
                    paths.to_pylist(),
                ),
                key=lambda row: int(row[0].value["page_id"]),
            )
            if not rows:
                raise ValueError("Daft emitted an empty PDF group")
            path = str(rows[0][2])
            if any(str(row_path) != path for _, _, row_path in rows):
                raise ValueError("Daft parent group mixes multiple PDFs")
            input_page_ids = tuple(
                int(page.value["page_id"]) for page, _, _ in rows
            )
            if input_page_ids != tuple(range(len(input_page_ids))):
                raise ValueError(
                    "non-contiguous or duplicate page ordinals: "
                    f"{input_page_ids}"
                )
            poison_page_ids = tuple(
                int(page.value["page_id"])
                for page, content, _ in rows
                if content.poisoned
            )
            drop_parent = (
                bool(poison_page_ids)
                and self.poison_policy == "drop_parent"
            )
            live = [row for row in rows if not row[1].poisoned]
            if not live and not drop_parent:
                raise ValueError("cannot assemble a PDF when every page is poisoned")
            page_values = [page.value for page, _, _ in live]
            content_values = [content.value for _, content, _ in live]
            page_ids = (
                ()
                if drop_parent
                else tuple(int(page["page_id"]) for page in page_values)
            )
            if drop_parent:
                output = {
                    "pdf": Path(path).stem,
                    "pages": 0,
                    "page_ids": (),
                    "input_pages": len(input_page_ids),
                    "poisoned_pages": len(poison_page_ids),
                    "poison_page_ids": poison_page_ids,
                    "dropped_parent": True,
                }
            elif self.metadata_only:
                output = {
                    "pdf": Path(path).stem,
                    "pages": len(page_values),
                    "page_ids": page_ids,
                    "input_pages": len(input_page_ids),
                    "poisoned_pages": len(poison_page_ids),
                    "poison_page_ids": poison_page_ids,
                }
            else:
                assert self.assembler is not None
                output = self.assembler.run(
                    [content_values],
                    [page_values],
                    [Path(path).stem],
                )[0]
                output.update(
                    input_pages=len(input_page_ids),
                    poisoned_pages=len(poison_page_ids),
                    poison_page_ids=list(poison_page_ids),
                )
            assemble_finished_s = time.time()
            return [
                _DocumentPayload(
                    value=output,
                    page_ids=page_ids,
                    batch_tokens=tuple(
                        content.batch_token for _, content, _ in rows
                    ),
                    batch_sizes=tuple(
                        content.batch_size for _, content, _ in rows
                    ),
                    render_started_s=tuple(
                        page.render_started_s for page, _, _ in rows
                    ),
                    render_finished_s=tuple(
                        page.render_finished_s for page, _, _ in rows
                    ),
                    actor_init_started_s=tuple(
                        content.actor_init_started_s
                        for _, content, _ in rows
                    ),
                    actor_ready_s=tuple(
                        content.actor_ready_s for _, content, _ in rows
                    ),
                    batch_started_s=tuple(
                        content.batch_started_s for _, content, _ in rows
                    ),
                    batch_finished_s=tuple(
                        content.batch_finished_s for _, content, _ in rows
                    ),
                    payload_publish_started_s=tuple(
                        content.payload_publish_started_s
                        for _, content, _ in rows
                    ),
                    payload_publish_finished_s=tuple(
                        content.payload_publish_finished_s
                        for _, content, _ in rows
                    ),
                    payload_block_bytes=tuple(
                        content.payload_block_bytes
                        for _, content, _ in rows
                    ),
                    assemble_started_s=assemble_started_s,
                    assemble_finished_s=assemble_finished_s,
                )
            ]

    @daft.cls(
        cpus=1,
        max_concurrency=args.reduce_replicas,
        ray_options=worker_resource_options(args.cpu_worker_resource),
    )
    class AssembleReferencePdf:
        def __init__(
            self,
            output_dir: str,
            metadata_only: bool,
            poison_policy: str,
        ) -> None:
            self.metadata_only = metadata_only
            self.poison_policy = poison_policy
            self.assembler = (
                None
                if metadata_only
                else MinerUAssembleDoc(output_dir=output_dir)
            )

        @daft.method.batch(return_dtype=DataType.python())
        def run(
            self,
            manifests: Series,
            paths: Series,
        ) -> list[_DocumentPayload]:
            import ray  # pyright: ignore[reportMissingImports]

            assemble_started_s = time.time()
            rows = sorted(
                zip(manifests.to_pylist(), paths.to_pylist()),
                key=lambda row: int(row[0].page_ordinal),
            )
            if not rows:
                raise ValueError("Daft emitted an empty PDF group")
            path = str(rows[0][1])
            if any(str(row_path) != path for _, row_path in rows):
                raise ValueError("Daft parent group mixes multiple PDFs")
            input_page_ids = tuple(
                int(manifest.page_ordinal) for manifest, _ in rows
            )
            if input_page_ids != tuple(range(len(input_page_ids))):
                raise ValueError(
                    "non-contiguous or duplicate page ordinals: "
                    f"{input_page_ids}"
                )

            blocks: dict[tuple[int, int], tuple[Any, ...]] = {}
            pages_and_contents = []
            for manifest, _ in rows:
                key = (int(manifest.store_slot), int(manifest.block_id))
                if key not in blocks:
                    nested_ref = ray.get(stores[key[0]].acquire.remote(key[1]))
                    blocks[key] = ray.get(nested_ref)
                pages_and_contents.append(
                    blocks[key][int(manifest.payload_row)]
                )
            page_payloads = [value[0] for value in pages_and_contents]
            poison_page_ids = tuple(
                int(manifest.page_ordinal)
                for manifest, _ in rows
                if manifest.poisoned
            )
            drop_parent = (
                bool(poison_page_ids)
                and self.poison_policy == "drop_parent"
            )
            live = [
                (payload, value[1])
                for (manifest, _), payload, value in zip(
                    rows, page_payloads, pages_and_contents, strict=True
                )
                if not manifest.poisoned
            ]
            if not live and not drop_parent:
                raise ValueError("cannot assemble a PDF when every page is poisoned")
            page_values = [page.value for page, _ in live]
            content_values = [content for _, content in live]
            page_ids = (
                ()
                if drop_parent
                else tuple(int(page["page_id"]) for page in page_values)
            )
            if drop_parent:
                output = {
                    "pdf": Path(path).stem,
                    "pages": 0,
                    "page_ids": (),
                    "input_pages": len(input_page_ids),
                    "poisoned_pages": len(poison_page_ids),
                    "poison_page_ids": poison_page_ids,
                    "dropped_parent": True,
                }
            elif self.metadata_only:
                output = {
                    "pdf": Path(path).stem,
                    "pages": len(page_values),
                    "page_ids": page_ids,
                    "input_pages": len(input_page_ids),
                    "poisoned_pages": len(poison_page_ids),
                    "poison_page_ids": poison_page_ids,
                }
            else:
                assert self.assembler is not None
                output = self.assembler.run(
                    [content_values],
                    [page_values],
                    [Path(path).stem],
                )[0]
                output.update(
                    input_pages=len(input_page_ids),
                    poisoned_pages=len(poison_page_ids),
                    poison_page_ids=list(poison_page_ids),
                )
            ray.get(
                [
                    stores[store_slot].release.remote(block_id)
                    for store_slot, block_id in blocks
                ]
            )
            assemble_finished_s = time.time()
            return [
                _DocumentPayload(
                    value=output,
                    page_ids=page_ids,
                    batch_tokens=tuple(
                        manifest.batch_token for manifest, _ in rows
                    ),
                    batch_sizes=tuple(
                        manifest.batch_size for manifest, _ in rows
                    ),
                    render_started_s=tuple(
                        page.render_started_s for page in page_payloads
                    ),
                    render_finished_s=tuple(
                        page.render_finished_s for page in page_payloads
                    ),
                    actor_init_started_s=tuple(
                        manifest.actor_init_started_s for manifest, _ in rows
                    ),
                    actor_ready_s=tuple(
                        manifest.actor_ready_s for manifest, _ in rows
                    ),
                    batch_started_s=tuple(
                        manifest.batch_started_s for manifest, _ in rows
                    ),
                    batch_finished_s=tuple(
                        manifest.batch_finished_s for manifest, _ in rows
                    ),
                    payload_publish_started_s=tuple(
                        manifest.payload_publish_started_s
                        for manifest, _ in rows
                    ),
                    payload_publish_finished_s=tuple(
                        manifest.payload_publish_finished_s
                        for manifest, _ in rows
                    ),
                    payload_block_bytes=tuple(
                        manifest.payload_block_bytes for manifest, _ in rows
                    ),
                    assemble_started_s=assemble_started_s,
                    assemble_finished_s=assemble_finished_s,
                )
            ]

    source_partitions = _effective_source_partitions(args, len(pdfs))
    source = daft.from_pydict(
        {
            "parent_id": list(range(len(pdfs))),
            # ``load_pdf_manifest`` deliberately returns an immutable tuple,
            # while Daft 0.7.x accepts column inputs as lists rather than
            # arbitrary Sequences.
            "pdf_path": list(pdfs),
        }
    ).into_partitions(source_partitions)
    renderer = RenderPdf(args.render_dpi)
    pages = source.with_column(
        "page",
        renderer.run(source["pdf_path"]),
    ).explode("page", ignore_empty_and_null=True)
    contents = pages.with_column(
        "content",
        ocr.run(pages["page"], pages["parent_id"]),
    )
    if reference_only:
        manifest = contents.select("parent_id", "pdf_path", "content")
        assembler = AssembleReferencePdf(
            os.path.abspath(args.output_dir),
            bool(
                args.smoke_no_model
                or args.assemble_mode == "metadata_only"
            ),
            args.poison_policy,
        )
        return manifest.groupby("parent_id").map_groups(
            assembler.run(manifest["content"], manifest["pdf_path"])
        )
    assembler = AssemblePdf(
        os.path.abspath(args.output_dir),
        bool(args.smoke_no_model or args.assemble_mode == "metadata_only"),
        args.poison_policy,
    )
    return contents.groupby("parent_id").map_groups(
        assembler.run(
            contents["content"], contents["page"], contents["pdf_path"]
        )
    )


def _explain(dataframe: Any) -> str:
    """Capture all three Daft plans without adding work to the timed region."""

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        dataframe.explain(show_all=True)
    return output.getvalue()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run one isolated Daft arm and persist metrics, plan, and outputs."""

    import ray  # pyright: ignore[reportMissingImports]

    flash_repo = os.path.abspath(args.flash_repo)
    if args.input_manifest:
        pdfs, page_counts = load_pdf_manifest(
            args.input_manifest,
            limit=args.limit,
        )
    else:
        pdfs = tuple(
            sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
        )
        if not pdfs:
            raise FileNotFoundError(f"no PDFs found under {flash_repo}")
        page_counts = pdf_page_counts(pdfs)
    poison_manifest = select_poison_pages(
        pdfs,
        page_counts,
        count=args.poison_count,
        page_id=args.poison_page_id,
        seed=args.poison_seed,
        exact_indices=(
            () if args.poison_pdf_index is None else (args.poison_pdf_index,)
        ),
    )
    runtime_env = _runtime_env(flash_repo)
    os.environ.update(runtime_env["env_vars"])
    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        init_kwargs: dict[str, Any] = {
            "include_dashboard": False,
            "runtime_env": runtime_env,
        }
        if args.ray_address is None:
            init_kwargs.update(
                num_cpus=args.num_cpus,
                num_gpus=0 if args.smoke_no_model else args.replicas,
                object_store_memory=int(args.object_store_gb * 1024**3),
            )
        ray.init(address=args.ray_address, **init_kwargs)
    daft.set_runner_ray(noop_if_initialized=True)

    stores: tuple[Any, ...] = ()
    if args.regroup_mode == "reference_only":
        if args.payload_stores <= 0:
            raise ValueError("payload_stores must be positive")
        store_class = _get_payload_store_class()
        stores = tuple(
            store_class.options(
                num_cpus=0,
                **worker_resource_options(args.cpu_worker_resource),
            ).remote()
            for _ in range(args.payload_stores)
        )
    dataframe = _build_dataframe(args, pdfs, stores, poison_manifest)
    plan = _explain(dataframe)
    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started_epoch = time.time()
    started = time.perf_counter()
    payload_store_stats: tuple[dict[str, int], ...] = ()
    try:
        collected = dataframe.collect().to_pydict()
        wall = time.perf_counter() - started
        finished_epoch = time.time()
        if stores:
            payload_store_stats = tuple(
                ray.get([store.stats.remote() for store in stores])
            )
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
        for store in stores:
            ray.kill(store, no_restart=True)
        if started_ray_here:
            ray.shutdown()

    rows = sorted(
        zip(collected["parent_id"], collected["content"]),
        key=lambda row: int(row[0]),
    )
    parent_ids = [int(parent_id) for parent_id, _ in rows]
    documents = [document for _, document in rows]
    if parent_ids != list(range(len(pdfs))):
        raise ValueError(f"Daft changed source identity/order: {parent_ids}")
    if not all(isinstance(value, _DocumentPayload) for value in documents):
        raise TypeError("Daft output did not preserve document payloads")
    outputs = [value.value for value in documents]
    successful_outputs = [
        output for output in outputs if not output.get("dropped_parent", False)
    ]
    pages = sum(len(value.page_ids) for value in documents)
    input_pages = sum(page_counts)
    poison_observed = sum(
        int(output.get("poisoned_pages", 0)) for output in outputs
    )
    poisoned_parent_pages = sum(entry.pdf_pages for entry in poison_manifest)
    drop_parent = args.poison_policy == "drop_parent"
    batches: dict[
        str,
        tuple[int, float, float, float, float, float, float, int],
    ] = {}
    for document in documents:
        for values in zip(
            document.batch_tokens,
            document.batch_sizes,
            document.actor_init_started_s,
            document.actor_ready_s,
            document.batch_started_s,
            document.batch_finished_s,
            document.payload_publish_started_s,
            document.payload_publish_finished_s,
            document.payload_block_bytes,
        ):
            token = values[0]
            observation = (
                int(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                float(values[6]),
                float(values[7]),
                int(values[8]),
            )
            previous = batches.setdefault(token, observation)
            if previous != observation:
                raise ValueError(f"conflicting Daft batch size for {token}")
    batch_sizes = tuple(value[0] for value in batches.values())
    dispatched_pages = sum(batch_sizes)
    histogram = {
        str(size): batch_sizes.count(size)
        for size in sorted(set(batch_sizes))
    }
    actor_init_started = tuple(value[1] for value in batches.values())
    actor_ready = tuple(value[2] for value in batches.values())
    batch_started = tuple(value[3] for value in batches.values())
    batch_finished = tuple(value[4] for value in batches.values())
    payload_publish_started = tuple(
        value[5] for value in batches.values() if value[5] > 0
    )
    payload_publish_finished = tuple(
        value[6] for value in batches.values() if value[6] > 0
    )
    payload_block_bytes = tuple(value[7] for value in batches.values())
    render_started = tuple(
        value for document in documents for value in document.render_started_s
    )
    render_finished = tuple(
        value for document in documents for value in document.render_finished_s
    )
    assemble_started = tuple(
        document.assemble_started_s for document in documents
    )
    assemble_finished = tuple(
        document.assemble_finished_s for document in documents
    )
    gpu_peak = (
        ()
        if args.smoke_no_model
        else gpu_memory_peaks(gpu_samples, args.replicas)
    )
    payload = {
        "engine": f"daft_{args.regroup_mode}",
        "daft_version": daft.__version__,
        "backend": "smoke_cpu" if args.smoke_no_model else "mineru_vllm",
        "assemble_mode": (
            "metadata_only" if args.smoke_no_model else args.assemble_mode
        ),
        "n_pdf": len(pdfs),
        "pages": pages,
        "input_pages": input_pages,
        "docs": len(outputs),
        "successful_docs": len(successful_outputs),
        "poison_policy": args.poison_policy,
        "poison_seed": args.poison_seed,
        "poison_injected": len(poison_manifest),
        "poison_observed": poison_observed,
        "poison_manifest": [entry.as_dict() for entry in poison_manifest],
        "poisoned_parent_pages": poisoned_parent_pages,
        "expected_successful_docs": (
            len(pdfs) - len(poison_manifest) if drop_parent else len(pdfs)
        ),
        "expected_output_pages": (
            input_pages - poisoned_parent_pages
            if drop_parent
            else input_pages - len(poison_manifest)
        ),
        "poison_report_contract_passed": (
            poison_observed == len(poison_manifest)
            and len(successful_outputs)
            == (len(pdfs) - len(poison_manifest) if drop_parent else len(pdfs))
            and pages
            == (
                input_pages - poisoned_parent_pages
                if drop_parent
                else input_pages - len(poison_manifest)
            )
        ),
        "batch_size": args.batch_size,
        "replicas": args.replicas,
        "render_replicas": args.render_replicas,
        "render_dpi": args.render_dpi,
        "reduce_replicas": args.reduce_replicas,
        "source_partitions": _effective_source_partitions(args, len(pdfs)),
        "regroup_mode": args.regroup_mode,
        "run_wall_s": round(wall, 3),
        "measured_wall_s": round(wall, 3),
        "end_to_end_wall_s": round(wall, 3),
        "timing_scope": {
            "ray_init_included": False,
            "actor_model_startup_included": True,
            "ray_shutdown_included": False,
            "driver_correctness_sort_included": False,
        },
        "pages_per_s": round(pages / max(wall, 1e-9), 4),
        "ocr_rpc_count": len(batches),
        "ocr_dispatched_pages": dispatched_pages,
        "ocr_model_pages": dispatched_pages - poison_observed,
        "ocr_model_pages_saved_vs_baseline": (
            input_pages - dispatched_pages + poison_observed
        ),
        "ocr_grains_per_rpc": (
            dispatched_pages / len(batches) if batches else 0.0
        ),
        "ocr_batch_fill_ratio": (
            dispatched_pages / (len(batches) * args.batch_size)
            if batches
            else 0.0
        ),
        "ocr_batch_histogram": histogram,
        "timeline_s": {
            "first_render_start": (
                min(render_started) - started_epoch if render_started else None
            ),
            "last_render_finish": (
                max(render_finished) - started_epoch
                if render_finished
                else None
            ),
            "first_ocr_actor_init": (
                min(actor_init_started) - started_epoch
                if actor_init_started
                else None
            ),
            "all_ocr_actors_ready": (
                max(actor_ready) - started_epoch if actor_ready else None
            ),
            "first_ocr_batch_start": (
                min(batch_started) - started_epoch if batch_started else None
            ),
            "last_ocr_batch_finish": (
                max(batch_finished) - started_epoch
                if batch_finished
                else None
            ),
            "post_ocr_to_collect_end": (
                finished_epoch - max(batch_finished)
                if batch_finished
                else None
            ),
            "first_payload_publish_start": (
                min(payload_publish_started) - started_epoch
                if payload_publish_started
                else None
            ),
            "last_payload_publish_finish": (
                max(payload_publish_finished) - started_epoch
                if payload_publish_finished
                else None
            ),
            "first_assemble_start": (
                min(assemble_started) - started_epoch
                if assemble_started
                else None
            ),
            "last_assemble_finish": (
                max(assemble_finished) - started_epoch
                if assemble_finished
                else None
            ),
            "post_assemble_to_collect_end": (
                finished_epoch - max(assemble_finished)
                if assemble_finished
                else None
            ),
        },
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "explicit_lineage_columns": ["parent_id", "page.page_id"],
        "opaque_payload_envelopes": [
            "_PagePayload",
            (
                "_ReferenceManifest"
                if stores
                else "_ContentPayload"
            ),
            "_DocumentPayload",
        ],
        "regroup_operator": (
            "RayShuffle: Hash" if "RayShuffle: Hash" in plan else "unknown"
        ),
        "full_value_regroup": args.regroup_mode == "full_value",
        "payload_block_count": len(batches) if stores else 0,
        "payload_object_bytes": sum(payload_block_bytes),
        "payload_store_stats": payload_store_stats,
        "global_output_sort": "* Sort" in plan,
        "output_dir": os.path.abspath(args.output_dir),
        "cpu_worker_resource": args.cpu_worker_resource,
        "gpu_worker_resource": args.gpu_worker_resource,
    }

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "daft_plan.txt").write_text(plan, encoding="utf-8")
    (artifact_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "poison_manifest.json").write_text(
        json.dumps(
            [entry.as_dict() for entry in poison_manifest],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "outputs.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (artifact_dir / "gpu_samples.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for sample in gpu_samples:
            handle.write(json.dumps(asdict(sample)) + "\n")
    result_path = Path(args.result_jsonl)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the 4/48/368 PDF Daft baseline CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument(
        "--regroup-mode",
        choices=("full_value", "reference_only"),
        default="full_value",
        help=(
            "controlled regroup representation: natural Daft full payload or "
            "an explicit reference-only expert arm"
        ),
    )
    parser.add_argument("--payload-stores", type=int, default=4)
    parser.add_argument(
        "--assemble-mode",
        choices=("full", "metadata_only"),
        default="full",
        help=(
            "keep full MinerU document materialization or retain only exact "
            "document/page identity as a terminal-work control"
        ),
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=None,
        help=(
            "Daft source partition count; defaults to render replicas. "
            "This is intentionally independent for a fair load-balance sweep."
        ),
    )
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--poison-count", type=int, default=0)
    parser.add_argument("--poison-pdf-index", type=int, default=None)
    parser.add_argument("--poison-page-id", type=int, default=0)
    parser.add_argument(
        "--poison-policy",
        choices=("skip_page", "drop_parent", "raise"),
        default="skip_page",
    )
    parser.add_argument("--poison-seed", default=DEFAULT_POISON_SEED)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--cpu-worker-resource", default=None)
    parser.add_argument("--gpu-worker-resource", default=None)
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--smoke-no-model", action="store_true")
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a Daft baseline arm and print its machine-readable summary."""

    payload = run_benchmark(build_parser().parse_args(argv))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - benchmark entrypoint
    raise SystemExit(main())


__all__ = ["build_parser", "run_benchmark"]
