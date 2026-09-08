"""Fair Ray Data 2.50 MinerU baseline and regroup control experiment.

Both arms use Ray Data's native ``flat_map -> map_batches -> groupby ->
map_groups`` pipeline, the same Arrow-compatible page representation, and the
same MinerU business UDFs.  The controlled variable is the value entering the
hash groupby:

* ``full_value`` keeps page images and OCR content in Dataset rows;
* ``reference_only`` groups a small manifest while an explicit owner actor
  retains coarse page/content blocks.

The reference arm is an expert control, not a claim about Ray Data's natural
API.  It deliberately counts the ownership protocol that application code must
provide when payload values no longer travel with Dataset rows.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ...multigrain_v3.benchmark.mineru import (
    DEFAULT_FLASH_REPO,
    DEFAULT_MODEL,
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    ResourceSampler,
    _runtime_env,
)
from ...multigrain_v3.benchmark.mineru_ray_data import (
    _page_from_row,
    _rows_from_batch,
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
class _DocumentObservation:
    """One business output plus non-invasive stage timing observations."""

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


_PAYLOAD_STORE_CLASS: Any = None


def _get_payload_store_class() -> Any:
    """Create the driver-owned coarse-block store for the expert arm."""

    global _PAYLOAD_STORE_CLASS
    if _PAYLOAD_STORE_CLASS is not None:
        return _PAYLOAD_STORE_CLASS

    import ray  # pyright: ignore[reportMissingImports]

    @ray.remote(max_concurrency=1, max_restarts=0)
    class PayloadStore:
        """Own payload ObjectRefs until all parent groups release them."""

        def __init__(self) -> None:
            self.next_id = 0
            self.refs: dict[int, Any] = {}
            self.remaining: dict[int, int] = {}
            self.rows: dict[int, int] = {}
            self.sizes: dict[int, int] = {}
            self.total_blocks = 0
            self.total_consumers = 0
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
            self.total_consumers += consumers
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
                "total_consumers": self.total_consumers,
                "total_rows": self.total_rows,
                "total_payload_bytes": self.total_payload_bytes,
            }

    _PAYLOAD_STORE_CLASS = PayloadStore
    return PayloadStore


def _source_blocks(args: argparse.Namespace, source_count: int) -> int:
    """Resolve the bounded source-block count independently of GPU batching."""

    requested = (
        args.source_blocks
        if args.source_blocks is not None
        else args.render_replicas
    )
    if requested <= 0:
        raise ValueError("source_blocks must be positive")
    return max(1, min(source_count, requested))


def _page_row(
    parent_id: int,
    pdf_path: str,
    ordinal: int,
    page: dict[str, Any],
    started_s: float,
    finished_s: float,
) -> dict[str, Any]:
    """Convert one MinerU page to Ray Data's Arrow-compatible row schema."""

    import numpy as np  # pyright: ignore[reportMissingImports]

    return {
        "parent_id": parent_id,
        "page_ordinal": ordinal,
        "pdf_path": pdf_path,
        "page_id": int(page["page_id"]),
        "image_rgb": np.asarray(page["img_pil"], dtype="uint8"),
        "scale": float(page["scale"]),
        "page_width": int(page["page_width"]),
        "page_height": int(page["page_height"]),
        "pdf_len": int(page["pdf_len"]),
        "render_started_s": started_s,
        "render_finished_s": finished_s,
    }


class _TimedRenderPdf:
    """Native Ray Data flat_map actor with explicit parent/ordinal columns."""

    def __init__(self, dpi: int = 200) -> None:
        self.renderer = MinerUPdfToPages(dpi=dpi)

    def __call__(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        parent_id = int(row["parent_id"])
        pdf_path = str(row["pdf_path"])
        started_s = time.time()
        pages = self.renderer.run([pdf_path])[0]
        finished_s = time.time()
        return [
            _page_row(
                parent_id,
                pdf_path,
                ordinal,
                page,
                started_s,
                finished_s,
            )
            for ordinal, page in enumerate(pages)
        ]


def _repeat(value: Any, count: int, *, dtype: Any = None):
    """Create one Ray Data output column without repeating call-site noise."""

    import numpy as np  # pyright: ignore[reportMissingImports]

    return np.asarray([value] * count, dtype=dtype)


class _TimedOcrPages:
    """One native Ray Data batched actor shared by both regroup arms."""

    def __init__(
        self,
        *,
        smoke: bool,
        regroup_mode: str,
        model: str,
        gpu_memory_utilization: float,
        stores: tuple[Any, ...],
        poison_pages: tuple[tuple[str, int], ...] = (),
        poison_policy: str = "skip_page",
    ) -> None:
        self.actor_init_started_s = time.time()
        self.smoke = smoke
        self.regroup_mode = regroup_mode
        self.stores = stores
        self.poison_pages = {
            (os.path.abspath(path), int(page_id))
            for path, page_id in poison_pages
        }
        self.poison_policy = poison_policy
        self.ocr = (
            None
            if smoke
            else MinerUVlmOcrPage(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        )
        self.actor_token = f"{os.getpid()}-{uuid.uuid4().hex}"
        self.calls = 0
        self.actor_ready_s = time.time()

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        import numpy as np  # pyright: ignore[reportMissingImports]
        import ray  # pyright: ignore[reportMissingImports]

        rows = _rows_from_batch(batch)
        pages = [_page_from_row(row) for row in rows]
        parent_ids = [int(row["parent_id"]) for row in rows]
        poisoned = [
            (
                os.path.abspath(str(row["pdf_path"])),
                int(row["page_id"]),
            )
            in self.poison_pages
            for row in rows
        ]
        if self.poison_policy == "raise" and any(poisoned):
            raise ValueError("injected deterministic poison page")
        token = f"{self.actor_token}-{self.calls}"
        self.calls += 1
        batch_started_s = time.time()
        live_indices = [index for index, bad in enumerate(poisoned) if not bad]
        if self.smoke:
            live_contents: list[Any] = [
                int(pages[index]["page_id"]) for index in live_indices
            ]
        else:
            assert self.ocr is not None
            live_contents = list(
                self.ocr.run([pages[index] for index in live_indices])
            )
        by_index = dict(zip(live_indices, live_contents, strict=True))
        contents = [
            None if bad else by_index[index]
            for index, bad in enumerate(poisoned)
        ]
        batch_finished_s = time.time()
        if len(contents) != len(rows):
            raise ValueError("MinerU OCR output count does not match input batch")

        count = len(rows)
        observations = {
            "poisoned": np.asarray(poisoned, dtype="bool"),
            "batch_token": _repeat(token, count, dtype=object),
            "batch_size_observed": _repeat(count, count, dtype="int64"),
            "actor_init_started_s": _repeat(
                self.actor_init_started_s, count, dtype="float64"
            ),
            "actor_ready_s": _repeat(
                self.actor_ready_s, count, dtype="float64"
            ),
            "batch_started_s": _repeat(
                batch_started_s, count, dtype="float64"
            ),
            "batch_finished_s": _repeat(
                batch_finished_s, count, dtype="float64"
            ),
        }
        if self.regroup_mode == "full_value":
            return {
                **batch,
                "content": np.asarray(contents, dtype=object),
                "payload_publish_started_s": _repeat(
                    0.0, count, dtype="float64"
                ),
                "payload_publish_finished_s": _repeat(
                    0.0, count, dtype="float64"
                ),
                "payload_block_bytes": _repeat(0, count, dtype="int64"),
                **observations,
            }

        if not self.stores:
            raise ValueError("reference_only mode requires payload stores")
        store_slot = min(parent_ids) % len(self.stores)
        publish_started_s = time.time()
        block_id, block_bytes = ray.get(
            self.stores[store_slot].store.remote(
                tuple(zip(pages, contents)),
                len(set(parent_ids)),
            )
        )
        publish_finished_s = time.time()
        return {
            "parent_id": batch["parent_id"],
            "page_ordinal": batch["page_ordinal"],
            "pdf_path": batch["pdf_path"],
            "render_started_s": batch["render_started_s"],
            "render_finished_s": batch["render_finished_s"],
            "store_slot": _repeat(store_slot, count, dtype="int64"),
            "block_id": _repeat(block_id, count, dtype="int64"),
            "payload_row": np.arange(count, dtype="int64"),
            "payload_publish_started_s": _repeat(
                publish_started_s, count, dtype="float64"
            ),
            "payload_publish_finished_s": _repeat(
                publish_finished_s, count, dtype="float64"
            ),
            "payload_block_bytes": _repeat(
                block_bytes, count, dtype="int64"
            ),
            **observations,
        }


def _ordered_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Restore one parent group and enforce its exact ordinal contract."""

    rows = sorted(
        _rows_from_batch(batch),
        key=lambda row: int(row["page_ordinal"]),
    )
    if not rows:
        raise ValueError("Ray Data emitted an empty PDF group")
    ordinals = [int(row["page_ordinal"]) for row in rows]
    if ordinals != list(range(len(rows))):
        raise ValueError(f"non-contiguous or duplicate page ordinals: {ordinals}")
    path = str(rows[0]["pdf_path"])
    if any(str(row["pdf_path"]) != path for row in rows):
        raise ValueError("Ray Data parent group mixes multiple PDFs")
    return rows


class _TimedAssemblePdf:
    """Native map_groups actor for natural rows or reference manifests."""

    def __init__(
        self,
        *,
        output_dir: str,
        metadata_only: bool,
        regroup_mode: str,
        stores: tuple[Any, ...],
        poison_policy: str,
    ) -> None:
        self.metadata_only = metadata_only
        self.regroup_mode = regroup_mode
        self.stores = stores
        self.poison_policy = poison_policy
        self.assembler = (
            None
            if metadata_only
            else MinerUAssembleDoc(output_dir=output_dir)
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        import numpy as np  # pyright: ignore[reportMissingImports]
        import ray  # pyright: ignore[reportMissingImports]

        assemble_started_s = time.time()
        rows = _ordered_rows(batch)
        blocks: dict[tuple[int, int], tuple[Any, ...]] = {}
        if self.regroup_mode == "full_value":
            pages = [_page_from_row(row) for row in rows]
            contents = [row["content"] for row in rows]
        else:
            pages = []
            contents = []
            for row in rows:
                key = (int(row["store_slot"]), int(row["block_id"]))
                if key not in blocks:
                    nested_ref = ray.get(
                        self.stores[key[0]].acquire.remote(key[1])
                    )
                    blocks[key] = ray.get(nested_ref)
                page, content = blocks[key][int(row["payload_row"])]
                pages.append(page)
                contents.append(content)

        path = str(rows[0]["pdf_path"])
        input_page_ids = tuple(int(page["page_id"]) for page in pages)
        if input_page_ids != tuple(range(len(input_page_ids))):
            raise ValueError(f"page payload ordinals changed: {input_page_ids}")
        poison_page_ids = tuple(
            int(page["page_id"])
            for row, page in zip(rows, pages, strict=True)
            if bool(row["poisoned"])
        )
        drop_parent = bool(poison_page_ids) and self.poison_policy == "drop_parent"
        live = [
            (page, content)
            for row, page, content in zip(rows, pages, contents, strict=True)
            if not bool(row["poisoned"])
        ]
        if not live and not drop_parent:
            raise ValueError("cannot assemble a PDF when every page is poisoned")
        pages = [page for page, _ in live]
        contents = [content for _, content in live]
        page_ids = (
            ()
            if drop_parent
            else tuple(int(page["page_id"]) for page in pages)
        )
        try:
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
                    "pages": len(pages),
                    "page_ids": page_ids,
                    "input_pages": len(input_page_ids),
                    "poisoned_pages": len(poison_page_ids),
                    "poison_page_ids": poison_page_ids,
                }
            else:
                assert self.assembler is not None
                output = self.assembler.run(
                    [contents],
                    [pages],
                    [Path(path).stem],
                )[0]
                output.update(
                    input_pages=len(input_page_ids),
                    poisoned_pages=len(poison_page_ids),
                    poison_page_ids=list(poison_page_ids),
                )
        finally:
            if blocks:
                ray.get(
                    [
                        self.stores[slot].release.remote(block_id)
                        for slot, block_id in blocks
                    ]
                )
        assemble_finished_s = time.time()
        document = _DocumentObservation(
            value=output,
            page_ids=page_ids,
            batch_tokens=tuple(str(row["batch_token"]) for row in rows),
            batch_sizes=tuple(
                int(row["batch_size_observed"]) for row in rows
            ),
            render_started_s=tuple(
                float(row["render_started_s"]) for row in rows
            ),
            render_finished_s=tuple(
                float(row["render_finished_s"]) for row in rows
            ),
            actor_init_started_s=tuple(
                float(row["actor_init_started_s"]) for row in rows
            ),
            actor_ready_s=tuple(float(row["actor_ready_s"]) for row in rows),
            batch_started_s=tuple(
                float(row["batch_started_s"]) for row in rows
            ),
            batch_finished_s=tuple(
                float(row["batch_finished_s"]) for row in rows
            ),
            payload_publish_started_s=tuple(
                float(row["payload_publish_started_s"]) for row in rows
            ),
            payload_publish_finished_s=tuple(
                float(row["payload_publish_finished_s"]) for row in rows
            ),
            payload_block_bytes=tuple(
                int(row["payload_block_bytes"]) for row in rows
            ),
            assemble_started_s=assemble_started_s,
            assemble_finished_s=assemble_finished_s,
        )
        return {
            "parent_id": np.asarray(
                [int(rows[0]["parent_id"])], dtype="int64"
            ),
            # Arrow transports one opaque terminal envelope identically in
            # both arms; observations never enter the controlled groupby.
            "result_blob": np.asarray(
                [
                    json.dumps(
                        asdict(document),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ],
                dtype=object,
            ),
        }


def build_dataset(
    args: argparse.Namespace,
    pdfs: Sequence[str],
    stores: tuple[Any, ...] = (),
    poison_manifest: tuple[PoisonPage, ...] = (),
):
    """Build the native Ray Data plan with one controlled representation."""

    import ray  # pyright: ignore[reportMissingImports]

    reference_only = args.regroup_mode == "reference_only"
    if reference_only != bool(stores):
        raise ValueError("reference_only mode requires payload stores")
    source = ray.data.from_items(
        [
            {"parent_id": index, "pdf_path": path}
            for index, path in enumerate(pdfs)
        ],
        override_num_blocks=_source_blocks(args, len(pdfs)),
    )
    pages = source.flat_map(
        _TimedRenderPdf,
        concurrency=args.render_replicas,
        num_cpus=1,
        fn_constructor_kwargs={"dpi": args.render_dpi},
        **worker_resource_options(args.cpu_worker_resource),
    )
    contents = pages.map_batches(
        _TimedOcrPages,
        batch_size=args.batch_size,
        batch_format="numpy",
        concurrency=args.replicas,
        num_gpus=0 if args.smoke_no_model else 1,
        num_cpus=1,
        fn_constructor_kwargs={
            "smoke": bool(args.smoke_no_model),
            "regroup_mode": args.regroup_mode,
            "model": args.model,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "stores": stores,
            "poison_pages": tuple(
                (entry.pdf_path, entry.page_id) for entry in poison_manifest
            ),
            "poison_policy": args.poison_policy,
        },
        **worker_resource_options(
            args.cpu_worker_resource
            if args.smoke_no_model
            else args.gpu_worker_resource
        ),
    )
    return contents.groupby(
        "parent_id",
        num_partitions=max(1, min(len(pdfs), args.reduce_replicas)),
    ).map_groups(
        _TimedAssemblePdf,
        batch_format="numpy",
        concurrency=args.reduce_replicas,
        num_cpus=1,
        fn_constructor_kwargs={
            "output_dir": os.path.abspath(args.output_dir),
            "metadata_only": bool(
                args.smoke_no_model
                or args.assemble_mode == "metadata_only"
            ),
            "regroup_mode": args.regroup_mode,
            "stores": stores,
            "poison_policy": args.poison_policy,
        },
        **worker_resource_options(args.cpu_worker_resource),
    )


def _batch_observations(
    documents: list[_DocumentObservation],
) -> dict[str, tuple[int, float, float, float, float, float, float, int]]:
    """Deduplicate row-repeated OCR observations by physical batch token."""

    result = {}
    for document in documents:
        values = zip(
            document.batch_tokens,
            document.batch_sizes,
            document.actor_init_started_s,
            document.actor_ready_s,
            document.batch_started_s,
            document.batch_finished_s,
            document.payload_publish_started_s,
            document.payload_publish_finished_s,
            document.payload_block_bytes,
        )
        for value in values:
            token = value[0]
            observation = (
                int(value[1]),
                float(value[2]),
                float(value[3]),
                float(value[4]),
                float(value[5]),
                float(value[6]),
                float(value[7]),
                int(value[8]),
            )
            previous = result.setdefault(token, observation)
            if previous != observation:
                raise ValueError(f"conflicting OCR observation for {token}")
    return result


def _decode_document(blob: Any) -> _DocumentObservation:
    """Decode the Arrow-safe terminal envelope produced by map_groups."""

    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    payload = json.loads(bytes(blob).decode("utf-8"))
    tuple_fields = {
        "page_ids",
        "batch_tokens",
        "batch_sizes",
        "render_started_s",
        "render_finished_s",
        "actor_init_started_s",
        "actor_ready_s",
        "batch_started_s",
        "batch_finished_s",
        "payload_publish_started_s",
        "payload_publish_finished_s",
        "payload_block_bytes",
    }
    for field in tuple_fields:
        payload[field] = tuple(payload[field])
    return _DocumentObservation(**payload)


def _ray_data_shuffle_observation(stats: str) -> dict[str, Any]:
    """Extract stable shuffle evidence while retaining the raw stats file."""

    operator = re.search(
        r"Operator \d+ Shuffle\([^\n]+executed in ([0-9.]+)s"
        r"(.*?)(?=\nOperator \d+ |\Z)",
        stats,
        flags=re.DOTALL,
    )
    if operator is None:
        return {
            "operator_wall_s": None,
            "map_output_bytes": None,
            "finalize_output_bytes": None,
        }

    body = operator.group(2)

    def output_bytes(suboperator: str) -> int | None:
        match = re.search(
            rf"Suboperator \d+ [^\n]*_{suboperator}:.*?"
            r"Output size bytes per block: [^\n]*?([0-9,]+) total",
            body,
            flags=re.DOTALL,
        )
        return int(match.group(1).replace(",", "")) if match else None

    return {
        "operator_wall_s": float(operator.group(1)),
        "map_output_bytes": output_bytes("shuffle"),
        "finalize_output_bytes": output_bytes("finalize"),
    }


def _ray_data_ocr_remote_tasks(stats: str) -> int | None:
    """Read Ray actor-task count separately from in-task UDF batch calls."""

    match = re.search(
        r"Operator \d+ MapBatches\(_TimedOcrPages\): "
        r"([0-9,]+) tasks executed",
        stats,
    )
    return int(match.group(1).replace(",", "")) if match else None


def _output_signatures(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hash business artifacts outside the measured timing window."""

    signatures = []
    for output in outputs:
        signature = {
            key: output[key]
            for key in ("pdf", "pages", "chars")
            if key in output
        }
        md_path = output.get("md_path")
        if md_path:
            markdown = Path(md_path)
            layout = markdown.parent / "layout.json"
            signature["markdown_sha256"] = hashlib.sha256(
                markdown.read_bytes()
            ).hexdigest()
            signature["layout_sha256"] = hashlib.sha256(
                layout.read_bytes()
            ).hexdigest()
        signatures.append(signature)
    return signatures


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep rollup JSONL/stdout small; full hashes remain in summary.json."""

    return {
        key: value
        for key, value in payload.items()
        if key != "output_signatures"
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run one isolated arm and persist plan, timings, resources, and output."""

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

    # Hash shuffle creates an internal SPREAD-scheduled aggregator actor whose
    # placement cannot inherit per-UDF worker resources.  Ray Data's configured
    # pull-based sort shuffle is the non-patching fallback on heterogeneous
    # clusters where that actor is not schedulable (for example, when the
    # control node is fully reserved).
    from ray.data.context import ShuffleStrategy  # pyright: ignore[reportMissingImports]

    ray.data.DataContext.get_current().shuffle_strategy = (
        ShuffleStrategy.HASH_SHUFFLE
        if args.shuffle_strategy == "hash"
        else ShuffleStrategy.SORT_SHUFFLE_PULL_BASED
    )

    stores: tuple[Any, ...] = ()
    sampler = ResourceSampler(args.rss_interval_s)
    sampler_started = False
    store_stats: tuple[dict[str, int], ...] = ()
    stats = ""
    driver_start = 0
    driver_peak = 0
    gpu_samples = ()
    try:
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
        dataset = build_dataset(args, pdfs, stores, poison_manifest)
        sampler.start()
        sampler_started = True
        started_epoch = time.time()
        started = time.perf_counter()
        rows = dataset.take_all()
        wall = time.perf_counter() - started
        finished_epoch = time.time()
        try:
            stats = dataset.stats()
        except Exception as exc:  # diagnostics must not invalidate the run
            stats = f"Ray Data stats unavailable: {exc!r}\n"
        if stores:
            store_stats = tuple(
                ray.get([store.stats.remote() for store in stores])
            )
    finally:
        if sampler_started:
            driver_start, driver_peak, gpu_samples = sampler.stop()
        for store in stores:
            ray.kill(store, no_restart=True)
        if started_ray_here:
            ray.shutdown()

    rows.sort(key=lambda row: int(row["parent_id"]))
    parent_ids = [int(row["parent_id"]) for row in rows]
    if parent_ids != list(range(len(pdfs))):
        raise ValueError(f"Ray Data changed source identity/order: {parent_ids}")
    documents = [_decode_document(row["result_blob"]) for row in rows]
    if not all(isinstance(value, _DocumentObservation) for value in documents):
        raise TypeError("Ray Data output did not preserve observations")
    outputs = [document.value for document in documents]
    successful_outputs = [
        output for output in outputs if not output.get("dropped_parent", False)
    ]
    pages = sum(len(document.page_ids) for document in documents)
    input_pages = sum(page_counts)
    poison_observed = sum(
        int(output.get("poisoned_pages", 0)) for output in outputs
    )
    poisoned_parent_pages = sum(entry.pdf_pages for entry in poison_manifest)
    drop_parent = args.poison_policy == "drop_parent"
    batches = _batch_observations(documents)
    sizes = tuple(value[0] for value in batches.values())
    dispatched_pages = sum(sizes)
    histogram = {
        str(size): sizes.count(size) for size in sorted(set(sizes))
    }
    render_started = tuple(
        value for document in documents for value in document.render_started_s
    )
    render_finished = tuple(
        value for document in documents for value in document.render_finished_s
    )
    actor_init = tuple(value[1] for value in batches.values())
    actor_ready = tuple(value[2] for value in batches.values())
    batch_started = tuple(value[3] for value in batches.values())
    batch_finished = tuple(value[4] for value in batches.values())
    publish_started = tuple(
        value[5] for value in batches.values() if value[5] > 0
    )
    publish_finished = tuple(
        value[6] for value in batches.values() if value[6] > 0
    )
    payload_bytes = tuple(value[7] for value in batches.values())
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
        "engine": f"ray_data_{args.regroup_mode}",
        "ray_version": ray.__version__,
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
        "reduce_replicas": args.reduce_replicas,
        "source_blocks": _source_blocks(args, len(pdfs)),
        "regroup_mode": args.regroup_mode,
        "shuffle_strategy": args.shuffle_strategy,
        "render_dpi": args.render_dpi,
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
        "ocr_batch_call_count": len(batches),
        "ocr_dispatched_pages": dispatched_pages,
        "ocr_model_pages": dispatched_pages - poison_observed,
        "ocr_model_pages_saved_vs_baseline": (
            input_pages - dispatched_pages + poison_observed
        ),
        "ocr_pages_per_batch_call": (
            dispatched_pages / len(batches) if batches else 0.0
        ),
        "ocr_remote_task_count": _ray_data_ocr_remote_tasks(stats),
        "ocr_batch_fill_ratio": (
            dispatched_pages / (len(batches) * args.batch_size)
            if batches
            else 0.0
        ),
        "ocr_batch_histogram": histogram,
        "timeline_s": {
            "first_render_start": min(render_started) - started_epoch,
            "last_render_finish": max(render_finished) - started_epoch,
            "first_ocr_actor_init": min(actor_init) - started_epoch,
            "all_ocr_actors_ready": max(actor_ready) - started_epoch,
            "first_ocr_batch_start": min(batch_started) - started_epoch,
            "last_ocr_batch_finish": max(batch_finished) - started_epoch,
            "post_ocr_to_collect_end": finished_epoch - max(batch_finished),
            "first_payload_publish_start": (
                min(publish_started) - started_epoch if publish_started else None
            ),
            "last_payload_publish_finish": (
                max(publish_finished) - started_epoch if publish_finished else None
            ),
            "first_assemble_start": min(assemble_started) - started_epoch,
            "last_assemble_finish": max(assemble_finished) - started_epoch,
            "post_assemble_to_collect_end": (
                finished_epoch - max(assemble_finished)
            ),
        },
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "explicit_lineage_columns": ["parent_id", "page_ordinal"],
        "page_representation": "Arrow-compatible uint8 image_rgb tensor",
        "regroup_operator": f"Ray Data {args.shuffle_strategy} groupby",
        "shuffle_observation": _ray_data_shuffle_observation(stats),
        "full_value_regroup": args.regroup_mode == "full_value",
        "payload_block_count": len(batches) if stores else 0,
        "payload_object_bytes": sum(payload_bytes),
        "payload_store_stats": store_stats,
        "output_dir": os.path.abspath(args.output_dir),
        "cpu_worker_resource": args.cpu_worker_resource,
        "gpu_worker_resource": args.gpu_worker_resource,
        "output_signatures": _output_signatures(outputs),
    }

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "ray_data_stats.txt").write_text(stats, encoding="utf-8")
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
        handle.write(json.dumps(_compact_payload(payload), ensure_ascii=False))
        handle.write("\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen 4/48/368-PDF Ray Data experiment CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--source-blocks", type=int, default=None)
    parser.add_argument(
        "--shuffle-strategy",
        choices=("hash", "sort"),
        default="hash",
        help="Ray Data groupby shuffle implementation.",
    )
    parser.add_argument(
        "--regroup-mode",
        choices=("full_value", "reference_only"),
        default="full_value",
    )
    parser.add_argument(
        "--assemble-mode",
        choices=("full", "metadata_only"),
        default="full",
    )
    parser.add_argument("--payload-stores", type=int, default=4)
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
    """Run one arm and print its machine-readable summary."""

    payload = run_benchmark(build_parser().parse_args(argv))
    print(json.dumps(_compact_payload(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
