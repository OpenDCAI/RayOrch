"""MinerU 的 Ray Data reference-only regroup baseline。

Ray Data 仍负责 PDF flat_map、GPU map_batches、manifest groupby 和 map_groups。OCR UDF
手工把每个物理 batch 的 page/content values 放进一个 coarse ObjectRef block，Dataset
只 shuffle parent、ordinal、ObjectRef 和 row selector。

这不是 V3 runtime：应用必须自行定义 lineage columns、调用 ray.put/ray.get、排序、去重
block refs，并承担对象生命周期与失败处理。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

from .mineru import (
    DEFAULT_FLASH_REPO,
    DEFAULT_MODEL,
    MinerUAssembleDoc,
    MinerUVlmOcrPage,
    ResourceSampler,
    _runtime_env,
)
from .mineru_ray_data import (
    RayDataRenderPdf,
    _page_from_row,
    _rows_from_batch,
)

_PAYLOAD_STORE_CLASS = None


def get_ray_payload_store_class():
    """延迟创建由 Driver 持有的 coarse payload ObjectRef owner actor。"""

    global _PAYLOAD_STORE_CLASS
    if _PAYLOAD_STORE_CLASS is not None:
        return _PAYLOAD_STORE_CLASS
    import ray

    @ray.remote(max_concurrency=1, max_restarts=0)
    class PayloadStore:
        """持有 coarse block ObjectRefs，直到所有 parent group 已消费。"""

        def __init__(self) -> None:
            """初始化 block id、引用和消费计数。"""

            self.next_id = 0
            self.refs: dict[int, Any] = {}
            self.remaining: dict[int, int] = {}
            self.rows: dict[int, int] = {}
            self.peak_blocks = 0
            self.peak_rows = 0

        def store(
            self,
            block: tuple[Any, ...],
            consumers: int,
        ) -> int:
            """把 block 放入 object store，并返回小型整数 token。"""

            if consumers <= 0:
                raise ValueError("payload block must have consumers")
            block_id = self.next_id
            self.next_id += 1
            self.refs[block_id] = ray.put(block)
            self.remaining[block_id] = consumers
            self.rows[block_id] = len(block)
            self.peak_blocks = max(self.peak_blocks, len(self.refs))
            self.peak_rows = max(
                self.peak_rows,
                sum(self.rows.values()),
            )
            return block_id

        def acquire(self, block_id: int):
            """返回由本 actor 拥有的 nested ObjectRef。"""

            return self.refs[block_id]

        def release(self, block_id: int) -> None:
            """一个 parent group 消费完成后归还 block consumer credit。"""

            self.remaining[block_id] -= 1
            if self.remaining[block_id] == 0:
                self.remaining.pop(block_id)
                self.rows.pop(block_id)
                self.refs.pop(block_id)

        def stats(self) -> dict[str, int]:
            """返回 observation-only store 水位。"""

            return {
                "live_blocks": len(self.refs),
                "live_rows": sum(self.rows.values()),
                "peak_blocks": self.peak_blocks,
                "peak_rows": self.peak_rows,
            }

    _PAYLOAD_STORE_CLASS = PayloadStore
    return PayloadStore


class RayDataOcrReferenceBlocks:
    """GPU actor：执行 OCR，并为本物理 batch 创建一个 page/content coarse block。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.8,
        stores: tuple[Any, ...] = (),
    ) -> None:
        """初始化 persistent MinerU OCR UDF 和 Driver-owned stores。"""

        if not stores:
            raise ValueError("reference baseline requires payload stores")
        self.ocr = MinerUVlmOcrPage(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.stores = stores

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """返回只含 logical manifest 和 ObjectRef row selectors 的小型 batch。"""

        import numpy as np
        import ray

        rows = _rows_from_batch(batch)
        pages = [_page_from_row(row) for row in rows]
        contents = self.ocr.run(pages)
        if len(contents) != len(rows):
            raise ValueError(
                "MinerU OCR output count does not match Ray Data input batch"
            )
        parent_ids = [int(value) for value in batch["parent_id"]]
        store_slot = min(parent_ids) % len(self.stores)
        block_id = ray.get(
            self.stores[store_slot].store.remote(
                tuple(zip(pages, contents)),
                len(set(parent_ids)),
            )
        )
        return {
            "parent_id": batch["parent_id"],
            "page_ordinal": batch["page_ordinal"],
            "pdf_path": batch["pdf_path"],
            "store_slot": np.asarray(
                [store_slot] * len(rows),
                dtype="int64",
            ),
            "block_id": np.asarray([block_id] * len(rows), dtype="int64"),
            "payload_row": np.arange(len(rows), dtype="int64"),
        }


class RayDataAssembleReferenceGroups:
    """按 parent regroup manifest，并按 coarse block 去重执行 ray.get。"""

    def __init__(
        self,
        output_dir: str,
        stores: tuple[Any, ...],
        parse_method: str = "vlm",
    ) -> None:
        """创建 assembler，并持有 Driver-owned store handles。"""

        self.stores = stores
        self.assembler = MinerUAssembleDoc(
            output_dir=output_dir,
            parse_method=parse_method,
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """恢复 ordered page/content values，并输出一个 document row。"""

        import numpy as np
        import ray

        rows = sorted(
            _rows_from_batch(batch),
            key=lambda row: int(row["page_ordinal"]),
        )
        if not rows:
            raise ValueError("Ray Data emitted an empty PDF group")
        ordinals = [int(row["page_ordinal"]) for row in rows]
        if ordinals != list(range(len(rows))):
            raise ValueError(
                f"non-contiguous or duplicate page ordinals: {ordinals}"
            )
        pdf_path = str(rows[0]["pdf_path"])
        if any(str(row["pdf_path"]) != pdf_path for row in rows):
            raise ValueError("Ray Data parent group mixes multiple PDFs")

        blocks: dict[tuple[int, int], tuple[Any, ...]] = {}
        pages = []
        contents = []
        for row in rows:
            key = (int(row["store_slot"]), int(row["block_id"]))
            if key not in blocks:
                store_ref = ray.get(
                    self.stores[key[0]].acquire.remote(key[1])
                )
                blocks[key] = ray.get(store_ref)
            page, content = blocks[key][int(row["payload_row"])]
            pages.append(page)
            contents.append(content)
        output = self.assembler.run(
            [contents],
            [pages],
            [Path(pdf_path).stem],
        )[0]
        ray.get(
            [
                self.stores[store_slot].release.remote(block_id)
                for store_slot, block_id in blocks
            ]
        )
        return {
            "parent_id": np.asarray(
                [int(rows[0]["parent_id"])],
                dtype="int64",
            ),
            "payload_blocks": np.asarray([len(blocks)], dtype="int64"),
            "output": np.asarray([output], dtype=object),
        }


def build_dataset(
    args: argparse.Namespace,
    pdfs: list[str],
    stores: tuple[Any, ...],
):
    """构造 reference-only Ray Data DAG。"""

    import ray

    source = ray.data.from_items(
        [
            {"parent_id": parent_id, "pdf_path": pdf_path}
            for parent_id, pdf_path in enumerate(pdfs)
        ],
        override_num_blocks=max(
            1,
            min(len(pdfs), args.render_replicas),
        ),
    )
    pages = source.flat_map(
        RayDataRenderPdf,
        concurrency=args.render_replicas,
        num_cpus=1,
        fn_constructor_kwargs={"dpi": 200},
    )
    manifest = pages.map_batches(
        RayDataOcrReferenceBlocks,
        batch_size=args.batch_size,
        batch_format="numpy",
        concurrency=args.replicas,
        num_gpus=1,
        num_cpus=1,
        fn_constructor_kwargs={
            "model": args.model,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "stores": stores,
        },
    )
    return manifest.groupby(
        "parent_id",
        num_partitions=max(
            1,
            min(len(pdfs), args.reduce_replicas),
        ),
    ).map_groups(
        RayDataAssembleReferenceGroups,
        batch_format="numpy",
        concurrency=args.reduce_replicas,
        num_cpus=1,
        fn_constructor_kwargs={
            "output_dir": os.path.abspath(args.output_dir),
            "stores": stores,
        },
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """运行 reference-only baseline，并保存 Ray Data operator stats。"""

    import ray

    flash_repo = os.path.abspath(args.flash_repo)
    pdfs = sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {flash_repo}")
    os.environ.update(_runtime_env(flash_repo)["env_vars"])
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=args.num_cpus,
            num_gpus=args.replicas,
            object_store_memory=int(args.object_store_gb * 1024**3),
            include_dashboard=False,
            runtime_env=_runtime_env(flash_repo),
        )

    store_class = get_ray_payload_store_class()
    stores = tuple(
        store_class.options(num_cpus=0).remote()
        for _ in range(args.payload_stores)
    )
    dataset = build_dataset(args, pdfs, stores)
    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    try:
        rows = dataset.take_all()
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
    store_stats = tuple(ray.get([store.stats.remote() for store in stores]))
    wall = time.perf_counter() - started
    rows.sort(key=lambda row: int(row["parent_id"]))
    outputs = [row["output"] for row in rows]
    pages = sum(int(output["pages"]) for output in outputs)
    gpu_peak = tuple(
        max((sample.memory_used[index] for sample in gpu_samples), default=0)
        for index in range(args.replicas)
    )
    payload = {
        "engine": "ray_data_reference",
        "n_pdf": len(pdfs),
        "pages": pages,
        "docs": len(outputs),
        "batch_size": args.batch_size,
        "replicas": args.replicas,
        "measured_wall_s": round(wall, 3),
        "end_to_end_wall_s": round(wall, 3),
        "pages_per_s": round(pages / wall, 4),
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "regroup_mode": "reference_only",
        "explicit_manifest_columns": [
            "parent_id",
            "page_ordinal",
            "store_slot",
            "block_id",
            "payload_row",
        ],
        "payload_store_stats": store_stats,
        "output_dir": os.path.abspath(args.output_dir),
    }
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        stats = dataset.stats()
    except Exception:
        stats = ""
    (artifact_dir / "ray_data_stats.txt").write_text(
        stats,
        encoding="utf-8",
    )
    (artifact_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with Path(args.result_jsonl).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    for store in stores:
        ray.kill(store)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造 reference-only MinerU Ray Data CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--payload-stores", type=int, default=4)
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、执行 benchmark 并打印 JSON。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
