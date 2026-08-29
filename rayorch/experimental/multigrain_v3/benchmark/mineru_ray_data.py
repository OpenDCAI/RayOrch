"""使用裸 Ray Data 复现 MinerU 的 PDF→Page→OCR→PDF 对比基线。

该 runner 刻意不依赖 Multigrain V3 runtime。为了获得与 V3 相同的 ordered Reduce
correctness，应用显式携带 ``parent_id`` 与 ``page_ordinal``，并在 Ray Data
``groupby().map_groups()`` 中恢复每个 PDF。
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

from .mineru import (
    DEFAULT_FLASH_REPO,
    DEFAULT_MODEL,
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    ResourceSampler,
    _is_hdfs_uri,
    _pdf_stem,
    _runtime_env,
)
from .profile_events import ProfileEventWriter
from ...multigrain_v3_6.benchmark.mineru import _select_pdfs


def _rows_from_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """把 Ray Data 的 numpy batch 转为普通 row 字典列表。"""

    if not batch:
        return []
    size = len(next(iter(batch.values())))
    return [
        {name: values[index] for name, values in batch.items()}
        for index in range(size)
    ]


def _page_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """把 Ray Data 的 columnar page row 恢复为 MinerU 原始 page record。"""

    from PIL import Image

    image_rgb = row.get("image_rgb")
    if image_rgb is not None:
        if not hasattr(image_rgb, "shape"):
            raise ValueError("image_rgb must be a numpy-compatible tensor")
        image = Image.fromarray(image_rgb, mode="RGB")
    else:
        image_jpeg = row.get("image_jpeg")
        if not isinstance(image_jpeg, (bytes, bytearray, memoryview)):
            raise ValueError("page row requires image_rgb or image_jpeg")
        with Image.open(BytesIO(bytes(image_jpeg))) as decoded:
            image = decoded.convert("RGB")
    return {
        "pdf_path": str(row["pdf_path"]),
        "page_id": int(row["page_id"]),
        "img_pil": image,
        "scale": float(row["scale"]),
        "page_width": int(row["page_width"]),
        "page_height": int(row["page_height"]),
        "pdf_len": int(row["pdf_len"]),
    }


def _encode_page_jpeg(image_rgb: Any, quality: int = 90) -> bytes:
    """Compress a rendered page before the global Ray Data shuffle."""

    from PIL import Image

    if not hasattr(image_rgb, "shape"):
        raise ValueError("image_rgb must be a numpy-compatible tensor")
    output = BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(
        output,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
    )
    return output.getvalue()


class RayDataRenderPdf:
    """Ray Data callable actor：渲染一个 PDF row，并显式附加 parent/ordinal。"""

    def __init__(
        self,
        dpi: int = 200,
        profile_dir: str | None = None,
        profile_system: str = "raydata",
    ) -> None:
        """创建与 V3 benchmark 相同的 PDF renderer。"""

        self.renderer = MinerUPdfToPages(
            dpi=dpi,
            profile_dir=profile_dir,
            profile_system=profile_system,
        )

    def __call__(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """把单个 PDF 展开为可被 Ray Data 跨 parent 重组的 page rows。"""

        parent_id = int(row["parent_id"])
        pdf_path = str(row["pdf_path"])
        pages = self.renderer.run([pdf_path])[0]
        if not pages:
            import numpy as np

            return [
                {
                    "parent_id": parent_id,
                    "page_ordinal": 0,
                    "pdf_path": pdf_path,
                    "page_id": -1,
                    "image_rgb": np.zeros((1, 1, 3), dtype="uint8"),
                    "scale": 1.0,
                    "page_width": 0,
                    "page_height": 0,
                    "pdf_len": 0,
                    "render_failed": True,
                }
            ]
        return [
            {
                "parent_id": parent_id,
                "page_ordinal": ordinal,
                "pdf_path": pdf_path,
                "page_id": int(page["page_id"]),
                "image_rgb": __import__("numpy").asarray(
                    page["img_pil"],
                    dtype="uint8",
                ),
                "scale": float(page["scale"]),
                "page_width": int(page["page_width"]),
                "page_height": int(page["page_height"]),
                "pdf_len": int(page["pdf_len"]),
                "render_failed": False,
            }
            for ordinal, page in enumerate(pages)
        ]


class RayDataOcrPages:
    """Ray Data GPU callable actor：对 page batch 执行真实 MinerU OCR。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.8,
        profile_dir: str | None = None,
        profile_system: str = "raydata",
    ) -> None:
        """初始化一个 persistent MinerU vLLM OCR 实例。"""

        self.ocr = MinerUVlmOcrPage(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            profile_dir=profile_dir,
            profile_system=profile_system,
        )

    def __call__(self, batch: dict[str, Any]) -> Any:
        """保留应用 lineage columns，并为每个 page 添加 OCR result。"""

        import numpy as np

        rows = _rows_from_batch(batch)
        valid_indexes = [
            index
            for index, row in enumerate(rows)
            if not bool(row.get("render_failed", False))
        ]
        pages = [_page_from_row(rows[index]) for index in valid_indexes]
        valid_contents = self.ocr.run(pages) if pages else []
        if len(valid_contents) != len(valid_indexes):
            raise ValueError(
                "MinerU OCR output count does not match Ray Data input batch"
            )
        contents: list[Any] = [None] * len(rows)
        for index, content in zip(valid_indexes, valid_contents, strict=True):
            contents[index] = content
        import pyarrow as pa

        image_jpeg = [_encode_page_jpeg(row["image_rgb"]) for row in rows]
        content_pickle = [
            pickle.dumps(content, protocol=pickle.HIGHEST_PROTOCOL)
            for content in contents
        ]
        # Explicit large_binary is required: a single 1/64 hash partition can
        # exceed Arrow binary's signed 32-bit (2 GiB) offset limit even after
        # JPEG compression.
        return pa.table({
            "parent_id": pa.array(batch["parent_id"]),
            "page_ordinal": pa.array(batch["page_ordinal"]),
            "pdf_path": pa.array(batch["pdf_path"]),
            "page_id": pa.array(batch["page_id"]),
            # Do not carry raw page tensors through groupby/shuffle.  At 200
            # DPI they exceed a terabyte for this corpus and OOM the final
            # HashShuffleAggregator.  JPEG keeps the downstream crop contract
            # while bounding shuffle and per-document reduce memory.
            "image_jpeg": pa.array(image_jpeg, type=pa.large_binary()),
            "scale": pa.array(batch["scale"]),
            "page_width": pa.array(batch["page_width"]),
            "page_height": pa.array(batch["page_height"]),
            "pdf_len": pa.array(batch["pdf_len"]),
            "render_failed": pa.array(batch.get(
                "render_failed",
                np.asarray([False] * len(rows), dtype="bool"),
            )),
            "content_pickle": pa.array(
                content_pickle, type=pa.large_binary()
            ),
        })


class RayDataSmokeOcrPages:
    """CPU smoke actor：验证 page transport/packing，而不加载 vLLM。"""

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """为每页生成稳定轻量摘要，同时保留显式 lineage columns。"""

        import numpy as np

        rows = _rows_from_batch(batch)
        contents = [
            {
                "page_id": int(row["page_id"]),
                "size": (
                    int(row["image_rgb"].shape[1]),
                    int(row["image_rgb"].shape[0]),
                ),
            }
            for row in rows
        ]
        return {
            "parent_id": batch["parent_id"],
            "page_ordinal": batch["page_ordinal"],
            "pdf_path": batch["pdf_path"],
            "content_page_id": np.asarray(
                [int(content["page_id"]) for content in contents],
                dtype="int64",
            ),
        }


class RayDataAssemblePdf:
    """Ray Data group callable：按 page ordinal 排序并组装一个 PDF。"""

    def __init__(
        self,
        output_dir: str,
        parse_method: str = "vlm",
        spool_batches: bool = False,
        remote_output_dir: str | None = None,
        profile_dir: str | None = None,
        profile_system: str = "raydata",
    ) -> None:
        """创建与 V3 benchmark 相同的 MinerU assemble UDF。"""

        constructor_kwargs: dict[str, Any] = {
            "output_dir": output_dir,
            "parse_method": parse_method,
        }
        if spool_batches:
            constructor_kwargs["spool_batches"] = True
        if remote_output_dir is not None:
            constructor_kwargs["remote_output_dir"] = remote_output_dir
        if profile_dir is not None:
            constructor_kwargs["profile_dir"] = profile_dir
            constructor_kwargs["profile_system"] = profile_system
        self.assembler = MinerUAssembleDoc(**constructor_kwargs)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """校验一个 parent group 的 ordinal 唯一性后输出单个 document row。"""

        import numpy as np

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
        render_failed = [
            bool(row.get("render_failed", False)) for row in rows
        ]
        if any(render_failed) and not all(render_failed):
            raise ValueError("Ray Data parent group mixes failed and valid pages")
        contents = (
            []
            if all(render_failed)
            else [
                row["content"]
                if "content" in row
                else pickle.loads(bytes(row["content_pickle"]))
                for row in rows
            ]
        )
        pages = [] if all(render_failed) else [_page_from_row(row) for row in rows]
        output = self.assembler.run(
            [contents],
            [pages],
            [_pdf_stem(pdf_path)],
        )[0]
        return {
            "parent_id": np.asarray(
                [int(rows[0]["parent_id"])],
                dtype="int64",
            ),
            "output": np.asarray([output], dtype=object),
        }


class RayDataSmokeAssemblePdf:
    """CPU smoke group callable：只验证 ordinal regroup，不写业务 artifacts。"""

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """按 ordinal 排序、校验连续性，并返回 PDF/page 数摘要。"""

        import numpy as np

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
        output = {
            "pdf": Path(pdf_path).stem,
            "pages": len(rows),
            "page_ids": tuple(
                int(row["content_page_id"]) for row in rows
            ),
        }
        return {
            "parent_id": np.asarray(
                [int(rows[0]["parent_id"])],
                dtype="int64",
            ),
            "output": np.asarray([output], dtype=object),
        }


def build_dataset(args: argparse.Namespace, pdfs: list[str]):
    """构造 lazy Ray Data DAG，保持 OCR batch cap 与 V3 配置一致。"""

    import ray

    rows = [
        {"parent_id": parent_id, "pdf_path": pdf_path}
        for parent_id, pdf_path in enumerate(pdfs)
    ]
    dataset = ray.data.from_items(
        rows,
        override_num_blocks=max(1, min(len(rows), args.render_replicas)),
    )
    resource_options = (
        {"resources": dict(args.actor_resources)}
        if getattr(args, "actor_resources", None)
        else {}
    )
    pages = dataset.flat_map(
        RayDataRenderPdf,
        concurrency=args.render_replicas,
        num_cpus=1,
        fn_constructor_kwargs={
            "dpi": 200,
            "profile_dir": getattr(args, "profile_dir", None),
            "profile_system": getattr(args, "profile_system", "raydata"),
        },
        **resource_options,
    )
    smoke = bool(getattr(args, "smoke_no_model", False))
    ocr_class = RayDataSmokeOcrPages if smoke else RayDataOcrPages
    contents = pages.map_batches(
        ocr_class,
        batch_size=args.batch_size,
        batch_format="numpy",
        concurrency=args.replicas,
        num_gpus=(
            0 if smoke else getattr(args, "gpus_per_ocr_actor", 1.0)
        ),
        num_cpus=1,
        fn_constructor_kwargs=(
            {}
            if smoke
            else {
                "model": args.model,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "profile_dir": getattr(args, "profile_dir", None),
                "profile_system": getattr(args, "profile_system", "raydata"),
            }
        ),
        **resource_options,
    )
    assemble_class = (
        RayDataSmokeAssemblePdf if smoke else RayDataAssemblePdf
    )
    return contents.groupby(
        "parent_id",
        num_partitions=max(1, min(len(rows), args.reduce_replicas)),
    ).map_groups(
        assemble_class,
        batch_format="numpy",
        concurrency=args.reduce_replicas,
        num_cpus=1,
        fn_constructor_kwargs=(
            {}
            if smoke
            else {
                "output_dir": str(args.output_dir),
                "spool_batches": bool(
                    getattr(args, "spool_batches", False)
                ),
                "remote_output_dir": getattr(
                    args, "remote_output_dir", None
                ),
                "profile_dir": getattr(args, "profile_dir", None),
                "profile_system": getattr(args, "profile_system", "raydata"),
            }
        ),
        **resource_options,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """运行 Ray Data baseline，并写与 V3 可并列比较的摘要 JSON。"""

    import ray

    flash_repo = os.path.abspath(args.flash_repo)
    raw_pdf_dirs = getattr(args, "pdf_dirs", None)
    if raw_pdf_dirs is None:
        raw_pdf_dirs = [getattr(args, "pdf_dir", None) or flash_repo]
    elif isinstance(raw_pdf_dirs, str):
        raw_pdf_dirs = [raw_pdf_dirs]
    pdf_dirs = [
        str(value)
        if _is_hdfs_uri(str(value))
        else str(Path(str(value)).expanduser().resolve())
        for value in raw_pdf_dirs
    ]
    output_dir = str(args.output_dir)
    completion_dir = str(getattr(args, "completion_dir", None) or output_dir)
    pdfs, skipped_existing, discovered_pdfs = _select_pdfs(
        pdf_dirs,
        completion_dir,
        limit=args.limit,
        skip_existing=bool(getattr(args, "skip_existing", False)),
        per_input_limits=getattr(args, "pdf_limits", None),
    )
    if not pdfs:
        raise FileNotFoundError("Ray Data has no pending PDFs to process")
    cold_start_epoch_s = float(
        getattr(args, "cold_e2e_start_epoch_s", 0.0) or time.time()
    )
    cold_start_monotonic_s = float(
        getattr(args, "cold_e2e_start_monotonic_s", 0.0)
        or time.perf_counter()
    )
    profile = ProfileEventWriter(
        getattr(args, "profile_driver_dir", None)
        or getattr(args, "profile_dir", None),
        system=getattr(args, "profile_system", "raydata"),
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
    os.environ.update(_runtime_env(flash_repo)["env_vars"])
    if not ray.is_initialized():
        init_kwargs: dict[str, Any] = {
            "address": getattr(args, "ray_address", None) or "local",
            "include_dashboard": False,
            "runtime_env": _runtime_env(flash_repo),
        }
        if init_kwargs["address"] == "local":
            init_kwargs.update(
                num_cpus=args.num_cpus,
                num_gpus=(
                    0
                    if getattr(args, "smoke_no_model", False)
                    else args.replicas
                ),
                object_store_memory=int(args.object_store_gb * 1024**3),
            )
        ray.init(**init_kwargs)
    profile.emit(
        {
            "type": "milestone",
            "name": "ray_initialized",
            "epoch_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
    )

    dataset = build_dataset(args, pdfs)
    profile.emit(
        {
            "type": "milestone",
            "name": "plan_built",
            "epoch_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
    )
    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    try:
        rows = dataset.take_all()
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
    wall = time.perf_counter() - started
    profile.emit(
        {
            "type": "milestone",
            "name": "collect_finished",
            "epoch_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
    )
    rows.sort(key=lambda row: int(row["parent_id"]))
    outputs = [row["output"] for row in rows]
    pages = sum(int(output["pages"]) for output in outputs)
    failed_docs = sorted(
        str(output["pdf"])
        for output in outputs
        if output.get("status") != "completed"
    )
    gpu_peak = tuple(
        max(
            (
                sample.memory_used[index]
                for sample in gpu_samples
                if index < len(sample.memory_used)
            ),
            default=0,
        )
        for index in range(args.replicas)
    )
    payload = {
        "engine": "ray_data",
        "status": "completed",
        "backend": (
            "smoke_cpu"
            if getattr(args, "smoke_no_model", False)
            else "mineru_vllm"
        ),
        "discovered_pdfs": discovered_pdfs,
        "skipped_existing": skipped_existing,
        "n_pdf": len(pdfs),
        "pages": pages,
        "docs": len(outputs),
        "failed_doc_count": len(failed_docs),
        "failed_docs": failed_docs,
        "batch_size": args.batch_size,
        "replicas": args.replicas,
        "gpus_per_ocr_actor": getattr(args, "gpus_per_ocr_actor", 1.0),
        "startup_s": 0.0,
        "measured_wall_s": round(wall, 3),
        "pages_per_s": round(pages / wall, 4),
        "end_to_end_wall_s": round(wall, 3),
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "explicit_lineage_columns": ["parent_id", "page_ordinal"],
        "pdf_dir": pdf_dirs[0] if len(pdf_dirs) == 1 else pdf_dirs,
        "pdf_dirs": pdf_dirs,
        "output_dir": output_dir,
        "profile_clock": {
            "e2e_start_epoch_s": cold_start_epoch_s,
            "e2e_start_monotonic_s": cold_start_monotonic_s,
            "benchmark_end_epoch_s": time.time(),
            "benchmark_end_monotonic_s": time.perf_counter(),
            "absolute_anchor_estimated": False,
        },
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
    result_path = Path(args.result_jsonl)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造裸 Ray Data MinerU baseline CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument(
        "--smoke-no-model",
        action="store_true",
        help="run real PDF render/lineage/regroup on CPU without loading vLLM",
    )
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--profile-system", default="raydata")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、运行 baseline，并打印 JSON 摘要。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
