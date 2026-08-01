"""使用裸 Ray Data 复现 MinerU 的 PDF→Page→OCR→PDF 对比基线。

该 runner 刻意不依赖 Multigrain V3 runtime。为了获得与 V3 相同的 ordered Reduce
correctness，应用显式携带 ``parent_id`` 与 ``page_ordinal``，并在 Ray Data
``groupby().map_groups()`` 中恢复每个 PDF。
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
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    ResourceSampler,
    _runtime_env,
)


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

    image_rgb = row["image_rgb"]
    if not hasattr(image_rgb, "shape"):
        raise ValueError("image_rgb must be a numpy-compatible tensor")
    return {
        "pdf_path": str(row["pdf_path"]),
        "page_id": int(row["page_id"]),
        "img_pil": Image.fromarray(image_rgb, mode="RGB"),
        "scale": float(row["scale"]),
        "page_width": int(row["page_width"]),
        "page_height": int(row["page_height"]),
        "pdf_len": int(row["pdf_len"]),
    }


class RayDataRenderPdf:
    """Ray Data callable actor：渲染一个 PDF row，并显式附加 parent/ordinal。"""

    def __init__(self, dpi: int = 200) -> None:
        """创建与 V3 benchmark 相同的 PDF renderer。"""

        self.renderer = MinerUPdfToPages(dpi=dpi)

    def __call__(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """把单个 PDF 展开为可被 Ray Data 跨 parent 重组的 page rows。"""

        parent_id = int(row["parent_id"])
        pdf_path = str(row["pdf_path"])
        pages = self.renderer.run([pdf_path])[0]
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
            }
            for ordinal, page in enumerate(pages)
        ]


class RayDataOcrPages:
    """Ray Data GPU callable actor：对 page batch 执行真实 MinerU OCR。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.8,
    ) -> None:
        """初始化一个 persistent MinerU vLLM OCR 实例。"""

        self.ocr = MinerUVlmOcrPage(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """保留应用 lineage columns，并为每个 page 添加 OCR result。"""

        import numpy as np

        rows = _rows_from_batch(batch)
        pages = [_page_from_row(row) for row in rows]
        contents = self.ocr.run(pages)
        if len(contents) != len(rows):
            raise ValueError(
                "MinerU OCR output count does not match Ray Data input batch"
            )
        return {
            "parent_id": batch["parent_id"],
            "page_ordinal": batch["page_ordinal"],
            "pdf_path": batch["pdf_path"],
            "page_id": batch["page_id"],
            "image_rgb": batch["image_rgb"],
            "scale": batch["scale"],
            "page_width": batch["page_width"],
            "page_height": batch["page_height"],
            "pdf_len": batch["pdf_len"],
            "content": np.asarray(contents, dtype=object),
        }


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

    def __init__(self, output_dir: str, parse_method: str = "vlm") -> None:
        """创建与 V3 benchmark 相同的 MinerU assemble UDF。"""

        self.assembler = MinerUAssembleDoc(
            output_dir=output_dir,
            parse_method=parse_method,
        )

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
        output = self.assembler.run(
            [[row["content"] for row in rows]],
            [[_page_from_row(row) for row in rows]],
            [Path(pdf_path).stem],
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
    pages = dataset.flat_map(
        RayDataRenderPdf,
        concurrency=args.render_replicas,
        num_cpus=1,
        fn_constructor_kwargs={"dpi": 200},
    )
    smoke = bool(args.smoke_no_model)
    ocr_class = RayDataSmokeOcrPages if smoke else RayDataOcrPages
    contents = pages.map_batches(
        ocr_class,
        batch_size=args.batch_size,
        batch_format="numpy",
        concurrency=args.replicas,
        num_gpus=0 if smoke else 1,
        num_cpus=1,
        fn_constructor_kwargs=(
            {}
            if smoke
            else {
                "model": args.model,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        ),
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
                "output_dir": os.path.abspath(args.output_dir),
            }
        ),
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """运行 Ray Data baseline，并写与 V3 可并列比较的摘要 JSON。"""

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
            num_gpus=0 if args.smoke_no_model else args.replicas,
            object_store_memory=int(args.object_store_gb * 1024**3),
            include_dashboard=False,
            runtime_env=_runtime_env(flash_repo),
        )

    dataset = build_dataset(args, pdfs)
    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    try:
        rows = dataset.take_all()
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
    wall = time.perf_counter() - started
    rows.sort(key=lambda row: int(row["parent_id"]))
    outputs = [row["output"] for row in rows]
    pages = sum(int(output["pages"]) for output in outputs)
    gpu_peak = tuple(
        max((sample.memory_used[index] for sample in gpu_samples), default=0)
        for index in range(args.replicas)
    )
    payload = {
        "engine": "ray_data",
        "backend": "smoke_cpu" if args.smoke_no_model else "mineru_vllm",
        "n_pdf": len(pdfs),
        "pages": pages,
        "docs": len(outputs),
        "batch_size": args.batch_size,
        "replicas": args.replicas,
        "measured_wall_s": round(wall, 3),
        "pages_per_s": round(pages / wall, 4),
        "end_to_end_wall_s": round(wall, 3),
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "explicit_lineage_columns": ["parent_id", "page_ordinal"],
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、运行 baseline，并打印 JSON 摘要。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
