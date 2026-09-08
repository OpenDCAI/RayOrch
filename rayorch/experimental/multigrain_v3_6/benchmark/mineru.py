"""使用 v3.6 compiler/runtime 回归真实 Flash-MinerU 368-PDF workload。

业务 UDF 与 V3 runner 完全复用；本模块只替换 ``RayModule + F.*`` authoring、
compiler 和 executor。这样性能差异只来自框架，而不是 render、VLM 或 assemble 内核。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, cast

from ...multigrain_v3.benchmark.mineru import (
    DEFAULT_FLASH_REPO,
    DEFAULT_MODEL,
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
    ResourceSampler,
    _runtime_env,
)
from .. import (
    F,
    Executor,
    GroupFailure,
    ItemOutcome,
    Pipeline,
    Port,
    RayModule,
)
from .mineru_poison import (
    DEFAULT_POISON_SEED,
    PoisonPage,
    PoisonedPage,
    gpu_memory_peaks,
    load_pdf_manifest,
    pdf_page_counts,
    select_poison_pages,
    worker_resource_options,
)


V3_GOLDEN_MEASURED_S = 587.781


class PoisoningMinerUVlmOcrPage(MinerUVlmOcrPage):
    """Real OCR wrapper that never sends designated bad pages to the model."""

    def __init__(
        self,
        *,
        model: str,
        gpu_memory_utilization: float,
        poison_pages: tuple[tuple[str, int, str], ...],
        poison_policy: str,
    ) -> None:
        super().__init__(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if poison_policy not in {"group_failure", "skip_page"}:
            raise ValueError("unsupported poison policy")
        self.poison_pages = {
            (os.path.abspath(path), int(page_id)): cause
            for path, page_id, cause in poison_pages
        }
        self.poison_policy = poison_policy

    def run(self, pages: list[dict[str, Any]]) -> list[Any]:
        poisoned = {
            index: self.poison_pages[key]
            for index, page in enumerate(pages)
            if (
                key := (
                    os.path.abspath(page["pdf_path"]),
                    int(page["page_id"]),
                )
            )
            in self.poison_pages
        }
        if not poisoned:
            return super().run(pages)

        live_indices = tuple(
            index for index in range(len(pages)) if index not in poisoned
        )
        live_results = (
            super().run([pages[index] for index in live_indices])
            if live_indices
            else []
        )
        by_index = dict(zip(live_indices, live_results, strict=True))
        return [
            (
                GroupFailure(poisoned[index])
                if self.poison_policy == "group_failure"
                else PoisonedPage(poisoned[index])
            )
            if index in poisoned
            else by_index[index]
            for index in range(len(pages))
        ]


class PoisonTolerantMinerUAssembleDoc(MinerUAssembleDoc):
    """Assemble a document after explicitly removing bad page/content pairs."""

    def run(
        self,
        grouped_contents: list[list[Any]],
        grouped_pages: list[list[dict[str, Any]]],
        stems: list[str],
    ) -> list[dict[str, Any]]:
        clean_contents = []
        clean_pages = []
        poison_ids = []
        for contents, pages in zip(grouped_contents, grouped_pages, strict=True):
            if len(contents) != len(pages):
                raise ValueError("MinerU content/page groups must align")
            live = [
                (content, page)
                for content, page in zip(contents, pages, strict=True)
                if not isinstance(content, PoisonedPage)
            ]
            if not live:
                raise ValueError("cannot assemble a PDF when every page is poisoned")
            clean_contents.append([content for content, _ in live])
            clean_pages.append([page for _, page in live])
            poison_ids.append(
                [
                    int(page["page_id"])
                    for content, page in zip(contents, pages, strict=True)
                    if isinstance(content, PoisonedPage)
                ]
            )
        outputs = super().run(clean_contents, clean_pages, stems)
        for output, original, page_ids in zip(
            outputs, grouped_pages, poison_ids, strict=True
        ):
            output.update(
                input_pages=len(original),
                poisoned_pages=len(page_ids),
                poison_page_ids=page_ids,
            )
        return outputs


class MetadataOnlyMinerUAssembleDoc:
    """Validate exact document membership without formatting Markdown/layout."""

    def run(
        self,
        grouped_contents: list[list[Any]],
        grouped_pages: list[list[dict[str, Any]]],
        stems: list[str],
    ) -> list[dict[str, Any]]:
        outputs = []
        for contents, pages, stem in zip(
            grouped_contents, grouped_pages, stems, strict=True
        ):
            if len(contents) != len(pages):
                raise ValueError("MinerU content/page groups must align")
            live_pages = [
                page
                for content, page in zip(contents, pages, strict=True)
                if not isinstance(content, PoisonedPage)
            ]
            poison_ids = [
                int(page["page_id"])
                for content, page in zip(contents, pages, strict=True)
                if isinstance(content, PoisonedPage)
            ]
            if not live_pages:
                raise ValueError("cannot assemble a PDF when every page is poisoned")
            page_ids = [int(page["page_id"]) for page in live_pages]
            outputs.append(
                {
                    "pdf": stem,
                    "pages": len(live_pages),
                    "page_ids": page_ids,
                    "input_pages": len(pages),
                    "poisoned_pages": len(poison_ids),
                    "poison_page_ids": poison_ids,
                }
            )
        return outputs


class MinerUV36Pipeline(Pipeline):
    """显式建模 PDF→Page→OCR→Document 的真实 MinerU Pipeline。"""

    def __init__(
        self,
        *,
        output_dir: str,
        batching_policy: str,
        model: str,
        replicas: int,
        batch_size: int,
        gpu_memory_utilization: float,
        render_replicas: int,
        reduce_replicas: int,
        runtime_env: dict[str, Any],
        render_dpi: int = 200,
        assemble_mode: str = "full",
        poison_manifest: tuple[PoisonPage, ...] = (),
        poison_policy: str = "group_failure",
        ready_queue_order: str = "fifo",
        cpu_worker_resource: str | None = None,
        gpu_worker_resource: str | None = None,
    ) -> None:
        """冻结四个计算 Call 的 actor、batch 和资源配置。"""

        if batching_policy not in {"any_parent", "single_parent"}:
            raise ValueError(
                "batching_policy must be any_parent or single_parent"
            )
        if ready_queue_order not in {"fifo", "unordered"}:
            raise ValueError("ready_queue_order must be fifo or unordered")
        if assemble_mode not in {"full", "metadata_only"}:
            raise ValueError("assemble_mode must be full or metadata_only")
        self.ready_queue_order = ready_queue_order
        self.render = (
            RayModule(MinerUPdfToPages)
            .pre_init(dpi=render_dpi)
            .ray_options(
                replicas=render_replicas,
                batch_size=1,
                num_cpus=1,
                runtime_env=runtime_env,
                **worker_resource_options(cpu_worker_resource),
            )
        )
        ocr = RayModule(
            MinerUVlmOcrPage
            if not poison_manifest
            else PoisoningMinerUVlmOcrPage
        )
        init_kwargs: dict[str, Any] = {
            "model": model,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if poison_manifest:
            init_kwargs.update(
                poison_pages=tuple(
                    (entry.pdf_path, entry.page_id, entry.cause)
                    for entry in poison_manifest
                ),
                poison_policy=poison_policy,
            )
        self.ocr = (
            ocr.pre_init(**init_kwargs)
            .ray_options(
                replicas=replicas,
                batch_size=batch_size,
                batching_policy=batching_policy,
                num_gpus=1.0,
                num_cpus=1,
                runtime_env=runtime_env,
                **worker_resource_options(gpu_worker_resource),
            )
        )
        self.metadata = RayModule(PdfMetadata).ray_options(
            replicas=1,
            batch_size=32,
            num_cpus=1,
            runtime_env=runtime_env,
            **worker_resource_options(cpu_worker_resource),
        )
        assemble_target = (
            MetadataOnlyMinerUAssembleDoc
            if assemble_mode == "metadata_only"
            else (
                PoisonTolerantMinerUAssembleDoc
                if poison_manifest and poison_policy == "skip_page"
                else MinerUAssembleDoc
            )
        )
        assemble = RayModule(assemble_target)
        if assemble_mode == "full":
            assemble = assemble.pre_init(output_dir=output_dir)
        self.assemble = (
            assemble
            .ray_options(
                replicas=reduce_replicas,
                batch_size=4,
                num_cpus=1,
                runtime_env=runtime_env,
                **worker_resource_options(cpu_worker_resource),
            )
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        pdfs: Port,
    ):
        """以 Port 关系声明 1:M、跨 parent page compute 和 ordered M:1。"""

        pages = F.expand(cast(Port, self.render(pdfs)))
        contents = cast(Port, self.ocr(pages))
        stems = cast(Port, self.metadata(pdfs))
        content_groups, ordered_page_groups = F.reduce_aligned(
            contents,
            pages,
            members=contents,
        )
        return self.assemble(content_groups, ordered_page_groups, stems)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """运行一次 v3.6 MinerU gate，并写入摘要和 GPU samples。"""

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
    pipeline = MinerUV36Pipeline(
        output_dir=os.path.abspath(args.output_dir),
        batching_policy=args.batching_policy,
        model=args.model,
        replicas=args.replicas,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        render_replicas=args.render_replicas,
        reduce_replicas=args.reduce_replicas,
        runtime_env=runtime_env,
        render_dpi=args.render_dpi,
        poison_manifest=poison_manifest,
        poison_policy=args.poison_policy,
        assemble_mode=args.assemble_mode,
        cpu_worker_resource=args.cpu_worker_resource,
        gpu_worker_resource=args.gpu_worker_resource,
        ready_queue_order=args.ready_queue_order,
    )
    compiled = pipeline.compile()
    ocr_call = next(
        call
        for call, spec in compiled.logical.calls.items()
        if spec.udf.target is pipeline.ocr.udf
    )

    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        init_kwargs: dict[str, Any] = {
            "runtime_env": runtime_env,
            "include_dashboard": False,
        }
        if args.ray_address is None:
            init_kwargs.update(
                num_cpus=args.num_cpus,
                num_gpus=args.replicas,
                object_store_memory=int(args.object_store_gb * 1024**3),
            )
        ray.init(address=args.ray_address, **init_kwargs)

    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    result = None
    end_to_end = 0.0
    try:
        with Executor(
            compiled,
            ready_queue_order=args.ready_queue_order,
        ) as executor:
            startup_s = time.perf_counter() - started
            result = executor.run(
                pdfs,
                microbatch_size=args.microbatch_size,
                max_active_microbatches=args.max_active_microbatches,
            )
        end_to_end = time.perf_counter() - started
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
        if started_ray_here:
            ray.shutdown()
    if result is None:  # pragma: no cover - exception path exits in the try block
        raise RuntimeError("v3.6 MinerU run produced no result")

    outputs = list(cast(Iterable[dict[str, Any] | ItemOutcome], result.outputs))
    successful_outputs = [
        output for output in outputs if isinstance(output, dict)
    ]
    pages = sum(int(output["pages"]) for output in successful_outputs)
    output_outcomes = {
        outcome.name: (
            len(successful_outputs)
            if outcome is ItemOutcome.PRESENT
            else sum(output is outcome for output in outputs)
        )
        for outcome in ItemOutcome
    }
    heavy = next(
        metrics for metrics in result.calls
        if metrics.call_index == ocr_call.value
    )
    gpu_peak = gpu_memory_peaks(gpu_samples, args.replicas)
    measured = result.elapsed_s
    total_pages = sum(page_counts)
    injected = len(poison_manifest)
    poisoned_parent_pages = sum(entry.pdf_pages for entry in poison_manifest)
    observed = (
        output_outcomes[ItemOutcome.SUPPRESSED.name]
        if args.poison_policy == "group_failure"
        else sum(int(output.get("poisoned_pages", 0)) for output in successful_outputs)
    )
    expected_docs = (
        len(pdfs) - injected
        if args.poison_policy == "group_failure"
        else len(pdfs)
    )
    expected_pages = (
        total_pages - poisoned_parent_pages
        if args.poison_policy == "group_failure"
        else total_pages - injected
    )
    physical_model_pages = heavy.grains - observed
    sibling_not_dispatched = (
        total_pages - heavy.grains
        if args.poison_policy == "group_failure"
        else 0
    )
    payload = {
        "engine": "multigrain_v3_6",
        "batching_policy": args.batching_policy,
        "ready_queue_order": args.ready_queue_order,
        "assemble_mode": args.assemble_mode,
        "n_pdf": len(pdfs),
        "pages": pages,
        "input_pages": total_pages,
        "docs": len(outputs),
        "successful_docs": len(successful_outputs),
        "output_outcomes": output_outcomes,
        "poison_policy": args.poison_policy,
        "poison_seed": args.poison_seed,
        "poison_injected": injected,
        "poison_observed": observed,
        "poison_manifest": [entry.as_dict() for entry in poison_manifest],
        "poisoned_parent_pages": poisoned_parent_pages,
        "expected_successful_docs": expected_docs,
        "expected_output_pages": expected_pages,
        "poison_report_contract_passed": (
            observed == injected
            and len(successful_outputs) == expected_docs
            and pages == expected_pages
        ),
        "batch_size": args.batch_size,
        "render_dpi": args.render_dpi,
        "replicas": args.replicas,
        "microbatch_size": args.microbatch_size,
        "max_active_microbatches": args.max_active_microbatches,
        "batch_policy": "immediate_work_conserving",
        "startup_s": round(startup_s, 3),
        "measured_wall_s": round(measured, 3),
        "end_to_end_wall_s": round(end_to_end, 3),
        "pages_per_s": round(pages / max(measured, 1e-9), 4),
        "rpc_count": result.rpc_count,
        "ocr_rpc_count": heavy.rpcs,
        "ocr_grains": heavy.grains,
        "ocr_model_pages": physical_model_pages,
        "ocr_model_pages_saved_vs_baseline": total_pages - physical_model_pages,
        "group_sibling_pages_not_dispatched": sibling_not_dispatched,
        "group_sibling_pages_computed_but_discarded": (
            max(
                0,
                poisoned_parent_pages - injected - sibling_not_dispatched,
            )
            if args.poison_policy == "group_failure"
            else 0
        ),
        "ocr_retries": heavy.retries,
        "ocr_grains_per_rpc": heavy.average_batch,
        "ocr_batch_fill_ratio": heavy.average_batch / args.batch_size,
        "ocr_batch_histogram": {
            str(size): heavy.batch_sizes.count(size)
            for size in sorted(set(heavy.batch_sizes))
        },
        "active_arenas_high_watermark": result.peak_active_microbatches,
        "actor_count": result.actor_count,
        "released_values": result.released_values,
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "ratio_vs_v3_golden": (
            measured / V3_GOLDEN_MEASURED_S if len(pdfs) == 368 else None
        ),
        "within_v3_golden_5_percent": (
            measured <= V3_GOLDEN_MEASURED_S * 1.05
            if len(pdfs) == 368
            else None
        ),
        "output_dir": os.path.abspath(args.output_dir),
        "cpu_worker_resource": args.cpu_worker_resource,
        "gpu_worker_resource": args.gpu_worker_resource,
    }

    artifact_root = Path(args.artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    with (artifact_root / "gpu_samples.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for sample in gpu_samples:
            handle.write(json.dumps(asdict(sample)) + "\n")
    (artifact_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_root / "poison_manifest.json").write_text(
        json.dumps(
            [entry.as_dict() for entry in poison_manifest],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result_path = Path(args.result_jsonl)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造 v3.6 MinerU 4/48/368 通用 CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batching-policy",
        choices=("any_parent", "single_parent"),
        default="any_parent",
    )
    parser.add_argument(
        "--ready-queue-order",
        choices=("fifo", "unordered"),
        default="fifo",
        help="per-Call READY ordering; unordered is the FIFO ablation",
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-active-microbatches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument(
        "--assemble-mode",
        choices=("full", "metadata_only"),
        default="full",
    )
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--poison-pdf-index", type=int, default=None)
    parser.add_argument("--poison-count", type=int, default=0)
    parser.add_argument("--poison-page-id", type=int, default=0)
    parser.add_argument(
        "--poison-policy",
        choices=("group_failure", "skip_page"),
        default="group_failure",
    )
    parser.add_argument("--poison-seed", default=DEFAULT_POISON_SEED)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--cpu-worker-resource", default=None)
    parser.add_argument("--gpu-worker-resource", default=None)
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、执行 gate，并打印摘要 JSON。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())


__all__ = [
    "MinerUV36Pipeline",
    "PoisoningMinerUVlmOcrPage",
    "PoisonTolerantMinerUAssembleDoc",
    "build_parser",
    "run_benchmark",
]
