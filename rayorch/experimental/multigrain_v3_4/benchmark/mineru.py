"""使用 v3.4 Port API 回归真实 Flash-MinerU PDF→Page→OCR→Document。

业务 UDF 与 v3 runner 完全复用，避免模型、render 或 assemble 逻辑漂移；本模块只替换
Pipeline 表达和执行器。4/48/368 三个 gate 因而能直接比较 v3 与 v3.4 的语义输出及
parent-bound/elastic 调度指标。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from .. import functional as F
from ..api import Pipeline, RayModule
from ..executor import Executor


class MinerUV33Pipeline(Pipeline):
    """显式建模 Expand/Reduce 的真实 MinerU v3.4 Pipeline。"""

    def __init__(
        self,
        *,
        output_dir: str,
        mode: str,
        model: str,
        replicas: int,
        batch_size: int,
        gpu_memory_utilization: float,
        render_replicas: int,
        reduce_replicas: int,
        runtime_env: dict[str, Any],
    ) -> None:
        """冻结四个 Call 的 actor、batch 和资源配置。"""

        self.render = (
            RayModule(MinerUPdfToPages)
            .pre_init(dpi=200)
            .ray_options(
                replicas=render_replicas,
                batch_size=1,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )
        self.ocr = (
            RayModule(MinerUVlmOcrPage)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            .ray_options(
                replicas=replicas,
                batch_size=batch_size,
                batch_scope=mode,
                num_gpus=1.0,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )
        self.metadata = RayModule(PdfMetadata).ray_options(
            replicas=1,
            batch_size=32,
            num_cpus=1,
            runtime_env=runtime_env,
        )
        self.assemble = (
            RayModule(MinerUAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(
                replicas=reduce_replicas,
                batch_size=4,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )

    def forward(self, pdfs):
        """以 Port 关系显式声明 1:M、page compute 与 M:1。"""

        page_groups = self.render(pdfs)
        pages = F.expand(page_groups)
        contents = self.ocr(pages)
        stems = self.metadata(pdfs)
        content_groups, ordered_page_groups = F.reduce_aligned(
            contents,
            pages,
            members=contents,
        )
        return self.assemble(content_groups, ordered_page_groups, stems)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """运行一次 v3.4 MinerU gate，并增量写入摘要与 GPU samples。"""

    flash_repo = os.path.abspath(args.flash_repo)
    pdfs = sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {flash_repo}")

    runtime_env = _runtime_env(flash_repo)
    pipeline = MinerUV33Pipeline(
        output_dir=os.path.abspath(args.output_dir),
        mode=args.mode,
        model=args.model,
        replicas=args.replicas,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        render_replicas=args.render_replicas,
        reduce_replicas=args.reduce_replicas,
        runtime_env=runtime_env,
    )
    compiled = pipeline.compile()
    ocr_call = next(
        call
        for call, spec in compiled.program.calls.items()
        if spec.kernel.target is MinerUVlmOcrPage
    )

    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    try:
        with Executor(
            compiled,
            address=args.ray_address,
            ray_init_kwargs={"runtime_env": runtime_env},
        ) as executor:
            # ready 屏障已经覆盖全部初始 actor；此处切分出与 v3
            # 相同口径的 startup 与 measured runtime。
            startup_s = time.perf_counter() - started
            result = executor.run(
                pdfs,
                arena_size=args.microbatch_size,
                max_in_flight=args.max_inflight_arenas,
            )
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
    end_to_end = time.perf_counter() - started

    outputs = result.outputs
    pages = sum(int(output["pages"]) for output in outputs)
    heavy = result.calls[ocr_call]
    gpu_count = max(args.replicas, 0)
    gpu_peak = tuple(
        max((sample.memory_used[index] for sample in gpu_samples), default=0)
        for index in range(gpu_count)
    )
    payload = {
        "engine": "multigrain_v3_4",
        "mode": args.mode,
        "n_pdf": len(pdfs),
        "pages": pages,
        "docs": len(outputs),
        "batch_size": args.batch_size,
        "replicas": args.replicas,
        "microbatch_size": args.microbatch_size,
        "max_inflight_arenas": args.max_inflight_arenas,
        "startup_s": round(startup_s, 3),
        "measured_wall_s": round(result.elapsed_s, 3),
        "end_to_end_wall_s": round(end_to_end, 3),
        "pages_per_s": round(pages / max(result.elapsed_s, 1e-9), 4),
        "rpc_count": result.rpc_count,
        "ocr_rpc_count": heavy.rpcs,
        "ocr_grains_per_rpc": heavy.average_batch,
        "active_arenas_high_watermark": result.max_active_arenas,
        "actor_count": result.actor_count,
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "output_dir": os.path.abspath(args.output_dir),
    }

    artifact_root = Path(args.artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    with (artifact_root / "gpu_samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in gpu_samples:
            handle.write(json.dumps(asdict(sample)) + "\n")
    (artifact_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result_path = Path(args.result_jsonl)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造 v3.4 MinerU 4/48/368 通用 CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("elastic", "parent_bound"), default="elastic")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-inflight-arenas", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--ray-address", default=None)
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


__all__ = ["MinerUV33Pipeline", "build_parser", "run_benchmark"]
