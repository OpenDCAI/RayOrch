"""当前 Flash-MinerU pipeline-parallel native baseline 的适配 CLI。

本模块不复制 Flash-MinerU scheduler，只调用其公开 `MineruEngine`，并输出可与 V3/Ray Data
并列保存的 JSON 摘要。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .mineru import DEFAULT_FLASH_REPO, DEFAULT_MODEL, ResourceSampler


def _ready_pipeline(engine: Any) -> None:
    """显式等待 Flash-MinerU 三个 Stage 的全部 actors 完成构造。"""

    import ray

    pipeline = engine._pipe
    refs = [
        actor.__ray_ready__.remote()
        for module in (
            pipeline.pdf2img,
            pipeline.process_img,
            pipeline.img2md,
        )
        for actor in module.actors
    ]
    if refs:
        ray.get(refs)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """运行 Flash-MinerU native DAG pipeline 并返回 startup-inclusive 摘要。"""

    flash_repo = os.path.abspath(args.flash_repo)
    if flash_repo not in sys.path:
        sys.path.insert(0, flash_repo)
    pdfs = sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {flash_repo}")

    from flash_mineru import MineruEngine

    sampler = ResourceSampler(args.rss_interval_s)
    started = time.perf_counter()
    engine = MineruEngine(
        model=str(Path(args.model).resolve()),
        save_dir=str(Path(args.output_dir).resolve()),
        batch_size=args.batch_size,
        replicas=args.replicas,
        num_gpus_per_replica=1.0,
        engine_gpu_util_rate_to_ray_cap=args.gpu_memory_utilization,
        inflight=args.inflight,
        dev_mode=False,
    )
    _ready_pipeline(engine)
    startup = time.perf_counter() - started
    sampler.start()
    measured = time.perf_counter()
    try:
        results = engine.run(pdfs)
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
    measured_wall = time.perf_counter() - measured
    gpu_peak = tuple(
        max((sample.memory_used[index] for sample in gpu_samples), default=0)
        for index in range(args.replicas)
    )
    payload = {
        "engine": "flash_mineru_native_dag",
        "n_pdf": len(pdfs),
        "result_batches": len(results),
        "pdf_batch_size": args.batch_size,
        "replicas": args.replicas,
        "inflight": args.inflight,
        "startup_s": round(startup, 3),
        "measured_wall_s": round(measured_wall, 3),
        "end_to_end_wall_s": round(startup + measured_wall, 3),
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with Path(args.result_jsonl).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造 Flash-MinerU native baseline CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--inflight", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、运行 native baseline 并打印 JSON。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
