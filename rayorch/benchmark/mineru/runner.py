"""Execute the MinerU graph and persist benchmark artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, cast

from rayorch import Executor

from .pipeline import MinerUPipeline
from .udfs import DEFAULT_MODEL
from .validate import (
    MINERU_GOLDEN_MEASURED_S,
    corpus_digest,
    golden_comparable,
    load_pdf_paths,
    validate_inputs,
)


DEFAULT_FLASH_REPO = os.environ.get("RAYORCH_MINERU_REPO", "./Flash-mineru")


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One observation-only GPU utilization and memory sample."""

    monotonic_s: float
    utilization: tuple[int | None, ...]
    memory_used: tuple[int, ...]


class ResourceSampler:
    """Collect driver RSS and GPU metrics without affecting scheduling."""

    def __init__(self, interval_s: float) -> None:
        import psutil  # pyright: ignore[reportMissingModuleSource]

        self.interval_s = interval_s
        self.process = psutil.Process()
        self.driver_start = int(self.process.memory_info().rss)
        self.driver_peak = self.driver_start
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> tuple[int, int, tuple[GpuSample, ...]]:
        self._stop.set()
        self._thread.join(timeout=max(2, self.interval_s * 2))
        return self.driver_start, self.driver_peak, tuple(self.samples)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.driver_peak = max(
                    self.driver_peak,
                    int(self.process.memory_info().rss),
                )
                self.samples.append(_gpu_sample())
            except Exception:
                continue


def _gpu_sample() -> GpuSample:
    """Best-effort sample of utilization and memory for every visible GPU."""

    try:
        import pynvml  # pyright: ignore[reportMissingImports]

        pynvml.nvmlInit()
        utilization = []
        memory = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                utilization.append(
                    int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                )
            except Exception:
                utilization.append(None)
            memory.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
        pynvml.nvmlShutdown()
        return GpuSample(time.monotonic(), tuple(utilization), tuple(memory))
    except Exception:
        return GpuSample(time.monotonic(), (), ())


def _gpu_memory_peaks(
    samples: Iterable[GpuSample],
    count: int,
) -> tuple[int, ...]:
    samples = tuple(samples)
    return tuple(
        max(
            (
                sample.memory_used[index]
                for sample in samples
                if index < len(sample.memory_used)
            ),
            default=0,
        )
        for index in range(count)
    )


def _development_runtime_env(flash_repo: str) -> dict[str, Any]:
    """Build the driver-level import environment for local source checkouts."""

    if os.environ.get("RAYORCH_PACKAGED_RUNTIME") == "1":
        return {}
    current = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        path for path in (flash_repo, os.getcwd(), current) if path
    )
    return {"env_vars": {"PYTHONPATH": pythonpath}}


def _ray_options(
    *,
    resource: str | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Add an optional cluster resource to ordinary Ray actor options."""

    if resource:
        options["resources"] = {resource: 0.001}
    return options


def _stage_options(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Translate runner configuration into the four physical stage contracts."""

    return {
        "render": _ray_options(
            replicas=args.render_replicas,
            batch_size=1,
            num_cpus=1,
            resource=args.cpu_worker_resource,
        ),
        "ocr": _ray_options(
            replicas=args.replicas,
            batch_size=args.batch_size,
            num_gpus=1.0,
            num_cpus=1,
            resource=args.gpu_worker_resource,
        ),
        "metadata": _ray_options(
            replicas=1,
            batch_size=32,
            num_cpus=1,
            resource=args.cpu_worker_resource,
        ),
        "assemble": _ray_options(
            replicas=args.reduce_replicas,
            batch_size=4,
            num_cpus=1,
            resource=args.cpu_worker_resource,
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run one MinerU regression and persist its summary and GPU samples."""

    import ray  # pyright: ignore[reportMissingImports]

    flash_repo = os.path.abspath(args.flash_repo)
    if args.input_manifest:
        pdfs = load_pdf_paths(args.input_manifest, limit=args.limit)
    else:
        pdfs = tuple(
            sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
        )
        if not pdfs:
            raise FileNotFoundError(f"no PDFs found under {flash_repo}")
    packaged_runtime = os.environ.get("RAYORCH_PACKAGED_RUNTIME") == "1"
    validate_inputs(
        pdfs,
        flash_repo=flash_repo,
        model=args.model,
        packaged_runtime=packaged_runtime,
    )
    input_digest = corpus_digest(pdfs)

    pipeline = MinerUPipeline(
        output_dir=os.path.abspath(args.output_dir),
        model=args.model,
        render_dpi=args.render_dpi,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **_stage_options(args),
    )
    compiled = pipeline.compile()
    ocr_call = next(
        call
        for call, spec in compiled.logical.calls.items()
        if spec.udf.target is pipeline.ocr.udf
    )

    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        init_kwargs: dict[str, Any] = {"include_dashboard": False}
        development_env = _development_runtime_env(flash_repo)
        if development_env:
            init_kwargs["runtime_env"] = development_env
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
        with Executor(compiled) as executor:
            startup_s = time.perf_counter() - started
            result = executor.run(
                pdfs,
                input_batch_size=args.input_batch_size,
                max_active_input_batches=args.max_active_input_batches,
            )
        end_to_end = time.perf_counter() - started
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
        if started_ray_here:
            ray.shutdown()
    if result is None:  # pragma: no cover - exception path exits in the try block
        raise RuntimeError("MinerU run produced no result")

    outputs = list(cast(Iterable[dict[str, Any]], result.outputs))
    if not all(isinstance(output, dict) for output in outputs):
        raise RuntimeError("MinerU regression produced a non-document outcome")
    pages = sum(int(output["pages"]) for output in outputs)
    heavy = next(
        metrics for metrics in result.calls if metrics.call_index == ocr_call.value
    )
    gpu_peak = _gpu_memory_peaks(gpu_samples, args.replicas)
    measured = result.elapsed_s
    expected_digest = args.golden_corpus_sha256
    comparable = golden_comparable(
        corpus_digest=input_digest,
        expected_digest=expected_digest,
        pdf_count=len(pdfs),
        page_count=pages,
    )
    payload = {
        "engine": "rayorch",
        "n_pdf": len(pdfs),
        "corpus_sha256": input_digest,
        "model": os.path.abspath(args.model),
        "pages": pages,
        "docs": len(outputs),
        "batch_size": args.batch_size,
        "render_dpi": args.render_dpi,
        "replicas": args.replicas,
        "input_batch_size": args.input_batch_size,
        "max_active_input_batches": args.max_active_input_batches,
        "scheduler": "completion_driven_ready_queue",
        "startup_s": round(startup_s, 3),
        "measured_wall_s": round(measured, 3),
        "end_to_end_wall_s": round(end_to_end, 3),
        "pages_per_s": round(pages / max(measured, 1e-9), 4),
        "rpc_count": result.rpc_count,
        "ocr_rpc_count": heavy.rpcs,
        "ocr_grain_dispatches": heavy.grain_dispatches,
        "ocr_grain_requeues": heavy.grain_requeues,
        "ocr_grains_per_rpc": heavy.average_batch,
        "ocr_batch_fill_ratio": heavy.average_batch / args.batch_size,
        "ocr_batch_histogram": {
            str(size): heavy.batch_sizes.count(size)
            for size in sorted(set(heavy.batch_sizes))
        },
        "peak_active_input_batches": result.peak_active_input_batches,
        "actor_count": result.actor_count,
        "released_values": result.released_values,
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "golden_corpus_match": (
            None if expected_digest is None else input_digest == expected_digest.lower()
        ),
        "golden_comparable": comparable,
        "ratio_vs_mineru_golden": (
            measured / MINERU_GOLDEN_MEASURED_S if comparable else None
        ),
        "within_mineru_golden_5_percent": (
            measured <= MINERU_GOLDEN_MEASURED_S * 1.05 if comparable else None
        ),
        "output_dir": os.path.abspath(args.output_dir),
        "cpu_worker_resource": args.cpu_worker_resource,
        "gpu_worker_resource": args.gpu_worker_resource,
    }
    _write_artifacts(
        payload,
        gpu_samples,
        artifact_dir=Path(args.artifact_dir),
        result_path=Path(args.result_jsonl),
    )
    return payload


def _write_artifacts(
    payload: dict[str, Any],
    gpu_samples: Iterable[Any],
    *,
    artifact_dir: Path,
    result_path: Path,
) -> None:
    """Persist one run without coupling validation or graph construction to I/O."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "gpu_samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in gpu_samples:
            handle.write(json.dumps(asdict(sample)) + "\n")
    (artifact_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the shared 4/48/368-document MinerU regression CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--input-batch-size", type=int, default=24)
    parser.add_argument("--max-active-input-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--cpu-worker-resource", default=None)
    parser.add_argument("--gpu-worker-resource", default=None)
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument(
        "--golden-corpus-sha256",
        default=None,
        help=(
            "expected corpus digest; the historical 368-PDF performance gate "
            "is evaluated only when this digest and the 7,072-page shape match"
        ),
    )
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the regression, and print its JSON summary."""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_benchmark"]
