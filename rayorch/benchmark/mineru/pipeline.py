"""Run the real MinerU workload on the released RayOrch runtime."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, cast

from .udfs import (
    DEFAULT_FLASH_REPO,
    DEFAULT_MODEL,
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
    ResourceSampler,
    _runtime_env,
)
from ...multigrain import (
    F,
    Executor,
    Pipeline,
    Port,
    RayModule,
)

MINERU_GOLDEN_MEASURED_S = 587.781
MINERU_GOLDEN_PDFS = 368
MINERU_GOLDEN_PAGES = 7_072


def _worker_resource_options(name: str | None) -> dict[str, Any]:
    """Return an optional custom Ray resource requirement for one actor."""

    return {} if not name else {"resources": {name: 0.001}}


def _gpu_memory_peaks(samples: Iterable[Any], count: int) -> tuple[int, ...]:
    """Compute per-device peak memory without assuming driver-side GPUs."""

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


def _load_pdf_paths(
    manifest_path: str,
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Load ordered PDF paths from a JSON array or JSONL manifest."""

    raw_text = Path(manifest_path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        entries = [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]
    else:
        entries = parsed if isinstance(parsed, list) else [parsed]

    paths: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"manifest entry {index} must contain a path")
        path = os.path.abspath(str(entry["path"]))
        if path in seen:
            raise ValueError(f"manifest entry {index} duplicates {path}")
        seen.add(path)
        paths.append(path)
        if limit is not None and limit > 0 and len(paths) >= limit:
            break
    if not paths:
        raise ValueError("PDF manifest is empty")
    return tuple(paths)


def _validate_inputs(
    pdfs: tuple[str, ...],
    *,
    flash_repo: str,
    model: str,
    packaged_runtime: bool = False,
) -> None:
    """Fail before Ray startup when required local workload inputs are absent."""

    missing_pdfs = [path for path in pdfs if not Path(path).is_file()]
    if missing_pdfs:
        preview = ", ".join(missing_pdfs[:3])
        suffix = "" if len(missing_pdfs) <= 3 else ", ..."
        raise FileNotFoundError(f"PDF inputs do not exist: {preview}{suffix}")
    if not packaged_runtime and not Path(flash_repo).is_dir():
        raise FileNotFoundError(f"Flash-MinerU repository does not exist: {flash_repo}")
    if not Path(model).exists():
        raise FileNotFoundError(f"MinerU model does not exist: {model}")
    stems = [Path(path).stem for path in pdfs]
    if len(stems) != len(set(stems)):
        raise ValueError("PDF filenames must have unique stems")


def _corpus_digest(pdfs: Iterable[str]) -> str:
    """Hash ordered PDF names and content without depending on mount paths."""

    digest = hashlib.sha256()
    for path in pdfs:
        absolute = os.path.abspath(path)
        digest.update(Path(absolute).name.encode("utf-8"))
        digest.update(b"\0")
        content = hashlib.sha256()
        with open(absolute, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                content.update(chunk)
        digest.update(content.digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _golden_comparable(
    *,
    corpus_digest: str,
    expected_digest: str | None,
    pdf_count: int,
    page_count: int,
) -> bool:
    """Return whether a run is comparable with the recorded MinerU baseline."""

    if expected_digest is None:
        return False
    expected = expected_digest.lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("--golden-corpus-sha256 must be a 64-character hex digest")
    return (
        corpus_digest == expected
        and pdf_count == MINERU_GOLDEN_PDFS
        and page_count == MINERU_GOLDEN_PAGES
    )


class MinerUPipeline(Pipeline):
    """Model the real PDF -> page -> OCR -> document MinerU pipeline."""

    def __init__(
        self,
        *,
        output_dir: str,
        model: str,
        replicas: int,
        batch_size: int,
        gpu_memory_utilization: float,
        render_replicas: int,
        reduce_replicas: int,
        runtime_env: dict[str, Any],
        render_dpi: int = 200,
        cpu_worker_resource: str | None = None,
        gpu_worker_resource: str | None = None,
    ) -> None:
        """Freeze actor, batch, and resource settings for the four Calls."""

        self.render = (
            RayModule(MinerUPdfToPages)
            .pre_init(dpi=render_dpi)
            .ray_options(
                replicas=render_replicas,
                batch_size=1,
                num_cpus=1,
                runtime_env=runtime_env,
                **_worker_resource_options(cpu_worker_resource),
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
                num_gpus=1.0,
                num_cpus=1,
                runtime_env=runtime_env,
                **_worker_resource_options(gpu_worker_resource),
            )
        )
        self.metadata = RayModule(PdfMetadata).ray_options(
            replicas=1,
            batch_size=32,
            num_cpus=1,
            runtime_env=runtime_env,
            **_worker_resource_options(cpu_worker_resource),
        )
        self.assemble = (
            RayModule(MinerUAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(
                replicas=reduce_replicas,
                batch_size=4,
                num_cpus=1,
                runtime_env=runtime_env,
                **_worker_resource_options(cpu_worker_resource),
            )
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        pdfs: Port,
    ):
        """Declare 1:M page work and ordered M:1 document reconstruction."""

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
    """Run one MinerU regression and persist its summary and GPU samples."""

    import ray  # pyright: ignore[reportMissingImports]

    flash_repo = os.path.abspath(args.flash_repo)
    if args.input_manifest:
        pdfs = _load_pdf_paths(
            args.input_manifest,
            limit=args.limit,
        )
    else:
        pdfs = tuple(
            sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
        )
        if not pdfs:
            raise FileNotFoundError(f"no PDFs found under {flash_repo}")
    _validate_inputs(
        pdfs,
        flash_repo=flash_repo,
        model=args.model,
        packaged_runtime=os.environ.get("RAYORCH_PACKAGED_RUNTIME") == "1",
    )
    corpus_digest = _corpus_digest(pdfs)

    runtime_env = _runtime_env(flash_repo)
    pipeline = MinerUPipeline(
        output_dir=os.path.abspath(args.output_dir),
        model=args.model,
        replicas=args.replicas,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        render_replicas=args.render_replicas,
        reduce_replicas=args.reduce_replicas,
        runtime_env=runtime_env,
        render_dpi=args.render_dpi,
        cpu_worker_resource=args.cpu_worker_resource,
        gpu_worker_resource=args.gpu_worker_resource,
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
        with Executor(compiled) as executor:
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
        raise RuntimeError("MinerU run produced no result")

    outputs = list(cast(Iterable[dict[str, Any]], result.outputs))
    if not all(isinstance(output, dict) for output in outputs):
        raise RuntimeError("MinerU regression produced a non-document outcome")
    pages = sum(int(output["pages"]) for output in outputs)
    heavy = next(
        metrics for metrics in result.calls
        if metrics.call_index == ocr_call.value
    )
    gpu_peak = _gpu_memory_peaks(gpu_samples, args.replicas)
    measured = result.elapsed_s
    expected_digest = args.golden_corpus_sha256
    golden_comparable = _golden_comparable(
        corpus_digest=corpus_digest,
        expected_digest=expected_digest,
        pdf_count=len(pdfs),
        page_count=pages,
    )
    payload = {
        "engine": "rayorch_multigrain",
        "n_pdf": len(pdfs),
        "corpus_sha256": corpus_digest,
        "model": os.path.abspath(args.model),
        "pages": pages,
        "docs": len(outputs),
        "batch_size": args.batch_size,
        "render_dpi": args.render_dpi,
        "replicas": args.replicas,
        "microbatch_size": args.microbatch_size,
        "max_active_microbatches": args.max_active_microbatches,
        "scheduler": "completion_driven_ready_queue",
        "startup_s": round(startup_s, 3),
        "measured_wall_s": round(measured, 3),
        "end_to_end_wall_s": round(end_to_end, 3),
        "pages_per_s": round(pages / max(measured, 1e-9), 4),
        "rpc_count": result.rpc_count,
        "ocr_rpc_count": heavy.rpcs,
        "ocr_grains": heavy.grains,
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
        "golden_corpus_match": (
            None
            if expected_digest is None
            else corpus_digest == expected_digest.lower()
        ),
        "golden_comparable": golden_comparable,
        "ratio_vs_mineru_golden": (
            measured / MINERU_GOLDEN_MEASURED_S if golden_comparable else None
        ),
        "within_mineru_golden_5_percent": (
            measured <= MINERU_GOLDEN_MEASURED_S * 1.05
            if golden_comparable
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
    result_path = Path(args.result_jsonl)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the shared 4/48/368-document MinerU regression CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-active-microbatches", type=int, default=3)
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


__all__ = [
    "MinerUPipeline",
    "build_parser",
    "run_benchmark",
]
