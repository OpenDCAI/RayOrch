"""Command-line runner for real and dependency-free V3 MinerU workloads."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .mineru import (
    DEFAULT_DUMMY_PAGE_COUNTS,
    DEFAULT_FLASH_REPO,
    DEFAULT_MODEL,
    MinerUDummyPipeline,
    MinerUV3Pipeline,
    discover_pdfs,
    execute_pipeline,
    jsonable,
    make_dummy_pdfs,
    real_runtime_env,
)


def _runtime_env_arg(value: str) -> dict[str, object]:
    """Parse an inline JSON object or an ``@path`` JSON file."""

    raw = value
    if value.startswith("@"):
        raw = Path(value[1:]).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"runtime_env must be valid JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("runtime_env JSON must be an object")
    return parsed


def _page_counts_arg(value: str) -> tuple[int, ...]:
    """Parse a comma-separated sequence of non-negative page counts."""

    try:
        counts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "dummy page counts must be comma-separated integers"
        ) from error
    if not counts or any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError(
            "dummy page counts must be non-empty and non-negative"
        )
    return counts


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit real/dummy MinerU benchmark CLI contract."""

    parser = argparse.ArgumentParser(
        description="Run the Multigrain V3 MinerU-shaped benchmark",
    )
    parser.add_argument(
        "--workload",
        choices=("dummy", "real"),
        default="dummy",
        help="dummy is CPU-only; real lazily loads Flash-MinerU and vLLM",
    )
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument(
        "--input-dir",
        default="",
        help="directory containing real PDF inputs; defaults to --flash-repo",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--render-batch-size", type=int, default=1)
    parser.add_argument(
        "--assemble-replicas",
        "--reduce-replicas",
        dest="assemble_replicas",
        type=int,
        default=4,
    )
    parser.add_argument("--assemble-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-wait-ms", type=float, default=5.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--runtime-env",
        type=_runtime_env_arg,
        default={},
        metavar="JSON|@PATH",
        help="Ray actor runtime_env as a JSON object or @JSON-file",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        default="./mineru_v3_output",
    )
    parser.add_argument(
        "--result-jsonl",
        "--result",
        dest="result_jsonl",
        default="./mineru_v3_results.jsonl",
    )
    parser.add_argument(
        "--dummy-page-counts",
        type=_page_counts_arg,
        default=DEFAULT_DUMMY_PAGE_COUNTS,
        metavar="N,N,...",
    )
    parser.add_argument("--dummy-delay-scale-s", type=float, default=0.0)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--num-cpus", type=int, default=16)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject resource values that cannot produce a valid actor graph."""

    positive = {
        "limit": args.limit,
        "replicas": args.replicas,
        "batch_size": args.batch_size,
        "render_replicas": args.render_replicas,
        "render_batch_size": args.render_batch_size,
        "assemble_replicas": args.assemble_replicas,
        "assemble_batch_size": args.assemble_batch_size,
        "dpi": args.dpi,
        "num_cpus": args.num_cpus,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(
            "arguments must be positive: " + ", ".join(sorted(invalid))
        )
    if args.max_batch_wait_ms < 0:
        raise ValueError("max_batch_wait_ms must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if args.dummy_delay_scale_s < 0:
        raise ValueError("dummy_delay_scale_s must be non-negative")


def _ensure_ray(args: argparse.Namespace):
    """Lazily initialize Ray and report whether this call owns shutdown."""

    import ray

    if ray.is_initialized():
        return ray, False
    init_options: dict[str, Any] = {
        "address": args.ray_address,
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }
    if args.ray_address == "local":
        init_options["num_cpus"] = args.num_cpus
    ray.init(**init_options)
    return ray, True


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Build, execute, and persist one selected MinerU benchmark run."""

    _validate_args(args)
    output_dir = str(Path(args.output_dir).expanduser().resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    runtime_env = dict(args.runtime_env)

    if args.workload == "real":
        flash_repo = Path(args.flash_repo).expanduser().resolve()
        if not flash_repo.is_dir():
            raise FileNotFoundError(
                f"Flash-MinerU repository does not exist: {flash_repo}"
            )
        input_dir = args.input_dir or str(flash_repo)
        source: list[object] = list(discover_pdfs(input_dir, args.limit))
        runtime_env = real_runtime_env(str(flash_repo), runtime_env)
        pipeline = MinerUV3Pipeline(
            output_dir=output_dir,
            model=args.model,
            replicas=args.replicas,
            batch_size=args.batch_size,
            max_batch_wait_ms=args.max_batch_wait_ms,
            gpu_memory_utilization=args.gpu_memory_utilization,
            render_replicas=args.render_replicas,
            render_batch_size=args.render_batch_size,
            assemble_replicas=args.assemble_replicas,
            assemble_batch_size=args.assemble_batch_size,
            runtime_env=runtime_env,
            dpi=args.dpi,
        )
    else:
        input_dir = args.input_dir
        source = list(
            make_dummy_pdfs(
                args.limit,
                page_counts=args.dummy_page_counts,
            )
        )
        pipeline = MinerUDummyPipeline(
            replicas=args.replicas,
            batch_size=args.batch_size,
            max_batch_wait_ms=args.max_batch_wait_ms,
            render_replicas=args.render_replicas,
            render_batch_size=args.render_batch_size,
            assemble_replicas=args.assemble_replicas,
            assemble_batch_size=args.assemble_batch_size,
            runtime_env=runtime_env,
            delay_scale_s=args.dummy_delay_scale_s,
        )

    ray, owns_ray = _ensure_ray(args)
    started = time.perf_counter()
    try:
        documents = execute_pipeline(pipeline, source)
    finally:
        if owns_ray:
            ray.shutdown()
    elapsed_s = time.perf_counter() - started
    result_path = Path(args.result_jsonl).expanduser().resolve()

    payload: dict[str, object] = {
        "workload": args.workload,
        "documents": jsonable(documents),
        "document_count": len(documents),
        "input_count": len(source),
        "elapsed_s": round(elapsed_s, 6),
        "replicas": args.replicas,
        "batch_size": args.batch_size,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "result_jsonl": str(result_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and print its machine-readable benchmark summary."""

    args = build_parser().parse_args(argv)
    payload = run_benchmark(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
