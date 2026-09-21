"""Command-line entrypoint for ``python -m rayorch.benchmarks.mineru_scale``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .benchmark import MinerUScaleBench


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the scale-oriented MinerU PDF Benchmark.",
    )
    parser.add_argument(
        "--input",
        dest="input_paths",
        action="append",
        required=True,
        help="PDF, directory, or HDFS URI; repeat for multiple roots",
    )
    parser.add_argument("--output", required=True, help="shared output root")
    parser.add_argument("--model", required=True, help="node-visible model path")
    parser.add_argument("--artifact-dir", help="shared Benchmark report root")
    parser.add_argument("--input-limit", type=int)
    parser.add_argument("--render-replicas", type=int, default=256)
    parser.add_argument("--ocr-replicas", type=int, default=128)
    parser.add_argument("--assemble-replicas", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-batch-size", type=int, default=24)
    parser.add_argument("--max-active-input-batches", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.32)
    parser.add_argument("--gpus-per-ocr-actor", type=float, default=0.5)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument(
        "--ray-address",
        help="Ray address, for example 'auto'; omitted starts local Ray",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--no-profile", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bench = MinerUScaleBench(
        input_paths=tuple(args.input_paths),
        output_dir=args.output,
        model=args.model,
        artifact_dir=args.artifact_dir,
        input_limit=args.input_limit,
        render_replicas=args.render_replicas,
        ocr_replicas=args.ocr_replicas,
        assemble_replicas=args.assemble_replicas,
        batch_size=args.batch_size,
        input_batch_size=args.input_batch_size,
        max_active_input_batches=args.max_active_input_batches,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpus_per_ocr_actor=args.gpus_per_ocr_actor,
        render_dpi=args.render_dpi,
    )
    report = bench.run(
        ray_address=args.ray_address,
        profile=not args.no_profile,
        run_id=args.run_id,
    )
    report.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
