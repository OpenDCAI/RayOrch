"""Manual CLI for reproducible paired elastic-rebatching experiments."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

from .report import write_reports
from .runner import run_paired_repetitions
from .workload import generate_workload


def _csv(value: str, cast):
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired parent-bound/elastic Multigrain V2.5 Ray experiments."
        )
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parents", type=int, default=100)
    parser.add_argument("--fanout-modes", default="uniform,lognormal,pareto")
    parser.add_argument("--fanout-scale", type=float, default=8.0)
    parser.add_argument("--service-modes", default="constant,lognormal")
    parser.add_argument("--mean-service-ms", type=float, default=5.0)
    parser.add_argument("--drop-probability", type=float, default=0.0)
    parser.add_argument("--actors", default="1,4")
    parser.add_argument("--batch-sizes", default="4,16")
    parser.add_argument("--wait-ms", default="2,5")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--num-cpus", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or Path("benchmark_results") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fanout_modes = _csv(args.fanout_modes, str)
    service_modes = _csv(args.service_modes, str)
    actors = _csv(args.actors, int)
    batch_sizes = _csv(args.batch_sizes, int)
    waits = _csv(args.wait_ms, float)
    seeds = _csv(args.seeds, int)
    combinations = tuple(
        itertools.product(
            fanout_modes,
            service_modes,
            actors,
            batch_sizes,
            waits,
            seeds,
        )
    )

    import ray

    init_kwargs = {
        "address": args.ray_address,
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }
    if args.ray_address == "local":
        init_kwargs["num_cpus"] = args.num_cpus
    ray.init(**init_kwargs)
    reports = []
    try:
        for index, (
            fanout_mode,
            service_mode,
            replica_count,
            batch_size,
            wait_ms,
            seed,
        ) in enumerate(combinations, start=1):
            print(
                json.dumps(
                    {
                        "case": index,
                        "total_cases": len(combinations),
                        "fanout": fanout_mode,
                        "service": service_mode,
                        "actors": replica_count,
                        "batch_size": batch_size,
                        "wait_ms": wait_ms,
                        "seed": seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            workload = generate_workload(
                seed=seed,
                parent_count=args.parents,
                fanout_mode=fanout_mode,
                fanout_scale=args.fanout_scale,
                service_mode=service_mode,
                mean_service_s=args.mean_service_ms / 1000.0,
                drop_probability=args.drop_probability,
            )
            reports.extend(
                run_paired_repetitions(
                    workload,
                    batch_size=batch_size,
                    elastic_wait_ms=wait_ms,
                    replicas=replica_count,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                )
            )
    finally:
        ray.shutdown()

    paths = write_reports(reports, output_dir=output_dir)
    print(
        json.dumps(
            {
                "trials": len(reports),
                "raw_jsonl": str(paths[0]),
                "summary_json": str(paths[1]),
                "summary_csv": str(paths[2]),
            },
            sort_keys=True,
        )
    )
    return 0
