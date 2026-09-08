"""Repeat the controlled V3.6/Ray Data/Daft cardinality matrix.

Every arm starts a fresh local Ray runtime.  The six engine/poison arms are
cyclically rotated between repetitions so cache or thermal drift cannot always
favor the same system.  Child summaries remain the source of truth; this runner
only adds repetition/order metadata and a machine-readable JSONL index.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_MODULE = (
    "rayorch.experimental.multigrain_v3_6.benchmark.mineru_cardinality"
)
_ARMS = (
    ("v36", 0),
    ("ray_data", 0),
    ("daft", 0),
    ("v36", 1),
    ("ray_data", 1),
    ("daft", 1),
)


def _arms(poison_counts: tuple[int, ...]) -> tuple[tuple[str, int], ...]:
    return tuple(
        (engine, poison)
        for poison in poison_counts
        for engine in ("v36", "ray_data", "daft")
    )


def _rotated_arms(
    repetition: int,
    arms: tuple[tuple[str, int], ...] = _ARMS,
) -> tuple[tuple[str, int], ...]:
    offset = repetition % len(arms)
    return arms[offset:] + arms[:offset]


def _command(
    args: argparse.Namespace,
    *,
    engine: str,
    poison_largest: int,
    artifact_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        _MODULE,
        "--engine",
        engine,
        "--input-manifest",
        os.path.abspath(args.input_manifest),
        "--poison-largest",
        str(poison_largest),
        "--replicas",
        str(args.replicas),
        "--batch-size",
        str(args.batch_size),
        "--work-ms",
        str(args.work_ms),
        "--expand-replicas",
        str(args.expand_replicas),
        "--reduce-replicas",
        str(args.reduce_replicas),
        "--source-blocks",
        str(args.source_blocks),
        "--page-partitions",
        str(args.page_partitions),
        "--microbatch-size",
        str(args.microbatch_size),
        "--max-active-microbatches",
        str(args.max_active_microbatches),
        "--ray-address",
        args.ray_address,
        "--artifact-dir",
        str(artifact_dir),
    ]
    if args.selection_largest is not None:
        command.extend(
            [
                "--selection-largest",
                str(args.selection_largest),
                "--healthy-max-pages",
                str(args.healthy_max_pages),
            ]
        )
    return command


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    if args.daft_pythonpath:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            args.daft_pythonpath
            if not existing
            else f"{args.daft_pythonpath}{os.pathsep}{existing}"
        )

    arms = _arms(tuple(args.poison_counts))
    protocol = {
        "input_manifest": os.path.abspath(args.input_manifest),
        "engines": ["v36", "ray_data", "daft"],
        "poison_largest": list(args.poison_counts),
        "selection_largest": args.selection_largest,
        "healthy_max_pages": args.healthy_max_pages,
        "repetitions": args.repetitions,
        "replicas": args.replicas,
        "batch_size": args.batch_size,
        "work_ms": args.work_ms,
        "expand_replicas": args.expand_replicas,
        "reduce_replicas": args.reduce_replicas,
        "source_blocks": args.source_blocks,
        "page_partitions": args.page_partitions,
        "competitor_input_mode": "preexpanded_balanced_pages",
        "microbatch_size": args.microbatch_size,
        "max_active_microbatches": args.max_active_microbatches,
        "ray_address": args.ray_address,
        "order": [
            [
                f"{engine}-p{poison}"
                for engine, poison in _rotated_arms(rep, arms)
            ]
            for rep in range(args.repetitions)
        ],
        "primary_wall_metric": "framework_wall_s",
    }
    (root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    indexed: list[dict[str, Any]] = []
    index_path = root / "results.jsonl"
    if index_path.exists():
        raise FileExistsError(f"refusing to append to existing matrix: {index_path}")
    for repetition in range(args.repetitions):
        for order_index, (engine, poison) in enumerate(
            _rotated_arms(repetition, arms)
        ):
            name = f"r{repetition}-{order_index}-{engine}-p{poison}"
            child_dir = root / name
            child_dir.mkdir(parents=True, exist_ok=True)
            command = _command(
                args,
                engine=engine,
                poison_largest=poison,
                artifact_dir=child_dir,
            )
            print(f"START {name}", flush=True)
            started = time.perf_counter()
            with (child_dir / "run.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
                while True:
                    try:
                        process.wait(timeout=args.heartbeat_s)
                        break
                    except subprocess.TimeoutExpired:
                        print(
                            f"RUNNING {name} outer_s={time.perf_counter() - started:.1f}",
                            flush=True,
                        )
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            summary = json.loads((child_dir / "summary.json").read_text())
            record = {
                "repetition": repetition,
                "order_index": order_index,
                "arm": f"{engine}-p{poison}",
                "outer_wall_s": round(time.perf_counter() - started, 6),
                **summary,
            }
            indexed.append(record)
            with index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"DONE {name} framework_wall_s={summary['framework_wall_s']}",
                flush=True,
            )
    return indexed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--poison-counts", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--selection-largest", type=int, default=None)
    parser.add_argument("--healthy-max-pages", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--work-ms", type=float, default=100.0)
    parser.add_argument("--expand-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--source-blocks", type=int, default=8)
    parser.add_argument("--page-partitions", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--max-active-microbatches", type=int, default=1)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--daft-pythonpath", default=None)
    parser.add_argument("--heartbeat-s", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not args.poison_counts or any(count < 0 for count in args.poison_counts):
        raise ValueError("poison_counts must be non-negative")
    if len(set(args.poison_counts)) != len(args.poison_counts):
        raise ValueError("poison_counts must be unique")
    if (args.selection_largest is None) != (args.healthy_max_pages is None):
        raise ValueError(
            "selection_largest and healthy_max_pages must be set together"
        )
    if args.heartbeat_s <= 0:
        raise ValueError("heartbeat_s must be positive")
    if args.page_partitions <= 0:
        raise ValueError("page_partitions must be positive")
    records = run_matrix(args)
    print(json.dumps({"completed_arms": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run_matrix"]
