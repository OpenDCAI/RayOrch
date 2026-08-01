"""生成 MinerU 四模式 benchmark 命令清单。

该工具不自动并发启动 GPU 作业；它生成顺序 shell commands，避免不同模式抢占同一组 H20。
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def build_commands(args: argparse.Namespace) -> list[dict[str, str | int]]:
    """为每次 repeat 生成 V3 elastic/parent/Ray Data/native 命令。"""

    root = Path(args.output_root).resolve()
    commands: list[dict[str, str | int]] = []
    engines = ("v3_elastic", "v3_parent_bound", "ray_data", "native")
    for repeat in range(args.repeats):
        run = f"r{repeat + 1}"
        common = [
            "--limit",
            str(args.limit),
            "--replicas",
            str(args.replicas),
            "--batch-size",
            str(args.model_batch_size),
        ]
        by_engine: dict[str, dict[str, str | int]] = {}
        for mode in ("elastic", "parent_bound"):
            prefix = root / "v3" / mode / run
            command = [
                "python",
                "-u",
                "-m",
                "rayorch.experimental.multigrain_v3.benchmark.mineru",
                "--mode",
                mode,
                *common,
                "--microbatch-size",
                str(args.microbatch_size),
                "--max-inflight-arenas",
                str(args.max_inflight),
                "--output-dir",
                str(prefix / "outputs"),
                "--artifact-dir",
                str(prefix / "artifacts"),
                "--result-jsonl",
                str(root / "results.jsonl"),
            ]
            by_engine[f"v3_{mode}"] = {
                "engine": f"v3_{mode}",
                "repeat": repeat + 1,
                "command": shlex.join(command),
            }
        prefix = root / "ray_data" / run
        by_engine["ray_data"] = {
            "engine": "ray_data",
            "repeat": repeat + 1,
            "command": shlex.join(
                    [
                        "python",
                        "-u",
                        "-m",
                        (
                            "rayorch.experimental.multigrain_v3."
                            "benchmark.mineru_ray_data"
                        ),
                        *common,
                        "--output-dir",
                        str(prefix / "outputs"),
                        "--artifact-dir",
                        str(prefix / "artifacts"),
                        "--result-jsonl",
                        str(root / "results.jsonl"),
                    ]
                ),
        }
        prefix = root / "native" / run
        by_engine["native"] = {
            "engine": "native",
            "repeat": repeat + 1,
            "command": shlex.join(
                    [
                        "python",
                        "-u",
                        "-m",
                        (
                            "rayorch.experimental.multigrain_v3."
                            "benchmark.mineru_native"
                        ),
                        "--limit",
                        str(args.limit),
                        "--replicas",
                        str(args.replicas),
                        "--batch-size",
                        str(args.native_pdf_batch_size),
                        "--inflight",
                        str(args.max_inflight),
                        "--output-dir",
                        str(prefix / "outputs"),
                        "--artifact-dir",
                        str(prefix / "artifacts"),
                        "--result-jsonl",
                        str(root / "results.jsonl"),
                    ]
                ),
        }
        order = (
            engines
            if repeat % 2 == 0
            else tuple(reversed(engines))
        )
        commands.extend(by_engine[engine] for engine in order)
    return commands


def build_parser() -> argparse.ArgumentParser:
    """构造 MinerU matrix command generator CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=368)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--native-pdf-batch-size", type=int, default=24)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-inflight", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    """打印 JSON command manifest。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(build_commands(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
