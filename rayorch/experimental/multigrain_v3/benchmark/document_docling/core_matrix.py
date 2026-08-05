"""生成可恢复的 Docling 368-PDF 四臂实验命令清单。

与 ``core_compare`` 的同进程 convenience matrix 不同，本模块让每个
``(repeat, arm)`` 都成为独立 Python 进程：

- 一臂完成后立即写 summary JSON、documents correctness artifact 和共享 JSONL；
- 中途失败只重跑缺失 arm；
- Native 不与 Ray actors/GPU context 共存；
- 四个 repeat 使用 Latin rotation，使每个 arm 各出现一次首位。

命令清单本身不并行执行，避免四臂争抢同一张 H20。
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from .core_compare import ARM_ORDER, BALANCED_ARM_ORDERS


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    """生成四臂 × repeats 的独立进程命令。"""

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    root = Path(args.output_root).resolve()
    commands: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        order = BALANCED_ARM_ORDERS[
            repeat % len(BALANCED_ARM_ORDERS)
        ]
        for position, arm in enumerate(order):
            run_dir = root / f"r{repeat + 1}" / arm
            command = [
                args.python,
                "-u",
                "-m",
                (
                    "rayorch.experimental.multigrain_v3.benchmark."
                    "document_docling.core_compare"
                ),
                "--arm",
                arm,
                "--matrix-repeat",
                str(repeat + 1),
                "--matrix-position",
                str(position + 1),
                "--manifest",
                str(Path(args.manifest).resolve()),
                "--device",
                args.device,
                "--ocr-device",
                args.ocr_device,
                "--num-threads",
                str(args.num_threads),
                "--stage-batch-size",
                str(args.stage_batch_size),
                "--table-core-batch-size",
                str(getattr(args, "table_core_batch_size", 0)),
                "--native-tuned-doc-concurrency",
                str(args.native_tuned_doc_concurrency),
                "--native-tuned-doc-batch-size",
                str(args.native_tuned_doc_batch_size),
                "--parse-replicas",
                str(args.parse_replicas),
                "--layout-replicas",
                str(args.layout_replicas),
                "--ocr-replicas",
                str(args.ocr_replicas),
                "--table-replicas",
                str(args.table_replicas),
                "--reduce-replicas",
                str(args.reduce_replicas),
                "--parse-batch-wait-ms",
                str(args.parse_batch_wait_ms),
                "--stage-batch-wait-ms",
                str(args.stage_batch_wait_ms),
                "--layout-num-gpus",
                str(args.layout_num_gpus),
                "--table-num-gpus",
                str(args.table_num_gpus),
                "--layout-actor-concurrency",
                str(args.layout_actor_concurrency),
                "--ocr-actor-concurrency",
                str(args.ocr_actor_concurrency),
                "--ocr-batch-mode",
                getattr(args, "ocr_batch_mode", "reference"),
                "--ocr-recognition-batch-size",
                str(getattr(args, "ocr_recognition_batch_size", 6)),
                "--table-actor-concurrency",
                str(args.table_actor_concurrency),
                "--table-batch-mode",
                getattr(args, "table_batch_mode", "reference"),
                "--table-batch-max-jobs",
                str(getattr(args, "table_batch_max_jobs", 16)),
                "--actor-num-cpus",
                str(args.actor_num_cpus),
                "--max-pending-per-actor",
                str(args.max_pending_per_actor),
                "--microbatch-size",
                str(args.microbatch_size),
                "--max-inflight-arenas",
                str(args.max_inflight_arenas),
                "--ray-num-cpus",
                str(args.ray_num_cpus),
                "--ray-num-gpus",
                str(args.ray_num_gpus),
                "--documents-output",
                str(run_dir / "documents.json.gz"),
                "--gpu-monitor-output",
                str(run_dir / "gpu_samples.jsonl"),
                "--output",
                str(run_dir / "summary.json"),
                "--append-jsonl",
                str(root / "results.jsonl"),
            ]
            if args.record_input_sha256:
                command.append("--record-input-sha256")
            if args.four_gpu_native:
                command.append("--four-gpu-native")
            commands.append(
                {
                    "repeat": repeat + 1,
                    "position": position + 1,
                    "arm": arm,
                    "command": shlex.join(command),
                    "summary": str(run_dir / "summary.json"),
                    "documents": str(run_dir / "documents.json.gz"),
                }
            )
    return commands


def build_parser() -> argparse.ArgumentParser:
    """构造 368-PDF 独立进程 matrix command generator。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--record-input-sha256", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--stage-batch-size", type=int, default=8)
    parser.add_argument("--table-core-batch-size", type=int, default=0)
    parser.add_argument(
        "--native-tuned-doc-concurrency",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--native-tuned-doc-batch-size",
        type=int,
        default=24,
    )
    parser.add_argument("--parse-replicas", type=int, default=4)
    parser.add_argument("--layout-replicas", type=int, default=1)
    parser.add_argument("--ocr-replicas", type=int, default=4)
    parser.add_argument("--table-replicas", type=int, default=3)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--parse-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--stage-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--layout-num-gpus", type=float, default=1.0)
    parser.add_argument("--table-num-gpus", type=float, default=1.0)
    parser.add_argument("--layout-actor-concurrency", type=int, default=1)
    parser.add_argument("--ocr-actor-concurrency", type=int, default=1)
    parser.add_argument(
        "--ocr-batch-mode",
        choices=("reference", "recognition_shadow", "recognition_accelerated"),
        default="reference",
    )
    parser.add_argument("--ocr-recognition-batch-size", type=int, default=6)
    parser.add_argument("--table-actor-concurrency", type=int, default=1)
    parser.add_argument(
        "--table-batch-mode",
        choices=(
            "reference",
            "encoder_shadow",
            "encoder_accelerated",
            "decoder_accelerated",
        ),
        default="reference",
    )
    parser.add_argument("--table-batch-max-jobs", type=int, default=16)
    parser.add_argument("--actor-num-cpus", type=float, default=1.0)
    parser.add_argument("--max-pending-per-actor", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-inflight-arenas", type=int, default=3)
    parser.add_argument("--ray-num-cpus", type=int, default=32)
    parser.add_argument("--ray-num-gpus", type=float, default=4.0)
    parser.add_argument(
        "--four-gpu-native",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """打印命令清单 JSON；调用方顺序执行 command 字段。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(build_commands(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
