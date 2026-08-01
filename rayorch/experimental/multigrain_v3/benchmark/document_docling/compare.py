"""Docling V3、裸 Ray Data 与 native 整文档 baseline 的 compare CLI。"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

from .native import run_native
from .ray_data import run_ray_data
from .v3 import run_v3


def _token_set(text: str) -> set[str]:
    """构造用于轻量 correctness 比较的小写 whitespace token set。"""

    return set(text.lower().split())


def _jaccard(left: str, right: str) -> float:
    """计算两个 Markdown token sets 的 Jaccard。"""

    lhs = _token_set(left)
    rhs = _token_set(right)
    union = lhs | rhs
    return len(lhs & rhs) / len(union) if union else 1.0


def _paths(args: argparse.Namespace) -> list[str]:
    """从显式 paths 或目录中确定有序 PDF 输入。"""

    if args.paths:
        return [str(Path(path).resolve()) for path in args.paths]
    return sorted(
        glob.glob(os.path.join(args.pdf_dir, "*.pdf"))
    )[: args.limit]


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    """顺序运行 V3、Ray Data 和 native，并比较输出。"""

    import ray

    paths = _paths(args)
    if not paths:
        raise FileNotFoundError("no PDF inputs")
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=args.num_cpus,
            include_dashboard=False,
        )
    common = {
        "scale": args.scale,
        "page_replicas": args.page_replicas,
        "page_batch_size": args.page_batch_size,
        "device": args.device,
        "num_threads": args.num_threads,
    }
    started = time.perf_counter()
    v3_result = run_v3(
        paths,
        **common,
        microbatch_size=args.microbatch_size,
        max_inflight_arenas=args.max_inflight_arenas,
    )
    v3_outputs = v3_result.get()
    v3_wall = time.perf_counter() - started

    started = time.perf_counter()
    ray_data_outputs = run_ray_data(
        paths,
        render_replicas=1,
        **common,
    )
    ray_data_wall = time.perf_counter() - started

    native_outputs: tuple[dict[str, Any], ...] = ()
    native_wall = None
    if not args.skip_native:
        native_outputs, native_wall = run_native(
            paths,
            device=args.device,
            num_threads=args.num_threads,
        )
    parity = [
        _jaccard(v3["markdown"], ray_data["markdown"])
        for v3, ray_data in zip(v3_outputs, ray_data_outputs)
    ]
    native_similarity = (
        [
            _jaccard(v3["markdown"], native["markdown"])
            for v3, native in zip(v3_outputs, native_outputs)
        ]
        if native_outputs
        else []
    )
    return {
        "documents": len(paths),
        "pages": [output["pages"] for output in v3_outputs],
        "v3_wall_s": round(v3_wall, 3),
        "ray_data_wall_s": round(ray_data_wall, 3),
        "native_wall_s": (
            round(native_wall, 3)
            if native_wall is not None
            else None
        ),
        "v3_ray_data_jaccard": parity,
        "v3_native_jaccard": native_similarity,
        "v3_rpc_count": v3_result.metrics["rpc_count"],
        "v3_grains_per_rpc": v3_result.metrics["grains_per_rpc"],
        "v3_batch_fill_ratio": v3_result.metrics["batch_fill_ratio"],
    }


def build_parser() -> argparse.ArgumentParser:
    """构造 Docling compare CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*")
    parser.add_argument("--pdf-dir", default=".")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--page-replicas", type=int, default=1)
    parser.add_argument("--page-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--max-inflight-arenas", type=int, default=2)
    parser.add_argument("--num-cpus", type=int, default=16)
    parser.add_argument("--skip-native", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行 Docling compare 并打印 JSON。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_compare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
