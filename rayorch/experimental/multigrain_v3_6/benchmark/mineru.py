"""使用 v3.6 compiler/runtime 回归真实 Flash-MinerU 368-PDF workload。

业务 UDF 与 V3 runner 完全复用；本模块只替换 ``RayModule + F.*`` authoring、
compiler 和 executor。这样性能差异只来自框架，而不是 render、VLM 或 assemble 内核。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, cast
from urllib.parse import urlsplit, urlunsplit

from ...multigrain_v3.benchmark.mineru import (
    DEFAULT_FLASH_REPO,
    DEFAULT_MODEL,
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
    ResourceSampler,
    _is_hdfs_uri,
    _pdf_stem,
    _runtime_env,
)
from ...multigrain_v3.benchmark.profile_events import ProfileEventWriter
from .. import F, Executor, Pipeline, Port, RayModule


V3_GOLDEN_MEASURED_S = 587.781


def _discover_pdfs(pdf_dir: str) -> list[str]:
    """按确定顺序返回一个输入目录中的全部 PDF。"""

    if _is_hdfs_uri(pdf_dir):
        from pyarrow import fs as pyarrow_fs

        filesystem, root_path = pyarrow_fs.FileSystem.from_uri(pdf_dir)
        root_info = filesystem.get_file_info(root_path)
        if root_info.type != pyarrow_fs.FileType.Directory:
            raise NotADirectoryError(
                f"PDF directory does not exist: {pdf_dir}"
            )
        selector = pyarrow_fs.FileSelector(
            root_path,
            recursive=True,
            allow_not_found=False,
        )
        parsed = urlsplit(pdf_dir)
        pdfs = sorted(
            urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    f"/{info.path.lstrip('/')}",
                    "",
                    "",
                )
            )
            for info in filesystem.get_file_info(selector)
            if info.type == pyarrow_fs.FileType.File
            and Path(info.path).suffix.lower() == ".pdf"
        )
        if not pdfs:
            raise FileNotFoundError(f"no PDFs found under {pdf_dir}")
        stems = [_pdf_stem(path) for path in pdfs]
        duplicate_stems = sorted(
            stem for stem in set(stems) if stems.count(stem) > 1
        )
        if duplicate_stems:
            raise ValueError(
                "HDFS PDF tree contains duplicate output stems: "
                + ", ".join(duplicate_stems[:5])
            )
        return pdfs

    root = Path(pdf_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"PDF directory does not exist: {root}")
    pdfs = [str(path) for path in sorted(root.glob("*.pdf"))]
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {root}")
    return pdfs


def _discover_pdf_inputs(
    pdf_dirs: str | Iterable[str],
    limits: Iterable[int] | None = None,
) -> list[str]:
    """合并多个输入目录并拒绝会覆盖同一输出目录的重复 stem。"""

    roots = [pdf_dirs] if isinstance(pdf_dirs, str) else list(pdf_dirs)
    if not roots:
        raise ValueError("at least one PDF input directory is required")
    root_limits = list(limits) if limits is not None else [None] * len(roots)
    if len(root_limits) != len(roots):
        raise ValueError("PDF input roots and limits must have equal lengths")
    if any(limit is not None and limit <= 0 for limit in root_limits):
        raise ValueError("per-input PDF limits must be positive")
    discovered = [
        path
        for root, root_limit in zip(roots, root_limits)
        for path in _discover_pdfs(root)[:root_limit]
    ]
    stems = [_pdf_stem(path) for path in discovered]
    duplicate_stems = sorted(
        stem for stem in set(stems) if stems.count(stem) > 1
    )
    if duplicate_stems:
        raise ValueError(
            "combined PDF inputs contain duplicate output stems: "
            + ", ".join(duplicate_stems[:5])
        )
    return discovered


def _output_is_complete(pdf_path: str, output_dir: str) -> bool:
    """校验 MinerU 文档归并后写出的两个最终文件。"""

    stem = _pdf_stem(pdf_path)
    if _is_hdfs_uri(output_dir):
        from pyarrow import fs as pyarrow_fs

        filesystem, root = pyarrow_fs.FileSystem.from_uri(output_dir)
        document_root = f"{root.rstrip('/')}/{stem}"
        markdown_path = f"{document_root}/vlm/{stem}.md"
        layout_path = f"{document_root}/vlm/layout.json"
        success_path = f"{document_root}/_SUCCESS"
        infos = filesystem.get_file_info(
            [markdown_path, layout_path, success_path]
        )
        if any(info.type != pyarrow_fs.FileType.File for info in infos):
            return False
        try:
            with filesystem.open_input_file(layout_path) as source:
                payload = json.loads(source.read().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and isinstance(
            payload.get("pdf_info"), list
        )

    document_dir = Path(output_dir).expanduser().resolve() / stem / "vlm"
    markdown = document_dir / f"{stem}.md"
    layout = document_dir / "layout.json"
    if (document_dir / ".rayorch-incomplete").exists():
        return False
    if not markdown.is_file() or not layout.is_file():
        return False
    try:
        payload = json.loads(layout.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("pdf_info"), list)


def _select_pdfs(
    pdf_dirs: str | Iterable[str],
    output_dir: str,
    *,
    limit: int,
    skip_existing: bool,
    per_input_limits: Iterable[int] | None = None,
) -> tuple[list[str], int, int]:
    """选择至多 ``limit`` 个待处理 PDF，并返回断点恢复统计。"""

    if limit <= 0:
        raise ValueError("limit must be positive")
    discovered = _discover_pdf_inputs(pdf_dirs, per_input_limits)
    pending = (
        [
            path
            for path in discovered
            if not _output_is_complete(path, output_dir)
        ]
        if skip_existing
        else discovered
    )
    return pending[:limit], len(discovered) - len(pending), len(discovered)


def _write_artifacts(
    args: argparse.Namespace,
    payload: dict[str, Any],
    gpu_samples: Iterable[Any],
) -> None:
    """持久化一次 benchmark 摘要、资源轨迹和结果记录。"""

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


class MinerUV36Pipeline(Pipeline):
    """显式建模 PDF→Page→OCR→Document 的真实 MinerU Pipeline。"""

    def __init__(
        self,
        *,
        output_dir: str,
        mode: str,
        model: str,
        replicas: int,
        batch_size: int,
        gpu_memory_utilization: float,
        render_replicas: int,
        reduce_replicas: int,
        runtime_env: dict[str, Any],
        actor_resources: dict[str, float] | None = None,
        actor_scheduling_strategy: str | None = None,
        gpus_per_ocr_actor: float = 1.0,
        spool_batches: bool = False,
        remote_output_dir: str | None = None,
        profile_dir: str | None = None,
        profile_system: str = "rayorch",
    ) -> None:
        """冻结四个计算 Call 的 actor、batch 和资源配置。"""

        if mode not in {"elastic", "parent_bound"}:
            raise ValueError("mode must be elastic or parent_bound")
        resource_options = (
            {"resources": dict(actor_resources)} if actor_resources else {}
        )
        if actor_scheduling_strategy:
            resource_options["scheduling_strategy"] = actor_scheduling_strategy
        self.render = (
            RayModule(MinerUPdfToPages)
            .pre_init(
                dpi=200,
                profile_dir=profile_dir,
                profile_system=profile_system,
            )
            .ray_options(
                replicas=render_replicas,
                batch_size=1,
                num_cpus=1,
                runtime_env=runtime_env,
                **resource_options,
            )
        )
        self.ocr = (
            RayModule(MinerUVlmOcrPage)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
                profile_dir=profile_dir,
                profile_system=profile_system,
            )
            .ray_options(
                replicas=replicas,
                batch_size=batch_size,
                batch_scope=mode,
                num_gpus=gpus_per_ocr_actor,
                num_cpus=1,
                runtime_env=runtime_env,
                **resource_options,
            )
        )
        self.metadata = RayModule(PdfMetadata).pre_init(
            profile_dir=profile_dir,
            profile_system=profile_system,
        ).ray_options(
            replicas=1,
            batch_size=32,
            num_cpus=1,
            runtime_env=runtime_env,
            **resource_options,
        )
        self.assemble = (
            RayModule(MinerUAssembleDoc)
            .pre_init(
                output_dir=output_dir,
                spool_batches=spool_batches,
                remote_output_dir=remote_output_dir,
                profile_dir=profile_dir,
                profile_system=profile_system,
            )
            .ray_options(
                replicas=reduce_replicas,
                batch_size=4,
                num_cpus=1,
                runtime_env=runtime_env,
                **resource_options,
            )
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        pdfs: Port,
    ):
        """以 Port 关系声明 1:M、跨 parent page compute 和 ordered M:1。"""

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
    """运行一次 v3.6 MinerU gate，并写入摘要和 GPU samples。"""

    import ray  # pyright: ignore[reportMissingImports]

    flash_repo = str(Path(args.flash_repo).expanduser().resolve())
    raw_pdf_dirs = getattr(args, "pdf_dirs", None)
    if raw_pdf_dirs is None:
        raw_pdf_dirs = [getattr(args, "pdf_dir", None) or flash_repo]
    elif isinstance(raw_pdf_dirs, str):
        raw_pdf_dirs = [raw_pdf_dirs]
    pdf_dirs = [
        str(value)
        if _is_hdfs_uri(str(value))
        else str(Path(str(value)).expanduser().resolve())
        for value in raw_pdf_dirs
    ]
    output_dir = (
        str(args.output_dir)
        if _is_hdfs_uri(str(args.output_dir))
        else str(Path(args.output_dir).expanduser().resolve())
    )
    skip_existing = bool(getattr(args, "skip_existing", False))
    completion_dir = str(getattr(args, "completion_dir", None) or output_dir)
    pdf_limits = getattr(args, "pdf_limits", None)
    pdfs, skipped_existing, discovered_pdfs = _select_pdfs(
        pdf_dirs,
        completion_dir,
        limit=args.limit,
        skip_existing=skip_existing,
        per_input_limits=pdf_limits,
    )
    if not pdfs:
        payload = {
            "engine": "multigrain_v3_6",
            "status": "already_complete",
            "checked_at_unix_s": time.time(),
            "mode": args.mode,
            "discovered_pdfs": discovered_pdfs,
            "skipped_existing": skipped_existing,
            "n_pdf": 0,
            "pages": 0,
            "docs": 0,
            "pdf_dir": pdf_dirs[0] if len(pdf_dirs) == 1 else pdf_dirs,
            "pdf_dirs": pdf_dirs,
            "output_dir": output_dir,
        }
        artifact_root = Path(args.artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "resume-status.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload

    (Path(args.artifact_dir) / "resume-status.json").unlink(missing_ok=True)

    cold_start_epoch_s = float(
        getattr(args, "cold_e2e_start_epoch_s", 0.0) or time.time()
    )
    cold_start_monotonic_s = float(
        getattr(args, "cold_e2e_start_monotonic_s", 0.0)
        or time.perf_counter()
    )
    profile = ProfileEventWriter(
        getattr(args, "profile_driver_dir", None)
        or getattr(args, "profile_dir", None),
        system=getattr(args, "profile_system", "rayorch"),
        stage="driver",
        role="collect",
    )
    profile.emit(
        {
            "type": "milestone",
            "name": "cold_e2e_started",
            "epoch_s": cold_start_epoch_s,
            "monotonic_s": cold_start_monotonic_s,
        }
    )

    runtime_env = _runtime_env(flash_repo)
    pipeline = MinerUV36Pipeline(
        output_dir=output_dir,
        mode=args.mode,
        model=args.model,
        replicas=args.replicas,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        render_replicas=args.render_replicas,
        reduce_replicas=args.reduce_replicas,
        runtime_env=runtime_env,
        actor_resources=getattr(args, "actor_resources", None),
        actor_scheduling_strategy=getattr(
            args, "actor_scheduling_strategy", None
        ),
        gpus_per_ocr_actor=getattr(args, "gpus_per_ocr_actor", 1.0),
        spool_batches=bool(getattr(args, "spool_batches", False)),
        remote_output_dir=getattr(args, "remote_output_dir", None),
        profile_dir=getattr(args, "profile_dir", None),
        profile_system=getattr(args, "profile_system", "rayorch"),
    )
    compiled = pipeline.compile()
    profile.emit(
        {
            "type": "milestone",
            "name": "plan_built",
            "epoch_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
    )
    ocr_call = next(
        call
        for call, spec in compiled.logical.calls.items()
        if spec.udf.target is MinerUVlmOcrPage
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
    profile.emit(
        {
            "type": "milestone",
            "name": "ray_initialized",
            "epoch_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
    )

    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    result = None
    end_to_end = 0.0
    try:
        with Executor(compiled) as executor:
            startup_s = time.perf_counter() - started
            profile.emit(
                {
                    "type": "milestone",
                    "name": "actors_ready",
                    "epoch_s": time.time(),
                    "monotonic_s": time.perf_counter(),
                }
            )
            result = executor.run(
                pdfs,
                microbatch_size=args.microbatch_size,
                max_active_microbatches=args.max_active_microbatches,
            )
        end_to_end = time.perf_counter() - started
        profile.emit(
            {
                "type": "milestone",
                "name": "executor_materialized",
                "epoch_s": time.time(),
                "monotonic_s": time.perf_counter(),
            }
        )
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
        if started_ray_here:
            ray.shutdown()
    if result is None:  # pragma: no cover - exception path exits in the try block
        raise RuntimeError("v3.6 MinerU run produced no result")

    outputs = list(cast(Iterable[dict[str, Any]], result.outputs))
    pages = sum(int(output["pages"]) for output in outputs)
    failed_docs = sorted(
        str(output["pdf"])
        for output in outputs
        if output.get("status") != "completed"
    )
    heavy = next(
        metrics for metrics in result.calls
        if metrics.call_index == ocr_call.value
    )
    gpu_peak = tuple(
        max(
            (
                sample.memory_used[index]
                for sample in gpu_samples
                if index < len(sample.memory_used)
            ),
            default=0,
        )
        for index in range(args.replicas)
    )
    measured = result.elapsed_s
    payload = {
        "engine": "multigrain_v3_6",
        "status": "completed",
        "mode": args.mode,
        "discovered_pdfs": discovered_pdfs,
        "skipped_existing": skipped_existing,
        "n_pdf": len(pdfs),
        "pages": pages,
        "docs": len(outputs),
        "failed_doc_count": len(failed_docs),
        "failed_docs": failed_docs,
        "batch_size": args.batch_size,
        "replicas": args.replicas,
        "gpus_per_ocr_actor": getattr(args, "gpus_per_ocr_actor", 1.0),
        "microbatch_size": args.microbatch_size,
        "max_active_microbatches": args.max_active_microbatches,
        "batch_policy": "immediate_work_conserving",
        "startup_s": round(startup_s, 3),
        "measured_wall_s": round(measured, 3),
        "end_to_end_wall_s": round(end_to_end, 3),
        "pages_per_s": round(pages / max(measured, 1e-9), 4),
        "rpc_count": result.rpc_count,
        "ocr_rpc_count": heavy.rpcs,
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
        "ratio_vs_v3_golden": (
            measured / V3_GOLDEN_MEASURED_S if len(pdfs) == 368 else None
        ),
        "within_v3_golden_5_percent": (
            measured <= V3_GOLDEN_MEASURED_S * 1.05
            if len(pdfs) == 368
            else None
        ),
        "pdf_dir": pdf_dirs[0] if len(pdf_dirs) == 1 else pdf_dirs,
        "pdf_dirs": pdf_dirs,
        "output_dir": output_dir,
        "profile_clock": {
            "e2e_start_epoch_s": cold_start_epoch_s,
            "e2e_start_monotonic_s": cold_start_monotonic_s,
            "benchmark_end_epoch_s": time.time(),
            "benchmark_end_monotonic_s": time.perf_counter(),
            "absolute_anchor_estimated": False,
        },
    }

    _write_artifacts(args, payload, gpu_samples)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造 v3.6 MinerU 4/48/368 通用 CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("elastic", "parent_bound"),
        default="elastic",
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--gpus-per-ocr-actor", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-active-microbatches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help=(
            "directory containing input PDFs; defaults to --flash-repo for "
            "backward compatibility"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip PDFs that already have both Markdown and layout outputs",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--profile-system", default="rayorch")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、执行 gate，并打印摘要 JSON。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())


__all__ = [
    "MinerUV36Pipeline",
    "build_parser",
    "run_benchmark",
]
