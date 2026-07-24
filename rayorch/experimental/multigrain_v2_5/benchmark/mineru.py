"""Real 4-GPU Flash-MinerU benchmark through the V2.5 public Pipeline API."""

from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..api import Expand, Map, Pipeline, Reduce
from ..executor import Executor


DEFAULT_FLASH_REPO = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/workspace/Flash-mineru"
)
DEFAULT_MODEL = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/model/MinerU2.5-2509-1.2B"
)
BASELINE_WALL_S = 891.98
PREVIOUS_V2_WALL_S = 519.95


class MinerUPdfToPages:
    """Render each PDF into first-class page records on CPU actors."""

    def __init__(self, dpi: int = 200) -> None:
        self.dpi = dpi

    def run(self, pdf_paths: list[str]) -> list[list[dict[str, Any]]]:
        from flash_mineru.mineru_core.utils.pdf_image_tools import (
            load_images_from_pdf,
        )

        groups = []
        for path in pdf_paths:
            with open(path, "rb") as handle:
                pdf_bytes = handle.read()
            images, pdf_doc = load_images_from_pdf(pdf_bytes, dpi=self.dpi)
            pages = []
            for page_id, image in enumerate(images):
                width, height = map(int, pdf_doc[page_id].get_size())
                pages.append(
                    {
                        "pdf_path": path,
                        "page_id": page_id,
                        "img_pil": image["img_pil"],
                        "scale": image.get("scale"),
                        "page_width": width,
                        "page_height": height,
                        "pdf_len": len(images),
                    }
                )
            pdf_doc.close()
            groups.append(pages)
        return groups


class MinerUVlmOcrPage:
    """Run the real MinerU vLLM page extraction on one GPU actor."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        from mineru_vl_utils import MinerUClient
        from vllm import LLM

        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=self.llm,
        )

    def run(self, pages: list[dict[str, Any]]) -> list[Any]:
        return list(
            self.client.batch_two_step_extract(
                images=[page["img_pil"] for page in pages]
            )
        )


class MinerUAssembleDoc:
    """Assemble ordered OCR contents with the aligned original page fiber."""

    def __init__(
        self,
        output_dir: str,
        parse_method: str = "vlm",
    ) -> None:
        self.output_dir = output_dir
        self.parse_method = parse_method

    def run(
        self,
        pdf_paths: list[str],
        grouped_contents: list[list[Any]],
        grouped_pages: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        from flash_mineru.mineru_core.data.data_reader_writer import (
            FileBasedDataWriter,
        )
        from flash_mineru.mineru_core.engine.model_output_to_middle_json import (
            result_to_middle_json,
        )
        from flash_mineru.mineru_core.engine.vlm_middle_json_mkcontent import (
            union_make as vlm_union_make,
        )
        from flash_mineru.mineru_core.utils.enum_class import MakeMode

        outputs = []
        for pdf_path, contents, pages in zip(
            pdf_paths,
            grouped_contents,
            grouped_pages,
        ):
            stem = Path(pdf_path).stem
            markdown_dir = Path(self.output_dir) / stem / self.parse_method
            image_dir = markdown_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_writer = FileBasedDataWriter(str(image_dir))
            markdown_writer = FileBasedDataWriter(str(markdown_dir))
            middle = result_to_middle_json(
                list(contents),
                list(pages),
                image_writer,
            )
            markdown = vlm_union_make(
                middle["pdf_info"],
                MakeMode.MM_MD,
                "images",
            )
            markdown_writer.write_string(f"{stem}.md", markdown)
            (markdown_dir / "layout.json").write_text(
                json.dumps(middle, indent=2),
                encoding="utf-8",
            )
            outputs.append(
                {
                    "pdf": stem,
                    "md_path": str(
                        (markdown_dir / f"{stem}.md").resolve()
                    ),
                    "chars": len(markdown),
                    "pages": len(pages),
                }
            )
        return outputs


class MinerUV25Pipeline(Pipeline):
    def __init__(
        self,
        *,
        output_dir: str,
        mode: str,
        model: str,
        replicas: int,
        batch_size: int,
        max_batch_wait_ms: float,
        gpu_memory_utilization: float,
        render_replicas: int,
        reduce_replicas: int,
        runtime_env: dict[str, Any],
    ) -> None:
        scope = "elastic" if mode == "elastic" else "parent_bound"
        self.render = (
            Expand(MinerUPdfToPages)
            .pre_init(dpi=200)
            .ray_options(
                replicas=render_replicas,
                batch_size=1,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )
        self.ocr = (
            Map(MinerUVlmOcrPage)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            .ray_options(
                replicas=replicas,
                batch_size=batch_size,
                max_batch_wait_ms=max_batch_wait_ms,
                batch_scope=scope,
                num_gpus=1.0,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )
        self.assemble = (
            Reduce(MinerUAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(
                replicas=reduce_replicas,
                batch_size=4,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )

    def forward(self, pdfs):
        pages = self.render(pdfs)
        contents = self.ocr(pages)
        return self.assemble(
            anchor=pdfs,
            members=contents,
            pages=pages,
        )


@dataclass(frozen=True, slots=True)
class RssSummary:
    driver_start: int
    driver_end: int
    driver_peak: int
    gpu_memory_peak: tuple[int, ...]
    gpu_samples: tuple["GpuSample", ...]


@dataclass(frozen=True, slots=True)
class GpuDeviceSample:
    index: int
    utilization_percent: int | None
    memory_used: int
    memory_total: int


@dataclass(frozen=True, slots=True)
class GpuSample:
    wall_time_s: float
    monotonic_s: float
    devices: tuple[GpuDeviceSample, ...]


class RssSampler:
    """Observation-only sampler; it never touches executor control state."""

    def __init__(self, interval_s: float = 1.0) -> None:
        import psutil

        self.interval_s = interval_s
        self.process = psutil.Process()
        self.start_rss = int(self.process.memory_info().rss)
        self.end_rss = self.start_rss
        self.peak_rss = self.start_rss
        self.gpu_peak: list[int] = []
        self.gpu_samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="mineru-rss-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> RssSummary:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 2))
        self.end_rss = int(self.process.memory_info().rss)
        self.peak_rss = max(self.peak_rss, self.end_rss)
        return RssSummary(
            self.start_rss,
            self.end_rss,
            self.peak_rss,
            tuple(self.gpu_peak),
            tuple(self.gpu_samples),
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.peak_rss = max(
                    self.peak_rss,
                    int(self.process.memory_info().rss),
                )
                devices = _gpu_device_samples()
                self.gpu_samples.append(
                    GpuSample(
                        wall_time_s=time.time(),
                        monotonic_s=time.monotonic(),
                        devices=devices,
                    )
                )
                if len(self.gpu_peak) < len(devices):
                    self.gpu_peak.extend(
                        [0] * (len(devices) - len(self.gpu_peak))
                    )
                for device in devices:
                    self.gpu_peak[device.index] = max(
                        self.gpu_peak[device.index],
                        device.memory_used,
                    )
            except Exception:
                continue


def _gpu_device_samples() -> tuple[GpuDeviceSample, ...]:
    try:
        import pynvml

        pynvml.nvmlInit()
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                utilization = int(
                    pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                )
            except Exception:
                utilization = None
            devices.append(
                GpuDeviceSample(
                    index=index,
                    utilization_percent=utilization,
                    memory_used=int(memory.used),
                    memory_total=int(memory.total),
                )
            )
        values = tuple(devices)
        pynvml.nvmlShutdown()
        return values
    except Exception:
        return ()


def _runtime_env(flash_repo: str) -> dict[str, Any]:
    current = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        path for path in (flash_repo, os.getcwd(), current) if path
    )
    return {"env_vars": {"PYTHONPATH": pythonpath}}


def _configure_timeline_profiling() -> None:
    os.environ.setdefault("RAY_PROFILING", "1")
    # Ray 2.50 documents zero here, but zero disables profile-event
    # reporting in the installed runtime. A short positive interval
    # preserves task spans for ray.timeline().
    if int(os.environ.get("RAY_task_events_report_interval_ms", "100")) <= 0:
        os.environ["RAY_task_events_report_interval_ms"] = "100"
    else:
        os.environ.setdefault(
            "RAY_task_events_report_interval_ms",
            "100",
        )


def run_mineru_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.timeline_dir:
        _configure_timeline_profiling()

    import ray

    flash_repo = os.path.abspath(args.flash_repo)
    pdfs = sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {flash_repo}")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(flash_repo, f"outputs_v2_5_{args.mode}_{args.limit}")
    )
    if not ray.is_initialized():
        ray.init(
            address="local",
            ignore_reinit_error=True,
            num_cpus=args.num_cpus,
            num_gpus=args.replicas,
            object_store_memory=int(args.object_store_gb * 1024**3),
            include_dashboard=False,
        )

    pipeline = MinerUV25Pipeline(
        output_dir=output_dir,
        mode=args.mode,
        model=args.model,
        replicas=args.replicas,
        batch_size=args.batch_size,
        max_batch_wait_ms=args.max_batch_wait_ms,
        gpu_memory_utilization=args.gpu_memory_utilization,
        render_replicas=args.render_replicas,
        reduce_replicas=args.reduce_replicas,
        runtime_env=_runtime_env(flash_repo),
    )
    sampler = RssSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    try:
        result = Executor(
            pipeline,
            microbatch_size=args.microbatch_size,
            max_inflight_arenas=args.max_inflight_arenas,
        ).run(pdfs)
    finally:
        rss = sampler.stop()
    outputs = result.get()
    end_to_end = time.perf_counter() - started
    artifact_paths = _write_observation_artifacts(
        args.timeline_dir,
        result.timeline,
        rss.gpu_samples,
    )

    pages = sum(int(output.get("pages", 0)) for output in outputs)
    metrics = result.metrics
    map_events = [
        event
        for event in result.timeline
        if event.node == 2 and event.worker_started_at is not None
    ]
    map_start = min(
        (event.worker_started_at for event in map_events),
        default=0.0,
    )
    map_stop = max(
        (event.worker_finished_at for event in map_events),
        default=0.0,
    )
    map_busy = sum(
        event.worker_finished_at - event.worker_started_at
        for event in map_events
    )
    map_capacity = max(map_stop - map_start, 0.0) * args.replicas
    map_bubble = (
        max(0.0, 1.0 - map_busy / map_capacity)
        if map_capacity
        else 0.0
    )
    payload = {
        "mode": args.mode,
        "n_pdf": len(pdfs),
        "pages": pages,
        "docs": len(outputs),
        "batch_size": args.batch_size,
        "max_batch_wait_ms": args.max_batch_wait_ms,
        "replicas": args.replicas,
        "microbatch_size": args.microbatch_size,
        "max_inflight_arenas": args.max_inflight_arenas,
        "startup_s": round(metrics["startup_time_s"], 3),
        "measured_wall_s": round(metrics["measured_wall_time_s"], 3),
        "end_to_end_wall_s": round(end_to_end, 3),
        "pdf_per_s": round(
            len(pdfs) / metrics["measured_wall_time_s"],
            4,
        ),
        "pages_per_s": round(
            pages / metrics["measured_wall_time_s"],
            4,
        ),
        "end_to_end_pdf_per_s": round(len(pdfs) / end_to_end, 4),
        "end_to_end_pages_per_s": round(pages / end_to_end, 4),
        "rpc_count": metrics["rpc_count"],
        "grains_per_rpc": metrics["grains_per_rpc"],
        "batch_fill_ratio": metrics["batch_fill_ratio"],
        "tail_rpc_fraction": metrics["tail_or_isolation_rpc_fraction"],
        "ocr_bubble_ratio": round(map_bubble, 4),
        "parent_p50_s": metrics["parent_completion_p50_s"],
        "parent_p95_s": metrics["parent_completion_p95_s"],
        "parent_p99_s": metrics["parent_completion_p99_s"],
        "live_blocks_at_delivery": metrics["live_blocks_at_delivery"],
        "live_blocks_high_watermark": metrics["live_blocks_high_watermark"],
        "active_arenas_high_watermark": metrics.get(
            "active_arenas_high_watermark",
            1.0,
        ),
        "live_blocks_across_arenas_high_watermark": metrics.get(
            "live_blocks_across_arenas_high_watermark",
            metrics["live_blocks_high_watermark"],
        ),
        "driver_rss_start": rss.driver_start,
        "driver_rss_end": rss.driver_end,
        "driver_rss_peak": rss.driver_peak,
        "worker_rss_peak": metrics["worker_rss_peak_bytes"],
        "gpu_memory_peak": rss.gpu_memory_peak,
        "observation_artifacts": artifact_paths,
        "baseline_s": BASELINE_WALL_S,
        "previous_v2_s": PREVIOUS_V2_WALL_S,
        "speedup_vs_baseline": (
            BASELINE_WALL_S / metrics["measured_wall_time_s"]
            if len(pdfs) == 368
            else None
        ),
        "speedup_vs_previous_v2": (
            PREVIOUS_V2_WALL_S / metrics["measured_wall_time_s"]
            if len(pdfs) == 368
            else None
        ),
        "output_dir": output_dir,
    }
    result_path = Path(args.result_jsonl)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def _write_observation_artifacts(
    timeline_dir: str,
    dispatch_timeline: tuple[Any, ...],
    gpu_samples: tuple[GpuSample, ...],
) -> dict[str, Any]:
    if not timeline_dir:
        return {}

    import ray

    root = Path(timeline_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dispatch_path = root / "dispatch_timeline.jsonl"
    with dispatch_path.open("w", encoding="utf-8") as handle:
        for event in dispatch_timeline:
            handle.write(
                json.dumps(asdict(event), ensure_ascii=False) + "\n"
            )
    gpu_path = root / "gpu_samples.jsonl"
    with gpu_path.open("w", encoding="utf-8") as handle:
        for sample in gpu_samples:
            handle.write(
                json.dumps(asdict(sample), ensure_ascii=False) + "\n"
            )
    ray_path = root / "ray_timeline.json"
    ray_events = ray.timeline()
    ray_path.write_text(json.dumps(ray_events), encoding="utf-8")
    return {
        "ray_timeline": str(ray_path),
        "dispatch_timeline": str(dispatch_path),
        "gpu_samples": str(gpu_path),
        "ray_timeline_events": len(ray_events),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["elastic", "parent_bound"], default="elastic")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-inflight-arenas", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-wait-ms", type=float, default=5.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=60.0)
    parser.add_argument("--rss-interval-s", type=float, default=1.0)
    parser.add_argument(
        "--timeline-dir",
        default="",
        help=(
            "write ray_timeline.json, dispatch_timeline.jsonl, and "
            "gpu_samples.jsonl; worker profiling is enabled before ray.init"
        ),
    )
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--result-jsonl",
        default=os.path.join(DEFAULT_FLASH_REPO, "v2_5_mineru_results.jsonl"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_mineru_benchmark(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0
